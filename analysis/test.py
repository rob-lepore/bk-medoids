from collections import Counter
import numpy as np
import json
import pandas as pd
from sklearn.metrics import consensus_score
from bkmedoids import BKmedoids
from graphics import * 
from utils import *
from gridsearch import GridSearch
import os
import time

grids_shift = [
    {  
        "distance_threshold": [1e-9, 1e-8],
        "outlier_threshold": [0.8, 0.9],
        "seed": list(range(2)),
    },
    { 
        "distance_threshold":  [0.03,0.04,0.05],
        "outlier_threshold": [0.65,0.7,0.75],
        "seed": list(range(2)),
    },
    { 
        "distance_threshold": [0.085,0.09,0.095],
        "outlier_threshold": [0.65,0.7,0.75],
        "seed": list(range(2)),
    },
    {
        "distance_threshold": [0.69,0.7,0.71],
        "outlier_threshold": [0.65,0.7,0.75],
        "seed": list(range(2)),
    },
]

grids_scale = [
    {  
        "distance_threshold": [1e-9, 1e-8],
        "outlier_threshold": [0.8, 0.9],
        "seed": list(range(2)),
    },
    { 
     "distance_threshold": [0.008,0.01,0.012],
        "outlier_threshold": [0.65,0.7,0.75],
        "seed": list(range(2))
    },
    { 
        "distance_threshold": [0.04,0.05,0.06],
        "outlier_threshold": [0.6,0.65,0.7],
        "seed": list(range(2)),
    },
    {
        "distance_threshold": [0.095,0.1,0.11],
        "outlier_threshold": [0.6,0.65,0.7],
        "seed": list(range(2)),
    },
]

patterns = ["col_const", "row_const", "scale", "shift"]
models = ["shift", "scale"]

def get_biclusters(bic_info):
    n_bics = bic_info.shape[0]
    M, N = 100, 100
    
    true_rows = [
        np.zeros((M,), dtype=bool) for _ in range(n_bics)
    ]
    true_cols = [
        np.zeros((N,), dtype=bool) for _ in range(n_bics)
    ]
    
    for _, row in bic_info.iterrows():
        bic_id = row["bicluster_id"]
        r_idx = row["rows"]     
        c_idx = row["columns"]  
        true_rows[bic_id][r_idx] = True
        true_cols[bic_id][c_idx] = True
        
    return (true_rows, true_cols)

def model_selection(noise_level):
    
    os.makedirs(f"results/model_selection/noise_{noise_level}", exist_ok=True)
    
    for model in models:
        config = {
            "threshold": 0.01,
            "max_it": 200,
            "empty_penalty": 1,
            "distance_func_args": [],
            "distance_func": model
        }
        
        best_params_list = []
        
        for patt in patterns:
            meta_path = f"datasets/{patt}/noise_{noise_level}/metadata.json"
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            meta_df = pd.DataFrame(metadata)
            
            for ds_id in range(5):
                ds = np.loadtxt(f"datasets/{patt}/noise_{noise_level}/ds_{ds_id}.tsv")
                
                bics_info = meta_df[meta_df["dataset_id"] == ds_id]
                y_true = get_biclusters(bics_info)
                
                grid = grids_shift[noise_level] if model == "shift" else grids_scale[noise_level]
                
                print(patt, ds_id)
                gs = GridSearch(config, grid)
                scores, solutions, times = gs.search(ds, k=8, p = 20, real=y_true)
                
                best = solutions[np.argmin(scores)]
                best_params = {k: best.params.__dict__[k] for k in grid.keys()}

                best_params_list.append({
                    "dataset_id": ds_id,
                    "noise_level": noise_level,
                    "consensus_score": 1 - np.min(scores),
                    "pattern": patt,
                    **best_params
                })
                
        best_df = pd.DataFrame(best_params_list)
        best_df.to_csv(f"results/model_selection/noise_{noise_level}/gridsearch_{model}.csv", index=False)

        common_params = {}
        for col in best_df.columns:
            if col not in ["dataset_id", "noise_level"]:
                most_common = Counter(best_df[col]).most_common(1)[0][0]
                common_params[col] = most_common

        with open(f"results/model_selection/noise_{noise_level}/best_params_{model}.json", "w") as f:
            json.dump({**config, **common_params}, f, indent=2)
    

def test(noise_level):
    n_ds = 25
    start_test = 5
    
    discovered_bics = []
    results = []
    for model in models:
        config_path = f"results/model_selection/noise_{noise_level}/best_params_{model}.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        for patt in patterns:
            meta_path = f"datasets/{patt}/noise_{noise_level}/metadata.json"
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            meta_df = pd.DataFrame(metadata)
            
            os.makedirs(f"results/test/noise_{noise_level}", exist_ok=True)
            
            
            
            for ds_id in range(start_test, n_ds):
                print(model, patt, ds_id)
                ds = np.loadtxt(f"datasets/{patt}/noise_{noise_level}/ds_{ds_id}.tsv")
                
                bics_info = meta_df[meta_df["dataset_id"] == ds_id]
                y_true =  get_biclusters(bics_info)
                
                bk = BKmedoids(ds, k=8, config=config)
            
                start = time.time()
                bk.run()
                ex_time = time.time() - start     
                    
                y_pred = bk.get_biclusters()
                
                for i, (r, c) in enumerate(zip(*y_pred)):
                    bic_rows = np.where(r == True)[0]
                    bic_cols = np.where(c == True)[0]
                    discovered_bics.append({
                        "dataset_id": ds_id,
                        "model": model,
                        "bicluster_id": i,
                        "pattern": patt,
                        "rows": bic_rows,
                        "cols": bic_cols,
                        "n_rows": len(bic_rows),
                        "n_cols": len(bic_cols),
                        "acv": acv(ds[np.ix_(bic_rows, bic_cols)])
                    })
                    
                score = consensus_score(y_pred, y_true)
                relevance = relevance_score(y_pred, y_true)
                recovery = recovery_score(y_pred, y_true)
                correlation_value = 1 - bk.evaluate_solution()
                
                results.append({
                    "dataset_id": ds_id,
                    "pattern": patt,
                    "model": model,
                    "consensus_score": score,
                    "relevance": relevance,
                    "recovery": recovery,
                    "acv": correlation_value,
                    "execution_time": ex_time
                })
                
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/test/noise_{noise_level}/results.csv", index=False)
    discovered_df = pd.DataFrame(discovered_bics)
    discovered_df.to_csv(f"results/test/noise_{noise_level}/discovered.csv", index=False)
                    

for noise_level in [3]:
    model_selection(noise_level)
    test(noise_level)