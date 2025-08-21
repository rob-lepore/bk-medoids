from gridsearch import GridSearch
import numpy as np
import pandas as pd
from utils import *
from sklearn.metrics import consensus_score

K=6

models = ["scale"]

config = {
    "threshold": 0.05,
    "max_it": 200,
    "distance_func_args": [],
    "distance_func": None,
    "distance_threshold":  1e-9,
    "outlier_threshold": 0.9,
}

grid = {
    "seed": list(range(3)),
}

n_ds_list = [3,4,4,4,4]

patterns_paths = ["colconst", "row_const", "shift", "scale", "shift_scale"]
patterns = ["col_const","row_const", "shift", "scale", "shift_scale"] 

y_true = ([],[])
for i in range(6):
    true_rows = np.zeros((600,), dtype=bool)
    true_cols = np.zeros((600,), dtype=bool)
    
    s = 100 * i
    e = 100 * (i+1)
    true_rows[list(range(s,e))] = 1
    true_cols[list(range(s,e))] = 1
    
    y_true[0].append(true_rows)
    y_true[1].append(true_cols)



for model in models:
    config["distance_func"] = model
    stats = []
    
    for i_p, patt_path in enumerate(patterns_paths):
        
        cs = []
        rel = []
        rec = []
        
        for ds_id in range(n_ds_list[i_p]):
            print(patt_path, ds_id)
            
            ds = pd.read_csv(f"ARBic_data/3.six_type/{patt_path}/{ds_id}", sep="\t", header=0, index_col=0).values
            ds[ds == 0.0] = 1e-10
            
            gs = GridSearch(config, grid)
            scores, solutions, _ = gs.search(ds, K, y_true, p=3)
            best = solutions[np.argmin(scores)]
            
            y_pred = best.get_biclusters()
            
            cs.append(consensus_score(y_pred, y_true))
            rel.append(relevance_score(y_pred, y_true))
            rec.append(recovery_score(y_pred, y_true))
        
        stats.append({
            "pattern": patterns[i_p],
            "consensus_score": np.average(cs),
            "consensus_score_std": np.std(cs),
            "relevance": np.average(rel),
            "relevance_std": np.std(rel),
            "recovery": np.average(rec),
            "recovery_std": np.std(rec),
        })
        
        
    # TREND DATASETS
    cs = []
    rel = []
    rec = []
    
    for ds_id in range(4):
        y_true = ([],[])

        with open(f"ARBic_data/3.six_type/trend/{ds_id}.biclusters") as f:
            lines = f.readlines()
            for l in range(0,len(lines),3):
                rows = [int(r) for r in lines[l].strip().split()]
                cols = [int(c) for c in lines[l+1].strip().split()]
                
                true_rows = np.zeros((600,), dtype=bool)
                true_cols = np.zeros((600,), dtype=bool)

                true_rows[rows] = 1
                true_cols[cols] = 1
                
                y_true[0].append(true_rows)
                y_true[1].append(true_cols)
        
        ds = pd.read_csv(f"ARBic_data/3.six_type/trend/{ds_id}", sep="\t", header=0, index_col=0).values
            
        gs = GridSearch(config, grid)
        scores, solutions, _ = gs.search(ds, K, y_true, p=3)
        best = solutions[np.argmin(scores)]
        
        y_pred = best.get_biclusters()
        
        cs.append(consensus_score(y_pred, y_true))
        rel.append(relevance_score(y_pred, y_true))
        rec.append(recovery_score(y_pred, y_true))
    
    stats.append({
        "pattern": "trend",
        "consensus_score": np.average(cs),
        "consensus_score_std": np.std(cs),
        "relevance": np.average(rel),
        "relevance_std": np.std(rel),
        "recovery": np.average(rec),
        "recovery_std": np.std(rec),
    })
    
    pd.DataFrame(stats).to_csv(f"results/comparison/bkmedoids_{model}/stats.csv")
