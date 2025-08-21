import pandas as pd
import numpy as np
import json
from sklearn.metrics import consensus_score
from bicgen import BicGen
from utils import *

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

patterns = ["col_const", "row_const", "scale", "shift", "shift_scale"]
n_ds_list = [3,4,4,4,4]

stats = []

for i,patt in enumerate(patterns):
    n_ds = n_ds_list[i]
    cs = []
    rel = []
    rec = []
    
    for ds_id in range(n_ds):

        with open(f"results/comparison/fabia/{patt}.json") as f:
            res = json.load(f)
        res_df = pd.DataFrame(res)
        res_df["dataset_id"] = res_df["dataset_id"].apply(lambda x: x[0])
        res_df = res_df[res_df["dataset_id"] == ds_id]
            
        y_pred = ([],[])
        for bic in range(6):
            rows = [int(r[4:]) for r in res_df.iloc[bic]["rows"]]
            cols = [int(c[4:]) for c in res_df.iloc[bic]["cols"]]

            pred_rows = np.zeros((600,), dtype=bool)
            pred_cols = np.zeros((600,), dtype=bool)

            pred_rows[rows] = 1
            pred_cols[cols] = 1
            
            y_pred[0].append(pred_rows)
            y_pred[1].append(pred_cols)

        cs.append(consensus_score(y_pred, y_true))
        rel.append(relevance_score(y_pred, y_true))
        rec.append(recovery_score(y_pred, y_true))
    stats.append({
        "pattern": patt,
        "consensus_score": np.average(cs),
        "consensus_score_std": np.std(cs),
        "relevance": np.average(rel),
        "relevance_std": np.std(rel),
        "recovery": np.average(rec),
        "recovery_std": np.std(rec),
    })

## TREND DATASETS

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
    
    with open(f"results/comparison/fabia/trend.json") as f:
        res = json.load(f)
    res_df = pd.DataFrame(res)
    res_df["dataset_id"] = res_df["dataset_id"].apply(lambda x: x[0])
    res_df = res_df[res_df["dataset_id"] == ds_id]
        
    y_pred = ([],[])
    for bic in range(6):
        rows = [int(r[4:]) for r in res_df.iloc[bic]["rows"]]
        cols = [int(c[4:]) for c in res_df.iloc[bic]["cols"]]

        pred_rows = np.zeros((600,), dtype=bool)
        pred_cols = np.zeros((600,), dtype=bool)

        pred_rows[rows] = 1
        pred_cols[cols] = 1
        
        y_pred[0].append(pred_rows)
        y_pred[1].append(pred_cols)
        
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
    
            

pd.DataFrame(stats).to_csv("results/comparison/fabia/stats.csv")
    
