import pandas as pd
from graphics import show_parallel_coordinates
from bkmedoids import BKmedoids
from sklearn.metrics import consensus_score
import numpy as np
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


stats = []

n_ds_list = [3,4,4,4,4]

patterns_paths = ["colconst", "row_const", "shift", "scale", "shift_scale"]
patterns = ["col_const","row_const", "scale", "shift", "shift_scale"]

for i_p, patt in enumerate(patterns_paths):
    cs = []
    rel = []
    rec = []
    
    for ds_id in range(n_ds_list[i_p]):
        path = f"ARBic_data/3.six_type/{patt}/{ds_id}.blocks"

        f = open(path, "r")
        lines = f.readlines()

        pred_rows = []
        pred_cols = []

        k=0
        for i in range(0,len(lines), 4):
            r = [int(n[4:]) for n in lines[i+1].strip().split(": ")[1].split(" ")]
            c = [int(n[4:]) for n in lines[i+2].strip().split(": ")[1].split(" ")]
            
            row_mask = np.zeros((600,), dtype=bool) 
            col_mask = np.zeros((600,), dtype=bool) 
            row_mask[r] = True
            col_mask[c] = True
            pred_rows.append(row_mask)
            pred_cols.append(col_mask)
            
            k+=1
            
        y_pred = (pred_rows, pred_cols)

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
    
    path = f"ARBic_data/3.six_type/trend/{ds_id}.blocks"

    f = open(path, "r")
    lines = f.readlines()

    pred_rows = []
    pred_cols = []

    k=0
    for i in range(0,len(lines), 4):
        r = [int(n[4:]) for n in lines[i+1].strip().split(": ")[1].split(" ")]
        c = [int(n[4:]) for n in lines[i+2].strip().split(": ")[1].split(" ")]
        print(r,c)
        
        row_mask = np.zeros((600,), dtype=bool) 
        col_mask = np.zeros((600,), dtype=bool) 
        row_mask[r] = True
        col_mask[c] = True
        pred_rows.append(row_mask)
        pred_cols.append(col_mask)
        
        k+=1
        
    y_pred = (pred_rows, pred_cols)
        
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
    


pd.DataFrame(stats).to_csv("results/comparison/arbic/stats.csv")

