from gridsearch import GridSearch
import numpy as np
import pandas as pd
from utils import *
from graphics import *
import json

K = 10

config = {
    "threshold": 0.01,
    "max_it": 50,
    "distance_func_args": [],
    "distance_func": "shift",
}

grid = {
    "distance_threshold":  [0.15],
    "outlier_threshold": [0.7],
    "seed": list(range(3)),
}

# ds = pd.read_csv(f"real_datasets/data/ecoli_dream5/E.tsv", sep="\t", header=0, index_col=0).values
# ds = ds.T
# print(ds.shape)

# gs = GridSearch(config, grid)
# scores, solutions, _ = gs.search(ds, K, p=3)
# best = solutions[np.argmin(scores)]

# y_pred = best.get_biclusters()
# with open("results/real_data/ecoli.json", "w") as f:
#     json.dump({
#         "loss": float(np.min(scores)),
#         "rows": [[int(n) for n in np.where(mask==True)[0]] for mask in y_pred[0]],
#         "cols": [[int(n) for n in np.where(mask==True)[0]] for mask in y_pred[1]],
#         }, f)
    
ds = pd.read_csv(f"real_datasets/data/yeast_dream5/E.tsv", sep="\t", header=0, index_col=0).values
ds = ds.T
print(ds.shape)

gs = GridSearch(config, grid)
scores, solutions, _ = gs.search(ds, K, p=3)
best = solutions[np.argmin(scores)]

y_pred = best.get_biclusters()
with open("results/real_data/yeast.json", "w") as f:
    json.dump({
        "loss": float(np.min(scores)),
        "rows": [[int(n) for n in np.where(mask==True)[0]] for mask in y_pred[0]],
        "cols": [[int(n) for n in np.where(mask==True)[0]] for mask in y_pred[1]],
        }, f)