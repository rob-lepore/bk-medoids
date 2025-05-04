from bkmedoids import BKmedoids
from utils import gene_standardization, grid_search
from graphics import show_biclusters, show_history, show_parallel_coordinates, show_biclusters_together, show_reordered
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_checkerboard
import time

if __name__ == "__main__":
    ds, rows, columns = make_checkerboard(
        shape=(100,100), n_clusters=(8,7), noise=2, random_state=1, shuffle=True
    )
    
    ds = pd.read_csv("./datasets/additive_data.tsv",index_col=0, sep = "\t").to_numpy()
    # noise = np.random.normal(loc=0, scale=0.2, size=ds.shape)
    # ds = ds + noise
    # ds = gene_standardization(ds, only_pos=False)
    
    configs = {
        "threshold": 1.e-3,
        "max_it": 20,
        "show_iterations": False,
        "row_exclusive": True,
        "column_exclusive": True,
        #"outlier_threshold": 0.9,
    }
    
    grid = {
        "seed": [0],# list(range(4)),
        "outlier_threshold": [0.000001],
        "row_out_th": [0.8],
        "col_out_th": [0.8]
    }
    
    start = time.time()
    scores, solutions = grid_search(ds, k=5, bk_config=configs, grid=grid, method = BKmedoids)
    best = solutions[np.argmin(scores)]
    
    orphan_rows = []
    for row in range(ds.shape[0]):
        orphan = True
        for c in best.medoids:
            if row in c.bicluster["rows"]:
                orphan = False
                break
        if orphan:
            orphan_rows.append(row)
            
    orphan_cols = []
    for col in range(ds.shape[1]):
        orphan = True
        for c in best.medoids:
            if col in c.bicluster["cols"]:
                orphan = False
                break
        if orphan:
            orphan_cols.append(col)
    
    
    print(f"\n\nExecution time: {time.time() - start:.2f} seconds. Score: {np.min(scores):.4f}. Iterations: {best.it}")
    print(f"Orphan rows: {100*len(orphan_rows)/ds.shape[0]:.2f}%. Orphan cols: {100*len(orphan_cols)/ds.shape[1]:.2f}%")
    print(f"Best solution: {best.params.__dict__}")
    # show_biclusters(best, path = "imgs/biclusters.png")
    show_biclusters_together(best, path = "imgs/biclusters.png")
    show_parallel_coordinates(best, "imgs/parallel_coordinates.png")
    show_history(best, "imgs/history.png")
    show_reordered(ds, best, path = "imgs/biclusters_reordered.png")
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(ds, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title("Dataset")
    fig.savefig("imgs/original.png")
    