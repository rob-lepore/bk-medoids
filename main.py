from bkmedoids import BKmedoids
from utils import gene_standardization, grid_search
from graphics import show_biclusters, show_history, show_parallel_coordinates, show_biclusters_together, show_reordered
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_checkerboard
import time

if __name__ == "__main__":
    # ds, rows, columns = make_checkerboard(
    #     shape=(500,500), n_clusters=(30,25), noise=2, random_state=1, shuffle=True
    # )
    # ds = gene_standardization(ds, only_pos=False)
    
    ds = pd.read_csv("./datasets/all_data.tsv",index_col=0, sep = "\t").to_numpy()
    # ds = ds + np.random.normal(loc=0, scale=0.1, size=ds.shape)
    
    configs = {
        "threshold": -1.e-3,
        "max_it": 40,
        "show_iterations": False,
        #"outlier_threshold": 0.9,
    }
    
    grid = {
        "seed": list(range(1,2)),
        "distance_threshold": [1e-4],
        "outlier_threshold": [0.8],
        "distance_func_args": [[2]]
    }
    
    start = time.time()
    scores, solutions = grid_search(ds, k=10, bk_config=configs, grid=grid, method = BKmedoids)
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
    
    with open("imgs/output.txt", "w") as file:
        for idx, m in enumerate(best.medoids):
            rows = m.bicluster["rows"]
            cols = m.bicluster["cols"]
            file.write(f"Bicluster {idx}: {len(rows)}x{len(cols)} -- {[ds[m.row, m.col]]}\n")
            file.write(f"Rows: {rows}\n")
            file.write(f"Cols: {cols}\n")
            
        
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(ds, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title("Dataset")
    fig.savefig("imgs/original.png")
    