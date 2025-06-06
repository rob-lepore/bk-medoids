from medoid import Medoid
from utils import gene_standardization, bistandardisation, make_positive
from graphics import show_biclusters, show_history, show_parallel_coordinates, show_biclusters_together, show_reordered
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_checkerboard
from bkmedoids import BKmedoids
from gridsearch import GridSearch
from test import Test
from bicgen import BicGen

def shuffle_matrix(data):
    rng = np.random.RandomState(0)
    row_idx_shuffled = rng.permutation(data.shape[0])
    col_idx_shuffled = rng.permutation(data.shape[1])
    return data[row_idx_shuffled][:, col_idx_shuffled], row_idx_shuffled, col_idx_shuffled


def test_gs():
    ds, rows, columns = BicGen(42).from_gbic("datasets/additive_small_bics.json")     
    ds = ds + np.random.RandomState(42).normal(loc=0, scale=0.1, size=ds.shape)
    # ds = make_positive(ds)
    ds, row_idx, col_idx = shuffle_matrix(ds)
    # ds = gene_standardization(ds)
    
    config = {
        "threshold": 0.0001,
        "max_it": 300,
        "empty_penalty": 5,
        "distance_func_args": [],
        "distance_func": "shift"
    }
    
    grid = {
        "seed": list(range(2)),
        "outlier_threshold": [0.5],
        "distance_threshold": [0.5],
    }
    
    start = time.time()
    gs = GridSearch(config, grid)
    scores, solutions, times = gs.search(ds, k=5)#,real=(rows[:,row_idx], columns[:,col_idx]))
    
    best = solutions[np.argmin(scores)]
    orphans = best.orphans()
    
    print(f"\n\nExecution time: {times[np.argmin(scores)]:.2f} seconds. Loss: {np.min(scores):.4f}. Iterations: {best.it}")
    print(f"Consensus score: {consensus_score(best.get_biclusters(), (rows[:,row_idx], columns[:,col_idx]))}")
    print(f"Orphan rows: {100*orphans[0]:.2f}%. Orphan cols: {100*orphans[1]:.2f}%")
    print(f"Best solution: {best.params.__dict__}")
    show_history(best, "imgs/history.png")
    show_reordered(best, path = "imgs/biclusters_reordered.png")
    show_parallel_coordinates(best, "imgs/parallel_coordinates.png")
    
    with open("imgs/output.txt", "w") as file:
        for idx, m in enumerate(best.medoids):
            rows = m.bicluster["rows"]
            cols = m.bicluster["cols"]
            file.write(f"Bicluster {idx}: {len(rows)}x{len(cols)} -- {[ds[m.row, m.col]]}\n")
            file.write(f"Rows: {rows}\n")
            file.write(f"Cols: {cols}\n")
            file.write(str(ds[np.ix_(rows,cols)])+"\n")
    
    fig, ax = plt.subplots()
    im = ax.imshow(ds, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title("Dataset")
    fig.savefig("imgs/original.png")
    
def test():
    ds = pd.read_csv("./datasets/all_data.tsv",index_col=0, sep = "\t").to_numpy()
    ds = ds + np.random.normal(loc=0, scale=0.1, size=ds.shape)

    config = {
        "threshold": 1.e-3,
        "max_it": 20,
        "empty_penalty": 6,
        "outlier_threshold": 0.8,
        "distance_threshold": 1e-1,
        "distance_func_args": [5],
        "seed": 0
    }
        
    bics = Test.test(ds, 10, config, 5)

    bk = BKmedoids(ds, 0, config)
    bk.medoids = bics
    show_parallel_coordinates(bk, "test_results.png")

from sklearn.metrics import consensus_score

if __name__ == "__main__":
    # test()
    test_gs()
    
    # gen = BicGen(42)
    # ds, rows, columns = gen.from_gbic("datasets/multiplicative_bics.json") 
    # ds, row_idx, col_idx = shuffle_matrix(ds)
    
    # config = {
    #     "threshold": 1.e-3,
    #     "max_it": 40,
    #     "empty_penalty": 5,
    #     "seed": 4,
    #     "outlier_threshold": 0.9,
    #     "distance_threshold": 1e-5,
    #     "distance_func_args": []
    # }

    # bk = BKmedoids(ds, 2, config)
    # bk.run()
    # bics = bk.get_biclusters()
    # print(bics[0].shape, bics[1].shape)
    
    # bics = []
    # for bic in gen.bics:
    #     start_r, end_r = bic[0]
    #     start_c, end_c = bic[1]
    #     position_rows = list(range(start_r,end_r))
    #     position_cols = list(range(start_c,end_c))
        
    #     b = ds[np.ix_(position_rows, position_cols)]
    #     m = Medoid([start_r,start_c])
    #     m.bicluster["rows"] = position_rows
    #     m.bicluster["cols"] = position_cols
    #     bics.append(m)

    # bk = BKmedoids(ds,0,{"seed":0})
    # bk.medoids = bics
    
    