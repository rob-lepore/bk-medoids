from sklearn.metrics import consensus_score
from bkmedoids import BKmedoids
from itertools import product
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
from copy import deepcopy


class GridSearch:
    def __init__(self, config: dict, grid: dict):
        self.config = config
        self.grid = grid
        
    def _search_chunk(self, chunk, dataset, k, real):
        results = []
        for combination in chunk:
            param_combination = dict(zip(list(self.grid.keys()), combination))
            config_copy = deepcopy(self.config)
            config_copy.update(param_combination)

            bk = BKmedoids(dataset, k, config=config_copy)

            start = time.time()
            bk.run()
            ex_time = time.time() - start

            if real is None:
                score = bk.evaluate_solution()
            else:
                score = 1 - consensus_score(bk.get_biclusters(), real)
            print(f"Grid combination: {param_combination}")
            print(f" -- Loss: {score:.4f}")
            print(f" -- Execution time: {ex_time:.3f} ({bk.it} iterations)")
            results.append((score, bk, ex_time))
        return results
    
    def search(self, dataset: np.ndarray, k: int, real: tuple = None, p: int = 1):
        prod = list(product(*self.grid.values()))
        print(f"Starting grid search: {len(prod)} combinations")
        
        chunks = [prod[i::p] for i in range(p)]

        with Pool(processes=p) as pool:
            func = partial(self._search_chunk, dataset=dataset, k=k, real=real)
            results = pool.map(func, chunks)

        scores, solutions, times = [], [], []
        for chunk_result in results:
            for score, sol, t in chunk_result:
                scores.append(score)
                solutions.append(sol)
                times.append(t)

        return scores, solutions, times
    