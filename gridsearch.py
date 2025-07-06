from sklearn.metrics import consensus_score
from bkmedoids import BKmedoids
from itertools import product
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
import ray

# ray.init(num_cpus=10)
# ray.init()

# @ray.remote
# def run_combination(combination, grid_keys, config, dataset, k, real):
#     param_combination = dict(zip(grid_keys, combination))
#     config_copy = deepcopy(config)
#     config_copy.update(param_combination)
    
#     bk = BKmedoids(dataset, k, config=config_copy)
#     start = time.time()
#     bk.run()
#     ex_time = time.time() - start
#     score = bk.evaluate_solution() if real is None else 1 - consensus_score(bk.get_biclusters(), real)
    
#     return score, bk, ex_time


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

            print(f" -- Loss: {score:.4f}")
            print(f" -- Execution time: {ex_time:.3f} ({bk.it} iterations)")
            results.append((score, bk, ex_time))
        return results
    
    
    # def search(self, dataset: np.ndarray, k: int, real: tuple = None):
    #     scores = []
    #     solutions = []
    #     times = []
    #     prod = list(product(*self.grid.values()))
        
    #     for it, combination in enumerate(prod):
    #         print(f"Search {it+1}/{len(prod)}")
            
    #         param_combination = dict(zip(self.grid.keys(), combination))
    #         self.config.update(param_combination)
            
    #         bk = BKmedoids(
    #             dataset,
    #             k,
    #             config=self.config
    #         )
            
    #         start = time.time()
    #         bk.run()
    #         ex_time = time.time() - start
    #         times.append(ex_time)
    #         if real is None:
    #             scores.append(bk.evaluate_solution())
    #         else:
    #             scores.append(1-consensus_score(bk.get_biclusters(), real))
    #         solutions.append(bk)    
            
    #         print(f" -- Loss: {scores[-1]:.4f}")
    #         print(f" -- Execution time: {ex_time:.3f} ({bk.it} iterations)")
    #     return scores, solutions, times
    
    def search(self, dataset: np.ndarray, k: int, real: tuple = None, p: int = 1):
        prod = list(product(*self.grid.values()))
        
        # Split combinations into p chunks
        chunks = [prod[i::p] for i in range(p)]

        # Prepare pool
        with Pool(processes=p) as pool:
            func = partial(self._search_chunk, dataset=dataset, k=k, real=real)
            results = pool.map(func, chunks)

        # Flatten results and extract scores, solutions, times
        scores, solutions, times = [], [], []
        for chunk_result in results:
            for score, sol, t in chunk_result:
                scores.append(score)
                solutions.append(sol)
                times.append(t)
                # print(f" -- Loss: {score:.4f}")
                # print(f" -- Execution time: {t:.3f} ({sol.it} iterations)")

        return scores, solutions, times
    
    # def search(self, dataset: np.ndarray, k: int, real: tuple = None):
    #     grid_keys = list(self.grid.keys())
    #     combinations = list(product(*self.grid.values()))
        
    #     # Submit tasks in parallel
    #     futures = [
    #         run_combination.remote(comb, grid_keys, self.config, dataset, k, real)
    #         for comb in combinations
    #     ]
        
    #     results = ray.get(futures)

    #     # Collect results
    #     scores, solutions, times = [], [], []
    #     for score, sol, t in results:
    #         scores.append(score)
    #         solutions.append(sol)
    #         times.append(t)
    #         print(f" -- Loss: {score:.4f}")
    #         print(f" -- Execution time: {t:.3f} ({sol.it} iterations)")
        
    #     return scores, solutions, times