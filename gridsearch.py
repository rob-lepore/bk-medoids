from bkmedoids import BKmedoids
from itertools import product
import numpy as np
import time

class GridSearch:
    def __init__(self, config: dict, grid: dict):
        self.config = config
        self.grid = grid
    
    def search(self, dataset: np.ndarray, k: int):
        scores = []
        solutions = []
        prod = list(product(*self.grid.values()))
        
        for it, combination in enumerate(prod):
            print(f"Search {it+1}/{len(prod)}")
            
            param_combination = dict(zip(self.grid.keys(), combination))
            self.config.update(param_combination)
            
            bk = BKmedoids(
                dataset,
                k,
                config=self.config
            )
            
            start = time.time()
            bk.run()
            ex_time = time.time() - start
            
            scores.append(bk.evaluate_solution())
            solutions.append(bk)    
            
            print(f" -- Score: {scores[-1]:.4f}")
            print(f" -- Execution time: {ex_time:.3f} ({bk.it} iterations)")
        return scores, solutions
            