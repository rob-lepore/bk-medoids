import numpy as np
import json

class BicGen:
    def __init__(self, gen_seed = None):
        if gen_seed is None: gen_seed = np.random.randint(0,10000)
        self.rng = np.random.RandomState(gen_seed)
        self.bics = []

    def generate_bicluster(self, coherence, m, n, seed, row_coeff = None, col_coeff=None):
        
        if row_coeff is not None and col_coeff is not None:
            row_coeff = np.array(row_coeff).reshape((m,1))
            col_coeff = np.array(col_coeff)
            
        if coherence == "additive-additive":
            return np.full((m,n),seed) + row_coeff + col_coeff
        elif coherence == "multiplicative-multiplicative":
            return np.full((m,n),seed) * row_coeff * col_coeff
        elif coherence == "constant-constant":
            return np.full((m,n),seed)
        elif coherence == "none-constant":
            return np.full((m,n), 1) * row_coeff
        elif coherence == "constant-none":
            return np.full((m,n), 1) * col_coeff
        elif coherence == "constant-additive":
            row_coeff = np.zeros((m,1))
            return np.full((m,n),seed) + row_coeff + col_coeff
        elif coherence == "additive-constant":
            col_coeff = np.zeros((1,n))
            return np.full((m,n),seed) + row_coeff + col_coeff
        elif coherence == "constant-multiplicative":
            row_coeff = np.ones((m,1))
            return np.full((m,n),seed) * row_coeff * col_coeff
        elif coherence == "multiplicative-constant":
            col_coeff = np.ones((1,n))
            return np.full((m,n),seed) * row_coeff * col_coeff
        else:
            raise ValueError(f"Unknown coherence \"{coherence}\"")
    
    def from_gbic(self, filename):
        with open(filename) as file:
            meta = json.load(file)
            N = meta["#DatasetColumns"]
            M = meta["#DatasetRows"]
            min_val = meta["#DatasetMinValue"]
            max_val = meta["#DatasetMaxValue"]
            
            data = self.rng.random((M,N)) * (max_val - min_val) + min_val
            
            prev_r, prev_c = 0, 0
            
            for bic_number, bic in meta["biclusters"].items():
                m = bic["#rows"]
                n = bic["#columns"]
                pattern = bic["RowPattern"].lower() + "-" + bic["ColumnPattern"].lower()
                if pattern == "constant-constant":
                    seed = float(bic["Data"][0][0])
                    row_coeff = None
                    col_coeff = None
                else:
                    row_coeff = [float(n) for n in bic["RowFactors"][1:-1].split(",")]
                    col_coeff = [float(n) for n in bic["ColumnFactors"][1:-1].split(",")]
                    seed = float(bic["Seed"])

                if prev_r+m >= M or prev_c+n >= N:
                    break 
                
                b = self.generate_bicluster(pattern, m, n, seed, row_coeff, col_coeff )
                position_rows = list(range(prev_r,prev_r+m))
                position_cols = list(range(prev_c,prev_c+n))
                
                self.bics.append(((prev_r,prev_r+m), (prev_c, prev_c+n), pattern))
                
                data[np.ix_(position_rows, position_cols)] = b
                prev_r += m
                prev_c += n
        return data
    
    def get_biclusters(self):
        pass
                
        