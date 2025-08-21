import numpy as np
import json

class BicGen:
    def __init__(self, gen_seed = None, noise = 0):
        if gen_seed is None: gen_seed = np.random.randint(0,10000)
        self.rng = np.random.RandomState(gen_seed)
        self.bics = []
        self.noise = noise

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
        elif coherence == "none-constant": # row constant
            return np.full((m,n), 1) * np.array(row_coeff).reshape((m,1))
        elif coherence == "constant-none": # column constant
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
            M = meta["#DatasetRows"]
            N = meta["#DatasetColumns"]
            
            data = self.rng.standard_normal((M,N))

            prev_r, prev_c = 0, 0
            
            rows = []
            cols = []
            true_biclusters = []
            patterns = []
            for _, bic in meta["biclusters"].items():
                m = bic["#rows"]
                n = bic["#columns"]
                pattern = bic["RowPattern"].lower() + "-" + bic["ColumnPattern"].lower()
                patterns.append(pattern)
                if pattern == "constant-constant":
                    seed = float(bic["Data"][0][0])
                    row_coeff = None
                    col_coeff = None
                elif pattern == "constant-none": # column constant bicluster
                    row_coeff = None
                    col_coeff = [float(n) for n in bic["Data"][0]]
                    seed = None
                elif pattern == "none-constant": # row constant bicluster
                    col_coeff = None
                    seed = None
                    row_coeff = [float(r[0]) for r in bic["Data"]] 
                else:
                    row_coeff = [float(n) for n in bic["RowFactors"][1:-1].split(",")]
                    col_coeff = [float(n) for n in bic["ColumnFactors"][1:-1].split(",")]
                    seed = float(bic["Seed"])

                if prev_r+m >= M or prev_c+n >= N:
                    break 
                
                b = self.generate_bicluster(pattern, m, n, seed, row_coeff, col_coeff )
                b += self.rng.normal(loc=0, scale=self.noise, size=b.shape)
                
                position_rows = list(range(prev_r, prev_r + m))
                position_cols = list(range(prev_c, prev_c + n))
                true_biclusters.append((position_rows, position_cols))
                
                self.bics.append(((prev_r,prev_r+m), (prev_c, prev_c+n), pattern))
                
                data[np.ix_(position_rows, position_cols)] = b
                prev_r += m
                prev_c += n
        
            row_idx_shuffled = self.rng.permutation(M)
            col_idx_shuffled = self.rng.permutation(N)
            data = data[row_idx_shuffled][:, col_idx_shuffled]
            
            row_pos = {old_idx: new_idx for new_idx, old_idx in enumerate(row_idx_shuffled)}
            col_pos = {old_idx: new_idx for new_idx, old_idx in enumerate(col_idx_shuffled)}

            for true_rows, true_cols in true_biclusters:
                position_rows = sorted([row_pos[r] for r in true_rows])
                position_cols = sorted([col_pos[c] for c in true_cols])
                row = np.zeros((M,), dtype=bool)
                row[position_rows] = True
                rows.append(row)
                col = np.zeros((N,), dtype=bool)
                col[position_cols] = True
                cols.append(col)            

        return data, np.array(rows), np.array(cols), patterns
        
                
        