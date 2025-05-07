import numpy as np
import json
from itertools import product


class Config:
  """Configuration class that contains train and model hyperparameters"""

  def __init__(self, params) -> None:
    self.params = params

  @classmethod
  def from_json(cls, cfg) -> "Config":
    params = json.loads(json.dumps(cfg), object_hook=HelperObject)
    return cls(params)

class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)
        

def variance_in_bicluster(bicluster, centroid_position):
    i_C, j_C = centroid_position
    
    m, n = bicluster.shape 

    V1 = bicluster  # Original matrix (m, n)
    V2 = np.broadcast_to(bicluster[:, j_C][:, None], (m, n))  # Column of centroid
    V3 = np.broadcast_to(bicluster[i_C, :][None, :], (m, n))  # Row of centroid
    V4 = np.full((m, n), bicluster[i_C, j_C])  # Single centroid value expanded

    V = np.stack([V1, V2, V3, V4], axis=-1)
    total_variance = np.sum(np.var(V, axis=-1, ddof=0))
    return total_variance

def global_variance(bk):
    total_var = 0
    total_size = 0
    
    for c in bk.centroids[1:] :
        rows = c.bicluster["rows"]
        columns = c.bicluster["cols"]
        bicluster = bk.dataset[np.ix_(rows, columns)]
        if bicluster.size <= 1:
            total_var += 0
        else:
            b_var = variance_in_bicluster(bicluster, (rows.index(c.row), columns.index(c.col)))
            total_var += b_var * bicluster.size  # Weighted by number of elements
        total_size += bicluster.size
    return total_var/total_size

def global_mean_squared_residue(bk):
    
    def mean_squared_residue(matrix):
        if matrix.size <= 1:
            return 1  # Handle empty biclusters safely
        
        row_means = matrix.mean(axis=1, keepdims=True)  # Mean of each row
        col_means = matrix.mean(axis=0, keepdims=True)  # Mean of each column
        overall_mean = matrix.mean()  # Global mean of the bicluster

        residue_matrix = matrix - row_means - col_means + overall_mean
        msr = np.mean(residue_matrix**2)
        
        return msr
    
    total_msr = 0
    total_size = 0
    dataset_variance = np.var(bk.dataset)
    
    empty_count = sum(1 for c in bk.centroids[1:] if c.size() <= 1)
    
    for c in bk.centroids[1:] :
        rows = c.bicluster["rows"]
        columns = c.bicluster["cols"]
        bicluster = bk.dataset[np.ix_(rows, columns)]
        msr = mean_squared_residue(bicluster)
        total_msr += msr * bicluster.size  # Weighted by number of elements
        total_size += bicluster.size
    
    penalty_factor = 1
    if empty_count > 0:
        empty_fraction = empty_count / bk.k
        # Apply stronger penalty using an exponent
        penalty_factor = (1 - empty_fraction) ** 3
        
    gmsr = total_msr / total_size if total_size > 0 else 0
    
    return max(0, 1 - gmsr / dataset_variance) * penalty_factor 

def gene_standardization(X, only_pos = False):
    row_means = np.mean(X, axis=1, keepdims=True)
    row_stds = np.std(X, axis=1, keepdims=True, ddof=0)
    X_norm = (X-row_means) /row_stds
    if only_pos:
        X_norm = X_norm - np.min(X_norm)
    return X_norm
        
def grid_search(dataset, k, bk_config, grid, method):
    scores = []
    solutions = []
    for combination in product(*grid.values()):
        param_combination = dict(zip(grid.keys(), combination))
        bk_config.update(param_combination)
        print(bk_config)
        
        bk = method(
            dataset,
            k= np.prod(k),
            config=bk_config
        )
        bk.run()
        
        scores.append(bk.evaluate_solution())
        solutions.append(bk)
        print("-- Score: ", scores[-1])
        #result_bics.extend([m for m in bk.medoids if len(m.bicluster["rows"])>2 and len(m.bicluster["cols"])>2])
        
    return scores, solutions

# def cosine_distance(v1, v2):
#     X = [1, v1[1] - v1[0] ]
#     Y = [1, v2[1] - v2[0] ]
#     return 1 - np.dot(X,Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    
def additive_cosine_distance_vectorized(stacked):
    delta1 = stacked[..., 1] - stacked[..., 0]  
    delta2 = stacked[..., 3] - stacked[..., 2]  
    X = np.stack([np.ones_like(delta1), delta1], axis=-1) 
    Y = np.stack([np.ones_like(delta2), delta2], axis=-1)
    dot = np.sum(X * Y, axis=-1)
    norm_X = np.linalg.norm(X, axis=-1)
    norm_Y = np.linalg.norm(Y, axis=-1)
    cosine_sim = dot / (norm_X * norm_Y + 1e-8)
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def multiplicative_cosine_distance_vectorized(stacked):
    X = np.stack([stacked[..., 1], stacked[..., 0]], axis=-1) 
    Y = np.stack([stacked[..., 3], stacked[..., 2]], axis=-1) 
    dot = np.sum(X * Y, axis=-1)
    norm_X = np.linalg.norm(X, axis=-1)
    norm_Y = np.linalg.norm(Y, axis=-1)
    cosine_sim = dot / (norm_X * norm_Y + 1e-8)
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def var_distance_vectorized(stacked):
    return np.var(stacked, axis=-1)

def combined_cosine_distance_vectorized(stacked):
    additive = additive_cosine_distance_vectorized(stacked)
    multiplicative = multiplicative_cosine_distance_vectorized(stacked)
    return np.minimum(additive, multiplicative)

def exp_shift_vectorized(stacked):
    delta1 = stacked[..., 1] - stacked[..., 0]  
    delta2 = stacked[..., 3] - stacked[..., 2]  
    diff = delta1-delta2
    a = 2
    return 1-np.exp(-np.abs(diff)/a)

def exp_scale_vectorized(stacked):
    delta1 = stacked[..., 1] / (stacked[..., 0]  + 1e-10)
    delta2 = stacked[..., 3] / (stacked[..., 2]  + 1e-10)
    # with log-ratio?
    diff = delta1-delta2
    a = 2
    return 1-np.exp(-np.abs(diff)/a)

def exp_combined_vectorized(stacked):
    shift = exp_shift_vectorized(stacked)
    scale = exp_scale_vectorized(stacked)
    return np.minimum(shift, scale)
