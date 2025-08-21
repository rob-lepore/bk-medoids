import numpy as np
import json
from scipy.linalg import norm
from scipy.sparse import dia_matrix, issparse

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
        
def bistochastic_normalize(X, max_iter=1000, tol=1e-5):
    """Normalize rows and columns of ``X`` simultaneously so that all
    rows sum to one constant and all columns sum to a different
    constant.
    """
    def make_nonnegative(X, min_value=0):
        min_ = X.min()
        if min_ < min_value:
            if issparse(X):
                raise ValueError(
                    "Cannot make the data matrix"
                    " nonnegative because it is sparse."
                    " Adding a value to every entry would"
                    " make it no longer sparse."
                )
            X = X + (min_value - min_)
        return X

    def scale_normalize(X):
        """Normalize ``X`` by scaling rows and columns independently.

        Returns the normalized matrix and the row and column scaling
        factors.
        """
        X = make_nonnegative(X)
        row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1))).squeeze()
        col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0))).squeeze()
        row_diag = np.where(np.isnan(row_diag), 0, row_diag)
        col_diag = np.where(np.isnan(col_diag), 0, col_diag)
        if issparse(X):
            n_rows, n_cols = X.shape
            r = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
            c = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
            an = r * X * c
        else:
            an = row_diag[:, np.newaxis] * X * col_diag
        return an, row_diag, col_diag
    
    
    # According to paper, this can also be done more efficiently with
    # deviation reduction and balancing algorithms.
    X = make_nonnegative(X)
    X_scaled = X
    for _ in range(max_iter):
        X_new, _, _ = scale_normalize(X_scaled)
        if issparse(X):
            dist = norm(X_scaled.data - X.data)
        else:
            dist = norm(X_scaled - X_new)
        X_scaled = X_new
        if dist is not None and dist < tol:
            break
    return X_scaled


def bistandardisation(X, max_iter=100, tol=1e-6):
    def standardize_rows(X):
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    def standardize_columns(X):
        return (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    X = X.copy()
    for i in range(max_iter):
        X_old = X.copy()
        X = standardize_rows(X)
        X = standardize_columns(X)
        if np.linalg.norm(X - X_old) < tol:
            break
    return X

def make_positive(X):
    m = X.min()
    return X - m

def variance_in_bicluster(bicluster, centroid_position):
    i_C, j_C = centroid_position
    
    m, n = bicluster.shape 

    V1 = bicluster 
    V2 = np.broadcast_to(bicluster[:, j_C][:, None], (m, n))
    V3 = np.broadcast_to(bicluster[i_C, :][None, :], (m, n)) 
    V4 = np.full((m, n), bicluster[i_C, j_C]) 

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
            total_var += b_var * bicluster.size  
        total_size += bicluster.size
    return total_var/total_size

def global_mean_squared_residue(bk):
    
    def mean_squared_residue(matrix):
        if matrix.size <= 1:
            return 1 
        
        row_means = matrix.mean(axis=1, keepdims=True)  
        col_means = matrix.mean(axis=0, keepdims=True)  
        overall_mean = matrix.mean()  

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
        total_msr += msr * bicluster.size  
        total_size += bicluster.size
    
    penalty_factor = 1
    if empty_count > 0:
        empty_fraction = empty_count / bk.k
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

def var_distance_vectorized(stacked):
    return np.var(stacked, axis=-1)

def linear_shift_vectorized(stacked):
    delta1 = stacked[..., 1] - stacked[..., 0]  
    delta2 = stacked[..., 3] - stacked[..., 2]  
    return np.abs(delta1-delta2)

def linear_scale_vectorized(stacked):
    delta1 = stacked[..., 0] * stacked[..., 3]  
    delta2 = stacked[..., 1] * stacked[..., 2]  
    return np.abs(delta1-delta2)

def slog_vectorized(stacked):
    def slog(x, eps=1e-20):
        return np.sign(x)*np.log(np.abs(x)+eps)
        
    delta1 = slog(stacked[..., 1]) - slog(stacked[..., 0])
    delta2 = slog(stacked[..., 3]) - slog(stacked[..., 2])
    return np.abs(delta1-delta2)

def slog_ratio_vectorized(stacked):
    a = stacked[..., 0]
    b = stacked[..., 1]
    c = stacked[..., 2]
    d = stacked[..., 3]
    s1 = np.sign(a*b)
    s2 = np.sign(c*d)
    
    delta1 = s1 * (np.log(np.abs(b/a)+1))
    delta2 = s2 * (np.log(np.abs(d/c)+1))
    return np.abs(delta1-delta2)

    
def acv(bic: np.ndarray) -> float:
    """
    Compute the average absolute Pearson correlation-based consistency metric for both rows and columns of bic array.
    Returns 1 minus the maximum of row-based and column-based consistency scores.
    Avoids computing correlations for pairs involving constant vectors.
    """
    m, n = bic.shape

    def consistency(matrix: np.ndarray) -> float:
        stds = matrix.std(axis=1)
        const_idx = np.where(np.isclose(stds, 0))[0]
        size = matrix.shape[0]

        corr = np.full((size, size), np.nan)

        for i in range(size):
            for j in range(i, size):
                if i in const_idx and j in const_idx:
                    r = 1.0
                elif i in const_idx or j in const_idx:
                    r = 0.0
                else:
                    r = np.corrcoef(matrix[i], matrix[j])[0, 1]
                corr[i, j] = corr[j, i] = r

        total = np.nansum(np.abs(corr)) - size 
        denom = size ** 2 - size
        return total / denom if denom > 0 else 0.0

    v1 = consistency(bic)
    v2 = consistency(bic.T)
    return max(v1, v2)

def jaccard_list(a, b):
    """Jaccard index between two index lists."""
    a, b = set(a), set(b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0

def match_score(rows1, cols1, rows2, cols2):
    """Compute the product of row and column Jaccard indices."""
    r_score = jaccard_list(np.where(rows1)[0], np.where(rows2)[0])
    c_score = jaccard_list(np.where(cols1)[0], np.where(cols2)[0])
    return r_score * c_score

def recovery_score(pred, true):
    pred_rows, pred_cols = pred
    true_rows, true_cols = true
    return np.mean([
        max([match_score(tr, tc, pr, pc) for pr, pc in zip(pred_rows, pred_cols)] or [0])
        for tr, tc in zip(true_rows, true_cols)
    ])

def relevance_score(pred, true):
    pred_rows, pred_cols = pred
    true_rows, true_cols = true
    return np.mean([
        max([match_score(pr, pc, tr, tc) for tr, tc in zip(true_rows, true_cols)] or [0])
        for pr, pc in zip(pred_rows, pred_cols)
    ]) 
