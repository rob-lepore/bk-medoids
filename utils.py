from matplotlib import patches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import json

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


def show_biclusters(solution, path = None):
    num_centroids = len(solution.centroids)+1

    fig, axes = plt.subplots(1, num_centroids, figsize=(num_centroids * 5, 3), sharey=True)
    axes[0].matshow(solution.dataset, cmap=plt.cm.Blues, norm=None)
    axes[0].set_title("Real")
    for i, c in enumerate(solution.centroids):
        solution_mat = np.zeros_like(solution.dataset)
        solution_mat[np.ix_(c.bicluster["rows"], c.bicluster["cols"])] = 1
        solution_mat[c.row, c.col] = 2
        axes[i+1].matshow(solution_mat, cmap=plt.cm.Blues, norm=None)
        axes[i+1].set_title(f"Bicluster {i+1}")
    
    if path is None:
        plt.show()
    else:
        fig.savefig(path)
        
def show_biclusters_together(solution, path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    m = np.zeros_like(solution.dataset)
    for idx, c in enumerate(solution.centroids):
        rows = c.bicluster["rows"]
        columns = c.bicluster["cols"]
        m[np.ix_(rows, columns)] = idx+1
    
    # Create a colormap: white for 0, then distinct colors for clusters
    num_clusters = len(solution.centroids)
    base_cmap = plt.get_cmap('tab20', num_clusters)
    colors = ['white'] + [base_cmap(i) for i in range(num_clusters)]
    cmap = ListedColormap(colors)

    # Define boundaries to match 0, 1, ..., num_clusters
    boundaries = np.arange(-0.5, num_clusters + 1, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Plot the matrix
    cax = ax.matshow(m, cmap=cmap, norm=norm)

    # Add colorbar with ticks only for cluster IDs (exclude 0 if desired)
    cbar = fig.colorbar(cax, ax=ax, ticks=np.arange(1, num_clusters + 1))
    cbar.ax.set_ylabel('Cluster ID')

    if path is None:
        plt.show()
    else:
        fig.savefig(path)
    

    
def show_parallel_coordinates(bk, path=None):
    
    num_centroids = len(bk.centroids)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_centroids, figsize=(num_centroids * 5, 3), sharey=True)

    if num_centroids == 1:  # Ensure axes is iterable when only one subplot
        axes = [axes]

    for ax, c in zip(axes, bk.centroids):
        rows = c.bicluster["rows"]
        columns = c.bicluster["cols"]
        subset = bk.dataset[np.ix_(rows, columns)]
        
        if subset.shape[1] != len(columns):
            raise ValueError(f"Mismatch: subset has {subset.shape[1]} columns, but expected {len(columns)}.")

        for i in range(len(rows)):  # Iterate properly over rows
            ax.plot(range(len(columns)), subset[i, :], marker='o' if len(columns)==1 else '', linestyle='-', alpha=0.6)
        
        ax.plot(range(len(columns)), subset[rows.index(c.row), :], marker='', linestyle='--', alpha=1.0, color='red')
        ax.plot([columns.index(c.col)], subset[rows.index(c.row), columns.index(c.col)], marker='x', color='red')

        ax.set_xlabel('Column Index')
        ax.set_title(f'Bicluster {bk.centroids.index(c)+1}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(np.min(bk.dataset),np.max(bk.dataset))
        ax.grid(True)

    # Set shared y-axis limits
    axes[0].set_ylabel('Value')
    axes[0].set_ylim(bk.dataset.min(), bk.dataset.max())

    if path is None:
        plt.show()
    else:
        fig.savefig(path)
    
from itertools import product
    
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
    return scores, solutions


def show_history(bk, path=None):
    scores = [sol for sol in bk.history]
    fig = plt.figure()
    plt.title("History")
    plt.ylabel("score")
    plt.xlabel("iteration")
    plt.plot(scores)
    if path is None:
        plt.show()
    else:
        fig.savefig(path)

def corr(X,Y):
    if len(X) == 2:
        if X[0] > X[1] and Y[0] > Y[1]:
            return 1
        if X[0] < X[1] and Y[0] < Y[1]:
            return 1
        if X[0] > X[1] and Y[0] < Y[1]:
            return -1
        if X[0] < X[1] and Y[0] > Y[1]:
            return -1
        if X[0] == X[1] and Y[0] == Y[1]:
            return 1
        if (X[0] == X[1] and Y[0] != Y[1]) or (X[0] != X[1] and Y[0] == Y[1]):
            return 0
        
    return np.corrcoef(X, Y)[0, 1]


def cosine_sim(v1,v2):
    X = [1, v1[1] - v1[0] ]
    Y = [1, v2[1] - v2[0] ]
    # X = v1
    # Y = v2
    return np.dot(X,Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

def cosine_distance(v1, v2):
    X = [1, v1[1] - v1[0] ]
    Y = [1, v2[1] - v2[0] ]
    return 1 - np.dot(X,Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    
def distance(data, i, j, Mi, Mj):
    r1 = [ data[i, j], data[i, Mj]]
    r2 = [ data[Mi,j], data[Mi, Mj] ]
    return cosine_distance(r1, r2)
    # return np.var([data[i, j], data[i, Mj], data[Mi,j], data[Mi, Mj]])
    
def submatrix_corr_score(bk):
    total = 0
    total_size = 0
    for c in bk.centroids:
      I = bk.centroids[0].bicluster["rows"]
      J = bk.centroids[0].bicluster["cols"]
      
      if len(I) <= 1 or len(J) <= 1: continue
      
      SiJ = []
      for i1 in I:
        r_1 = bk.dataset[i1, J]
        S = (np.sum([np.abs(corr(r_1, bk.dataset[i2, J])) for i2 in I if i1 != i2])) / (len(I)-1)
        SiJ.append(1-S)
      
      S_row = np.min(SiJ)
      
      SIj = []
      for j1 in J:
        c_1 = bk.dataset[I, j1]
        S = (np.sum([np.abs(corr(c_1, bk.dataset[I, j2])) for j2 in J if j1 != j2])) / (len(J)-1)
        SIj.append(1-S)
      
      S_col = np.min(SIj)
      total += (min(S_row, S_col) * c.size() )
      total_size += c.size()
    return total / total_size if total_size > 0 else 0

def show_reordered(data, bk, path=None):
    
    row_indices = []
    col_indices = []
    bicluster_positions = []
    for c in bk.centroids:
        bicluster_positions.append((len(row_indices), len(col_indices), len(c.bicluster["rows"]), len(c.bicluster["cols"])))
        row_indices.extend(c.bicluster["rows"])
        col_indices.extend(c.bicluster["cols"])
    remaining_rows = [i for i in range(data.shape[0]) if i not in row_indices]
    remaining_cols = [j for j in range(data.shape[1]) if j not in col_indices]
    new_row_order = row_indices + remaining_rows
    new_col_order = col_indices + remaining_cols

    reordered =  data[np.ix_(new_row_order, new_col_order)]

    fig, ax = plt.subplots()
    ax.imshow(reordered, cmap='Blues')

    for r_start, c_start, r_len, c_len in bicluster_positions:
        rect = patches.Rectangle(
            (c_start - 0.5, r_start - 0.5),  # x, y (adjust for pixel grid)
            c_len, r_len,                   # width, height
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title("Biclustered Matrix")
    if path is not None:
        fig.savefig(path)
    else:
        plt.show()