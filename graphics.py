from matplotlib import patches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np


def show_reordered(data, bk, path=None):
    
    row_indices = []
    col_indices = []
    bicluster_positions = []
    for c in bk.medoids:
        bicluster_positions.append((len(row_indices), len(col_indices), len(c.bicluster["rows"]), len(c.bicluster["cols"])))
        row_indices.extend(c.bicluster["rows"])
        col_indices.extend(c.bicluster["cols"])
    remaining_rows = [i for i in range(data.shape[0]) if i not in row_indices]
    remaining_cols = [j for j in range(data.shape[1]) if j not in col_indices]
    new_row_order = row_indices + remaining_rows
    new_col_order = col_indices + remaining_cols

    reordered =  data[np.ix_(new_row_order, new_col_order)]

    fig, ax = plt.subplots()
    im = ax.imshow(reordered, cmap='viridis')
    fig.colorbar(im, ax=ax)

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
        

def show_history(bk, path=None):
    scores = [sol for sol in bk.history]
    fig = plt.figure()
    plt.title("History")
    plt.ylabel("score")
    plt.xlabel("iteration")
    plt.plot(scores)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks on x-axis
    if path is None:
        plt.show()
    else:
        fig.savefig(path)
        

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
    for idx, c in enumerate(solution.medoids):
        rows = c.bicluster["rows"]
        columns = c.bicluster["cols"]
        m[np.ix_(rows, columns)] = idx+1
    
    # Create a colormap: white for 0, then distinct colors for clusters
    num_clusters = len(solution.medoids)
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
    
    medoids = [m for m in bk.medoids if len(m.bicluster["rows"])>2 and len(m.bicluster["cols"])>2]
    num_centroids = len(medoids)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_centroids, figsize=(num_centroids * 5, 5), sharey=True)

    if num_centroids == 1:  # Ensure axes is iterable when only one subplot
        axes = [axes]

    for ax, c in zip(axes, medoids):
        rows = c.bicluster["rows"]
        columns = c.bicluster["cols"]
        
        subset = bk.dataset[np.ix_(rows, columns)]
        
        if subset.shape[1] != len(columns):
            raise ValueError(f"Mismatch: subset has {subset.shape[1]} columns, but expected {len(columns)}.")

        for i in range(len(rows)):  # Iterate properly over rows
            ax.plot(range(len(columns)), subset[i, :], marker='o' if len(columns)==1 else '', linestyle='-', alpha=0.4)
        
        ax.plot(range(len(columns)), subset[rows.index(c.row), :], marker='', linestyle='--', alpha=1.0, color='red')
        ax.plot([columns.index(c.col)], subset[rows.index(c.row), columns.index(c.col)], marker='x', color='red')

        ax.set_xlabel('Column Index')
        ax.set_title(f'Bicluster {bk.medoids.index(c)} ({len(c.bicluster["rows"])}x{len(c.bicluster["cols"])})')
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