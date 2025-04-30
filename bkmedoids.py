import numpy as np
from utils import Config, distance
from centroid import Centroid


class BKmedoids:
  def __init__(self, dataset: np.ndarray, k: int | tuple, config: dict):
    self.dataset = dataset
    self.k = k
    
    self.history = []
    
    self.params = Config.from_json(config).params

    self.rng = np.random.RandomState(self.params.seed)
    self.centroids = []
    self.it = 0
    self.velocity = [np.inf]*3
    
  def move_centroid(self, c: Centroid): 
    
    def loss(bicluster, Mi, Mj):
      s = 0
      for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
          s += distance(bicluster, i, j, Mi, Mj)
      return s
      
    rows = c.bicluster["rows"]
    cols = c.bicluster["cols"]

    if len(rows) <= 1 or len(cols) <= 1: 
      random_row = self.rng.choice(self.dataset.shape[0])
      random_col = self.rng.choice(self.dataset.shape[1])
      return (random_row, random_col)

    bicluster = self.dataset[np.ix_(rows, cols)]
    
    min_loss = loss(bicluster, rows.index(c.row), cols.index(c.col))
    Mi, Mj = c.row, c.col
    for i in range(len(rows)):
      for j in range(len(cols)):
        s = loss(bicluster, i, j)
        if s < min_loss:
          min_loss = s
          Mi, Mj = rows[i], cols[j]
    
    return (Mi, Mj)
    
    # Compute average row and column
    # avg_row = np.mean(bicluster, axis=0)
    # avg_col = np.mean(bicluster, axis=1)

    # # Compute distances of each row to the average row
    # row_distances = np.linalg.norm(bicluster - avg_row, axis=1)
    # row_index = np.argmin(row_distances)

    # # Compute distances of each column to the average column
    # col_distances = np.linalg.norm(bicluster.T - avg_col, axis=1)
    # col_index = np.argmin(col_distances)
    
    # updated_position = (rows[row_index], cols[col_index])
    # return updated_position
  
  def get_closest_centroid(self, i, j):
    distances = []
    for c in self.centroids:
      if i == c.row or j == c.col: 
        distances.append(0)
        continue
    
      distances.append(distance(self.dataset, i,j,c.row,c.col))
    
    if np.min(distances) < self.params.outlier_threshold:
      return np.argmin(distances)
    else:
      return None  
      
  def run(self):
    
    # Initialize k centroids, all in different rows and columns
    rows = self.rng.choice(self.dataset.shape[0], size=self.k, replace=False)
    cols = self.rng.choice(self.dataset.shape[1], size=self.k, replace=False)
    self.centroids = [Centroid([r,c]) for r,c in zip(rows, cols)]
    
    while np.mean(self.velocity) > self.params.threshold and self.it < self.params.max_it:

      for c in self.centroids:
          c.empty()
        
      # Assign points to the closest centroid
      for i in range(self.dataset.shape[0]):
        for j in range(self.dataset.shape[1]):
          
          best_centroid = self.get_closest_centroid(i,j)
          if best_centroid is not None:
            self.centroids[best_centroid].add(row = i, col = j)        
      
      
      # Row exclusiveness
      if self.params.row_exclusive:
        for i in range(self.dataset.shape[0]):
          counts = []  # Completeness of row i
          for c in self.centroids:
            counts.append(0)
            if i not in c.bicluster["rows"] or len(c.bicluster["cols"]) == 0:
              continue
            for p in c.points:
              if p[0] == i:
                counts[-1] += 1
            #counts[-1] /= len(c.bicluster["cols"])
          idx_best = np.argmax(counts)
          for idx, c in enumerate(self.centroids):
            if idx != idx_best and i in c.bicluster["rows"]:
              c.bicluster["rows"].remove(i)
        
      # Column exclusiveness
      if self.params.column_exclusive:
        for j in range(self.dataset.shape[1]):
          counts = []  # Completeness of column j
          for c in self.centroids:
            counts.append(0)
            if j not in c.bicluster["cols"] or len(c.bicluster["rows"]) == 0:
              continue
            for p in c.points:
              if p[1] == j:
                counts[-1] += 1
            #counts[-1] /= len(c.bicluster["rows"])
          idx_best = np.argmax(counts)
          for idx, c in enumerate(self.centroids):
            if idx != idx_best and j in c.bicluster["cols"]:
              c.bicluster["cols"].remove(j)

      
      for c in self.centroids:
        c.add(c.row, c.col)
        c.remove_outliers(self.params.outlier_threshold, self.params.row_out_th, self.params.col_out_th, self.dataset)

        if self.it < self.params.max_it-1:
          updated_position = self.move_centroid(c)
          c.update(updated_position)
        c.add(c.row, c.col)
          
      
      self.history.append(self.evaluate_solution())
      if self.params.show_iterations:
          print([f"centroid {i}: {c.position}" for i, c in enumerate(self.centroids)])
      
      velocity_time = 4
      self.velocity =  self.history[-velocity_time:]
      if len(self.velocity) == velocity_time:
        self.velocity = [np.abs(self.velocity[i]-self.velocity[i-1]) for i in range(1,velocity_time)]
      else: self.velocity = [np.inf]*velocity_time
        
      self.it += 1
    self.history.append(self.evaluate_solution())
  
  def evaluate_solution(self):
    #return submatrix_corr_score(self)
    
    score = []
    tot_size = 0
    for c in self.centroids:
      rows = c.bicluster["rows"]
      cols = c.bicluster["cols"]
      
      correlations = []
      for i in rows:
        for j in cols:
          correlations.append(distance(self.dataset, i,j,c.row, c.col)) 
      score.append(np.mean(correlations)*c.size())
      tot_size += c.size()
    return np.sum(score)/tot_size
    
    
  def get_biclusters(self):
    rows = []
    cols = []
    for c in self.centroids[1:]:
      rows_indices = c.bicluster["rows"]
      cols_indices = c.bicluster["cols"]
      rows_bool = np.zeros(self.dataset.shape[0], dtype=bool)
      rows_bool[rows_indices] = True
      cols_bool = np.zeros(self.dataset.shape[1], dtype=bool)
      cols_bool[cols_indices] = True
      rows.append(rows_bool.tolist())
      cols.append(cols_bool.tolist())
    return np.array(rows), np.array(cols)