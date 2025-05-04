import numpy as np
from utils import Config, cosine_distance_vectorized, var_distance_vectorized
from medoid import Medoid
import time
from scipy.stats import mode


class BKmedoids:
  def __init__(self, dataset: np.ndarray, k: int | tuple, config: dict):
    self.dataset = dataset
    self.k = k
    
    self.history = []
    
    self.params = Config.from_json(config).params

    self.rng = np.random.RandomState(self.params.seed)
    self.medoids = []
    self.it = 0
    self.velocity = [np.inf]*3
    
    self.distance_func = cosine_distance_vectorized
  
  def get_closest_medoid(self):
    N, M = self.dataset.shape
    positions = np.array([m.position for m in self.medoids])
    K = positions.shape[0]

    Mi = positions[:, 0]  
    Mj = positions[:, 1]  

    a = self.dataset[np.newaxis, :, :]                    
    b = self.dataset[:, Mj]                               
    b = np.transpose(b, (1, 0))[:, :, np.newaxis]  

    c = self.dataset[Mi, :]                              
    c = c[:, np.newaxis, :]                     

    d = self.dataset[Mi, Mj]                               
    d = d[:, np.newaxis, np.newaxis]           

    a_broadcast = np.broadcast_to(a, (K, N, M))
    b_broadcast = np.broadcast_to(b, (K, N, M))
    c_broadcast = np.broadcast_to(c, (K, N, M))
    d_broadcast = np.broadcast_to(d, (K, N, M))

    stacked = np.stack([a_broadcast, b_broadcast, c_broadcast, d_broadcast], axis=-1) 
    distances = self.distance_func(stacked) 

    min_index = np.argmin(distances, axis=0)  
    min_values = np.min(distances, axis=0)  
    min_index[min_values >= self.params.outlier_threshold] = -1
    return min_index
  
  def find_medoid(self, ds: np.ndarray):
    N, M = ds.shape

    rows = np.arange(N)
    cols = np.arange(M)
    Mi_all, Mj_all = np.meshgrid(rows, cols, indexing='ij')
    Mi_all = Mi_all.ravel()  
    Mj_all = Mj_all.ravel()  

    K = Mi_all.size

    a = ds[np.newaxis, :, :]
    a = np.broadcast_to(a, (K, N, M))
    
    b = ds[:, Mj_all]                                       
    b = b.T[:, :, np.newaxis]                               
    b = np.broadcast_to(b, (K, N, M))

    c = ds[Mi_all, :]                                       
    c = c[:, np.newaxis, :]                                 
    c = np.broadcast_to(c, (K, N, M))

    d = ds[Mi_all, Mj_all]                                  
    d = d[:, np.newaxis, np.newaxis]                       
    d = np.broadcast_to(d, (K, N, M))

    stacked = np.stack([a, b, c, d], axis=-1)

    D = self.distance_func(stacked)

    losses = D.sum(axis=(1, 2))

    best_k   = np.argmin(losses)

    return int(Mi_all[best_k]), int(Mj_all[best_k])
    
  def is_over(self):
    velocity_time = 4
    self.velocity =  self.history[-velocity_time:]
    if len(self.velocity) == velocity_time:
      self.velocity = [np.abs(self.velocity[i]-self.velocity[i-1]) for i in range(1,velocity_time)]
    else: self.velocity = [np.inf]*velocity_time
    return np.mean(self.velocity) < self.params.threshold or self.it >= self.params.max_it
      
  def run(self):
    
    # Initialize k centroids, all in different rows and columns
    rows = self.rng.choice(self.dataset.shape[0], size=self.k, replace=False)
    cols = self.rng.choice(self.dataset.shape[1], size=self.k, replace=False)
    self.medoids = [Medoid([r,c]) for r,c in zip(rows, cols)]
    
    while not self.is_over():

      for m in self.medoids:
          m.empty()
      
      # Compute closest medoid for each data value
      closest_medoid = self.get_closest_medoid()
      
      # Enforce row and column exclusiveness
      for i, row in enumerate(closest_medoid):
        valid = row[row != -1]
        if len(valid) > 0:
            assigned_medoid = mode(valid, keepdims=False).mode
            self.medoids[int(assigned_medoid)].bicluster["rows"].append(i)

      for j, col in enumerate(closest_medoid.T):
          valid = col[col != -1]
          if len(valid) > 0:
              assigned_medoid = mode(valid, keepdims=False).mode
              self.medoids[int(assigned_medoid)].bicluster["cols"].append(j)
              
      # Remove outlier rows and columns
      for m in self.medoids:
        # m.add(m.row, m.col)
        m.remove_outliers(self.distance_func,self.params.outlier_threshold, self.params.row_out_th, self.params.col_out_th, self.dataset)
        
      # Update medoids
      for m in self.medoids:
        rows = m.bicluster["rows"]
        cols = m.bicluster["cols"]

        if m.size() <= 1: 
          random_row = self.rng.choice(self.dataset.shape[0])
          random_col = self.rng.choice(self.dataset.shape[1])
          updated_position = (random_row, random_col)
        else:
          bicluster = self.dataset[np.ix_(rows, cols)]
          i,j = self.find_medoid(bicluster)
          updated_position = rows[i], cols[j]
        
        m.update(updated_position)
        m.add(m.row, m.col)
          
      if self.params.show_iterations:
          print([f"Medoid {i}: {m.position}" for i, m in enumerate(self.medoids)])
        
      self.history.append(self.evaluate_solution())
      self.it += 1
    self.history.append(self.evaluate_solution())
  
  
  def evaluate_solution(self):    
    score = 0.0
    tot_size = 0

    for m in self.medoids:
        rows = np.asarray(m.bicluster["rows"])
        cols = np.asarray(m.bicluster["cols"])

        if len(rows) <= 1 or len(cols) <= 1:
            score += 10
            continue

        ii, jj = np.meshgrid(rows, cols, indexing='ij')

        a = self.dataset[ii, jj]                          
        b = self.dataset[ii, m.col]                       
        c = self.dataset[m.row, jj]                       
        d = self.dataset[m.row, m.col] * np.ones_like(a) 

        stacked = np.stack([a, b, c, d], axis=-1)         
        dists = self.distance_func(stacked)                  

        score += np.mean(dists) * m.size()
        tot_size += m.size()

    return score / tot_size if tot_size > 0 else np.inf
