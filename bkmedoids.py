import numpy as np
from medoid import Medoid
import time
from scipy.stats import mode
from utils import *
from scipy.stats import pearsonr


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
    
    self.distance_func = linear_scale_vectorized
  
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
    distances = self.distance_func(stacked, *self.params.distance_func_args) 

    min_index = np.argmin(distances, axis=0)  
    min_values = np.min(distances, axis=0)  
    # print(min_values.shape)
    min_index[min_values >= self.params.distance_threshold] = -1
    # print(min_index)
    return min_index
  
  def find_medoid(self, ds: np.ndarray):
    N, M = ds.shape
    min_loss = float('inf')
    best_i, best_j = 0, 0

    for i in range(N):
        for j in range(M):
            # Prepare stacked tensor for current (i, j)
            a = ds
            b = np.tile(ds[:, j][:, np.newaxis], (1, M))
            c = np.tile(ds[i, :][np.newaxis, :], (N, 1))
            d = np.full((N, M), ds[i, j])

            stacked = np.stack([a, b, c, d], axis=-1)
            D = self.distance_func(stacked, *self.params.distance_func_args)
            loss = D.sum()

            if loss < min_loss:
                min_loss = loss
                best_i, best_j = i, j

    return best_i, best_j
    
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
      medoids_rows = [m.row for m in self.medoids]
      for i, row in enumerate(closest_medoid):
        if i in medoids_rows:
          continue
        valid = row[row != -1]
        if len(valid) > 0:
            assigned_medoid = mode(valid, keepdims=False).mode
            self.medoids[int(assigned_medoid)].add_row(i)
        # correlations = []
        # for k,m in enumerate(self.medoids):
        #   valid_cols = np.where(row == k)[0]
        #   # print(len(valid_cols))
        #   if (len(valid_cols)>2):
        #     corr = abs(np.corrcoef(self.dataset[i,valid_cols], self.dataset[m.row,valid_cols])[0][1])
        #     # print(corr)
        #     correlations.append(corr)
        # # print(i, len(correlations))
        # if(len(correlations)>0):
        #   print(np.argmax(correlations))
        #   self.medoids[np.argmax(correlations)].add_row(i)
        # else:
          # assigned_medoid = mode(row[row!=-1], keepdims=False).mode
          # self.medoids[int(assigned_medoid)].add_row(i)
              
      medoids_cols = [m.col for m in self.medoids]
      for j, col in enumerate(closest_medoid.T):
        if j in medoids_cols:
          continue
        valid = col[col != -1]
        if len(valid) > 0:
            assigned_medoid = mode(valid, keepdims=False).mode
            self.medoids[int(assigned_medoid)].add_column(j)
        
        # correlations = []
        # for k,m in enumerate(self.medoids):
        #   valid_rows = np.where(col == k)[0]
        #   # print(len(valid_rows))
        #   if (len(valid_rows)>2):
        #     corr = abs(np.corrcoef(self.dataset[valid_rows,j], self.dataset[valid_rows, m.col])[0])
        #     correlations.append(corr)
        # # print(j, len(correlations))
        # if(len(correlations)>0):
        #   self.medoids[np.argmax(correlations)].add_column(j)
        # else:
        #   assigned_medoid = mode(col[col!=-1], keepdims=False).mode
        #   self.medoids[int(assigned_medoid)].add_column(j)
              
      # Remove outlier rows and columns
      for m in self.medoids:
        m.add(m.row, m.col)
        m.remove_outliers(self.distance_func, self.params.distance_func_args, self.params.distance_threshold, self.params.outlier_threshold, self.dataset)
        
      # Update medoids
      for m in self.medoids:
        rows = m.bicluster["rows"]
        cols = m.bicluster["cols"]

        if len(rows) <= 1 or len(cols) <= 1 : 
          occupied_rows = [m.row for m in self.medoids]
          occupied_cols = [m.col for m in self.medoids]
          free_rows = [i for i in range(self.dataset.shape[0]) if i not in occupied_rows]
          free_cols = [i for i in range(self.dataset.shape[1]) if i not in occupied_cols]
          
          
          random_row = self.rng.choice(free_rows)
          random_col = self.rng.choice(free_cols)
          updated_position = (random_row, random_col)
        else:
          bicluster = self.dataset[np.ix_(rows, cols)]
          i,j = self.find_medoid(bicluster)
          updated_position = rows[i], cols[j]
        
        m.update(updated_position)
          
      self.history.append(self.evaluate_solution())
      self.it += 1
    self.history.append(self.evaluate_solution())
  
  
  def evaluate_solution(self):    
    score = 0.0
    tot_size = 0
    empties = 0

    for m in self.medoids:
        rows = np.asarray(m.bicluster["rows"])
        cols = np.asarray(m.bicluster["cols"])

        if len(rows) <= 2 or len(cols) <= 2:
            empties += 1
            # if len(rows)>0:
            continue
          
        score += acv(self.dataset[np.ix_(rows, cols)])

        # ii, jj = np.meshgrid(rows, cols, indexing='ij')

        # a = self.dataset[ii, jj]                          
        # b = self.dataset[ii, m.col]                       
        # c = self.dataset[m.row, jj]                       
        # d = self.dataset[m.row, m.col] * np.ones_like(a) 

        # stacked = np.stack([a, b, c, d], axis=-1)         
        # dists = self.distance_func(stacked, *self.params.distance_func_args)                  

        # score += np.sum(dists) * m.size()
        # tot_size += m.size()

    return score + empties*self.params.empty_penalty# (score / tot_size) + empties*self.params.empty_penalty if tot_size > 0 else np.inf
  
  
  def orphans(self):
    included_rows = len([r for m in self.medoids for r in m.bicluster["rows"]])
    included_cols = len([c for m in self.medoids for c in m.bicluster["cols"]])
    return 1-(included_rows/self.dataset.shape[0]), 1-(included_cols/self.dataset.shape[1])
  
  def get_biclusters(self):
    pass