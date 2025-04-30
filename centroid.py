from matplotlib import pyplot as plt
import numpy as np
from utils import distance

class Centroid:
  def __init__(self, position: list):
    self.position = np.array(position)
    self.row = position[0]
    self.col = position[1]
    self.bicluster = {"rows": [], "cols": []}
    self.points = []
    # self.data = {"row": data_row, "column": data_column}
    
  def add(self, row, col):
    self.points.append((row,col))
    if row not in self.bicluster["rows"]:
      self.bicluster["rows"].append(row)
      self.bicluster["rows"].sort()
    if col not in self.bicluster["cols"]:
      self.bicluster["cols"].append(col) 
      self.bicluster["cols"].sort() 
      
  def update(self, position):
    self.position = np.array(position)
    self.row = position[0]
    self.col = position[1]
  
  def empty(self):
    self.bicluster = {"rows": [], "cols": []}
    self.points = []
    
  def size(self):
    return len(self.bicluster["rows"]) * len(self.bicluster["cols"])
  def __str__(self) -> str:
    return f"row: {self.row}, column: {self.col}, b-rows: {self.bicluster['rows']}, b-cols: {self.bicluster['cols']}"
  def __repr__(self) -> str:
    return self.__str__()
  
  # def remove_outliers(self, out_threshold, row_threshold_perc, col_threshold_perc, dataset):
  #   if len(self.bicluster["rows"]) <= 1 or len(self.bicluster["cols"]) <= 1:
  #       return 
    
  #   rows = self.bicluster["rows"]
  #   cols = self.bicluster["cols"]
  #   bicluster = dataset[np.ix_(rows, cols)]
  #   i_C, j_C = rows.index(self.position[0]), cols.index(self.position[1])
    
  #   m, n = bicluster.shape 

  #   correlations = np.zeros_like(bicluster)
  #   for i in range(m):
  #     for j in range(n):
  #       correlations[i,j] = cosine_sim(bicluster[i,[j, j_C]], bicluster[i_C,[j, j_C]]) 
  #       # print(bicluster[i,[j, j_C]], bicluster[i_C,[j, j_C]], correlations[i,j])
    
  #   row_mask = (correlations < out_threshold).sum(axis=1) > row_threshold_perc * n
  #   col_mask = (correlations < out_threshold).sum(axis=0) > col_threshold_perc * m
    
  #   row_indices = np.where(row_mask)[0].tolist()
  #   col_indices = np.where(col_mask)[0].tolist()
  #   self.bicluster["rows"] = [self.bicluster["rows"][i] for i in range(m) if i not in row_indices]
  #   self.bicluster["cols"] = [self.bicluster["cols"][i] for i in range(n) if i not in col_indices]
    
  def remove_outliers(self, out_threshold, row_threshold_perc, col_threshold_perc, dataset):
    if len(self.bicluster["rows"]) <= 1 or len(self.bicluster["cols"]) <= 1:
        return 
    
    for it in range(200):
      rows = self.bicluster["rows"]
      cols = self.bicluster["cols"]
      bicluster = dataset[np.ix_(rows, cols)]
      try:
        i_C,  j_C = rows.index(self.position[0]), cols.index(self.position[1])
      except:
        print("Error: ", self.position[0], self.position[1], rows, cols, it)
        return
      m, n = bicluster.shape 

      correlations = np.zeros_like(bicluster)
      for i in range(m):
        for j in range(n):
          if i == i_C or j == j_C:
            correlations[i,j] = 0
          else:
            correlations[i,j] = distance(bicluster, i, j, i_C, j_C) 
      
      row_mask = (correlations > out_threshold).sum(axis=1) > row_threshold_perc * n
      col_mask = (correlations > out_threshold).sum(axis=0) > col_threshold_perc * m
      
      row_indices = np.where(row_mask)[0].tolist()
      col_indices = np.where(col_mask)[0].tolist()
      self.bicluster["rows"] = [self.bicluster["rows"][i] for i in range(m) if i not in row_indices]
      self.bicluster["cols"] = [self.bicluster["cols"][i] for i in range(n) if i not in col_indices]
      
      if len(row_indices) == 0 and len(col_indices) == 0:
        break
      