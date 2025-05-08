from matplotlib import pyplot as plt
import numpy as np

class Medoid:
  def __init__(self, position: list):
    self.position = np.array(position)
    self.row = position[0]
    self.col = position[1]
    self.bicluster = {"rows": [], "cols": []}
    self.points = []
    
  def add_row(self, row):
    if row not in self.bicluster["rows"]:
      self.bicluster["rows"].append(row)
      self.bicluster["rows"].sort()
      
  def add_column(self, col):
    if col not in self.bicluster["cols"]:
      self.bicluster["cols"].append(col) 
      self.bicluster["cols"].sort() 
    
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
    
  def remove_outliers(self, distance_func, distance_func_args, distance_threshold, threshold_perc, dataset):
    if len(self.bicluster["rows"]) <= 1 or len(self.bicluster["cols"]) <= 1:
        return
    
    for _ in range(200):
        rows = self.bicluster["rows"]
        cols = self.bicluster["cols"]
        if len(rows) <= 1 or len(cols) <= 1:
            break

        bicluster = dataset[np.ix_(rows, cols)]
       
        Mi = rows.index(self.position[0])
        Mj = cols.index(self.position[1])
                
        m, n = bicluster.shape
        
        a = bicluster
        b = np.tile(bicluster[:, Mj][:, np.newaxis], (1, n)) 
        c = np.tile(bicluster[Mi, :][np.newaxis, :], (m, 1))  
        d = np.full((m, n), bicluster[Mi, Mj])

        stacked = np.stack([a, b, c, d], axis=-1) 
        distances = distance_func(stacked, *distance_func_args)    

        distances[Mi, :] = 0
        distances[:, Mj] = 0

        row_mask = (distances > distance_threshold).sum(axis=1) > threshold_perc * n
        col_mask = (distances > distance_threshold).sum(axis=0) > threshold_perc * m

        row_indices = np.where(row_mask)[0]
        col_indices = np.where(col_mask)[0]

        self.bicluster["rows"] = [rows[i] for i in range(m) if i not in row_indices]
        self.bicluster["cols"] = [cols[i] for i in range(n) if i not in col_indices]

        if len(row_indices) == 0 and len(col_indices) == 0:
            break
      