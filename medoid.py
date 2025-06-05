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
      
  # def remove_outliers(self, distance_func, distance_func_args, distance_threshold, threshold_perc, dataset: np.ndarray):
  #     print("start acv: ", acv(dataset[np.ix_( self.bicluster["rows"],self.bicluster["cols"])]))
  #     for _ in range(1):
  #       rows = self.bicluster["rows"]
  #       cols = self.bicluster["cols"]
        
  #       if len(rows) <= 2 or len(cols) <= 2: break
        
  #       m_row = dataset[self.row, cols]
  #       m_col = dataset[rows, self.col].flatten()
        
  #       bic = dataset[np.ix_(rows,cols)]
        
  #       # print(self.size(), " -> ", end="")
        
  #       is_m_row_const = np.isclose(np.std(m_row), 0)
  #       to_remove_r = []
  #       for i, row in enumerate(bic):
  #         is_row_const = np.isclose(np.std(row), 0)
  #         if is_row_const and is_m_row_const:
  #             r = 1.0
  #         elif is_row_const or is_m_row_const:
  #             r = 0.0
  #         else:
  #             r = np.abs(np.corrcoef(row, m_row)[0, 1])
  #         # print(f"{r:.3}", end=" ")
  #         if (r < distance_threshold): to_remove_r.append(rows[i])

  #       is_m_col_const = np.isclose(np.std(m_col), 0)
  #       to_remove_c = []
  #       for j, col in enumerate(bic.T):
  #         is_col_const = np.isclose(np.std(col), 0)
  #         if is_col_const and is_m_col_const:
  #             r = 1.0
  #         elif is_col_const or is_m_col_const:
  #             r = 0.0
  #         else:
  #             r = np.abs(np.corrcoef(row, m_row)[0, 1])
  #         if (r < distance_threshold): to_remove_c.append(cols[j])
        
  #       if len(to_remove_r) == 0 and len(to_remove_c) == 0: break
        
  #       self.bicluster["rows"] = [r for r in self.bicluster["rows"] if r not in to_remove_r]
  #       self.bicluster["cols"] = [c for c in self.bicluster["cols"] if c not in to_remove_c]
  #       # print(self.size())
  #     print("end acv", acv(dataset[np.ix_(self.bicluster["rows"],self.bicluster["cols"])]))
      