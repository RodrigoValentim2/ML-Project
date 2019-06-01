import pandas as pd
import numpy as np
from scipy.spatial import distance 
from numba import jit
import numpy as np

class KNN:
    def __init__(self):
        self.k = 3
        self.X_train = []
        self.y_trains = []
        self.evidence = 0
        self.count_labels_y = np.array([])
        self.priori = []
        self.posteriori_ = []
        
        
    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
       
        __, counts = np.unique(self.y_train, return_counts=True)
        self.priori = counts/len(self.y_train)
        
 
    def posteriori(self, X_test):
        
        poster = []
        for x in X_test:
            n = self.neighbors(x ,self.X_train ,self.y_train, self.k)
            poster.append(self.calc_posteriori(n, X_test))
                                 
  
        return np.array(poster)             
                               
    def predict(self, X_test):
        post =  self.posteriori(X_test)
        y = []                         
        for p in  post:
            y.append(np.argmax(p))
        return y           
              
                                 
    def calc_posteriori(self, n, x):
        
        dens = []
        
        for i in  range(0, 10):
            dens.append(n.count(i)/len(self.y_train))
           
            
            

        return dens 
    
    def neighbors(self, x ,X_train ,y_train,k):
        distances = []
        y_neighbors = []
 

        for x_t, y in zip(X_train, y_train):
            distances.append(distance.euclidean(x, x_t))
            y_neighbors.append(y)

        df = pd.DataFrame()
        df['y'] = y_neighbors
        df['distance'] =  distances
        df = df.sort_values('distance').head(self.k)
        x_neighbors = list(df['y'].values)
      
       
       
        return x_neighbors