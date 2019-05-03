import pandas as pd
import numpy as np
from scipy.spatial import distance 
from numba import jit
import numpy as np

class KNN:
    def __init__(self):
        self.k = 3
        self.X_trains = []
        self.y_trains = []
        self.evidence = 0
        self.count_labels_y = np.array([])
        
        
    def fit(self, X_trains, y_trains, k):
        self.X_trains = X_trains
        self.y_trains = y_trains
        self.k = k
        
    @jit(nopython=True)   
    def posteriori(self, X_test):
        
        densities_view1  = []
        for x in (X_test[0]):
            n = self.neighbors(x ,self.X_trains[0] ,self.y_trains[0], self.k)
            densities_view1.append(self.den_calculation(n, X_test[0]))
                                 
        densities_view2  = []
        for x in (X_test[1]):
            n = self.neighbors(x ,self.X_trains[1] ,self.y_trains[1], self.k)
            densities_view2.append(self.den_calculation(n, X_test[1]))
         
        densities_view3  = []
        for x in (X_test[2]):
            n = self.neighbors(x ,self.X_trains[2] ,self.y_trains[2], self.k)
            densities_view3.append(self.den_calculation(n, X_test[2]))
        
        evidence = np.array(densities_view1)+np.array(densities_view2)+np.array(densities_view3)
     
        evidence =  evidence.sum() 
       
        posteriori_view1 =  densities_view1 / evidence
        posteriori_view2 =  densities_view2 /evidence
        posteriori_view3 =  densities_view3 / evidence
             
        return posteriori_view1, posteriori_view2, posteriori_view3                 
                               
    def predict(self, X_tests):
        posteriori1, posteriori2, posteriori3 = self.posteriori(X_tests)
        posteriori_final = posteriori1+posteriori2+posteriori3 
        y = []                         
        for p in posteriori_final:
            y.append(np.argmax(p))
        return y           
              
                                 
    def den_calculation(self, n, x):
        #volume = ((mt.pi)**(x[1]/2))/mt.factorial((x.shape[1]/2))
        #volume = volume.sum()
        dens = []
        
        for i in  range(0, 10):
            dens.append(n.count(i)/200)

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