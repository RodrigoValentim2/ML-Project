import numpy as np
from sklearn.naive_bayes import GaussianNB
import math as mt
from numpy.linalg import inv, det


class NaiveBayes:
    
    def __init__(self):
            self.densities_view1  = []
            self.densities_view2  = []
            self.densities_view3  = []
            self.parameters_views = []
            self.evidence = 0
            self.posteriori_view1 = []
            self.posteriori_view2 = []
            self.posteriori_view3 = [] 
            
    def fit(self, X_train_views, y_train_views):
      
         
        for x_train, y_train in zip(X_train_views, y_train_views):
            nb = GaussianNB() 
            nb.fit(x_train, y_train);
            parameter_view = []
            parameter_view.append(nb.class_prior_)
            parameter_view.append(nb.sigma_)
            parameter_view.append(nb.theta_)
            parameter_view.append(nb.classes_)
            
            self.parameters_views.append(parameter_view)
            
       

    def calc_density(self,x,parameter):
        apriori = parameter[0]
        sigma = parameter[1]
        mean = parameter[2]
        classes =  len(parameter[3])
        x = x
        densities = []

        for c in range(0,classes):


            ##determinant
            inver = inv(np.identity(x.shape[0]) * sigma[c])
            determinant= det(inver)

            part_one_equation = mt.pow(2*mt.pi, -x.shape[0]/2)*mt.pow(determinant,0.5)

            ## values for exp calculation
            value1_exp = ((x -mean[c]).T)
            value1_exp = np.dot(value1_exp, inver)

            value2_exp = (x-mean[c])

            ##calc  exp
            exp = np.exp(-0.5*(np.dot(value1_exp, value2_exp)))

            #Result conditional x priori

            rest = part_one_equation*exp

            densities.append(rest*apriori[c])

        return densities
    
    def calc_evidence(self, densities_view1, densities_view2, densities_view3):
        self.evidence = np.array(densities_view1).sum() + np.array(densities_view2).sum()+ np.array(densities_view3).sum()

        return self.evidence
    
    def posteriori(self,view):
    
        view = np.array(view)
    
        return  np.array(view/self.evidence)

    def predict(self,X_tests):
        
        ##denstities 
        self.densities_view1 =  [self.calc_density(xi, self.parameters_views[0]) for xi in X_tests[0]]
        self.densities_view2 =  [self.calc_density(xi, self.parameters_views[1]) for xi in X_tests[1]]
        self.densities_view3 =  [self.calc_density(xi, self.parameters_views[2]) for xi in X_tests[2]]
        
        ##evidence
        self.evidece = self.calc_evidence(self.densities_view1, self.densities_view2, self.densities_view3)
        
        
        
        ##Posteriori
        self.posteriori_view1 = self.posteriori(self.densities_view1)
        self.posteriori_view2 = self.posteriori(self.densities_view2)
        self.posteriori_view3 = self.posteriori(self.densities_view3)
        
        #Sum rule
        posteriori_final = (self.posteriori_view1+ self.posteriori_view2+ self.posteriori_view3)
        
        #y predict
        #y_pred_view1 = [np.argmax(x) for x in self.posteriori_view1]
        #y_pred_view2 = [np.argmax(x) for x in self.posteriori_view2]
        #y_pred_view3 = [np.argmax(x) for x in self.posteriori_view3]
        
        y_pred_final = [np.argmax(x) for x in posteriori_final]
        
  
    
        return y_pred_final
 