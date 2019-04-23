from sklearn.naive_bayes import GaussianNB

class NaiveBayes():
    
    def fit(X_tran, y_train):
        
        nb = GaussianNB()
        nb.fit(X_train, y_train);


        priori = nb.class_prior_
        sigma = nb.sigma_
        mean = nb.theta_
        classes = nb.classes_


    def calc_density(x, priori, mean, sigma, classes):
        apriori = priori
        mean = mean
        sigma = sigma
        classes = classes
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

            densities.append(rest*priori[c])


        return densities
    
    def calc_evidence(densities_view1, densities_view2, densities_view3):
        evidence = np.array(densities_view1).sum() + np.array(densities_view2).sum()+ np.array(densities_view3).sum()

        return evidence
    
    def posteriori(view, evidence):
    
        view = np.array(view)
    
        return  np.array(view/evidence)

    def predict(posteriori):
    
        y_pred = []
        for post in posteriori:
            y_pred.append(np.argmax(post))
        return y_pred    