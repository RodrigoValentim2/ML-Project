import pandas as pd
import numpy as np
import sys
from pytictoc import TicToc
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import random


# -------------------------------------------------------------------------------
# multi-view fuzzy c-medoid
# RETURNS:
#    (one vector for each cluster)
#    G - medoid vectors
#    W - relevance weight vectors
#    U - membership degree vectors
def mvfuzzy3(D: np.array, K, m, T, err):
    t_iteration = 0
    n_elems = D.shape[0]
    p_views = D.shape[2]
    
    # initialize medoid vector G (3 views x K clusters)
    #     the elements will be represented by a tuple, (p, n), where:
    #        p: view of the element
    #        n: number (row) of the element
    #     that's all we need when accessing the dissimilarity matrix
    rand_elements = random.sample_without_replacement(n_elems, K * p_views)
    G_medoids = np.zeros(shape=[K, p_views], dtype=int)
    for k in range(0, K):
        for p in range(0, p_views):
            G_medoids[k, p] = rand_elements[k*p_views + p] # p cols, k lines
    
    # initialize weigth vector
    W_weights = np.ones(shape=[K, p_views], dtype=float)

    # compute initial membership degree vector
    t = TicToc()
    t.tic()
    U_membDegree = calc_membership_degree(D, G_medoids, W_weights, K, m)
    t.toc("Membership calculation: ")
    return (G_medoids, U_membDegree)


def calc_membership_degree(D, G_medoids, W_weights, K, m):
    n_elems = D.shape[0]
    p_views = D.shape[2]

    # initialize membership degree vector
    U_membDegree = np.zeros(shape=[n_elems, K], dtype=float)
    for i in range(0, n_elems):
        for k in range(0, K):
            u_ik = 0
            for h in range(0, K):
                A_ik = 0
                B_ih = 0
                for j in range(0, p_views):
                    A_ik += W_weights[k, j] * D[i, G_medoids[k, j], j]
                    B_ih += W_weights[h, j] * D[i, G_medoids[h, j], j]
                u_ik += (A_ik/B_ih)**(1/(m-1))
            U_membDegree[i, k] = 1/u_ik
    return U_membDegree


# -------------------------------------------------------------------------------
# MAIN - for testing
def main():
    # read the data
    mfeat_fac = pd.read_csv("mfeat/mfeat-fac", sep="\\s+", header=None, dtype=float)
    mfeat_fou = pd.read_csv("mfeat/mfeat-fou", sep="\\s+", header=None, dtype=float)
    mfeat_kar = pd.read_csv("mfeat/mfeat-kar", sep="\\s+", header=None, dtype=float)

    # normalize
    scaler = preprocessing.MinMaxScaler()
    norm_fac = scaler.fit_transform(mfeat_fac)
    norm_fou = scaler.fit_transform(mfeat_fou)
    norm_kar = scaler.fit_transform(mfeat_kar)

    # compute dissimilarity matrices
    t = TicToc()
    t.tic()
    D_fac = euclidean_distances(norm_fac)
    D_fou = euclidean_distances(norm_fou)
    D_kar = euclidean_distances(norm_kar)
    t.toc("Dissimilarity calculation:")

    D_all = np.zeros((2000, 2000, 3))
    D_all[:,:,0] = D_fac
    D_all[:,:,1] = D_fou
    D_all[:,:,2] = D_kar
    
    mvfuzzy3(D_all, 10, 1.6, 150, 10**-10)
    # return mvfuzzy3(D_all, 10, 1.6, 150, 10**-10)

