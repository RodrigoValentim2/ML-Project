import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import random


# read the data
mfeat_fac = pd.read_csv("mfeat/mfeat-fac", sep="\s+", header=None)
mfeat_fou = pd.read_csv("mfeat/mfeat-fou", sep="\s+", header=None)
mfeat_kar = pd.read_csv("mfeat/mfeat-kar", sep="\s+", header=None)

# normalize
scaler = preprocessing.MinMaxScaler()
norm_fac = scaler.fit_transform(mfeat_fac)
norm_fou = scaler.fit_transform(mfeat_fou)
norm_kar = scaler.fit_transform(mfeat_kar)

# compute dissimilarity matrices
D_fac = euclidean_distances(norm_fac)
D_fou = euclidean_distances(norm_fou)
D_kar = euclidean_distances(norm_kar)

# -------------------------------------------------------------------------------
# multi-view fuzzy c-medoid
# RETURNS:
#    (one vector for each cluster)
#    G - medoid vectors
#    W - relevance weight vectors
#    U - membership degree vectors
def mvfuzzy3(D1: pd.DataFrame, D2: pd.DataFrame, D3: pd.DataFrame, n_elems, K, m, T, err):
    t_iteration = 0
    p_views = 3
    n_elems = D1.shape[0]
    
    # initialize medoid vector G (3 views x K clusters)
    #     the elements will be represented by a tuple, (p, n), where:
    #        p: view of the element
    #        n: number (row) of the element
    #     that's all we need when accessing the dissimilarity matrix
    rand_elements = random.sample_without_replacement(n_elems, K * p_views)
    G_medoids = pd.DataFrame(np.zeros(shape=[K, p_views], dtype=int))
    for k in range(0, K):
        for p in range(0, p_views):
            G_medoids[p][k] = rand_elements[k*p_views + p] # p cols, k lines
    
    # initialize weigth vector
    W_weights = pd.DataFrame(np.ones(shape=[K, p_views], dtype=float))

    # compute initial membership degree vector
    U_membDegree = calc_membership_degree(D1, D2, D3, G_medoids, W_weights, K, m)
    return (G_medoids, U_membDegree)


def calc_membership_degree(D1, D2, D3, G_medoids, W_weights, K, m):
    n_elems = D1.shape[0]
    D_all = np.zeros((n_elems, n_elems, 3))
    D_all[:,:,0] = D1
    D_all[:,:,1] = D2
    D_all[:,:,2] = D3

    # initialize membership degree vector
    U_membDegree = pd.DataFrame(np.zeros(shape=[n_elems, K], dtype=float))
    for i in range(0, n_elems):
        for k in range(0, K):
            u_ik = 0
            for h in range(0, K):
                A_ik = 0
                B_ih = 0
                for j in range(0, 3):
                    A_ik += W_weights[j][k]*D_all[i, G_medoids[j][k], j]
                    B_ih += W_weights[j][h]*D_all[i, G_medoids[j][h], j]
                u_ik += (A_ik/B_ih)**(1/(m-1))
            U_membDegree[k][i] = 1/u_ik
    return U_membDegree


# -------------------------------------------------------------------------------
# MAIN - for testing
def main():
    return 0
