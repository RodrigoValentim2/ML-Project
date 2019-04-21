import pandas as pd
import numpy as np
import math
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
def mvfuzzy(D: np.array, K, m, T, err):
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
    U_membDegree = calc_membership_degree(D, G_medoids, W_weights, K, m)

    # compute initial adequacy
    J_adequacy = calc_adequacy(D, G_medoids, W_weights, U_membDegree, K, m)

    U_previous = U_membDegree
    J_previous = J_adequacy
    for t_iteration in range(1, 150):
        # find best medoid vectors
        G_t = calc_best_medoids(D, U_previous, K, m)

        # find new best relevance weights
        W_t = calc_best_weights(D, U_previous, G_t, K, m)

        # find new membership degree vector (best fuzzy partition)
        U_t = calc_membership_degree(D, G_t, W_t, K, m)

        # find new adequacy and determine if meets criteria
        J_t = calc_adequacy(D, G_t, W_t, U_t, K, m)
        J_adequacy_difference = abs(J_previous - J_t)
        if J_adequacy_difference < m:
            break
    return (G_t, W_t, U_t)
    return (G_medoids, U_membDegree, J_adequacy)


def calc_best_medoids(D, U_membDegree, K, m):
    p_views = D.shape[2]

    # must specify as int because numpy default is float
    G_best_medoids = np.zeros((K, p_views), dtype=int)
    for k in range(0, K):
        for j in range(0, p_views):
            # multiply column U_k to every column of D_j
            # Akj_matrix: shape = (n_elems, n_elems)
            Akj_matrix = (U_membDegree[:, k] ** m)* D[:, :, j]
            #
            # sum column-wise (second dimension => axis=1)
            # A_kj: shape = (n_elems)
            A_kj = np.sum(Akj_matrix, axis=1)
            G_best_medoids[k, j] = np.argmin(A_kj)
    return G_best_medoids


def calc_membership_degree(D, G_medoids, W_weights, K, m):
    n_elems = D.shape[0]
    p_views = D.shape[2]

    # initialize membership degree vector and subparts
    U_membDegree = np.zeros(shape=[n_elems, K], dtype=float)
    for k in range(0, K):
        # compute A column for current k (outter k)
        A_k = np.zeros((n_elems))
        for j in range(0, p_views):
            A_k += W_weights[k, j] * D[:, G_medoids[k, j], j]
        # compute B for each h (inner k) + already calculate U
        for h in range(0, K):
            B_h = np.zeros((n_elems))
            for j in range(0, p_views):
                B_h += W_weights[h, j] * D[:, G_medoids[h, j], j]
            # accumulate U for each h (inner k)
            U_membDegree[:, k] += (A_k/B_h)**(1/(m-1))
    # final operation (after inner k sum) on each element of U
    U_membDegree = 1/U_membDegree
    return U_membDegree


def calc_adequacy(D, G_medoids, W_weights, U_membDegree, K, m):
    n_elems = D.shape[0]
    p_views = D.shape[2]

    Jk_mat = np.zeros((n_elems, K))
    for k in range(0, K):
        for j in range(0, p_views):
            Jk_mat[:, k] += W_weights[k, j] * D[:, G_medoids[k, j], j]
        Jk_mat[:, k] = (U_membDegree[:, k] ** m) * Jk_mat[:, k]
    return Jk_mat.sum()


def calc_best_weights(D, U_previous, G_medoids, K, m):
    p_views = D.shape[2]

    W_weights = np.ones(shape=[K, p_views])
    for k in range(0, K):
        for j in range(0, p_views):
            # A: upper member of Eq. 5
            A_kj = 1
            for h in range(0, p_views):
                # C: summatory in upper member of Eq. 5
                Ckj_h = (U_previous[:, k] ** m) * D[:, G_medoids[k, h], h]
                A_kj = A_kj * np.sum(Ckj_h, axis=0)
            A_kj = A_kj ** (1/p_views)
            # B: bottom member of Eq. 5
            Bkj_column = (U_previous[:, k] ** m) * D[:, G_medoids[k, j], j]
            B_kj = np.sum(Bkj_column, axis=0)
            W_weights[k, j] = A_kj / B_kj
    return W_weights
    