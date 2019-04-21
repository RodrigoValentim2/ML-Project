#!/usr/bin/env python

import pandas as pd
import numpy as np
import mvfuzzy as mvf
import mvfuzzy_iterative as mvf_iterative
import math
from pytictoc import TicToc
from sklearn import preprocessing
from sklearn.utils import random
from sklearn.metrics.pairwise import euclidean_distances


# CONSTANTS
RANDOM_SEED = 128476


def test_matrix_iterative(D: np.array, K, m, T, err):
    """
    Compares the performance of the iterative vs matrix implementations of the
    several equations to compute the fuzzy partition.
    """
    n_elems = D.shape[0]
    p_views = D.shape[2]
    t = TicToc()

    # Initial medoids selection
    rand_elements = random.sample_without_replacement(n_elems, K * p_views, random_state=RANDOM_SEED)
    G_medoids = np.zeros(shape=[K, p_views], dtype=int)
    for k in range(0, K):
        for p in range(0, p_views):
            G_medoids[k, p] = rand_elements[k*p_views + p]  # p cols, k lines

    # Initial weight vector
    W_weights = np.ones(shape=[K, p_views], dtype=float)

    # ---------------------------------------------------------------------------
    # Membership degree vector calculation
    # iterative
    print("---------------------------------------------")
    print("Membership Vector (U | Eq. 6)")
    t.tic()
    U_membDegree_iterative = mvf_iterative.calc_membership_degree(
        D, G_medoids, W_weights, K, m)
    elapsed = t.tocvalue()
    print_formatted("Iterative: ", elapsed)

    # matrix (optimized)
    t.tic()
    U_membDegree_matrix = mvf.calc_membership_degree(
        D, G_medoids, W_weights, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix: ", elapsed)

    # check if matrices are identical
    areEqual_U = np.isclose(U_membDegree_iterative, U_membDegree_matrix)
    areEqual_U = areEqual_U.all()
    print_formatted("Iterative == Matrix:", areEqual_U)

    # ---------------------------------------------------------------------------
    # Adequacy calculation
    #
    # iterative
    print("---------------------------------------------")
    print("Adequacy (J | Eq. 1)")
    t.tic()
    J_adequacy_iterative = mvf_iterative.calc_adequacy(
        D, G_medoids, W_weights, U_membDegree_iterative, K, m)
    elapsed = t.tocvalue()
    print_formatted("Iterative: ", elapsed)

    # matrix (optimized)
    t.tic()
    J_adequacy_matrix = mvf.calc_adequacy(
        D, G_medoids, W_weights, U_membDegree_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix: ", elapsed)

    # check if results are identical
    # must use math.isclose() to compare floats to deal with aproximation errors
    areEqual_J = math.isclose(J_adequacy_iterative, J_adequacy_matrix)
    print_formatted("Iterative == Matrix:", areEqual_J)

    # ---------------------------------------------------------------------------
    # Best medoid vector calculation
    #
    # iterative
    print("---------------------------------------------")
    print("Best Medoids Vector (G | Eq. 4)")
    t.tic()
    G_bestMedoids_iterative = mvf_iterative.calc_best_medoids(D, U_membDegree_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Iterative:", elapsed)

    # matrix (optimized)
    t.tic()
    G_bestMedoids_matrix = mvf.calc_best_medoids(D, U_membDegree_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix:", elapsed)

    # check if results are identical
    areEqual_G = G_bestMedoids_iterative == G_bestMedoids_matrix
    areEqual_G = areEqual_G.all()
    print_formatted("Iterative == Matrix:", areEqual_G)

    # ---------------------------------------------------------------------------
    # Best weights vector calculation
    #
    # iterative
    print("---------------------------------------------")
    print("Best Weights Vector (W | Eq. 5)")
    t.tic()
    W_bestWeights_iterative = mvf_iterative.calc_best_weights(D, U_membDegree_matrix, G_bestMedoids_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Iterative:", elapsed)

    # matrix (optimized)
    t.tic()
    W_bestWeights_matrix = mvf.calc_best_weights(D, U_membDegree_matrix, G_bestMedoids_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix:", elapsed)

    # check if results are identical
    areEqual_W = np.isclose(W_bestWeights_iterative, W_bestWeights_matrix)
    areEqual_W = areEqual_W.all()
    print_formatted("Iterative == Matrix:", areEqual_W)
    np.savetxt("W_iterative", W_bestWeights_iterative)
    np.savetxt("W_matrix", W_bestWeights_matrix)


# should be improved, maybe generalized and always require the suffix
def print_formatted(name: str, value, suffix=True):
    if(type(value) is float):
        if(suffix):
            suffix = " seconds"
        else:
            suffix = ""
        value_str = "{:.8f}{}".format(value, suffix)
    else:
        value_str = str(value)
    print("{:25}{}".format(name, value_str))


def test_fuzzy_partition(D: np.array, K, m, T, err):
    """
    Calculate the fuzzy partition for a given input, measures the time and
    saves the results
    """
    t = TicToc()
    print("---------------------------------------------")
    print("Calculating fuzzy partition...", end="", flush=True)
    t.tic()
    result = mvf.calc_fuzzy_partition(D, K, m, T, err)
    elapsed = t.tocvalue()
    (last_iteration, J_last, J_diff, G_medoids, W_weights, U_memb) = result
    print("done!")
    print_formatted("Last Iteration:", last_iteration, suffix=False)
    print_formatted("Last J_t:", J_last, suffix=False)
    print_formatted("Last J_diff:", J_diff, suffix=False)
    print_formatted("Elapsed time:", elapsed)
    np.save("G_medoids", G_medoids)
    np.save("W_weights", W_weights)
    np.save("U_membership", U_memb)


# -------------------------------------------------------------------------------
# MAIN - for testing
def main():
    # read the data
    mfeat_fac = pd.read_csv(
        "mfeat/mfeat-fac", sep="\\s+", header=None, dtype=float)
    mfeat_fou = pd.read_csv(
        "mfeat/mfeat-fou", sep="\\s+", header=None, dtype=float)
    mfeat_kar = pd.read_csv(
        "mfeat/mfeat-kar", sep="\\s+", header=None, dtype=float)

    # normalize
    scaler = preprocessing.MinMaxScaler()
    norm_fac = scaler.fit_transform(mfeat_fac)
    norm_fou = scaler.fit_transform(mfeat_fou)
    norm_kar = scaler.fit_transform(mfeat_kar)

    # compute dissimilarity matrices
    D = np.zeros((2000, 2000, 3))
    D[:, :, 0] = euclidean_distances(norm_fac)
    D[:, :, 1] = euclidean_distances(norm_fou)
    D[:, :, 2] = euclidean_distances(norm_kar)

    for i in range(0, 15):
        # test_matrix_iterative(D, 10, 1.6, 150, 10**-10)
        test_fuzzy_partition(D, 10, 1.6, 150, 10**-10)


if __name__ == '__main__':
    main()
