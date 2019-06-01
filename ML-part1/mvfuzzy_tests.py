#!/usr/bin/env python

import pandas as pd
import numpy as np
from mvfuzzy import MVFuzzy
import mvfuzzy_iterative as mvf_iterative
import math
from pytictoc import TicToc
from sklearn import preprocessing
from sklearn.utils import random
from sklearn.metrics.pairwise import euclidean_distances


# CONSTANTS
RANDOM_SEED = 495924220


def test_matrix_iterative(D: np.array, K, m, T, err):
    """
    Compares the performance of the iterative vs matrix implementations of the
    several equations to compute the fuzzy partition.
    """
    n_elems = D.shape[1]
    p_views = D.shape[0]
    t = TicToc()
    mvf = MVFuzzy()

    # Initial medoids selection
    np.random.seed(RANDOM_SEED)
    G_medoids = np.random.randint(n_elems, size=(K, p_views))

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
    U_membDegree_matrix = mvf._calc_membership_degree(
        D, G_medoids, W_weights, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix: ", elapsed)

    # check if matrices are identical
    areEqual_U = U_membDegree_iterative == U_membDegree_matrix
    areEqual_U = areEqual_U.all()
    print_formatted("U | Iterative == Matrix:", areEqual_U)
    print_formatted("Diff: ", str(np.sum(abs(U_membDegree_iterative - U_membDegree_matrix))))

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
    J_adequacy_matrix = mvf._calc_adequacy_np(
        D, G_medoids, W_weights, U_membDegree_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix: ", elapsed)

    # check if results are identical
    areEqual_J = J_adequacy_iterative == J_adequacy_matrix
    print_formatted("J | Iterative == Matrix:", areEqual_J)
    print_formatted("Diff: ", str(abs(J_adequacy_iterative - J_adequacy_matrix)))

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
    G_bestMedoids_matrix = mvf._calc_best_medoids(D, U_membDegree_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix:", elapsed)

    # check if results are identical
    areEqual_G = G_bestMedoids_iterative == G_bestMedoids_matrix
    areEqual_G = areEqual_G.all()
    print_formatted("G | Iterative == Matrix:", areEqual_G)

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
    W_bestWeights_matrix = mvf._calc_best_weights(D, U_membDegree_matrix, G_bestMedoids_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix:", elapsed)

    # check if results are identical
    areEqual_W = W_bestWeights_iterative == W_bestWeights_matrix
    areEqual_W = areEqual_W.all()
    print_formatted("W | Iterative == Matrix:", areEqual_W)
    print_formatted("Diff: ", str(np.sum(abs(W_bestWeights_iterative - W_bestWeights_matrix))))

    # ---------------------------------------------------------------------------
    # Membership with weights
    # iterative
    print("---------------------------------------------")
    print("WEIGHTED Membership Vector (U | Eq. 6)")
    t.tic()
    U_membDegree_iterative = mvf_iterative.calc_membership_degree(
        D, G_bestMedoids_matrix, W_bestWeights_iterative, K, m)
    elapsed = t.tocvalue()
    print_formatted("Iterative: ", elapsed)

    # matrix (optimized)
    t.tic()
    U_membDegree_matrix = mvf._calc_membership_degree(
        D, G_bestMedoids_matrix, W_bestWeights_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Matrix: ", elapsed)

    # check if matrices are identical
    areEqual_U = U_membDegree_iterative == U_membDegree_matrix
    areEqual_U = areEqual_U.all()
    print_formatted("Uw | Iterative == Matrix:", areEqual_U)
    print_formatted("Diff: ", str(np.sum(abs(U_membDegree_iterative - U_membDegree_matrix))))


# should be improved, maybe generalized and always require the suffix
def print_formatted(name: str, value, suffix=True, end='\n'):
    if(type(value) is float):
        if(suffix):
            suffix = " seconds"
        else:
            suffix = ""
        value_str = "{:.8f}{}".format(value, suffix)
    else:
        value_str = str(value)
    print("{:25}{}".format(name, value_str), end=end)


def test_fuzzy_partition(D: np.array, K, m, T, err):
    """
    Calculate the fuzzy partition for a given input, measures the time and
    saves the results
    """
    t = TicToc()
    mvf = MVFuzzy()
    print("---------------------------------------------")
    print("Calculating fuzzy partition...", end="", flush=True)
    t.tic()
    mvf.run(D, K, m, T, err)
    elapsed = t.tocvalue()
    print("done!")
    print_formatted("Last Iteration:", mvf.lastIteration, suffix=False)
    print_formatted("Last J_t:", mvf.lastAdequacy, suffix=False)
    print_formatted("Elapsed time:", elapsed)
    np.save("G_medoids", mvf.bestMedoidVectors)
    np.save("W_weights", mvf.bestWeightVectors)
    np.save("U_membership", mvf.bestMembershipVectors)


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
    scaler = preprocessing.StandardScaler()
    norm_fac = scaler.fit_transform(mfeat_fac)
    norm_fou = scaler.fit_transform(mfeat_fou)
    norm_kar = scaler.fit_transform(mfeat_kar)

    # compute dissimilarity matrices
    D = np.zeros((3, 2000, 2000), dtype=float)
    D[0, :, :] = euclidean_distances(norm_fac)
    D[1, :, :] = euclidean_distances(norm_fou)
    D[2, :, :] = euclidean_distances(norm_kar)

    for i in range(10):
        test_matrix_iterative(D, 10, 1.6, 150, 10**-10)
        # test_fuzzy_partition(D, 10, 1.6, 150, 10**-10)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")


if __name__ == '__main__':
    main()
