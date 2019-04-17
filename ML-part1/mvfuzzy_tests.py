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

def test_mvfuzzy(D: np.array, K, m, T, err):
    n_elems = D.shape[0]
    p_views = D.shape[2]
    t = TicToc()

    # Initial medoids selection
    rand_elements = random.sample_without_replacement(n_elems, K * p_views)
    G_medoids = np.zeros(shape=[K, p_views], dtype=int)
    for k in range(0, K):
        for p in range(0, p_views):
            G_medoids[k, p] = rand_elements[k*p_views + p]  # p cols, k lines

    # Initial weight vector
    W_weights = np.ones(shape=[K, p_views], dtype=float)

    # ---------------------------------------------------------------------------
    # Membership degree vector calculation
    # iterative
    t.tic()
    U_membDegree_iterative = mvf_iterative.calc_membership_degree(
        D, G_medoids, W_weights, K, m)
    elapsed = t.tocvalue()
    print_formatted("Membership vector (iterative): ", elapsed)

    # matrix (optimized)
    t.tic()
    U_membDegree_matrix = mvf.calc_membership_degree(
        D, G_medoids, W_weights, K, m)
    elapsed = t.tocvalue()
    print_formatted("Membership vector (matrix): ", elapsed)

    # check if matrices are identical
    areEqual_U = U_membDegree_iterative == U_membDegree_matrix
    areEqual_U = areEqual_U.any()
    print_formatted("Membership iterative == matrix:", areEqual_U)

    # ---------------------------------------------------------------------------
    # Adequacy calculation
    # iterative
    t.tic()
    J_adequacy_iterative = mvf_iterative.calc_adequacy(
        D, G_medoids, W_weights, U_membDegree_iterative, K, m)
    elapsed = t.tocvalue()
    print_formatted("Adequacy (iterative): ", elapsed)

    # matrix (optimized)
    t.tic()
    J_adequacy_matrix = mvf.calc_adequacy(
        D, G_medoids, W_weights, U_membDegree_matrix, K, m)
    elapsed = t.tocvalue()
    print_formatted("Adequacy (matrix): ", elapsed)

    # check if results are identical
    # must use math.isclose() to compare floats to deal with aproximation errors
    areEqual_J = math.isclose(J_adequacy_iterative, J_adequacy_matrix)
    print_formatted("Adequacy iterative == matrix:", areEqual_J)


def print_formatted(name: str, value, suffix=True):
    if(type(value) is float):
        if(suffix):
            suffix = " seconds"
        else:
            suffix = ""
        value_str = "{:.8f}{}".format(value, suffix)
    else:
        value_str = str(value)
    print("{:35}{}".format(name, value_str))


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

    return test_mvfuzzy(D, 10, 1.6, 150, 10**-10)


if __name__ == '__main__':
    main()
