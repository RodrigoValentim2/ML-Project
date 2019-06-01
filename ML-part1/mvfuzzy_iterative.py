##############################################
# loop, iterative, non-matrix, calculations  #
# to verify that matrix versions are correct #
##############################################
import numpy as np


def calc_membership_degree(D, G_medoids, W_weights, K, m):
    p_views = D.shape[0]
    n_elems = D.shape[1]

    # initialize membership degree vector
    U_membDegree = np.zeros(shape=[n_elems, K], dtype=float)
    for i in range(n_elems):
        for k in range(K):
            u_ik = 0.
            # calc A_ik
            A_ik = 0.
            for j in range(p_views):
                A_ik += W_weights[k, j] * D[j, i, G_medoids[k, j]]
            # calc B_ik
            for h in range(K):
                B_ih = 0.
                for j in range(p_views):
                    B_ih += W_weights[h, j] * D[j, i, G_medoids[h, j]]
                u_ik += (A_ik/(B_ih + 1e-25))**(1./(m-1.))
            # calc final U_ik
            U_membDegree[i, k] = 1./(u_ik+ 1e-25)
    return U_membDegree


def calc_adequacy(D, G_medoids, W_weights, U_membDegree, K, m):
    p_views = D.shape[0]
    n_elems = D.shape[1]

    J_adequacy = 0.0
    for k in range(K):
        for i in range(n_elems):
            dw_k = 0.0
            for j in range(p_views):
                dw_k += W_weights[k, j] * D[j, i, G_medoids[k, j]]
            J_adequacy += (U_membDegree[i, k] ** m) * dw_k
    return J_adequacy


def calc_best_medoids(D, U_membDegree, K, m):
    p_views = D.shape[0]
    n_elems = D.shape[1]

    # must specify as int because numpy default is float
    G_best_medoids = np.zeros((K, p_views), dtype=int)
    for k in range(K):
        for j in range(p_views):
            sum_previous = float("inf")
            for h in range(n_elems):
                sum_h = 0
                for i in range(n_elems):
                    sum_h += (U_membDegree[i, k] ** m) * D[j, i, h]
                if sum_h < sum_previous:
                    l_kj = h
                    sum_previous = sum_h
            G_best_medoids[k, j] = l_kj
    return G_best_medoids


def calc_best_weights(D, U_previous, G_medoids, K, m):
    p_views = D.shape[0]
    n_elems = D.shape[1]

    W_weights = np.ones(shape=[K, p_views], dtype=float)
    for k in range(K):
        for j in range(p_views):
            # A: upper member of Eq. 5
            A_kj = 1.
            for h in range(p_views):
                # summatory in upper member of Eq. 5
                sum_h = 0.0
                for i in range(n_elems):
                    sum_h += (U_previous[i, k] ** m) * D[h, i, G_medoids[k, h]]
                A_kj *= sum_h
            A_kj = A_kj ** (1/p_views)
            # B: bottom member of Eq. 5
            B_kj = 0.0
            for i in range(n_elems):
                B_kj += (U_previous[i, k] ** m) * D[j, i, G_medoids[k, j]]
            W_weights[k, j] = A_kj / B_kj
    return W_weights
