##############################################
# loop, iterative, non-matrix, calculations  #
# to verify that matrix versions are correct #
##############################################
import numpy as np


def calc_membership_degree(D, G_medoids, W_weights, K, m):
    n_elems = D.shape[0]
    p_views = D.shape[2]

    # initialize membership degree vector
    U_membDegree = np.zeros(shape=[n_elems, K], dtype=float)
    for i in range(0, n_elems):
        for k in range(0, K):
            u_ik = 0
            # calc A_ik
            A_ik = 0
            for j in range(0, p_views):
                A_ik += W_weights[k, j] * D[i, G_medoids[k, j], j]
            # calc B_ik
            for h in range(0, K):
                B_ih = 0
                for j in range(0, p_views):
                    B_ih += W_weights[h, j] * D[i, G_medoids[h, j], j]
                u_ik += (A_ik/B_ih)**(1/(m-1))
            # calc final U_ik
            U_membDegree[i, k] = 1/u_ik
    return U_membDegree


def calc_adequacy(D, G_medoids, W_weights, U_membDegree, K, m):
    n_elems = D.shape[0]
    p_views = D.shape[2]

    J_adequacy = 0
    for k in range(0, K):
        for i in range(0, n_elems):
            dw_k = 0
            for j in range(0, p_views):
                dw_k += W_weights[k, j] * D[i, G_medoids[k, j], j]
            J_adequacy += (U_membDegree[i, k] ** m) * dw_k
    return J_adequacy
