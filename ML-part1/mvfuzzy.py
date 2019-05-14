import pandas as pd
import numpy as np
from pytictoc import TicToc
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import random


class MVFuzzy:
    """
    Multi-view fuzzy c-medoid (MVFCMddV) clustering method

    How to Run
    ----------
    Instantiate and call "run" with the appropriate parameters:
        K: int
            number of clusters
        D: numpy.array
            Dissimilarity matrix (n, n, p), where 'p' is the number of views
            and 'n' the number of elements
        m: float
            the fuzziness coefficient
        T: int
            maximum number of iterations
        err: float
            stop condition to determine when convergence has been achieved

    Output
    ------
    The final fuzzy partition with it's weights and medoid vectors will be
    stored in the class members:

        bestMedoidVectors: numpy.array
            Matrix G: medoid vectors of the last iteration, dimensions (K, p)

        bestWeightVectors: numpy.array
            Matrix W: the weight vectors of the last iteration, dimensions (K, p)

        bestMembershipVectors: numpy.array
            Matrix U: the membership vectors of the last iteration, dimensions (n, K)

    Final State on Last Iteration
    -----------------------------
    The final state of the iterations is also saved and may be accessed by the
    respective members:

        lastIteration: int
            the number of interations until reach convergence

        lastAdequacy: float
            the value of the last adequacy when reached convergence

    Algorithm Details
    -----------------
    DE CARVALHO, Francisco de AT; DE MELO, Filipe M.; LECHEVALLIER, Yves. A multi-view relational fuzzy c-medoid vectors clustering algorithm. Neurocomputing, v. 163, p. 115-123, 2015.
    """

    def __init__(self):
        self.bestMedoidVectors = np.empty
        self.bestWeightVectors = np.empty
        self.bestMembershipVectors = np.empty
        self.lastIteration = np.empty
        self.lastAdequacy = 0.0

    def run(self, D: np.array, K, m, T, err):
        """Finds the best fuzzy partition. Read the class docs for more details."""
        p_views = D.shape[0]
        n_elems = D.shape[1]

        # initialize medoid vector G (3 views x K clusters)
        G_medoids = np.random.randint(n_elems, size=(K, p_views))

        # initialize weight vector
        W_weights = np.ones(shape=[K, p_views], dtype=float)

        # compute initial membership degree vector
        U_membDegree = self._calc_membership_degree(D, G_medoids, W_weights, K, m)

        # print('Ite  | Empty Clusters'); print('T: 0 | '); self.bestMembershipVectors = U_membDegree; self.printEmptyClasses(K, True)

        # compute initial adequacy
        J_adequacy = self._calc_adequacy(D, G_medoids, W_weights, U_membDegree, K, m)
        U_previous = U_membDegree
        J_previous = J_adequacy
        J_t = 0.0
        J_adequacy_difference = 0.0
        for t in range(1, T+1):
            self.lastIteration = t
            G_t = self._calc_best_medoids(D, U_previous, K, m)
            W_t = self._calc_best_weights(D, U_previous, G_t, K, m)
            U_t = self._calc_membership_degree(D, G_t, W_t, K, m)
            J_t = self._calc_adequacy(D, G_t, W_t, U_t, K, m)
            J_adequacy_difference = abs(J_previous - J_t)
            if J_adequacy_difference < err:
                break
            else:
                U_previous = U_t
                J_previous = J_t
            # self.bestMembershipVectors = U_t; print('T:', t, '|', end='', flush=True);  self.printEmptyClasses(K, True); print('', flush=True)
        self.lastAdequacy = J_t
        self.bestMedoidVectors = G_t
        self.bestWeightVectors = W_t
        self.bestMembershipVectors = U_t

    def toCrispPartition(self):
        """Returns an array with the cluster number where g_ik is maximum for each element"""
        return np.argmax(self.bestMembershipVectors, axis=1)

    def hasEmptyCluster(self, K):
        crisp_byClass = self.toCrispPartitionByClass(K)
        for partition in crisp_byClass:
            if len(partition) == 0:
                return True
        return False

    def toCrispPartitionByClass(self, K):
        crisp_partition = self.toCrispPartition()
        n_elems = crisp_partition.shape[0]
        partition_byClass = [[] for x in range(K)]
        for i in range(n_elems):
            k_cluster = crisp_partition[i]
            partition_byClass[k_cluster - 1].append(i)
        return partition_byClass

    def printCrispPartitionByClass(self, K):
        partition_byClass = self.toCrispPartitionByClass(K)
        for k in range(K):
            cur_list = partition_byClass[k]
            print("Cluster {} ({} elements):\n{}".format(k+1, len(cur_list), cur_list))
            print("-----------")

    def printEmptyClasses(self, K):
        partition_byClass = self.toCrispPartitionByClass(K)
        for k in range(K):
            cur_list = partition_byClass[k]
            len(cur_list)
            if(len(cur_list) == 0):
                print(" {}".format(k+1), end='', flush=True)

    def _calc_best_medoids(self, D, U_membDegree, K, m):
        """Calculate the best medoids vector according to Eq. 4"""
        p_views = D.shape[0]

        # must specify as int because numpy default is float
        G_best_medoids = np.zeros((K, p_views), dtype=int)
        for k in range(K):
            for j in range(p_views):
                # multiply column U_k to every column of D_j
                # Lkj_matrix:
                #     each row h for each column i
                #     shape = (n_elems, n_elems)
                #     Eq. 4 = (h, i)
                Lkj_matrix = (U_membDegree[:, k] ** m) * D[j, :, :]
                #
                # for each row, sum all columns (fix axis=0 and sum along axis=1)
                # L_kj: shape = (n_elems) | Eq.4: (h)
                L_kj = np.sum(Lkj_matrix, axis=1)
                G_best_medoids[k, j] = np.argmin(L_kj)
        return G_best_medoids

    def _calc_membership_degree(self, D, G_medoids, W_weights, K, m):
        """(Python sum) Calculate the best membership degree vectors according to Eq. 6"""
        p_views = D.shape[0]
        n_elems = D.shape[1]

        # initialize membership degree vector and subparts
        U_membDegree = np.zeros(shape=[n_elems, K], dtype=float)
        for k in range(K):
            # A: numerator
            # compute A column for current k (outter k)
            A_k = np.zeros((n_elems), dtype=float)
            for j in range(p_views):
                A_k += W_weights[k, j] * D[j, :, G_medoids[k, j]]
            #
            # B: denominator
            # compute B for each h (inner k) + already calculate U_k
            for h in range(K):
                B_h = np.zeros((n_elems), dtype=float)
                for j in range(p_views):
                    B_h += W_weights[h, j] * D[j, :, G_medoids[h, j]]
                #
                # accumulate U for each h (inner k)
                U_membDegree[:, k] += (A_k/(B_h + 1e-25))**(1./(m-1.))
        # final operation (after inner k sum) on each element of U
        U_membDegree = 1./(U_membDegree + 1e-25)
        return U_membDegree

    def _calc_adequacy(self, D, G_medoids, W_weights, U_membDegree, K, m):
        """(Python sum) Calculate the adequacy vectors according to Eq. 1 using"""
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

    def _calc_adequacy_np(self, D, G_medoids, W_weights, U_membDegree, K, m):
        """(NumPy sum) Calculate the adequacy vectors according to Eq. 1 using"""
        p_views = D.shape[0]
        n_elems = D.shape[1]

        Jk_mat = np.zeros((n_elems, K), dtype=float)
        for k in range(K):
            for j in range(p_views):
                Jk_mat[:, k] += W_weights[k, j] * D[j, :, G_medoids[k, j]]
            Jk_mat[:, k] = (U_membDegree[:, k] ** m) * Jk_mat[:, k]
        return Jk_mat.sum()

    def _calc_best_weights(self, D, U_previous, G_medoids, K, m):
        """(Python sum) Calculate the best weights vectors according to Eq. 5"""
        p_views = D.shape[0]

        W_weights = np.ones(shape=[K, p_views], dtype=float)
        for k in range(K):
            for j in range(p_views):
                # A: numerator of Eq. 5
                A_kj = 1.
                for h in range(p_views):
                    # C: summatory in numerator of Eq. 5
                    Ckj_h = (U_previous[:, k] ** m) * D[h, :, G_medoids[k, h]]
                    A_kj = A_kj * sum(Ckj_h)
                A_kj = A_kj ** (1./p_views)
                #
                # B: denominator of Eq. 5
                Bkj_column = (U_previous[:, k] ** m) * D[j, :, G_medoids[k, j]]
                B_kj = sum(Bkj_column)
                W_weights[k, j] = A_kj / (B_kj + 1e-25)
        return W_weights
