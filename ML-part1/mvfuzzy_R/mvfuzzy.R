#'    """
#'    Multi-view fuzzy c-medoid (MVFCMddV) clustering method
#'    
#'    How to Run
#'    ----------
#'    Instantiate and call "run" with the appropriate parameters:
#'        K: int
#'            number of clusters
#'        D: numpy.array
#'            Dissimilarity matrix (n, n, p), where 'p' is the number of views
#'            and 'n' the number of elements
#'        m: float
#'            the fuzziness coefficient
#'        T: int
#'            maximum number of iterations
#'        err: float
#'            stop condition to determine when convergence has been achieved
#'
#'    Output
#'    ------
#'    The final fuzzy partition with it's weights and medoid vectors will be
#'    stored in the class members:
#'
#'        bestMedoidVectors: numpy.array
#'            Matrix G: medoid vectors of the last iteration, dimensions (K, p)
#'
#'        bestWeightVectors: numpy.array
#'            Matrix W: the weight vectors of the last iteration, dimensions (K, p)
#'
#'        bestMembershipVectors: numpy.array
#'            Matrix U: the membership vectors of the last iteration, dimensions (n, K)
#'    
#'    Final State on Last Iteration
#'    -----------------------------
#'    The final state of the iterations is also saved and may be accessed by the
#'    respective members:
#'
#'        lastIteration: int
#'            the number of interations until reach convergence
#'
#'        lastAdequacy: float
#'            the value of the last adequacy when reached convergence
#'
#'    Algorithm Details
#'    -----------------
#'    DE CARVALHO, Francisco de AT; DE MELO, Filipe M.; LECHEVALLIER, Yves. A multi-view relational fuzzy c-medoid vectors clustering algorithm. Neurocomputing, v. 163, p. 115-123, 2015.

#' Finds the best fuzzy partition. Read the class docs for more details.
#' @param D has dimensions (n_elem, n_elem, p_views)
mv.fuzzy <- function(D, K, m, T, err) {
  bestMedoidVectors = c();
  bestWeightVectors = c();
  bestMembershipVectors = c();
  lastIteration = c()
  lastAdequacy = 0.0

  n_elems = dim(D)[1]
  p_views = dim(D)[3]
        
  # initialize medoid vector G (3 views x K clusters)
  ## G_medoids = array(sample(seq(1, n_elems), K*p_views), dim=c(K,p_views))

  # np.random.seed(2)
  G_medoids =  c(c(1192,  527,  493),
                 c(1608, 1558,  299),
                 c( 466, 1099,  360),
                 c(1287,  674,  433),
                 c( 607,  587,  725),
                 c(1071,  831, 1311),
                 c( 730,  404,  124),
                 c(1652,  805,  679),
                 c(1219, 1126,  772),
                 c( 938,  875,   51))
  G_medoids = matrix(G_medoids, nrow=K, ncol=p_views, byrow=T)
  
  cat('Initial Medois: \n'); print(G_medoids); cat('\n\n')

  # initialize weight vector
  W_weights = array(rep(1.0, K*p_views), dim=c(K,p_views))

  # compute initial membership degree vector
  U_membDegree = calc_membership_degree(D, G_medoids, W_weights, K, m)
  
  # compute initial adequacy
  J_adequacy = calc_adequacy(D, G_medoids, W_weights, U_membDegree, K, m)
  cat('Initial Adequacy: \n'); print(J_adequacy); cat('\n\n')
  return(0)
  U_previous = U_membDegree
  J_previous = J_adequacy
  J_t = 0.0
  J_adequacy_difference = 0.0
  for(t in seq(1, T+1)) {
    lastIteration = t
    cat('t: ', t, '\n')
    cat('G\n')
    G_t = calc_best_medoids(D, U_previous, K, m)
    cat('W\n')
    W_t = calc_best_weights(D, U_previous, G_t, K, m)
    cat('U\n')
    U_t = calc_membership_degree(D, G_t, W_t, K, m)
    cat('J: ')
    J_t = calc_adequacy(D, G_t, W_t, U_t, K, m)
    cat(J_t, '\n')
    J_adequacy_difference = abs(J_previous - J_t)
    if(J_adequacy_difference < err)
         break
    else
      U_previous = U_t
    J_previous = J_t
  }
  result = list( 
    lastAdequacy = J_t,
    bestMedoidVectors = G_t,
    bestWeightVectors = W_t,
    bestMembershipVectors = U_t
  )
  return(result)
}

##     def toCrispPartition(:
##         """Returns an array with the cluster number where g_ik is maximum for each element"""
##         return np.argmax(bestMembershipVectors, axis=1)


##     def hasEmptyCluster( K):
##         crips_byClass = toCrispPartitionByClass(K)
##         for partition in crips_byClass:
##             if len(partition) == 0:
##                 return True
##         return False


##     def toCrispPartitionByClass( K):
##         crisp_partition = toCrispPartition()
##         n_elems = crisp_partition.shape[0]
##         partition_byClass = [[] for x in range(K)]
##         for i in range(n_elems):
##             k_cluster = crisp_partition[i]
##             partition_byClass[k_cluster - 1].append(i)
##         return partition_byClass


##     def printCrispPartitionByClass( K, onlyEmpty):
##         partition_byClass = toCrispPartitionByClass(K)
##         for k in range(K):
##             cur_list = partition_byClass[k]
##             if(onlyEmpty):
##                 len(cur_list)
##                 if(len(cur_list) == 0):
##                     print(" {}".format(k+1), end='', flush=True)
##             else:
##                 print("Cluster {} ({} elements):\n{}".format(k+1, len(cur_list), cur_list))
##                 print("-----------")


#' Calculate the best medoids vector according to Eq. 4
calc_best_medoids  <- function(D, U_membDegree, K, m) {
  p_views = dim(D)[3]

  # must specify as int because numpy default is float
  G_best_medoids = array(0, dim=c(10, 3))
  for(k in seq(K)) {
    for(j in seq(p_views)) {
      Akj_matrix = (U_membDegree[,k] ^ m) * D[,, j]
      A_kj = colSums(Akj_matrix)
      G_best_medoids[k, j] = which.min(A_kj)
    }
  }
  return(G_best_medoids)
}

#' Calculate the best membership degree vectors according to Eq. 6
calc_membership_degree <- function(D, G_medoids, W_weights, K, m) {
  n_elems = dim(D)[1]
  p_views = dim(D)[3]

  # initialize membership degree vector and subparts
  U_membDegree = array(0.0, dim=c(n_elems, K))
  for(k in seq(K)) {
    A_k = array(0.0, dim=c(n_elems))
    for(j in seq(p_views))
      A_k = + W_weights[k, j] * D[, G_medoids[k, j], j]
    for(h in seq(K)) {
      B_h = array(0.0, dim=c(n_elems))
      for(j in seq(p_views)) {
        B_h = + W_weights[h, j] * D[, G_medoids[h, j], j]
      }
      U_membDegree[, k] = U_membDegree[, k] + (A_k/(B_h + 1e-25))^(1.0/(m-1.0))
    }
  }
  U_membDegree = 1.0/(U_membDegree + 1e-25)
  U<<-U_membDegree
  return(U_membDegree)
}

#' Calculate the adequacy vectors according to Eq. 1
calc_adequacy <- function( D, G_medoids, W_weights, U_membDegree, K, m) {
  n_elems = dim(D)[1]
  p_views = dim(D)[3]

  Jk_mat = array(0.0, dim=c(n_elems, K))
  for(k in seq(K)) {
    for(j in seq(p_views))
      Jk_mat[, k] = Jk_mat[, k] + W_weights[k, j] * D[,G_medoids[k, j], j]
    Jk_mat[, k] = (U_membDegree[, k] ^ m) * Jk_mat[, k]
  }
  #Jk_mat2<<-Jk_mat
  return(sum(Jk_mat))
}

#' Calculate the best weights vectors according to Eq. 5
calc_best_weights <- function( D, U_previous, G_medoids, K, m) {
  p_views = dim(D)[3]

  W_weights = array(1.0, dim=c(K, p_views))
  for(k in seq(K)) {
    for(j in seq(p_views)) {
      A_kj = 1.0
      for(h in seq(p_views)) {
        Ckj_h = (U_previous[, k] ^ m) * D[, G_medoids[k, h], h]
        A_kj = A_kj * sum(Ckj_h)
      }
      A_kj = A_kj ^ (1.0/p_views)
      Bkj_column = (U_previous[, k] ^ m) * D[, G_medoids[k, j], j]
      B_kj = sum(Bkj_column)
      W_weights[k, j] = A_kj / (B_kj + 1e-25)
    }
  }
  return(W_weights)
}
