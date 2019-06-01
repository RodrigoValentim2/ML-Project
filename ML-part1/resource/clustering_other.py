import numpy


def argminIndex(list,exclude):
    argmin = min(list)
    argmax = max(list)
    indexList = []
    for i in range(len(list)):
        if (list[i] == argmin) :
            if i in exclude:
                list[i] = argmax
                i = argminIndex(list,exclude)
            indexList.append(i)
    return indexList[0]


def J(m,E,K,G,L,U, n, p): #equacao 1: funcao objetivo
	val = 0
	for k in range(K):
		for i in range(n):
			summ = 0
			for j in range(p):
				summ += L[k][j] * E[j][i][G[k][j]]
			val += (U[i][k] ** m) * summ
	return val

def update_G(m,E,K,G,L,U, n, p): #equacao 4: atualiza G
	nG=[] #new G -> G(t) updated
	for k in range(K):
		g_k = []
		for j in range(p):
			arglist = []
			for h in range (n):
				summ = 0
				for i in range (n):
					summ += ((U[i][k] ** m) * (E[j][i][h]))
				arglist.append(summ)
			l = argminIndex(arglist,g_k)
			g_k_j = l
			g_k.append(g_k_j)
		nG.append(g_k)
	return numpy.array(nG)

def update_L(m,E,K,G,L,U, n, p): #equacao 5: atualiza L
	nL = []
	for k in range(K):
		l_k = []
		for j in range(p):
			prod = 1
			for h in range(p):
				summ = 0
				for i in range(n):
					summ += ( (U[i][k]**m) * E[h][i][G[k][h]] )
				prod *= summ
			numerator = prod
			denominator = 0
			for i in range(n):
				denominator += ( (U[i][k]**m) * E[j][i][G[k][j]] )
			l_k_j = (float(numerator) ** (1./p))/denominator
			l_k.append(l_k_j)
		nL.append(l_k)
	return numpy.array(nL)

def update_U(m,E,K,G,L,U, n, p): #equacao 6: atualiza U
	nU = []
	for i in range(n):
		u_i = []
		for k in range(K):
			summ = 0
			for h in range(K):
				numerator = 0
				denominator = 0
				for j in range(p):
					numerator += L[k][j] * E[j][i][G[k][j]]
				for j in range(p):
					denominator+= L[h][j] * E[j][i][G[h][j]]
				summ += ( float(numerator) /(denominator + 1e-25 )) ** (1./(m-1.))
			u_i_k = ( summ  + 1e-25  ) ** -1.
			u_i.append(u_i_k)
		nU.append(u_i)
	return numpy.array(nU)