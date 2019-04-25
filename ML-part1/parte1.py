#%% [markdown]
# ## 1. Setup
# Carrega bibliotecas e funções para cálculo das partições fuzzy, além de definir
# os parâmetros para a execução do algoritmo MVFCMddV.
#
# Também uma seed é fixada para facilitar análises posteriores, ela foi obtida
# executando:
# ```pyton
# import sys
# import random
# random.SystemRandom().randint(0, 2**32-1)
# ```

#%% Setup
from mvfuzzy import MVFuzzy
import pandas as pd
import numpy as np
import copy
from pytictoc import TicToc
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score

RANDOM_SEED = 495924220
PARAM_K = 10
PARAM_m = 1.6
PARAM_T = 150
PARAM_e = 10**-10

#%% [markdown]
# ## 2. Prepara os dados de entrada

#%%
# lê os dados
mfeat_fac = pd.read_csv(
    "mfeat/mfeat-fac", sep="\\s+", header=None, dtype=float)
mfeat_fou = pd.read_csv(
    "mfeat/mfeat-fou", sep="\\s+", header=None, dtype=floatiteration)
mfeat_kar = pd.read_csv(
    "mfeat/mfeat-kar", sep="\\s+", header=None, dtype=float)

# calcula partição à priori
apriori_partition = 1 + (np.array(range(0, 2000)) // 200)

# normalizaiteration
scaler = preprocessing.MinMaxScaler()
norm_fac = scaler.fit_transform(mfeat_fac)
norm_fou = scaler.fit_transform(mfeat_fou)
norm_kar = scaler.fit_transform(mfeat_kar)

# calcula as matrizes de dissimilaridade
D = np.zeros((2000, 2000, 3))
D[:, :, 0] = euclidean_distances(norm_fac)
D[:, :, 1] = euclidean_distances(norm_fou)
D[:, :, 2] = euclidean_distances(norm_kar)

#%% [markdown]
# ## 3. Execução do algoritmos
# 1. Fixa seed inicial para prover repetibilidade
# 2. Executa 100 vezes
# 3. Guarda resultado para aquele com menor J (função objetivo)
#
# > ainda é possível que resultado varie caso, durante a execução, o numpy seja chamado em outro código (execução em paralelo), pelo que entendi do FAQ. Porém isso nunca ocorrerá em nosso cenário, logo a reprodutibilidade é garantida em nosso cenário.

#%% Executa algoritmo MVFCMddV
t = TicToc()
best_result = MVFuzzy()
mvf = MVFuzzy()
best_iteration = 0
np.random.seed(RANDOM_SEED)
J_previous = float("Inf")
t.tic()
for i in range(0, 100):
    print("Current iteration:", i, end="\r", flush=True)
    mvf.run(D, PARAM_K, PARAM_m, PARAM_T, PARAM_e)
    if mvf.lastAdequacy < J_previous:
        J_previous = mvf.lastAdequacy
        best_result = copy.copy(mvf)
        best_iteration = i + 1
t.toc("Fuzzy algorithm 100x: ")

#%% Salva arrays finais em arquivo
np.save("fuzzy_bestMedoids", best_result.bestMedoidVectors)
np.save("fuzzy_bestMembership", best_result.bestMembershipVectors)
np.save("fuzzy_bestWeights", best_result.bestWeightVectors)

#%% [markdown]
# ## 4. Resultados do Particionamento com MVFCMddV

#%% Avalia resultados e gera listas para relatório
crisp_mvf_partition = best_result.toCrispPartition()
rand_score = adjusted_rand_score(apriori_partition, crisp_mvf_partition)
final_medoids_vector = best_result.bestMedoidVectors

#%% Índice Rand Ajustadopartition_byCluster[k]:
print("Adjusted Rand Score:", rand_score)
print("Best iteration (from 100):", best_iteration)

#%% Melhor Partição: Vetor de Medoids ($G$)
print(final_medoids_vector)

#%% Melhor Partição: elementos por classe
partition_byCluster = [[] for x in range(0, PARAM_K)]
n_elems = crisp_mvf_partition.shape[0]
for i in range(0, n_elems):
    k_cluster = crisp_mvf_partition[i]
    partition_byCluster[k_cluster - 1].append(i)

for k in range(0, PARAM_K):
    cur_list = partition_byCluster[k]
    print("Cluster {} ({} elements):\n{}".format(k+1, len(cur_list), cur_list))
    print("-----------")

#%% salva em CSV a partição crisp final
pd.DataFrame(crisp_mvf_partition).to_csv("fuzzy_crisp_partition.csv", index=False,header=False)
