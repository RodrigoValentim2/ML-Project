#!/usr/bin/env python
# %% [markdown]
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

# %% Setup
from mvfuzzy import MVFuzzy
import pandas as pd
import numpy as np
import copy
import io_utils as io
from pathlib import Path
from pytictoc import TicToc
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score

# RANDOM_SEED = 495924220
RANDOM_SEED = 572953998
# RANDOM_SEED = 2
PARAM_K = 10
PARAM_m = 1.6
PARAM_T = 150
PARAM_e = 10**-10
REPETITIONS = 100

OUTPUT_DIR = Path("results") / "seed_{}".format(RANDOM_SEED)

# %% [markdown]
# ## 2. Prepara os dados de entrada

# %%
# lê os dados
mfeat_fac = pd.read_csv(
    "mfeat/mfeat-fac", sep="\\s+", header=None, dtype=float)
mfeat_fou = pd.read_csv(
    "mfeat/mfeat-fou", sep="\\s+", header=None, dtype=float)
mfeat_kar = pd.read_csv(
    "mfeat/mfeat-kar", sep="\\s+", header=None, dtype=float)

# calcula partição à priori
apriori_partition = (np.array(range(0, 2000)) // 200)

# normalization
scaler = preprocessing.StandardScaler()
norm_fac = scaler.fit_transform(mfeat_fac)
norm_fou = scaler.fit_transform(mfeat_fou)
norm_kar = scaler.fit_transform(mfeat_kar)

# calcula as matrizes de dissimilaridade
D = np.zeros((3, 2000, 2000))
D[0, :, :] = euclidean_distances(norm_fac)
D[1, :, :] = euclidean_distances(norm_fou)
D[2, :, :] = euclidean_distances(norm_kar)

# %% [markdown]
# ## 3. Execução do algoritmos
# 1. Fixa seed inicial para prover repetibilidade
# 2. Executa 100 vezes
# 3. Guarda resultado para aquele com menor J (função objetivo)
#
# > ainda é possível que resultado varie caso, durante a execução, o numpy seja chamado em outro código (execução em paralelo), pelo que entendi do FAQ. Porém isso nunca ocorrerá em nosso cenário, logo a reprodutibilidade é garantida em nosso cenário.

# %% Executa algoritmo MVFCMddV
t = TicToc()
best_result = MVFuzzy()
mvf = MVFuzzy()
best_iteration = 0
np.random.seed(RANDOM_SEED)
J_previous = float("Inf")
t.tic()
print("{:5}| {:15}  | {:15}  | {:5}  | {}".format("It", "J_t", "Best J_t", "Has Empty", "Last t"))
for i in range(REPETITIONS):
    # print('\n---------------------')
    # print("Current iteration:", i)  # , end="\r", flush=True)
    # print("Current iteration: {}  |  J: {}".format(i, mvf.lastAdequacy), end="\r", flush=True)
    mvf.run(D, PARAM_K, PARAM_m, PARAM_T, PARAM_e)
    if (mvf.lastAdequacy < J_previous) and not mvf.hasEmptyCluster(PARAM_K):
        J_previous = mvf.lastAdequacy
        best_result = copy.copy(mvf)
        best_iteration = i + 1
        Jbest_print = "{:<16.8f}".format(mvf.lastAdequacy)
    else:
        Jbest_print = "{:<16}".format('')
    if mvf.hasEmptyCluster(PARAM_K):
        has_empty_str = "True: {}".format(mvf.getEmptyClasses(PARAM_K))
    else:
        has_empty_str = ""
    print("{:5}| {:<16.8f} | {Jbest} | {:10} | {}"
          .format(i+1, mvf.lastAdequacy, has_empty_str, mvf.lastIteration, Jbest=Jbest_print),flush=True)
t.toc("Fuzzy algorithm 100x: ")

# %% [markdown]
# ## 4. Resultados do Particionamento com MVFCMddV

# %% Avalia resultados e gera listas para relatório
crisp_mvf_partition = best_result.toCrispPartition()
rand_score = adjusted_rand_score(apriori_partition, crisp_mvf_partition)
final_medoids_vector = best_result.bestMedoidVectors
print("Adjusted Rand Score:", rand_score)
print("Best iteration (from 100):", best_iteration)
print("Last Adequacy: ", best_result.lastAdequacy)

# %% Vetores finais
print("----------------------------")
print("FINAL MEDOID VECTORS (G)")
print(final_medoids_vector)

print("----------------------------")
print("FINAL MEMBERSHIP VECTORS (U)")
print(best_result.bestMembershipVectors)

print("----------------------------")
print("BEST PARTITION (by class)")
partition_byCluster = best_result.toCrispPartitionByClass(PARAM_K)
best_result.printCrispPartitionByClass(PARAM_K)

# %% Salva resultados finais em arquivo
io.prepare_outdir(OUTPUT_DIR)

# formato numpy para testes futuros
np.save(OUTPUT_DIR / "fuzzy_bestMedoids",    best_result.bestMedoidVectors)
np.save(OUTPUT_DIR / "fuzzy_bestMembership", best_result.bestMembershipVectors)
np.save(OUTPUT_DIR / "fuzzy_bestWeights",    best_result.bestWeightVectors)

# formato CSV para relatório
pd.DataFrame(crisp_mvf_partition).to_csv(
    str(OUTPUT_DIR / "fuzzy_crisp_partition.csv"),
    index=True, header=False)
