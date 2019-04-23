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
import mvfuzzy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
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
    "mfeat/mfeat-fou", sep="\\s+", header=None, dtype=float)
mfeat_kar = pd.read_csv(
    "mfeat/mfeat-kar", sep="\\s+", header=None, dtype=float)

# normaliza
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

#%% Executa algoritmo MVFCMddV
np.random.seed(RANDOM_SEED)
mvfuzzy.calc_fuzzy_partition(D, PARAM_K, PARAM_m, PARAM_T, PARAM_e)

#%%