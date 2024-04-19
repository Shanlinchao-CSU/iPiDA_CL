
import numpy as np

import pandas as pd
import torch.nn as nn
import torch

pirna_idx = pd.read_csv('../Independent test data/piRNA.csv').values
disease_idx = pd.read_csv('../Independent test data/disease.csv').values
n = len(disease_idx)
m = len(pirna_idx)

ass1 = pd.read_csv('../Independent test data/positive.csv', delimiter=',', dtype=int).to_numpy()
mirna_interaction_profile1 = ass1
disease_interaction_profile1 = ass1.T

mirna_gip_similarity = np.zeros((m, m))
theta_r = 0
for i in range(m):
    a = mirna_interaction_profile1[i]
    tem2 = np.linalg.norm(a) ** 2
    theta_r += tem2
theta_r = n / theta_r
print('theta_r = {}'.format(theta_r))

for i in range(m):
    for j in range(m):
        if i == j:
            mirna_gip_similarity[i, j] = 1
        else:
            temp1 = mirna_interaction_profile1[i] - mirna_interaction_profile1[j]
            temp1 = np.linalg.norm(temp1) ** 2
            temp = - theta_r * temp1
            temp = np.exp(temp)
            mirna_gip_similarity[i, j] = mirna_gip_similarity[j, i] = temp
np.save('../dataset/piSim.npy', mirna_gip_similarity)

# disease-GIP
#%%


#%%

disease_gip_similarity = np.zeros((n, n))
disease_gip_similarity.shape

#%%

theta_r = 0
for i in range(n):
    tem2 = np.linalg.norm(disease_interaction_profile1[i]) ** 2
    theta_r += tem2
theta_r = m / theta_r
print('theta_r = {}'.format(theta_r))

for i in range(n):
    for j in range(n):
        if i == j:
            disease_gip_similarity[i, j] = 1
        else:
            temp1 = disease_interaction_profile1[i] - disease_interaction_profile1[j]
            temp1 = np.linalg.norm(temp1) ** 2
            temp = - theta_r * temp1
            temp = np.exp(temp)
            disease_gip_similarity[i, j] = disease_gip_similarity[j, i] = temp
np.save('../dataset/DiseaSim.npy', disease_gip_similarity)

#%% md
