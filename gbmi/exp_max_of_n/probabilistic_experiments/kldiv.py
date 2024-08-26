# %%
import torch
from gbmi.exp_max_of_n.train import MAX_OF_4_CONFIG
import numpy as np
from gbmi.exp_max_of_n.train import MAX_OF_10_CONFIG
from gbmi.model import train_or_load_model
from math import *
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import binom

rundata, model = train_or_load_model(MAX_OF_4_CONFIG(123))

# rundata, model = train_or_load_model(MAX_OF_10_CONFIG)
length = 4
attn_scale_0 = model.blocks[0].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E
W_U = model.W_U
W_U = W_U - W_U.mean(dim=1, keepdim=True)
W_K_0 = model.W_K[0, 0]
W_V_0 = model.W_V[0, 0]
W_O_0 = model.W_O[0, 0]
W_Q_0 = model.W_Q[0, 0]
W_Q_0 = model.W_Q[0, 0]
EQKP = (W_E @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKP = (W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKE = (W_pos @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)
EQKE = (W_E @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)
PVOU = W_pos @ W_V_0 @ W_O_0 @ W_U
EVOU = W_E @ W_V_0 @ W_O_0 @ W_U
o = W_O_0
v = W_V_0
at = W_E @ W_Q_0 @ W_K_0.T @ W_E.T + W_pos[length - 1] @ W_Q_0 @ W_K_0 @ W_E.T


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


accuracies = []


# %%
def kl_divergence(p, q):

    return p @ torch.log(p / q)


kl_divergences = []
for i in range(1000):
    sequence = torch.randint(low=0, high=64, size=(4,))
    if len(list(sequence)) != len(set(list(sequence))):
        continue
    logits, cache = model.run_with_cache(sequence)
    softmax = cache["attn", 0].squeeze()[-1]
    kl = kl_divergence(softmax, torch.tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4]))
    kl_divergences.append(kl)

kl_clusters = []
distance = 0.01
for i in range(1000):
    kl = kl_divergences[i]
    distinct_cluster = True
    for l in kl_clusters:
        if abs(l - kl) < distance:
            distinct_cluster = False
    if distinct_cluster:
        kl_clusters.append(kl)
accounted_sequences = []
for i in range(1000):
    sequence = torch.randint(low=0, high=64, size=(4,))
    sequence_list = [i.item() for i in list(sequence)]
    if len(sequence_list) != len(set(sequence_list)):
        continue
    logits, cache = model.run_with_cache(sequence)
    softmax = cache["attn", 0].squeeze()[-1]
    kl = kl_divergence(softmax, torch.tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4]))

    for f in kl_clusters:
        if f.item() in accounted_sequences:
            continue
        if abs(kl - f) < distance:
            print("HERE")
            accounted_sequences.append(f.item())
            print(sequence)
            print(f)
# %%
