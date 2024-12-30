# %%
from gbmi.exp_indhead.train import ABCAB8_1H
from torch import where
from gbmi.model import train_or_load_model
import torch
from torch import tensor
from math import *
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
import copy
from inspect import signature

import plotly.express as px


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
runtime_model_1, model = train_or_load_model(ABCAB8_1H, force="load")
model.to(device)

W_pos = model.W_pos
W_E = model.W_E
n_ctx = W_pos.shape[0]
d_voc = W_E.shape[0]
d_model = W_E.shape[1]


# %%
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E
W_K_1 = model.W_K[1, 0]
W_U = model.W_U
W_V_1 = model.W_V[1, 0]
W_K_0 = model.W_K[0, 0]
W_V_0 = model.W_V[0, 0]
W_O_0 = model.W_O[0, 0]
W_Q_1 = model.W_Q[1, 0]
W_Q_0 = model.W_Q[0, 0]
W_O_1 = model.W_O[1, 0]
W_Q_0 = model.W_Q[0, 0]
o = W_O_0
v = W_V_0
q_1 = W_Q_1
k_1 = W_K_1
v_1 = W_V_1
o_1 = W_O_1
# %%


EQKP = (W_E @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKP = (W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKE = (W_pos @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)
EQKE = (W_E @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)


# %%

pos_pattern_pres = []
for index in range(1, 9):
    pos_pattern_pres.append(
        torch.softmax(PQKP[index - 1, :index] + EQKP[:, :index], dim=1)
    )

other_parts = torch.exp(PQKE[-index] + EQKE)


# %%
pvo = torch.zeros(8, 64)
for index in range(1, 9):
    pvo[index - 1] = W_pos[index - 1] + (
        (W_pos[:index] @ v @ o) * (pos_pattern_pres[index - 1].mean(dim=0)).unsqueeze(1)
    ).sum(dim=0)


# %%
pvoqkpvo = (pvo @ q_1 @ k_1.T @ pvo.T) / (attn_scale_1)
eqkpvo = (W_E @ q_1 @ k_1.T @ pvo.T) / (attn_scale_1)
evoqkpvo = (W_E @ v @ o @ q_1 @ k_1.T @ pvo.T) / (attn_scale_1)
# %%
index = 6
pvo_pattern = torch.softmax(
    eqkpvo[:, :index] + evoqkpvo[:, :index].mean() + pvoqkpvo[index - 1, :index], dim=1
)
show(pvo_pattern)
# %%
pvoqke = (pvo @ q_1 @ k_1.T @ W_E.T) / (attn_scale_1)
eqke = (W_E @ q_1 @ k_1.T @ W_E.T) / (attn_scale_1)
evoqke = (W_E @ v @ o @ q_1 @ k_1.T @ W_E.T) / (attn_scale_1)
pvoqkevo = (W_pos @ v @ o @ q_1 @ k_1.T @ (W_E @ v @ o).T) / (attn_scale_1)
evoqkevo = (W_E @ v @ o @ q_1 @ k_1.T @ (W_E @ v @ o).T) / (attn_scale_1)
# %%
# e in itself
show(pvoqkevo)
show(evoqkevo)
show(eqkevo)
show(pvoqke)
show(eqke)  # a -> b
# a -> a
show(evoqke)  # c -> a
# c - > a
# %%
pvoqkevo = (pvo @ q_1 @ k_1.T @ (W_E @ v @ o).T) / (attn_scale_1)
eqkevo = (W_E @ q_1 @ k_1.T @ (W_E @ v @ o).T) / (attn_scale_1)
evoqkevo = (W_E @ v @ o @ q_1 @ k_1.T @ (W_E @ v @ o).T) / (attn_scale_1)
show(torch.exp(evoqkevo))
show(eqkevo)
show(torch.exp(pvoqkevo[1:-1]))
# %%
