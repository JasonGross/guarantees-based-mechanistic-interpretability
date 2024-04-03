# %%
from gbmi import utils

from gbmi.exp_indhead.train import ABCAB8_1H

from gbmi.model import train_or_load_model
import torch
import einops
from torch import tensor
from math import *
from tqdm.auto import tqdm
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
import pandas as pd
from gbmi.utils import ein
from functools import partial
from inspect import signature
from typing import Callable, Optional, List
from torch import Tensor


def armin(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return ein.apply(f, collect=lambda xs, d: xs.max(d).indices, sizes=sizes)


runtime_model_1, model = train_or_load_model(ABCAB8_1H, force="load")
model.to("cpu")
n_ctx = model.W_pos.shape[0]
d_voc = model.W_E.shape[0]
e_p = model.W_E.unsqueeze(dim=0) + model.W_pos.unsqueeze(dim=1)
everything = (
    einops.einsum(
        e_p,
        model.blocks[0].attn.W_Q.squeeze(dim=0),
        model.blocks[0].attn.W_K.squeeze(dim=0),
        e_p,
        "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
    )
    / model.blocks[0].attn.attn_scale
)
print(everything.shape)
print(n_ctx)
print(d_voc)
# %%
table = torch.zeros((d_voc, d_voc, n_ctx, d_voc)) + float("nan")
for p in range(2, n_ctx + 1):
    tmp = torch.zeros((p, d_voc))
    for t_q in range(d_voc):
        tmp[-1, :] = everything[p - 1, t_q, p - 1, t_q]
        for t_k in range(d_voc):
            tmp[-2, :] = everything[p - 1, t_q, p - 2, t_k]
            tmp[:-2, :] = everything[p - 1, t_q, : p - 2, :]
            tmp_sm = tmp.softmax(dim=0)
            table[t_q, t_k, p - 1, :] = tmp_sm[-2, :]
            if p < n_ctx and tmp_sm[-2, :].min(dim=-1).values <= 0.7:
                print(
                    p,
                    t_q,
                    t_k,
                    tmp_sm[-2, :].min(dim=-1).indices,
                    tmp_sm[:, tmp_sm[-2, :].min(dim=-1).indices],
                    tmp[:, tmp_sm[-2, :].min(dim=-1).indices],
                )
v = model.W_V[0, 0]
z = model.W_O[0, 0]
q_1 = model.W_Q[0, 0]
k_1 = model.W_K[0, 0]
everything_1_1 = ein.array(
    lambda a, c, i_2, j, x: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ (k_1.T)
    @ (e_p[j, x].T)
)
everything_1_2 = ein.array(
    lambda a, c, i_2, j, y: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ k_1.T
    @ ((e_p[j - 1, y]) @ v @ z).T,
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
everything_1_b = ein.array(
    lambda a, c, i_2, i_1, b: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ k_1.T
    @ (e_p[i_1, b] + (e_p[i_1 - 1, a]) @ v @ z).T,
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)

armintable_1_1 = ein.array(
    lambda a, c, i_2, j: armin(lambda x: everything_1_1[a, c, i_2, j, x])
)
armintable_1_2 = ein.array(
    lambda a, c, i_2, j: armin(
        lambda y: torch.where(y != a, everything_1_2[a, c, i_2, j, y], -torch.inf),
        sizes=[e_p.shape[1]],
    ),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0]],
)
worst_possible = ein.array(
    lambda a, c, i_2: ein.sum(
        lambda j: torch.exp(
            everything_1_1[a, c, i_2, j, armintable_1_1[a, c, i_2, j]]
            + everything_1_2[a, c, i_2, j, armintable_1_2[a, c, i_2, j]]
        )
    ),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0]],
)
print(
    ein.array(
        lambda a, c, i_2, j: everything_1_1[a, c, i_2, j, armintable_1_2[a, c, i_2, j]]
    )
)
