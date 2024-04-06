# %%
from gbmi import utils

from gbmi.exp_indhead.train import ABCAB8_1H
from torch import where
from gbmi.model import train_or_load_model
import torch
import einops
from torch import tensor
from math import *

# from tqdm.auto import tqdm
from tqdm import tqdm
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
import pandas as pd
from gbmi.utils import ein
from functools import partial
from inspect import signature
from typing import Callable, Optional, List
from torch import Tensor
import numpy as np
import plotly.express as px


def armin(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return ein.apply(f, collect=lambda xs, d: xs.max(d).indices, sizes=sizes)


device = "cuda" if torch.cuda.is_available() else "cpu"
runtime_model_1, model = train_or_load_model(ABCAB8_1H, force="load")
model.to(device)
c = 10
d = 10
W_pos = model.W_pos
W_E = model.W_E
epsilon = 0.05
n_ctx = W_pos.shape[0]
d_voc = W_E.shape[0]
d_model = W_E.shape[1]


def noise(M):
    return epsilon * (torch.rand_like(M) - 0.5)


def add_noise(*ms):
    for m in ms:
        m += noise(m.shape)


# %%
W_E = ein.array(lambda i, j: i == j, sizes=[d_voc, d_model]).float().to(device) + noise(
    W_E
)
# %%
W_pos = (
    ein.array(lambda i, j: ((i + d_voc) == j) * 1.0, sizes=[n_ctx, d_model])
    .float()
    .to(device)
) + noise(W_pos)

# %%


W_O_0 = model.W_O[0, 0]
W_O_0 = (
    ein.array(lambda i, j: ((i + n_ctx + d_voc) == j) * 1.0, sizes=[d_voc, d_model])
    .float()
    .to(device)
)
W_O_0 = W_O_0 + noise(W_O_0)
W_V_0 = model.W_V[0, 0]
W_V_0 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
W_V_0 = W_V_0 + noise(W_V_0)
W_V_1 = model.W_V[1, 0]
W_V_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
W_V_1 = W_V_1 + noise(W_V_1)
W_O_1 = model.W_O[1, 0]
W_O_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_voc, d_model]).float().to(device)
)
W_O_1 = W_O_1 + noise(W_O_1)
W_Q_0 = model.W_Q[0, 0]
W_Q_0 = (
    ein.array(lambda i, j: where((i + d_voc + 1) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .to(device)
    .T
)
W_Q_0 = W_Q_0 + noise(W_Q_0)
W_Q_1 = model.W_Q[1, 0]
W_Q_1 = (
    ein.array(lambda i, j: where(i == j, d, 0), sizes=[d_voc, d_model])
    .float()
    .T.to(device)
)
W_Q_1 = W_Q_1 + noise(W_Q_1)
W_K_0 = model.W_K[0, 0]
W_K_0 = (
    ein.array(lambda i, j: where((i + d_voc) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .T
).to(device)
W_K_0 = W_K_0 + noise(W_K_0)
W_K_1 = model.W_K[1, 0]
W_K_1 = (
    ein.array(
        lambda i, j: where((i + n_ctx + d_voc) == j, d, 0), sizes=[d_voc, d_model]
    )
    .float()
    .T
).to(device)
W_K_1 = W_K_1 + noise(W_K_1)
# %%
# px.imshow((W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T).cpu())
# %%
W_U = model.W_U
W_U = ein.array(lambda i, j: i == j, sizes=[d_model, d_voc]).float().to(device)
W_U = W_U + noise(W_U)
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale


e_p = W_E.unsqueeze(dim=0) + W_pos.unsqueeze(dim=1)
print(e_p.shape)
everything = (
    einops.einsum(
        e_p,
        W_Q_0,
        W_K_0,
        e_p,
        "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
    )
    / attn_scale_0
)
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
                continue
# %%
o = W_O_0
v = W_V_0
print(v.shape, "w_v_0")
print(o.shape, "w_o_0")
q_1 = W_Q_1
k_1 = W_K_1
# %%
"""
everything_1_1 = ein.array(
    lambda a, c, i_2, j, x: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ (k_1.T)
    @ (e_p[j, x].T)
    * (1 / sqrt(32))
)
"""
everything_1_1 = ein.array(
    lambda a, c, i_2, j, x: torch.where(
        (j < i_2) & (x != a),
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o)
        @ q_1
        @ (k_1.T)
        @ (e_p[j, x].T)
        * (1 / attn_scale_1),
        -torch.inf,
    ),
    device=device,
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
# %%


everything_1_2 = ein.array(
    lambda a, c, i_2, j, y: torch.where(
        (j >= 1) & (j < i_2) & (y != a),
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o)
        @ q_1
        @ k_1.T
        @ ((e_p[j - 1, y]) @ v @ o).T
        * (1 / attn_scale_1),
        -torch.inf,
    ),
    device=device,
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
# %%


# %%
# px.imshow((W_E @ q_1 @ k_1.T @ o.T @ v.T @ W_E.T).cpu())

# %%
# px.imshow((W_E @ v).cpu())
# %%

everything_1_b = ein.array(
    lambda a, c, i_2, i_1, b: where(
        torch.logical_and(i_2 > i_1, i_1 >= 1),
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o)
        @ q_1
        @ k_1.T
        @ ((e_p[i_1, b] + (e_p[i_1 - 1, a]) @ v @ o).T)
        * (1 / attn_scale_1),
        torch.inf,
    ),
    device=device,
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
# %%
"""
armintable_1_1 = ein.array(
    lambda a, c, i_2, j: ein.max(
        lambda x: torch.where(
            torch.logical_and(torch.logical_and(x != c, x != a), i_2 > j),
            everything_1_1[a, c, i_2, j, x],
            -torch.inf,
        ),
        sizes=[e_p.shape[1]],
    ),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0]],
)
armintable_1_2 = ein.array(
    lambda a, c, i_2, j: ein.max(
        lambda y: torch.where(
            torch.logical_and(torch.logical_and(y != c, i_2 > j), y != a),
            everything_1_2[a, c, i_2, j, y],
            -torch.inf,
        ),
        sizes=[e_p.shape[1]],
    ),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0]],
)
"""
# %%


attn = torch.zeros((d_voc, d_voc, d_voc, n_ctx, n_ctx))
last_dim_indices_1_1 = torch.arange(
    everything_1_1.shape[-1], device=everything_1_1.device
)
last_dim_indices_1_2 = torch.arange(
    everything_1_2.shape[-1], device=everything_1_2.device
)


def make_inner_val(a, c, i_2, j):
    return torch.where(
        last_dim_indices_1_1 == a,
        -torch.inf,
        everything_1_1[a, c, i_2, j],
    ).max(dim=-1).values + torch.where(
        j != 0,
        torch.where(
            last_dim_indices_1_2 == a,
            -torch.inf,
            everything_1_2[a, c, i_2, j],
        )
        .max(dim=-1)
        .values,
        0.0,
    )


# %%
x = torch.tensor([1, 2, 3])

ein.array(lambda i: x[i])

# %%
# pre-softmax
attn_pattern = ein.array(
    lambda a, b, c, i_2, i_1, j: torch.where(
        (i_2 >= 2) & (i_1 < (i_2 - 2)),
        torch.where(
            j < i_2,
            torch.where(
                j == i_1 + 1,
                everything_1_b[a, c, i_2, (i_1 + 1) % n_ctx, b],
                make_inner_val(a, c, i_2, j),
            ),
            -torch.inf,
        ),
        -torch.inf,
    ),
    device=device,
    sizes=[d_voc, d_voc, d_voc, n_ctx, n_ctx, n_ctx],
)
attn = attn_pattern.softmax(dim=-1)
# %%
attn_correct = ein.array(
    lambda a, b, c, i_2, i_1: torch.where(
        i_1 + 1 < n_ctx, attn[a, b, c, i_2, i_1, (i_1 + 1) % n_ctx], torch.nan
    ),
    device=device,
    sizes=attn.shape[:-1],
)

# vals = ein.array(
#     lambda a, b, c, i_2, i_1, j: torch.where(
#         (i_2 >= 2) & (i_1 < (i_2 - 2)),
#         torch.where(
#             0 <= j - 1 < i_1 + 1,
#             make_inner_val(a, c, i_2, j - 1),
#             torch.where(
#                 i_1 + 1 <= j - 1 < i_2 - 1,
#                 make_inner_val(a, c, j + 1 - 1),
#                 torch.where(
#                     j - 1 == -1, everything_1_b[a, c, i_2, i_1 + 1, b], -torch.inf
#                 ),
#             ),
#         ),
#         -torch.inf,
#     ),
#     device=device,
#     sizes=[d_voc, d_voc, d_voc, n_ctx, n_ctx, n_ctx],
# )


# attn = vals.softmax(dim=-1)

# for a in tqdm(range(d_voc)):
#     for b in range(d_voc):
#         for c in range(d_voc):
#             for i_2 in range(2, n_ctx):
#                 for i_1 in range(0, i_2 - 2):
#                     vals = []
#                     for j in range(i_2):
#                         if j != i_1 + 1:

#                             vals.append(
#                                 torch.where(
#                                     last_dim_indices_1_1 == a,
#                                     -torch.inf,
#                                     everything_1_1[a, c, i_2, j],
#                                 )
#                                 .max(dim=-1)
#                                 .values
#                                 + torch.where(
#                                     torch.tensor(j != 0, device=device),
#                                     torch.where(
#                                         last_dim_indices_1_2 == a,
#                                         -torch.inf,
#                                         everything_1_2[a, c, i_2, j],
#                                     )
#                                     .max(dim=-1)
#                                     .values,
#                                     0.0,
#                                 )
#                             )
#                     vals.append(everything_1_b[a, c, i_2, i_1 + 1, b])
#                     vals = torch.tensor(vals)
#                     attn[a, b, c, i_2, i_1] = vals.softmax(dim=0)[-1]


# for a in tqdm(range(d_voc)):
#     for b in range(d_voc):
#         for c in range(d_voc):
#             for i_2 in range(2, n_ctx):
#                 for i_1 in range(0, i_2 - 2):
#                     vals = []
#                     for j in range(i_2):
#                         if j != i_1 + 1:
#                             if j != 0:
#                                 condition_1_1 = last_dim_indices_1_1 == a

#                                 condition_1_2 = last_dim_indices_1_2 == a

#                                 modified_1_2 = torch.where(
#                                     condition_1_2,
#                                     -torch.inf,
#                                     everything_1_2,
#                                 )
#                                 modified_1_1 = torch.where(
#                                     condition_1_1,
#                                     -torch.inf,
#                                     everything_1_1,
#                                 )

#                                 vals.append(
#                                     modified_1_1[a, c, i_2, j].max(dim=-1).values
#                                     + modified_1_2[a, c, i_2, j].max(dim=-1).values
#                                 )
#                             if j == 0:

#                                 condition_1_1 = last_dim_indices_1_1 == a

#                                 modified_1_1 = torch.where(
#                                     condition_1_1,
#                                     -torch.inf,
#                                     everything_1_1,
#                                 )

#                                 vals.append(
#                                     modified_1_1[a, c, i_2, j].max(dim=-1).values
#                                 )

#                     vals.append(everything_1_b[a, c, i_2, i_1 + 1, b])
#                     vals = torch.tensor(vals)
#                     attn[a, b, c, i_2, i_1] = vals.softmax(dim=0)[-1]

# %%
