# %%
from functools import partial
from inspect import signature
from math import *
from typing import Callable, List, Optional

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch import Tensor, tensor, where

# from tqdm.auto import  tqdm
from tqdm import tqdm

from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1H
from gbmi.model import train_or_load_model
from gbmi.utils import ein
from gbmi.utils.sequences import generate_all_sequences


def armin(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return ein.apply(f, collect=lambda xs, d: xs.max(d).indices, sizes=sizes)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cuda")
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
# W_O_0 = W_O_0 + noise(W_O_0)
W_V_0 = model.W_V[0, 0]
W_V_0 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
# W_V_0 = W_V_0 + noise(W_V_0)
W_V_1 = model.W_V[1, 0]
W_V_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
# W_V_1 = W_V_1 + noise(W_V_1)
W_O_1 = model.W_O[1, 0]
W_O_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_voc, d_model]).float().to(device)
)
# W_O_1 = W_O_1 + noise(W_O_1)
W_Q_0 = model.W_Q[0, 0]
W_Q_0 = (
    ein.array(lambda i, j: where((i + d_voc + 1) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .to(device)
    .T
)
# W_Q_0 = W_Q_0 + noise(W_Q_0)

W_Q_1 = (
    ein.array(lambda i, j: where(i == j, d, 0), sizes=[d_voc, d_model])
    .float()
    .T.to(device)
)
# W_Q_1 = W_Q_1 + noise(W_Q_1)

W_K_0 = (
    ein.array(lambda i, j: where((i + d_voc) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .T
).to(device)
# W_K_0 = W_K_0 + noise(W_K_0)

W_K_1 = (
    ein.array(
        lambda i, j: where((i + n_ctx + d_voc) == j, d, 0), sizes=[d_voc, d_model]
    )
    .float()
    .T
).to(device)
# W_K_1 = W_K_1 + noise(W_K_1)
# %%
# px.imshow((W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T).cpu())
# %%
W_U = ein.array(lambda i, j: i == j, sizes=[d_model, d_voc]).float().to(device)
# W_U = W_U + noise(W_U)
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale
"""
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
"""
e_p = W_E.unsqueeze(dim=0) + W_pos.unsqueeze(dim=1)

everything = (
    einops.einsum(
        e_p,
        W_Q_0,
        W_K_0,
        e_p,
        "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
    )
    / 1
)
# %%
table = torch.zeros((d_voc, d_voc, n_ctx - 2, d_voc)) + float(
    "nan"
)  # p Represents the position of 'b' at index + 1
for p in range(2, n_ctx):  #
    tmp = torch.zeros((p, d_voc))
    for t_q in range(d_voc):
        tmp[-1, :] = everything[p - 1, t_q, p - 1, t_q]
        for t_k in range(d_voc):
            tmp[-2, :] = everything[p - 1, t_q, p - 2, t_k]
            tmp[:-2, :] = everything[p - 1, t_q, : p - 2, :]
            if p == n_ctx:
                print()
                print(tmp, "TMP")
            tmp_sm = tmp.softmax(dim=0)
            table[t_q, t_k, p - 2, :] = tmp_sm[
                -2, :
            ]  # Table represents post softmax attention paid to t_k, if the final entry is spammed everywhere, and t_q is used as the first entry, at pth poisition

# everything looks like EQKE, table looks like you're indexing by query, key, position (of key?), and other token in the sequence.
# They you're computing softmax of d_voc - 2 copies of the other token, one copy of t_k in p-2, and the query in p-1.
# Then you store the post-softmax attention paid to t_k.
#
#
#
##       xEQKE^tx^t
#
##
#                               t_q vocab paying attention to t_k another letter, if other one gets spammed
#
##
#
#
#
##
#
#
#
#
#
#
#
#
attn_1 = table.min(dim=1).values.min(dim=2).values

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
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o) @ q_1 @ (k_1.T) @ (e_p[j, x].T),
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
        @ ((e_p[j - 1, y]) @ v @ o).T,
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
print(q_1.device)
everything_1_b = ein.array(
    lambda a, c, i_2, i_1, b: where(
        torch.logical_and(i_2 > i_1, i_1 >= 1),
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o)
        @ q_1
        @ k_1.T
        @ ((e_p[i_1, b] + (e_p[i_1 - 1, a]) @ v @ o).T),
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
PVOU = W_pos @ W_V_1 @ W_O_1 @ W_U
PVOVOU = W_pos @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOVOU = W_E @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOU = W_E @ W_V_1 @ W_O_1 @ W_U
# %%
px.imshow(PVOU.detach().cpu())


# %%
def compute_worst_case_scenario_pos(M, b, attn_on_b):

    attn_to_others = torch.tensor(M.max(dim=0).values) * (1.0 - attn_on_b)
    attn_to_others[b] = attn_on_b * M.min(dim=0).values[b]
    return attn_to_others


def compute_worst_case_scenario_evou(b, attn_on_b):
    attn_to_others = torch.tensor(EVOU.max(dim=0).values) * (1.0 - attn_on_b)
    attn_to_others[b] = attn_on_b * EVOU[b][b]

    return attn_to_others


attn_to_b = 0.6
pre_softmax = (
    compute_worst_case_scenario_pos(PVOU, 1, attn_to_b)
    + compute_worst_case_scenario_pos(PVOVOU, 1, attn_to_b)
    + compute_worst_case_scenario_pos(EVOVOU, 1, attn_to_b)
    + compute_worst_case_scenario_evou(1, attn_to_b)
)
print(pre_softmax.softmax(dim=0))


# %%
term_1 = (
    einops.einsum(
        e_p,
        W_Q_1,
        W_K_1,
        e_p,
        "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
    )
    / 1
)
term_2 = (
    einops.einsum(
        e_p,
        W_V_0,
        W_O_0,
        W_Q_1,
        W_K_1,
        e_p,
        "q_pos q_val k, k l, l m, m n, o n, k_pos k_val o -> q_pos q_val k_pos k_val",
    )
    / 1
)

term_3 = (
    einops.einsum(
        e_p,
        W_Q_1,
        W_K_1,
        W_O_0,
        W_V_0,
        e_p,
        "q_pos q_val k, k l, m l, n m, o n, k_pos k_val o -> q_pos q_val k_pos k_val",
    )
    / 1
)

term_4 = (
    einops.einsum(
        e_p,
        W_V_0,
        W_O_0,
        W_Q_1,
        W_K_1,
        W_O_0,
        W_V_0,
        e_p,
        "q_pos q_val k, k l, l m, m n, o n, p o, q p, k_pos k_val q -> q_pos q_val k_pos k_val",
    )
    / 1
)


# %%
def least_attention(a, b, i_1, i_2, j):

    if j != i_1 and j != 0 and j != 1:
        t_1 = term_1[i_2, a, j, :].max()
        c = torch.max(term_2[: i_2 - 1, :, j, :].max(), term_2[i_2, a, j, :].max())
        if c > 0:
            t_2 = (1 - attn_1[:, i_2 - 1].min()) * c
        else:
            t_2 = 0
        c = term_2[i_2 - 1, :, j, :].max()
        if c > 0:
            t_2 += c
        else:
            t_2 += attn_1[:, i_2 - 1].min() * c

        """
        if a != 0 and a != d_voc - 1:
            c = torch.tensor(
                [
                    term_3[i_2, a, : j - 1, :a].max(),
                    term_3[i_2, a, : j - 1, a + 1 :].max(),
                    term_3[i_2, a, j, :].max(),
                ]
            ).max()
        if a == 0:
            c = torch.tensor(
                [
                    term_3[i_2, a, : j - 1, a + 1 :].max(),
                    term_3[i_2, a, j, :].max(),
                ]
            ).max()
        """

        c = torch.max(term_3[i_2, a, : j - 1, :].max(), term_3[i_2, a, j, :].max())

        if c > 0:
            t_3 = (1 - attn_1[:, j - 1].min()) * c
        else:
            t_3 = 0

        if a != 0 and a != d_voc - 1:
            c = torch.max(
                term_3[i_2, a, j - 1, :a].max(), term_3[i_2, a, j - 1, a + 1 :].max()
            )
        if a == 0:
            c = term_3[i_2, a, j - 1, a + 1 :].max()

        if a == d_voc - 1:
            c = term_3[i_2, a, j - 1, :a].max()

        if c > 0:
            t_3 += c
        else:
            t_3 += attn_1[:, j - 1].min() * c
        c = (
            (1 - attn_1[:, j - 1].min())
            * (1 - attn_1[:, i_2 - 1].min())
            * torch.tensor(
                [
                    term_4[: i_2 - 1, :, : j - 1, :].max(),
                    term_4[i_2, :, : j - 1, :].max(),
                    term_4[: i_2 - 1, :, j, :].max(),
                    term_4[i_2, :, j, :].max(),
                ]
            ).max()
        )
        d = (1 - attn_1[:, i_2 - 1].min()) * torch.max(
            term_4[: i_2 - 1, :, j - 1, :].max(), term_4[i_2, a, j - 1, :].max()
        )
        e = (1 - attn_1[:, j - 1].min()) * torch.max(
            term_4[i_2 - 1, :, : j - 1, :].max(), term_4[i_2 - 1, :, j, :].max()
        )

        if c > 0:
            t_4 = c
        else:
            t_4 = 0

        if d > 0:
            t_4 += d

        if e > 0:
            t_4 += e

        c = term_4[i_2 - 1, :, j - 1, :].max()
        if c > 0:
            t_4 += c
        else:
            t_4 += attn_1[:, i_2 - 1].min() * attn_1[:, j - 1].min() * c

    if j != i_1 and j == 1:
        t_1 = term_1[i_2, a, j, :].max()
        c = torch.max(term_2[: i_2 - 1, :, j, :].max(), term_2[i_2, a, j, :].max())
        if c > 0:
            t_2 = (1 - attn_1[:, i_2 - 1].min()) * c
        else:
            t_2 = 0
        c = term_2[i_2 - 1, :, j, :].max()
        if c > 0:
            t_2 += c
        else:
            t_2 += attn_1[:, i_2 - 1].min() * c
        c = term_3[i_2, a, j, :].max()
        if c > 0:
            t_3 = (1 - attn_1[:, j - 1].min()) * c
        else:
            t_3 = 0

        if a != 0 and a != d_voc - 1:

            c = torch.max(
                term_3[i_2, a, j - 1, :a].max(), term_3[i_2, a, j - 1, a + 1 :].max()
            )

        if a == 0:
            c = term_3[i_2, a, j - 1, a + 1 :].max()

        if a == d_voc - 1:
            c = term_3[i_2, a, j - 1, :a].max()

        if c > 0:
            t_3 += c
        else:
            t_3 += attn_1[:, j - 1].min() * c
        c = (
            (1 - attn_1[:, j - 1].min())
            * (1 - attn_1[:, i_2 - 1].min())
            * torch.tensor(
                [
                    term_4[: i_2 - 1, :, j, :].max(),
                    term_4[i_2, :, j, :].max(),
                ]
            ).max()
        )
        d = (1 - attn_1[:, i_2 - 1].min()) * torch.max(
            term_4[: i_2 - 1, :, j - 1, :].max(), term_4[i_2, a, j - 1, :].max()
        )
        e = (1 - attn_1[:, j - 1].min()) * term_4[i_2 - 1, :, j, :].max()

        if c > 0:
            t_4 = c
        else:
            t_4 = 0

        if d > 0:
            t_4 += d

        if e > 0:
            t_4 += e

        c = term_4[i_2 - 1, :, j - 1, :].max()
        if c > 0:
            t_4 += c
        else:
            t_4 += attn_1[:, i_2 - 1].min() * attn_1[:, j - 1].min() * c

    if j != i_1 and j == 0:
        t_1 = term_1[i_2, a, j, :].max()
        c = torch.max(term_2[: i_2 - 1, :, j, :].max(), term_2[i_2, a, j, :].max())
        if c > 0:
            t_2 = (1 - attn_1[:, i_2 - 1].min()) * c
        else:
            t_2 = 0
        c = term_2[i_2 - 1, :, j, :].max()
        if c > 0:
            t_2 += c
        else:
            t_2 += attn_1[:, i_2 - 1].min() * c

        if a != 0 and a != d_voc - 1:
            c = torch.max(
                term_3[i_2, a, j, :a].max(), term_3[i_2, a, j - 1, a + 1 :].max()
            )
        if a == 0:
            c = term_3[i_2, a, j, a + 1 :].max()

        if a == d_voc - 1:
            c = term_3[i_2, a, j, :a].max()

        if c > 0:
            t_3 = c
        else:
            t_3 = 0

        c = (1 - attn_1[:, i_2 - 1].min()) * torch.tensor(
            [
                term_4[: i_2 - 1, :, j, :].max(),
                term_4[i_2, :, j, :].max(),
            ]
        ).max()

        e = term_4[i_2 - 1, :, j, :].max()

        if c > 0:
            t_4 = c
        else:
            t_4 = 0

        if e > 0:
            t_4 += e

    if j == i_1 and j != 1:
        t_1 = term_1[i_2, a, j, b].min()
        c = torch.min(term_2[: i_2 - 1, :, j, b].min(), term_2[i_2, a, j, b].min())
        if c < 0:
            t_2 = (1 - attn_1[:, i_2 - 1].min()) * c
        else:
            t_2 = 0
        c = term_2[i_2 - 1, :, j, b].min()
        if c < 0:
            t_2 += c
        else:
            t_2 += attn_1[:, i_2 - 1].min() * c
        c = torch.min(term_3[i_2, a, : j - 1, a].min(), term_3[i_2, a, j, b].min())
        if c < 0:
            t_3 = (1 - attn_1[:, j - 1].min()) * c
        else:
            t_3 = 0
        c = term_3[i_2, a, j - 1, a].min()
        if c < 0:
            t_3 += c
        else:
            t_3 += attn_1[:, j - 1].min() * c
        c = (
            (1 - attn_1[:, j - 1].min())
            * (1 - attn_1[:, i_2 - 1].min())
            * torch.tensor(
                [
                    term_4[: i_2 - 1, :, : j - 1, :].min(),
                    term_4[i_2, :, : j - 1, :].min(),
                    term_4[: i_2 - 1, :, j, b].min(),
                    term_4[i_2, :, j, b].min(),
                ]
            ).max()
        )
        d = (1 - attn_1[:, i_2 - 1].min()) * torch.min(
            term_4[: i_2 - 1, :, j - 1, a].min(), term_4[i_2, a, j - 1, a].min()
        )
        e = (1 - attn_1[:, j - 1].min()) * torch.min(
            term_4[i_2 - 1, :, : j - 1, :].min(), term_4[i_2 - 1, :, j, b].min()
        )

        if c < 0:
            t_4 = c
        else:
            t_4 = 0

        if d < 0:
            t_4 += d

        if e < 0:
            t_4 += e

        c = term_4[i_2 - 1, :, j - 1, :].min()
        if c < 0:
            t_4 += c
        else:
            t_4 += attn_1[:, i_2 - 1].min() * attn_1[:, j - 1].min() * c

    if j == i_1 and j == 1:
        t_1 = term_1[i_2, a, j, b].min()
        c = torch.min(term_2[: i_2 - 1, :, j, b].min(), term_2[i_2, a, j, b].min())
        if c < 0:
            t_2 = (1 - attn_1[:, i_2 - 1].min()) * c
        else:
            t_2 = 0
        c = term_2[i_2 - 1, :, j, b].min()
        if c < 0:
            t_2 += c
        else:
            t_2 += attn_1[:, i_2 - 1].min() * c
        c = term_3[i_2, a, j, b].min()
        if c < 0:
            t_3 = (1 - attn_1[:, j - 1].min()) * c
        else:
            t_3 = 0
        c = term_3[i_2, a, j - 1, a].min()
        if c < 0:
            t_3 += c
        else:
            t_3 += attn_1[:, j - 1].min() * c
        c = (
            (1 - attn_1[:, j - 1].min())
            * (1 - attn_1[:, i_2 - 1].min())
            * torch.tensor(
                [
                    term_4[: i_2 - 1, :, j, b].min(),
                    term_4[i_2, :, j, b].min(),
                ]
            ).max()
        )
        d = (1 - attn_1[:, i_2 - 1].min()) * torch.min(
            term_4[: i_2 - 1, :, j - 1, a].min(), term_4[i_2, a, j - 1, a].min()
        )
        e = (1 - attn_1[:, j - 1].min()) * term_4[i_2 - 1, :, j, b].min()

        if c < 0:
            t_4 = c
        else:
            t_4 = 0

        if d < 0:
            t_4 += d

        if e < 0:
            t_4 += e

        c = term_4[i_2 - 1, :, j - 1, :].min()
        if c < 0:
            t_4 += c
        else:
            t_4 += attn_1[:, i_2 - 1].min() * attn_1[:, j - 1].min() * c
    # print(t_1, t_2, t_3, t_4)
    return t_1 + t_2 + t_3 + t_4


"""

def least_attention(a,b,i_1,i_2,j):
    if j!=i_1 and j!=0:
        t_1 = term_1[i_2,a,j,:].max()
        t_2=0
        for i in range(i_2+1):
            c=term_2[i,:,j,:].max()
            if i!=i_2-1:
                if c>0:
                    t_2+=(1-attn_1[:,i_2-1].min())*c
            else:
                if c>0:
                    t_2+=c

        t_3=0
        for l in range(j+1):
            c=term_3[i_2,a,l,:].max()
            if l!=j-1
                if c>0:
                    t_3+=(1-attn_1[:,j-1].min())*c
            else:
                if c>0:
                    t_3+=c
        t_4=0
        for k in range(i_2+1):
            for l in range(j+1):
                c=term_4[k,:,l,:].max()
                if k!=i_2-1 and l!=j-1:
                    if c>0:
                        t_4+=(1-attn_1[:,i_2-1].min())*(1-attn_1[:,j-1].min())*c
                if k==i_2-1 and l!=j-1:
                    if c>0:
                        t_4+=(1-attn_1[:,j-1].min())*c
                if k!=i_2-1 and l==j-1:
                    if c>0:
                        t_4+=(1-attn_1[:,i_2-1].min())*c
                if k!=i_2-1 and l!=j-1:
                    if c>0:
                        t_4+=c
    if j!=i_1 and j==0:
        t_1 = term_1[i_2,a,j,:].max()
        t_2=0
        for i in range(i_2+1):
            c=term_2[i,:,j,:].max()
            if i!=i_2-1:
                if c>0:
                    t_2+=(1-attn_1[:,i_2-1].min())*c
            else:
                if c>0:
                    t_2+=c


        t_3=term_3[i_2,:,0,:].max()


        t_4=0
        for k in range(i_2+1):

            c=term_4[k,:,0,:].max()
            if k!=i_2-1:
                if c>0:
                    t_4+=(1-attn_1[:,i_2-1].min())*c
            if k==i_2-1:
                if c>0:
                    t_4+=c
    if j==i_1:
        t_1=term_1[i_2,a,j,:].min()
        t_2=0
        for i in range(i_2+1):
            c=term_2[i,:,j,b].min()
            if i!=i_2-1:
                if c<0:
                    t_2+=(1-attn_1[:,i_2-1].min())*c
            else:
                if c<0:
                    t_2+=c
        t_3=0
        for l in range(j+1):
            c=term_3[i_2,a,l,:].max()
            if l!=j-1
                if c>0:
                    t_3+=(1-attn_1[:,j-1].min())*c
            else:
                if c>0:
                    t_3+=c

"""

# %%
for j in range(7):
    print(least_attention(3, 2, 2, 6, j))
# %%

bound = ein.array(
    lambda a, b, i_2, i_1, j: torch.where(
        (i_1 < i_2) & (i_1 > 0) & (i_2 + 1 > j),
        least_attention(a, b, i_1, i_2, j),
        -torch.inf,
    ),
    device=device,
    sizes=[
        e_p.shape[1],
        e_p.shape[1],
        e_p.shape[0] - 1,
        e_p.shape[0] - 1,
        e_p.shape[0],
    ],
)

# %%

bound = (
    torch.zeros((e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[0]))
    - torch.inf
)

for a in tqdm(range(e_p.shape[1])):
    for b in tqdm(range(e_p.shape[1])):
        for i_2 in range(e_p.shape[0] - 1):
            for i_1 in range(e_p.shape[0] - 1):
                for j in range(i_2 + 1):
                    if (i_1 < i_2) & (i_1 > 1) & (i_2 + 1 > j):
                        bound[a, b, i_2, i_1, j] = least_attention(a, b, i_1, i_2, j)


# %%

bound_soft = bound.softmax(dim=-1)
bound_2 = einops.einsum(
    bound_soft,
    "a b i_2 i_1 i_1 -> a b i_2 i_1",
)


# %%
