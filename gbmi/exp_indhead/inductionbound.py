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

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cuda")
runtime_model_1, model = train_or_load_model(ABCAB8_1H, force="load")
model.to(device)
n_ctx = W_pos.shape[0]
d_voc = W_E.shape[0]
d_model = W_E.shape[1]
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
# %%
e_p = W_E.unsqueeze(dim=0) + W_pos.unsqueeze(dim=1)

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
term_1 = (
    einops.einsum(
        e_p,
        W_Q_1,
        W_K_1,
        e_p,
        "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
    )
    / attn_scale_1
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
    / attn_scale_1
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
    / attn_scale_1
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
    / attn_scale_1
)


def diff_1(a, b, i_1, i_2, j):
    if j == i_1:
        return 0
    else:
        return term_1[i_2, a, j, :].max() - term_1[i_2, a, i_1, b]


def diff_2(a, b, i_1, i_2, j):
    if j == i_1:
        return 0
    diff = term_2[:, :, j, :] - term_2[:, :, i_1, b]
    c = torch.max(diff[: i_2 - 1, :, :].max(), term_2[i_2, a, :].max())
    if c > 0:
        t_2 = (1 - attn_1[:, i_2 - 1].min()) * c
    else:
        t_2 = 0
    c = diff[i_2 - 1, :, :].max()
    if c > 0:
        t_2 += c
    else:
        t_2 += attn_1[:, i_2 - 1].min() * c
    return t_2


def diff_3(a, b, i_1, i_2, j):
    if j == i_1:
        return 0
    if j != 0 and j != 1:
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
    if j == 1:
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
    if j == 0:
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
    if i_1 != 1:
        c = torch.min(term_3[i_2, a, : i_1 - 1, a].min(), term_3[i_2, a, i_1, b].min())
        if c < 0:
            t_3 -= (1 - attn_1[:, i_1 - 1].min()) * c

        c = term_3[i_2, a, i_1 - 1, a].min()
        if c < 0:
            t_3 -= c
        else:
            t_3 -= attn_1[:, i_1 - 1].min() * c
    if i_1 == 1:
        c = term_3[i_2, a, i_1, b].min()
        if c < 0:
            t_3 -= (1 - attn_1[:, i_1 - 1].min()) * c

        c = term_3[i_2, a, i_1 - 1, a].min()
        if c < 0:
            t_3 -= c
        else:
            t_3 -= attn_1[:, i_1 - 1].min() * c

    return t_3


def diff_4(a, b, i_1, i_2, j):
    if j == i_1:
        return 0
    if j != 0 and j != 1:
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
    if j == 1:
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
    if j == 0:
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

    if i_1 != 1:
        c = (
            (1 - attn_1[:, i_1 - 1].min())
            * (1 - attn_1[:, i_2 - 1].min())
            * torch.tensor(
                [
                    term_4[: i_2 - 1, :, : i_1 - 1, :].min(),
                    term_4[i_2, :, : i_1 - 1, :].min(),
                    term_4[: i_2 - 1, :, i_1, b].min(),
                    term_4[i_2, :, i_1, b].min(),
                ]
            ).max()
        )
        d = (1 - attn_1[:, i_2 - 1].min()) * torch.min(
            term_4[: i_2 - 1, :, i_1 - 1, a].min(), term_4[i_2, a, i_1 - 1, a].min()
        )
        e = (1 - attn_1[:, i_1 - 1].min()) * torch.min(
            term_4[i_2 - 1, :, : i_1 - 1, :].min(), term_4[i_2 - 1, :, i_1, b].min()
        )

        if c < 0:
            t_4 -= c

        if d < 0:
            t_4 -= d

        if e < 0:
            t_4 -= e

        c = term_4[i_2 - 1, :, i_1 - 1, :].min()
        if c < 0:
            t_4 -= c
        else:
            t_4 -= attn_1[:, i_2 - 1].min() * attn_1[:, i_1 - 1].min() * c
    if i_1 == 1:
        c = (
            (1 - attn_1[:, i_1 - 1].min())
            * (1 - attn_1[:, i_2 - 1].min())
            * torch.tensor(
                [
                    term_4[: i_2 - 1, :, i_1, b].min(),
                    term_4[i_2, :, i_1, b].min(),
                ]
            ).max()
        )
        d = (1 - attn_1[:, i_2 - 1].min()) * torch.min(
            term_4[: i_2 - 1, :, i_1 - 1, a].min(), term_4[i_2, a, i_1 - 1, a].min()
        )
        e = (1 - attn_1[:, i_1 - 1].min()) * term_4[i_2 - 1, :, i_1, b].min()

        if c < 0:
            t_4 -= c

        if d < 0:
            t_4 -= d

        if e < 0:
            t_4 -= e

        c = term_4[i_2 - 1, :, i_1 - 1, :].min()
        if c < 0:
            t_4 -= c
        else:
            t_4 -= attn_1[:, i_2 - 1].min() * attn_1[:, i_1 - 1].min() * c
    return t_4


def least_attention(a, b, i_1, i_2, j):
    return (
        diff_1(a, b, i_1, i_2, j)
        + diff_2(a, b, i_1, i_2, j)
        + diff_3(a, b, i_1, i_2, j)
        + diff_4(a, b, i_1, i_2, j)
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


bound_soft = bound.softmax(dim=-1)
bound_2 = einops.einsum(
    bound_soft,
    "a b i_2 i_1 i_1 -> a b i_2 i_1",
)
