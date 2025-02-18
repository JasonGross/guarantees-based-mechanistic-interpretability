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

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cuda")
runtime_model_1, model_1 = train_or_load_model(ABCAB8_1H, force="load")
model_1.to(device)

W_pos_1 = model_1.W_pos
W_E_1 = model_1.W_E
n_ctx = W_pos_1.shape[0]
d_voc = W_E_1.shape[0]
d_model = W_E_1.shape[1]
attn_scale_0 = model_1.blocks[0].attn.attn_scale
attn_scale_1 = model_1.blocks[1].attn.attn_scale


# %%


def terms(model):
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

    e_p = W_E.unsqueeze(dim=0) + W_pos.unsqueeze(dim=1)

    term_0 = (
        einops.einsum(
            e_p,
            W_Q_0,
            W_K_0,
            e_p,
            "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
        )
        / attn_scale_0
    )

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

    term_5 = einops.einsum(e_p, W_U, "q_pos q_val k, k l -> q_pos q_val l")
    term_6 = einops.einsum(
        e_p, W_V_0, W_O_0, W_U, "q_pos q_val k, k l, l m, m n -> q_pos q_val n"
    )
    term_7 = einops.einsum(
        e_p, W_V_1, W_O_1, W_U, "q_pos q_val k, k l, l m, m n -> q_pos q_val n"
    )
    term_8 = einops.einsum(
        e_p,
        W_V_0,
        W_O_0,
        W_V_1,
        W_O_1,
        W_U,
        "q_pos q_val k, k l, l m, m n, n p, p q -> q_pos q_val q",
    )

    return (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8)


def trivial_heuristic(matrices):
    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices
    reduced_3 = einops.einsum(term_3, "q_pos q_val k_pos k_val -> q_pos q_val k_pos")
    reduced_7 = einops.einsum(term_3, "q_pos q_val q_val -> q_pos q_val")
    a_indices = torch.arange(term_3.shape[1]).view(1, -1, 1, 1)
    b_indices = torch.arange(term_3.shape[1]).view(1, 1, 1, -1)
    mask = a_indices == b_indices
    T_3 = term_3.masked_fill(mask, float("-inf"))
    c_indices = torch.arange(term_7.shape[1]).view(1, -1, 1)
    d_indices = torch.arange(term_7.shape[1]).view(1, 1, -1)
    mask = c_indices == d_indices
    T_7 = term_7.masked_fill(mask, float("-inf"))
    return torch.tensor(
        [
            term_1.abs().max(),
            term_2.abs().max(),
            term_4.abs().max(),
            term_5.abs().max(),
            term_6.abs().max(),
            term_8.abs().max(),
            T_3.max() - reduced_3.min(),
            T_7.max() - reduced_7.min(),
        ]
    ).max()


def first_layer_attention(matrices):
    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices
    table = torch.zeros((d_voc, d_voc, n_ctx - 2, d_voc)) + float(
        "nan"
    )  # p Represents the position of 'b' at index + 1

    for p in range(2, n_ctx):  #
        tmp = torch.zeros((p, d_voc))
        for t_q in range(d_voc):
            tmp[-1, :] = term_0[p - 1, t_q, p - 1, t_q]

            for t_k in range(d_voc):
                tmp[-2, :] = term_0[p - 1, t_q, p - 2, t_k]
                tmp[:-2, :] = term_0[p - 1, t_q, : p - 2, :]
                tmp_sm = tmp.softmax(dim=0)
                table[t_q, t_k, p - 2, :] = tmp_sm[-2, :]
    attn_1 = table.min(dim=1).values.min(dim=2).values
    return attn_1


def diff_1(a, i_1, i_2, j, dic, matrices):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if j == i_1:
        return 0
    else:
        return term_1[i_2, a, j, dic[j]].max() - term_1[i_2, a, i_1, dic[i_1]].min()


def diff_3(a, i_1, i_2, j, dic, matrices, attn_1):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if j == i_1:
        return 0
    if j != 0 and j != 1:
        c = term_3[i_2, a, j - 1, dic[j - 1]].max()
        # new=c.clone()
        t_3 = c * attn_1[dic[j], j - 1].min()
        for i in range(0, j - 1):
            c = torch.max(c, term_3[i_2, dic[i_2], i, dic[i]].max())
        c = torch.max(c, term_3[i_2, dic[i_2], j, dic[j]].max())
        t_3 += (1 - attn_1[dic[j], j - 1].min()) * c
        print(a, i_1, i_2, j)
        print(c)

        # print(t_3)
    if j == 1:
        c = term_3[i_2, a, j - 1, dic[j - 1]].max()
        # new=c.clone()
        t_3 = c * attn_1[dic[j], j - 1].min()
        c = torch.max(c, term_3[i_2, a, j, dic[j]].max())
        t_3 += (1 - attn_1[dic[j], j - 1].min()) * c

    if j == 0:

        t_3 = term_3[i_2, a, j, dic[j]].max()
        # print(t_3)
    if i_1 != 1:
        c = term_3[i_2, a, i_1 - 1, a]
        # new=c.clone()
        t_3 = t_3 - c * attn_1[dic[i_1], i_1 - 1].min()

        for i in range(0, i_1 - 1):
            c = torch.min(c, term_3[i_2, dic[i_2], i, dic[i]].min())
        c = torch.min(c, term_3[i_2, dic[i_2], i_1, dic[i_1]].min())
        t_3 = t_3 - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c

    if i_1 == 1:
        c = term_3[i_2, a, i_1 - 1, a]
        # new=c.clone()
        t_3 = t_3 - c * attn_1[dic[i_1], i_1 - 1].min()
        c = torch.min(c, term_3[i_2, a, i_1, dic[i_1]].min())
        t_3 = t_3 - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c

    return t_3


def diff_2_4(a, i_1, i_2, j, dic, matrices, attn_1):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if j == i_1:
        return 0
    for k in range(i_2 + 1):
        if j != 0 and j != 1:
            c = term_4[k, dic[k], j - 1][..., dic[j - 1]].max()
            # new = c.clone()
            d = c * attn_1[dic[j], j - 1].min()

            for i in range(0, j - 1):

                c = torch.max(c, term_4[k, dic[k], i][..., dic[i]].max())
            c = torch.max(c, term_4[k, dic[k], j][..., dic[j]].max())
            d = d + (1 - attn_1[dic[j], j - 1].min()) * c
            if k == 0:
                print(c)

        if j == 0:

            d = term_4[k, dic[k], j][..., dic[j]].max()

        if j == 1:
            c = term_4[k, dic[k], j - 1][..., dic[j - 1]].max()
            # new=c.clone()
            d = c * attn_1[dic[j], j - 1].min()
            c = torch.max(c, term_4[k, dic[k], j][..., dic[j]].max())
            d = d + (1 - attn_1[dic[j], j - 1].min()) * c

        # print(d)
        if i_1 != 1:
            c = term_4[k, dic[k], i_1 - 1, a].min()
            # new=c.clone()
            d = d - attn_1[dic[i_1], i_1 - 1].min() * c

            for i in range(0, i_1 - 1):

                c = torch.min(c, term_4[k, dic[k], i][..., dic[i]].min())
            c = torch.min(c, term_4[k, dic[k], i_1][..., dic[i_1]].min())
            d = d - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c

        if i_1 == 1:
            c = term_4[k, dic[k], i_1 - 1, a].min()
            # new=c.clone()
            d = d - attn_1[dic[i_1], i_1 - 1].min() * c

            c = torch.min(c, term_4[k, dic[k], i_1][..., dic[i_1]].min())
            d = d - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c

        # print(d)

        if type(dic[j]) == int:
            d = (
                d
                + (
                    term_2[k, dic[k], j][..., dic[j]]
                    - term_2[k, dic[k], i_1][..., dic[i_1]].min(dim=-1).values
                ).max()
            )

        else:
            d = (
                d
                + (
                    term_2[k, dic[k], j][..., dic[j]].max(dim=-1).values
                    - term_2[k, dic[k], i_1][..., dic[i_1]].min(dim=-1).values
                ).max()
            )

        if k == 0:

            f = d

        if k != 0:
            f = torch.max(f, d)

        if k == i_2 - 1:

            g = d.clone()

    t_4 = g * attn_1[dic[i_2], i_2 - 1]
    t_4 = t_4 + (1 - attn_1[dic[i_2], i_2 - 1]) * f

    return t_4


def diff_2_3_4(a, i_1, i_2, j, dic, matrices, attn_1):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if j == i_1:
        return 0
    for k in range(i_2 + 1):
        if j != 0 and j != 1:
            c = (
                term_4[k, dic[k], j - 1][..., dic[j - 1]].max()
                + term_3[i_2, a, j - 1, dic[j - 1]].max()
            )
            # new = c.clone()
            d = c * attn_1[dic[j], j - 1].min()

            for i in range(0, j - 1):

                c = torch.max(
                    c,
                    term_4[k, dic[k], i][..., dic[i]].max()
                    + term_3[i_2, dic[i_2], i, dic[i]].max(),
                )
            c = torch.max(
                c,
                term_4[k, dic[k], j][..., dic[j]].max()
                + term_3[i_2, dic[i_2], j, dic[j]].max(),
            )
            d = d + (1 - attn_1[dic[j], j - 1].min()) * c
            if k == 0:
                print(c)

        if j == 0:

            d = (
                term_4[k, dic[k], j][..., dic[j]].max()
                + term_3[i_2, a, j, dic[j]].max()
            )

        if j == 1:
            c = (
                term_4[k, dic[k], j - 1][..., dic[j - 1]].max()
                + term_3[i_2, a, j - 1, dic[j - 1]].max()
            )
            # new=c.clone()
            d = c * attn_1[dic[j], j - 1].min()
            c = torch.max(
                c,
                term_4[k, dic[k], j][..., dic[j]].max()
                + term_3[i_2, a, j, dic[j]].max(),
            )
            d = d + (1 - attn_1[dic[j], j - 1].min()) * c

        # print(d)
        if i_1 != 1:
            c = term_4[k, dic[k], i_1 - 1, a].min() + term_3[i_2, a, i_1 - 1, a]
            # new=c.clone()
            d = d - attn_1[dic[i_1], i_1 - 1].min() * c

            for i in range(0, i_1 - 1):

                c = torch.min(
                    c,
                    term_4[k, dic[k], i][..., dic[i]].min()
                    + term_3[i_2, dic[i_2], i, dic[i]].min(),
                )
            c = torch.min(
                c,
                term_4[k, dic[k], i_1][..., dic[i_1]].min()
                + term_3[i_2, dic[i_2], i_1, dic[i_1]].min(),
            )
            d = d - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c

        if i_1 == 1:
            c = term_4[k, dic[k], i_1 - 1, a].min() + term_3[i_2, a, i_1 - 1, a]
            # new=c.clone()
            d = d - attn_1[dic[i_1], i_1 - 1].min() * c

            c = torch.min(
                c,
                term_4[k, dic[k], i_1][..., dic[i_1]].min()
                + term_3[i_2, a, i_1, dic[i_1]].min(),
            )
            d = d - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c

        # print(d)

        if type(dic[j]) == int:
            d = (
                d
                + (
                    term_2[k, dic[k], j][..., dic[j]]
                    - term_2[k, dic[k], i_1][..., dic[i_1]].min(dim=-1).values
                ).max()
            )

        else:
            d = (
                d
                + (
                    term_2[k, dic[k], j][..., dic[j]].max(dim=-1).values
                    - term_2[k, dic[k], i_1][..., dic[i_1]].min(dim=-1).values
                ).max()
            )

        if k == 0:

            f = d

        if k != 0:
            f = torch.max(f, d)

        if k == i_2 - 1:

            g = d.clone()

    t_4 = g * attn_1[dic[i_2], i_2 - 1]
    t_4 = t_4 + (1 - attn_1[dic[i_2], i_2 - 1]) * f

    return t_4


def least_attention(a, i_1, i_2, j, dic, matrices, attn_1):

    g = diff_3(a, i_1, i_2, j, dic, matrices, attn_1)
    f = diff_2_4(a, i_1, i_2, j, dic, matrices, attn_1)
    e = diff_2_3_4(a, i_1, i_2, j, dic, matrices, attn_1)

    # print(a, i_1, i_2, j)
    # print(e)
    # print(f+g)
    # print(e-f-g)
    return diff_1(a, i_1, i_2, j, dic, matrices) + f + g


def second_layer_attention(matrices, attn_1):
    term_0 = matrices[0]
    bound = (
        torch.zeros(
            (
                term_0.shape[1],
                term_0.shape[0],
                term_0.shape[0],
                term_0.shape[0],
            )
        )
        - torch.inf
    )

    for a in range(0, term_0.shape[1]):

        for i_2 in range(3, term_0.shape[0] - 1):
            for i_1 in range(2, i_2):
                for j in range(i_2 + 1):
                    if (i_1 < i_2) & (i_1 > 0) & (i_2 + 1 > j):
                        dic = {
                            i_2: a,
                            i_1 - 1: a,
                        }
                        for i in range(8):
                            dic.setdefault(i, torch.arange(26)[torch.arange(26) != a])
                        bound[a, i_2, i_1, j] = least_attention(
                            a, i_1, i_2, j, dic, matrices, attn_1
                        )

    bound_soft = bound.softmax(dim=-1)
    bound_2 = einops.einsum(
        bound_soft,
        "a i_2 i_1 i_1 ->a i_2 i_1",
    )
    return bound_2


def loss_diff_1(b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if n == b:
        return 0

    if n is None:

        n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        return (
            term_5[i_2, dic[i_2]][..., n] - term_5[i_2, dic[i_2], b].unsqueeze(dim=-1)
        ).max()

    return (term_5[i_2, dic[i_2], n] - term_5[i_2, dic[i_2], b]).max()


def loss_diff_2(b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if n == b:
        return 0

    if n is None:

        n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        c = (
            term_6[i_2 - 1, dic[i_2 - 1]][..., n]
            - term_6[i_2 - 1, dic[i_2 - 1], b].unsqueeze(dim=-1)
        ).max()
        ld_2 = c * attn_1[dic[i_2], i_2 - 1].min()

        for i in range(i_2 - 1):
            c = torch.max(
                c,
                (
                    term_6[i, dic[i]][..., n] - term_6[i, dic[i], b].unsqueeze(dim=-1)
                ).max(),
            )
        c = torch.max(
            c,
            (
                term_6[i_2, dic[i_2]][..., n]
                - term_6[i_2, dic[i_2], b].unsqueeze(dim=-1)
            ).max(),
        )
        ld_2 += (1 - attn_1[dic[i_2], i_2 - 1].min()) * c
        return ld_2

    c = (term_6[i_2 - 1, dic[i_2 - 1], n] - term_6[i_2 - 1, dic[i_2 - 1], b]).max()
    ld_2 = c * attn_1[dic[i_2], i_2 - 1].min()

    for i in range(i_2 - 1):
        c = torch.max(
            c,
            (term_6[i, dic[i], n] - term_6[i, dic[i], b]).max(),
        )
    c = torch.max(
        c,
        (term_6[i_2, dic[i_2], n] - term_6[i_2, dic[i_2], b]).max(),
    )
    ld_2 += (1 - attn_1[dic[i_2], i_2 - 1].min()) * c
    return ld_2


def loss_diff_3(b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None):
    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices
    if n == b:
        return 0

    if n is None:
        n = torch.arange(d_voc)[torch.arange(d_voc) != b]
        c = (
            term_7[i_1, dic[i_1]][..., n] - term_7[i_1, dic[i_1], b].unsqueeze(dim=-1)
        ).max()
        ld_3 = c * bound_2[dic[i_2], i_2, i_1].min()
        for i in range(i_1):
            c = torch.max(
                c,
                (
                    term_7[i, dic[i]][..., n] - term_7[i, dic[i], b].unsqueeze(dim=-1)
                ).max(),
            )
        for i in range(i_2, i_1, -1):
            c = torch.max(
                c,
                (
                    term_7[i, dic[i]][..., n] - term_7[i, dic[i], b].unsqueeze(dim=-1)
                ).max(),
            )

        ld_3 += (1 - bound_2[dic[i_2], i_2, i_1].min()) * c
        return ld_3

    c = (term_7[i_1, dic[i_1], n] - term_7[i_1, dic[i_1], b]).max()
    ld_3 = c * bound_2[dic[i_2], i_2, i_1].min()
    for i in range(i_1):
        c = torch.max(
            c,
            (term_7[i, dic[i], n] - term_7[i, dic[i], b]).max(),
        )
    for i in range(i_2, i_1, -1):
        c = torch.max(
            c,
            (term_7[i, dic[i], n] - term_7[i, dic[i], b]).max(),
        )

    ld_3 += (1 - bound_2[dic[i_2], i_2, i_1].min()) * c
    return ld_3


def loss_diff_4(b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if n == b:
        return 0

    if n is None:

        n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        for k in range(i_2 + 1):
            if k != 0 and k != 1:
                c = (
                    term_8[k - 1, dic[k - 1]][..., n]
                    - term_8[k - 1, dic[k - 1], b].unsqueeze(dim=-1)
                ).max()
                d = c * attn_1[dic[k], k - 1].min()
                for i in range(k - 1):
                    c = torch.max(
                        c,
                        (
                            term_8[i, dic[i]][..., n]
                            - term_8[i, dic[i], b].unsqueeze(dim=-1)
                        ).max(),
                    )
                c = torch.max(
                    c,
                    (
                        term_8[k, dic[k]][..., n]
                        - term_8[k, dic[k], b].unsqueeze(dim=-1)
                    ).max(),
                )
                d += (1 - attn_1[dic[k], k - 1].min()) * c

            if k == 0:
                d = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()

            if k == 1:
                c = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()
                d = c * attn_1[dic[k], k - 1].min()
                c = torch.max(
                    c,
                    (
                        term_8[1, dic[1]][..., n]
                        - term_8[1, dic[1], b].unsqueeze(dim=-1)
                    ).max(),
                )
                d += (1 - attn_1[dic[k], k - 1].min()) * c
            if k == 0:
                f = d
            if k != 0:
                f = torch.max(f, d)
            if k == i_1:
                g = d
        ld_4 = g * (bound_2[dic[i_2], i_2, i_1].min())
        ld_4 += (1 - bound_2[dic[i_2], i_2, i_1].min()) * f
        return ld_4

    for k in range(i_2 + 1):
        if k != 0 and k != 1:
            c = (term_8[k - 1, dic[k - 1], n] - term_8[k - 1, dic[k - 1], b]).max()
            d = c * attn_1[dic[k], k - 1].min()
            for i in range(k - 1):
                c = torch.max(
                    c,
                    (term_8[i, dic[i], n] - term_8[i, dic[i], b]).max(),
                )
            c = torch.max(
                c,
                (term_8[k, dic[k], n] - term_8[k, dic[k], b]).max(),
            )
            d += (1 - attn_1[dic[k], k - 1].min()) * c

        if k == 0:
            d = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()

        if k == 1:
            c = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()
            d = c * attn_1[dic[k], k - 1].min()
            c = torch.max(
                c,
                (term_8[1, dic[1], n] - term_8[1, dic[1], b]).max(),
            )
            d += (1 - attn_1[dic[k], k - 1].min()) * c
        if k == 0:
            f = d
        if k != 0:
            f = torch.max(f, d)
        if k == i_1:
            g = d
    ld_4 = g * (bound_2[dic[i_2], i_2, i_1].min())
    ld_4 += (1 - bound_2[dic[i_2], i_2, i_1].min()) * f
    return ld_4


def loss_diff_3_4(b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None):

    (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8) = matrices

    if n == b:
        return 0

    if n is None:

        n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        for k in range(i_2 + 1):
            if k != 0 and k != 1:
                c = (
                    term_8[k - 1, dic[k - 1]][..., n]
                    - term_8[k - 1, dic[k - 1], b].unsqueeze(dim=-1)
                ).max()
                d = c * attn_1[dic[k], k - 1].min()
                for i in range(k - 1):
                    c = torch.max(
                        c,
                        (
                            term_8[i, dic[i]][..., n]
                            - term_8[i, dic[i], b].unsqueeze(dim=-1)
                        ).max(),
                    )
                c = torch.max(
                    c,
                    (
                        term_8[k, dic[k]][..., n]
                        - term_8[k, dic[k], b].unsqueeze(dim=-1)
                    ).max(),
                )
                d += (1 - attn_1[dic[k], k - 1].min()) * c

            if k == 0:
                d = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()

            if k == 1:
                c = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()
                d = c * attn_1[dic[k], k - 1].min()
                c = torch.max(
                    c,
                    (
                        term_8[1, dic[1]][..., n]
                        - term_8[1, dic[1], b].unsqueeze(dim=-1)
                    ).max(),
                )
                d += (1 - attn_1[dic[k], k - 1].min()) * c

            d = (
                d
                + (
                    term_7[k, dic[k]][..., n] - term_7[k, dic[k], b].unsqueeze(dim=-1)
                ).max()
            )

            if k == 0:
                f = d
            if k != 0:
                f = torch.max(f, d)
            if k == i_1:
                g = d
        ld_4 = g * (bound_2[dic[i_2], i_2, i_1].min())
        ld_4 += (1 - bound_2[dic[i_2], i_2, i_1].min()) * f
        return ld_4

    for k in range(i_2 + 1):
        if k != 0 and k != 1:
            c = (term_8[k - 1, dic[k - 1], n] - term_8[k - 1, dic[k - 1], b]).max()
            d = c * attn_1[dic[k], k - 1].min()
            for i in range(k - 1):
                c = torch.max(
                    c,
                    (term_8[i, dic[i], n] - term_8[i, dic[i], b]).max(),
                )
            c = torch.max(
                c,
                (term_8[k, dic[k], n] - term_8[k, dic[k], b]).max(),
            )
            d += (1 - attn_1[dic[k], k - 1].min()) * c

        if k == 0:
            d = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()

        if k == 1:
            c = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()
            d = c * attn_1[dic[k], k - 1].min()
            c = torch.max(
                c,
                (term_8[1, dic[1], n] - term_8[1, dic[1], b]).max(),
            )
            d += (1 - attn_1[dic[k], k - 1].min()) * c

        d = d + (term_7[k, dic[k], n] - term_7[k, dic[k], b]).max()

        if k == 0:
            f = d
        if k != 0:
            f = torch.max(f, d)
        if k == i_1:
            g = d
    ld_4 = g * (bound_2[dic[i_2], i_2, i_1].min())
    ld_4 += (1 - bound_2[dic[i_2], i_2, i_1].min()) * f
    return ld_4


def total_bound(b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None):
    return (
        loss_diff_1(b, i_1, i_2, dic, matrices, attn_1, bound_2, n)
        + loss_diff_2(b, i_1, i_2, dic, matrices, attn_1, bound_2, n)
        + loss_diff_3_4(b, i_1, i_2, dic, matrices, attn_1, bound_2, n)
        # + loss_diff_3(b, i_1, i_2, dic, matrices, attn_1, bound_2, n)
        # + loss_diff_4(b, i_1, i_2, dic, matrices, attn_1, bound_2, n)
    )


def loss_bound(model):
    matrices = terms(model)
    term_0 = matrices[0]
    attn_1 = first_layer_attention(matrices)
    bound_2 = second_layer_attention(matrices, attn_1)

    out = torch.zeros((d_voc, n_ctx, n_ctx)) + torch.inf
    # b i_2 i_1

    for b in range(term_0.shape[1]):

        for i_2 in range(3, term_0.shape[0] - 1):
            for i_1 in range(2, i_2):

                if (i_1 < i_2) & (i_1 > 0):
                    dic = {i_1: b}
                    for i in range(8):
                        dic.setdefault(i, torch.arange(26))

                    out[b, i_2, i_1] = total_bound(
                        b, i_1, i_2, dic, matrices, attn_1, bound_2, n=None
                    )

    out_2 = 1 / (1 + ((d_voc - 1) * torch.exp(out)))

    return (attn_1, bound_2, out, out_2)


def good_loss_bound(model):
    matrices = terms(model)
    term_0 = matrices[0]
    attn_1 = first_layer_attention(matrices)
    bound_2 = second_layer_attention(matrices, attn_1)

    out = torch.zeros((d_voc, n_ctx, n_ctx, d_voc))
    # b i_2 i_1

    for b in range(term_0.shape[1]):
        for n in range(term_0.shape[1]):
            for i_2 in range(3, term_0.shape[0] - 1):
                for i_1 in range(2, i_2):

                    if (i_1 < i_2) & (i_1 > 0):
                        dic = {i_1: b}
                        for i in range(8):
                            dic.setdefault(i, torch.arange(26))

                        out[b, i_2, i_1, n] = total_bound(
                            b, i_1, i_2, dic, matrices, attn_1, bound_2, n
                        )

    out_2 = einops.einsum(out.softmax(dim=-1), "b i_2 i_1 b -> b i_2 i_1")

    out_3 = einops.einsum(
        out - out.max(dim=-1).values.unsqueeze(dim=-1), "b i_2 i_1 b -> b i_2 i_1"
    )

    return (out, out_2, out_3)


# %%
runtime_model_1, model_1 = train_or_load_model(ABCAB8_1H, force="load")
# %%
optimiser = torch.optim.AdamW(
    model_1.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=1.0
)

counter = 0
# %%
matrices = terms(model_1)
loss = 1 - first_layer_attention(matrices).min()
while loss > 0.02:
    print(1 - loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    matrices = terms(model_1)
    loss = 1 - first_layer_attention(matrices).min()
    counter += 1
    print(counter)
# %%


optimiser = torch.optim.AdamW(
    model_1.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=1.0
)
# %%
torch.autograd.set_detect_anomaly(False)
# %%
weights_1 = torch.zeros((d_voc, n_ctx, n_ctx))
for a in range(d_voc):
    for i_2 in range(3, n_ctx - 1):
        for i_1 in range(2, i_2):
            weights_1[a, i_2, i_1] = (d_voc - 1) ** (i_2 - 1)
# %%
matrices = terms(model_1)
attn_1 = first_layer_attention(matrices)
a = second_layer_attention(matrices, attn_1)
loss = 1 - (torch.nansum(a * weights_1) / (weights_1.sum()))
print(a[~torch.isnan(a)].min())
print(torch.nansum(a * weights_1) / (weights_1.sum()))
print(a[~torch.isnan(a)].max())
while loss > 0.5:
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)
    optimiser.step()
    optimiser.zero_grad()
    matrices = terms(model_1)
    attn_1 = first_layer_attention(matrices)
    a = second_layer_attention(matrices, attn_1)
    loss = 1 - (torch.nansum(a * weights_1) / (weights_1.sum()))
    counter += 1
    print(counter)
    print(a[~torch.isnan(a)].min())
    print(torch.nansum(a * weights_1) / (weights_1.sum()))
    print(a[~torch.isnan(a)].max())


# %%
valid = (
    ein.array(
        lambda i, j, k: where(k > 1, where(j > k, where(j < 7, 1, 0), 0), 0),
        sizes=[d_voc, n_ctx, n_ctx],
    )
    .bool()
    .to(device)
)
weights_2 = ein.array(
    lambda i, j, k: where(k > 1, where(j > k, where(j < 7, 1, 0), 0), 0)
    * ((d_voc - 1) * ((d_voc - 1) ** (j - 2))),
    sizes=[d_voc, n_ctx, n_ctx],
).to(device)
# %%
optimiser = torch.optim.AdamW(
    model_1.parameters(), lr=5e-2, betas=(0.9, 0.999), weight_decay=1.0
)
# %%
# optimiser = torch.optim.SGD(model_1.parameters(), lr=100)
# %%
bound = loss_bound(model_1)[3]
loss = 1 - (torch.nansum(bound * weights_2) / (weights_2.sum()))
print(bound[valid].min())
print(torch.nansum(bound * weights_2) / (weights_2.sum()))
print(bound[valid].max())
while loss > 0.05:
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)
    optimiser.step()
    optimiser.zero_grad()
    bound = loss_bound(model_1)[1]
    loss = 1 - (torch.nansum(bound * weights_2) / (weights_2.sum()))
    counter += 1
    print(counter)
    print(bound[valid].min())
    print(torch.nansum(bound * weights_2) / (weights_2.sum()))
    print(bound[valid].max())


# %%
for i in range(10):
    print(i)
    a = loss_bound(model_1)
    loss = 1 - ((torch.nansum(a[3] * weights_2) / (weights_2.sum())))
    print(a[0].min())
    print(torch.nansum(a[3] * weights_2) / (weights_2.sum()))
    print(torch.nansum(a[1] * weights_1) / (weights_1.sum()))
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()


# %%
ModelMatrixLoggingOptions.all(
    use_subplots=True, add_mean={-1: None, 0: "tok_to_pos", 1: None}
).plot_matrices_from_model(model)


# %%
import torch
import matplotlib.pyplot as plt

# Example tensor with more than 2 dimensions
# Create a 3x4x5 tensor with random elements from a normal distribution

# Flatten the tensor to 1D so that we can plot the histogram of all elements
flattened_tensor = (bound_2[mask]).flatten().detach().cpu().numpy()

# Plot the histogram
plt.hist(flattened_tensor, bins=1000, edgecolor="black")

# Add labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Tensor Elements")

# Show the plot
plt.show()

# %%
import torch as t


def sample(a, b, i, d_voc):
    # i goes from 1 to n_ctx-3
    # randomly fill with tokens which are not equal to a
    seq = t.randint(low=0, high=d_voc - 1, size=(i + 3,))
    seq = seq + (seq >= a).int()

    # fill last position with a
    seq[-1] = a

    # pick position of first a
    m = t.randint(low=0, high=i, size=(1,)).item()

    # fill position m with b
    seq[m + 1] = a
    seq[m + 2] = b
    return seq


def sample_acc_and_loss(model, batch_size=15000):
    d_vocab = model.W_E.shape[0]
    n_ctx = model.W_pos.shape[0]

    acc = 0
    loss = 0

    loss_CE = t.nn.CrossEntropyLoss()

    # Compute probability of each sequence length
    sample_seq_length = t.arange(1, n_ctx - 3)
    prob_sample_seq_len = t.tensor([i * (d_vocab - 1) ** i for i in sample_seq_length])
    prob_sample_seq_len = prob_sample_seq_len / prob_sample_seq_len.sum()

    # sample the sequence length
    sampled = sample_seq_length[
        torch.multinomial(prob_sample_seq_len, num_samples=batch_size, replacement=True)
    ]

    # sample a
    sample_a = t.randint(0, d_vocab, (batch_size,))

    with t.no_grad():
        for i in range(batch_size):
            # sample a
            a = sample_a[i].item()

            # sample b unequal to a
            b = t.randint(0, d_vocab - 1, (1,)).item()
            b = b + (b >= a)
            length = sampled[i]

            # sample sequence
            seq = sample(a, b, length, d_vocab)

            # measure accuracy and loss
            logit = model(seq).squeeze()[-1]
            acc += logit.argmax() == b
            loss += loss_CE(logit.unsqueeze(0), t.tensor([b]))

    return acc / batch_size, loss / batch_size


# %%
