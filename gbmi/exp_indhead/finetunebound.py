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
def loss_bound(model, s):

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

    if s == -1:
        return (term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8)

    if s == 0:
        reduced_3 = einops.einsum(
            term_3, "q_pos q_val k_pos k_val -> q_pos q_val k_pos"
        )
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
    # Table represents post softmax attention paid to t_k, if the final entry is spammed everywhere, and t_q is used as the first entry, at pth poisition

    # term_0 looks like EQKE, table looks like you're indexing by query, key, position (of key?), and other token in the sequence.
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

    if s == 1:
        return attn_1

    # attn_1=torch.ones(attn_1.shape)

    def diff_1(a, i_1, i_2, j, dic):

        if j == i_1:
            return 0
        else:
            return term_1[i_2, a, j, dic[j]].max() - term_1[i_2, a, i_1, dic[i_1]].min()

    def diff_3(a, i_1, i_2, j, dic):

        if j == i_1:
            return 0
        if j != 0 and j != 1:
            c = term_3[i_2, dic[i_2], 0, dic[0]].max()
            for i in range(1, j - 1):
                c = torch.max(c, term_3[i_2, dic[i_2], i, dic[i]].max())
            c = torch.max(c, term_3[i_2, dic[i_2], j, dic[j]].max())

            t_3 = torch.where(c > 0, (1 - attn_1[dic[j], j - 1].min()) * c, 0)

            """
            if a != 0 and a != d_voc - 1:
                c = torch.max(
                    term_3[i_2, a, j - 1, :a].max(), term_3[i_2, a, j - 1, a + 1 :].max()
                )
            if a == 0:
                c = term_3[i_2, a, j - 1, a + 1 :].max()

            if a == d_voc - 1:
                c = term_3[i_2, a, j - 1, :a].max()
            """
            c = term_3[i_2, a, j - 1, dic[j - 1]].max()
            t_3 = torch.where(c > 0, t_3 + c, t_3 + (attn_1[dic[j], j - 1].min() * c))

            # print(t_3)
        if j == 1:
            c = term_3[i_2, a, j, dic[j]].max()

            t_3 = torch.where(c > 0, (1 - attn_1[dic[j], j - 1].min()) * c, 0)

            """
            if a != 0 and a != d_voc - 1:

                c = torch.max(
                    term_3[i_2, a, j - 1, :a].max(), term_3[i_2, a, j - 1, a + 1 :].max()
                )

            if a == 0:
                c = term_3[i_2, a, j - 1, a + 1 :].max()

            if a == d_voc - 1:
                c = term_3[i_2, a, j - 1, :a].max()
            """
            c = term_3[i_2, a, j - 1, dic[j - 1]].max()

            t_3 = torch.where(c > 0, t_3 + c, t_3 + (attn_1[dic[j], j - 1].min() * c))

            # print(t_3)
        if j == 0:

            t_3 = term_3[i_2, a, j, dic[j]].max()
            # print(t_3)
        if i_1 != 1:
            c = term_3[i_2, dic[i_2], 0, dic[0]].min()
            for i in range(1, i_1 - 1):
                c = torch.min(c, term_3[i_2, dic[i_2], i, dic[i]].min())
            c = torch.min(c, term_3[i_2, dic[i_2], i_1, dic[i_1]].min())

            # c = torch.min(term_3[i_2, a, : i_1 - 1, a].min(), term_3[i_2, a, i_1, b].min())

            t_3 = torch.where(
                c < 0, t_3 - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c, t_3
            )

            c = term_3[i_2, a, i_1 - 1, a]

            t_3 = torch.where(c < 0, t_3 - c, t_3 - attn_1[dic[i_1], i_1 - 1].min() * c)

            # print(t_3)
        if i_1 == 1:
            c = term_3[i_2, a, i_1, dic[i_1]].min()

            torch.where(c < 0, t_3 - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c, t_3)

            c = term_3[i_2, a, i_1 - 1, a]

            torch.where(c < 0, t_3 - c, t_3 - attn_1[dic[i_1], i_1 - 1].min() * c)

            # print(t_3)

        return t_3

    def diff_2_4(a, i_1, i_2, j, dic):
        if j == i_1:
            return 0
        for k in range(i_2 + 1):
            if j != 0 and j != 1:

                c = term_4[k, dic[k], 0][..., dic[0]].max()
                for i in range(1, j - 1):

                    c = torch.max(c, term_4[k, dic[k], i][..., dic[i]].max())
                c = torch.max(c, term_4[k, dic[k], j][..., dic[j]].max())

                d = torch.where(c > 0, (1 - attn_1[dic[j], j - 1].min()) * c, 0)

                c = term_4[k, dic[k], j - 1][..., dic[j - 1]].max()

                d = torch.where(c > 0, d + c, d + attn_1[dic[j], j - 1].min() * c)

            if j == 0:

                d = term_4[k, dic[k], j][..., dic[j]].max()

            if j == 1:
                c = term_4[k, dic[k], j][..., dic[j]].max()

                d = torch.where(c > 0, (1 - attn_1[dic[j], j - 1].min()) * c, 0)

                c = term_4[k, dic[k], j - 1][..., dic[j - 1]].max()

                d = torch.where(c > 0, d + c, d + attn_1[dic[j], j - 1].min() * c)

            # print(d)
            if i_1 != 1:

                c = term_4[k, dic[k], 0][..., dic[0]].min()

                for i in range(1, i_1 - 1):

                    c = torch.min(c, term_4[k, dic[k], i][..., dic[i]].min())
                c = torch.min(c, term_4[k, dic[k], i_1][..., dic[i_1]].min())

                d = torch.where(c < 0, d - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c, d)

                c = term_4[k, dic[k], i_1 - 1, a].min()

                d = torch.where(c < 0, d - c, d - attn_1[dic[i_1], i_1 - 1].min() * c)

            if i_1 == 1:
                c = term_4[k, dic[k], i_1][..., dic[i_1]].min()

                d = torch.where(c < 0, d - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c, d)

                c = term_4[k, dic[k], i_1 - 1, a].min()

                d = torch.where(c < 0, d - c, d - attn_1[dic[i_1], i_1 - 1].min() * c)

            # print(d)

            if type(dic[j]) == int:
                d += (
                    term_2[k, dic[k], j][..., dic[j]]
                    - term_2[k, dic[k], i_1][..., dic[i_1]].min(dim=-1).values
                ).max()

            else:
                d += (
                    term_2[k, dic[k], j][..., dic[j]].max(dim=-1).values
                    - term_2[k, dic[k], i_1][..., dic[i_1]].min(dim=-1).values
                ).max()

            if k == 0:

                f = d

            if k != 0 and k != i_2 - 1:
                f = torch.max(f, d)

            if k == i_2 - 1:

                g = d

        t_4 = torch.where(f > 0, (1 - attn_1[dic[i_2], i_2 - 1]) * f, 0)

        t_4 = torch.where(g > 0, t_4 + g, t_4 + g * attn_1[dic[i_2], i_2 - 1])

        return t_4

    def least_attention(a, i_1, i_2, j, dic):
        e = diff_2_4(a, i_1, i_2, j, dic)

        return diff_1(a, i_1, i_2, j, dic) + diff_3(a, i_1, i_2, j, dic) + e

    bound = (
        torch.zeros(
            (
                e_p.shape[1],
                e_p.shape[0],
                e_p.shape[0],
                e_p.shape[0],
            )
        )
        - torch.inf
    )

    for a in range(e_p.shape[1]):

        for i_2 in range(e_p.shape[0] - 1):
            for i_1 in range(i_2):
                for j in range(i_2 + 1):
                    if (i_1 < i_2) & (i_1 > 0) & (i_2 + 1 > j):
                        dic = {
                            i_2: a,
                            i_1 - 1: a,
                        }
                        for i in range(8):
                            dic.setdefault(i, torch.arange(26)[torch.arange(26) != a])
                        bound[a, i_2, i_1, j] = least_attention(a, i_1, i_2, j, dic)

    bound_soft = bound.softmax(dim=-1)
    bound_2 = einops.einsum(
        bound_soft,
        "a i_2 i_1 i_1 ->a i_2 i_1",
    )

    if s == 2:
        return (attn_1, bound, bound_2)

    def loss_diff_1(b, i_1, i_2, dic, n=None):

        if n == b:
            return 0

        if n is None:

            n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        return (
            term_5[i_2, dic[i_2]][..., n] - term_5[i_2, dic[i_2], b].unsqueeze(dim=-1)
        ).max()

    def loss_diff_2(b, i_1, i_2, dic, n=None):

        if n == b:
            return 0

        if n is None:

            n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        c = (term_6[0, dic[0]][..., n] - term_6[0, dic[0], b].unsqueeze(dim=-1)).max()

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
        ld_2 = torch.where(c > 0, (1 - attn_1[dic[i_2], i_2 - 1].min()) * c, 0)

        c = (
            term_6[i_2 - 1, dic[i_2 - 1]][..., n]
            - term_6[i_2 - 1, dic[i_2 - 1], b].unsqueeze(dim=-1)
        ).max()
        ld_2 = torch.where(
            c > 0, ld_2 + c, ld_2 + (c * attn_1[dic[i_2], i_2 - 1].min())
        )
        return ld_2

    def loss_diff_3(b, i_1, i_2, dic, n=None):
        if n == b:
            return 0

        if n is None:
            n = torch.arange(d_voc)[torch.arange(d_voc) != b]
        c = (term_7[0, dic[0]][..., n] - term_7[0, dic[0], b].unsqueeze(dim=-1)).max()
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

        ld_3 = torch.where(c > 0, (1 - bound_2[dic[i_2], i_2, i_1].min()) * c, 0)

        c = (
            term_7[i_1, dic[i_1]][..., n] - term_7[i_1, dic[i_1], b].unsqueeze(dim=-1)
        ).max()
        ld_3 = torch.where(
            c > 0, ld_3 + c, ld_3 + (c * bound_2[dic[i_2], i_2, i_1].min())
        )
        return ld_3

    def loss_diff_4(b, i_1, i_2, dic, n=None):

        if n == b:
            return 0

        if n is None:

            n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        for k in range(i_2 + 1):
            if k != 0 and k != 1:
                c = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()
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
                d = torch.where(c > 0, (1 - attn_1[dic[k], k - 1].min()) * c, 0)
                c = (
                    term_8[k - 1, dic[k - 1]][..., n]
                    - term_8[k - 1, dic[k - 1], b].unsqueeze(dim=-1)
                ).max()
                d = torch.where(c > 0, d + c, d + c * attn_1[dic[k], k - 1].min())

            if k == 0:
                d = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()

            if k == 1:
                c = (
                    term_8[1, dic[1]][..., n] - term_8[1, dic[1], b].unsqueeze(dim=-1)
                ).max()
                d = torch.where(c > 0, (1 - attn_1[dic[k], k - 1].min()) * c, 0)
                c = (
                    term_8[0, dic[0]][..., n] - term_8[0, dic[0], b].unsqueeze(dim=-1)
                ).max()
                d = torch.where(c > 0, d + c, d + c * attn_1[dic[k], k - 1].min())
            if k == 0:
                f = d
            if k != 0 and k != i_1:
                f = torch.max(f, d)
            if k == i_1:
                g = d

        ld_4 = torch.where(f > 0, (1 - bound_2[dic[i_2], i_2, i_1].min()) * f, 0)
        ld_4 = torch.where(
            g > 0, ld_4 + g, ld_4 + g * (bound_2[dic[i_2], i_2, i_1].min())
        )
        return ld_4

    def total_bound(b, i_1, i_2, dic, n=None):
        return (
            loss_diff_1(b, i_1, i_2, dic, n)
            + loss_diff_2(b, i_1, i_2, dic, n)
            + loss_diff_3(b, i_1, i_2, dic, n)
            + loss_diff_4(b, i_1, i_2, dic, n)
        )

    if s == 3:

        out = torch.zeros((d_voc, n_ctx, n_ctx)) + torch.inf
        # b i_2 i_1

        for b in range(e_p.shape[1]):

            for i_2 in range(e_p.shape[0] - 1):
                for i_1 in range(1, i_2):

                    if (i_1 < i_2) & (i_1 > 0):
                        dic = {i_1: b}
                        for i in range(8):
                            dic.setdefault(i, torch.arange(26))

                        out[b, i_2, i_1] = total_bound(b, i_1, i_2, dic)

        out_2 = 1 / (1 + ((d_voc - 1) * torch.exp(out)))

        return (attn_1, bound, bound_2, out, out_2)

    out = torch.zeros((d_voc, n_ctx, n_ctx, d_voc)) + torch.inf
    # b i_2 i_1

    for b in range(e_p.shape[1]):
        for n in range(e_p.shape[1]):
            for i_2 in range(e_p.shape[0] - 1):
                for i_1 in range(1, i_2):

                    if (i_1 < i_2) & (i_1 > 0):
                        dic = {i_1: b}
                        for i in range(8):
                            dic.setdefault(i, torch.arange(26))

                        out[b, i_2, i_1, n] = total_bound(b, i_1, i_2, dic, n)

    out_2 = einops.einsum(out.softmax(dim=-1), "b i_2 i_1 b -> b i_2 i_1")

    out_3 = einops.einsum(
        out - out.max(dim=-1).values.unsqueeze(dim=-1), "b i_2 i_1 b -> b i_2 i_1"
    )

    return (attn_1, bound, bound_2, out, out_2, out_3)


# %%
optimiser = torch.optim.AdamW(
    model_1.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=1.0
)

counter = 0
# %%
loss = loss_bound(model_1, 0)
for i in range(100):
    print(loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    loss = loss_bound(model_1, 0)
    counter += 1
    print(counter)


# %%
loss = 1 - loss_bound(model_1, 1).min()
while loss > 0.02:
    print(1 - loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    loss = 1 - loss_bound(model_1, 1).min()
    counter += 1
    print(counter)
# %%

a = loss_bound(model_1, 2)[2]
loss = 1 - a[~torch.isnan(a)].mean()
while loss > 0.1:
    print(1 - loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    a = loss_bound(model_1, 2)[2]
    loss = 1 - a[~torch.isnan(a)].mean()
    counter += 1
    print(counter)
# %%
a = loss_bound(model_1, 2)[2]
loss = 1 - a[~torch.isnan(a)].min()
while loss > 0.5:
    print(1 - loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    a = loss_bound(model_1, 2)[2]
    loss = 1 - a[~torch.isnan(a)].min()
    counter += 1
    print(counter)


# %%
valid = (
    ein.array(
        lambda i, j, k: where(k > 0, where(j > k, where(j < 7, 1, 0), 0), 0),
        sizes=[d_voc, n_ctx, n_ctx],
    )
    .bool()
    .to(device)
)
optimiser = torch.optim.AdamW(
    model_1.parameters(), lr=1, betas=(0.9, 0.999), weight_decay=0
)
# %%
optimiser = torch.optim.SGD(model_1.parameters(), lr=100)
# %%
a = loss_bound(model_1, 3)[4]
loss = 1 - a[valid].mean()
print(a[valid].min())
print(a[valid].mean())
print(a[valid].max())
for i in range(1):
    print(i + 1)

    loss.backward()
    optimiser.step()
    for param in model_1.parameters():
        if param.requires_grad:
            print(param.grad.norm())  # Check gradient norms

    optimiser.zero_grad()
    a = loss_bound(model_1, 3)[4]
    loss = 1 - a[valid].mean()
    print(a[valid].min())
    print(a[valid].mean())
    print(a[valid].max())
    if i % 10 == 1:
        r = loss_bound(model_1, 4)[5]
        print(r[valid].min())
        print(r[valid].mean())
        print(r[valid].max())

# %%
'''
def least_attention_2(a, b, i_1, i_2, j):

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

        t_3 = term_3[i_2, a, j, :].max()

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
    print(t_1, t_2, t_3, t_4)
    return t_1 + t_2 + t_3 + t_4


# %%
bound_a = (
    torch.zeros((e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[0]))
    - torch.inf
)

for a in tqdm(range(e_p.shape[1])):
    for b in tqdm(range(e_p.shape[1])):
        for i_2 in range(e_p.shape[0] - 1):
            for i_1 in range(e_p.shape[0] - 1):
                for j in range(i_2 + 1):
                    if (i_1 < i_2) & (i_1 > 0) & (i_2 + 1 > j) & (a != b):
                        bound_a[a, b, i_2, i_1, j] = least_attention_2(
                            a, b, i_1, i_2, j
                        )

# %%
bound_soft_a = bound_a.softmax(dim=-1)
bound_2_a = einops.einsum(
    bound_soft_a,
    "a b i_2 i_1 i_1 -> a b i_2 i_1",
)
'''

"""
def diff_2(a, b, i_1, i_2, j):
    if j == i_1:
        return 0
    diff = term_2[:, :, j, :] - term_2[:, :, i_1, b].unsqueeze(dim=-1)
    c = torch.max(diff[: i_2 - 1, :, :].max(), diff[i_2, a, :].max())
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


def diff_2_2(a, b, i_1, i_2, j):
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

    c = torch.min(term_2[: i_2 - 1, :, i_1, b].min(), term_2[i_2, a, i_1, b].min())
    if c < 0:
        t_2 -= (1 - attn_1[:, i_2 - 1].min()) * c

    c = term_2[i_2 - 1, :, i_1, b].min()
    if c < 0:
        t_2 -= c
    else:
        t_2 -= attn_1[:, i_2 - 1].min() * c

    return t_2

"""


"""

def diff_4(a, b, i_1, i_2, j):
    if j == i_1:
        return 0
    diff = []
    for k in range(i_2 + 1):
        if j != 0 and j != 1:
            c = torch.max(term_4[k, :, : j - 1, :].max(), term_4[k, :, j, :].max())
            if c > 0:
                d = (1 - attn_1[:, j - 1].min()) * c
            else:
                d = 0
            c = term_4[k, :, j - 1, :].max()
            if c > 0:
                d += c
            else:
                d += attn_1[:, j - 1].min() * c
        if j == 0:
            d = term_4[k, :, j, :].max()

        if j == 1:
            c = term_4[k, :, j, :].max()
            if c > 0:
                d = (1 - attn_1[:, j - 1].min()) * c
            else:
                d = 0
            c = term_4[k, :, j - 1, :].max()
            if c > 0:
                d += c
            else:
                d += attn_1[:, j - 1].min() * c
        if i_1 != 1:
            c = torch.min(term_4[k, :, : i_1 - 1, :].min(), term_4[k, :, i_1, b].min())
            if c < 0:
                d -= (1 - attn_1[:, i_1 - 1].min()) * c
            c = term_4[k, :, i_1 - 1, a].min()
            if c < 0:
                d -= c
            else:
                d -= attn_1[:, i_1 - 1].min() * c
        if i_1 == 1:
            c = term_4[k, :, i_1, b].min()
            if c < 0:
                d -= (1 - attn_1[:, i_1 - 1].min()) * c
            c = term_4[k, :, i_1 - 1, a].min()
            if c < 0:
                d -= c
            else:
                d -= attn_1[:, i_1 - 1].min() * c

        diff.append(d)
    diff = torch.tensor(diff)
    c = torch.max(diff[: i_2 - 1].max(), diff[i_2])
    if c > 0:
        t_4 = (1 - attn_1[:, i_2 - 1].min()) * c
    else:
        t_4 = 0
    c = diff[i_2 - 1]
    if c > 0:
        t_4 += c
    else:
        t_4 += c * attn_1[:, i_2 - 1].min()
    return t_4



def diff_4_2(a, b, i_1, i_2, j):
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
"""

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
