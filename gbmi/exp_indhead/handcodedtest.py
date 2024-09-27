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
n_ctx = W_pos.shape[0]
d_voc = W_E.shape[0]
d_model = W_E.shape[1]


def noise(M):
    return epsilon * (torch.rand_like(M) - 0.5)


def add_noise(*ms):
    for m in ms:
        m += noise(m.shape)


mean_bound = []
for i in range(0, 1):
    epsilon = i / 100
    W_E = ein.array(lambda i, j: i == j, sizes=[d_voc, d_model]).float().to(device)
    W_E = W_E + noise(W_E)

    W_pos = (
        ein.array(lambda i, j: ((i + d_voc) == j) * 1.0, sizes=[n_ctx, d_model])
        .float()
        .to(device)
    )

    W_pos = W_pos + noise(W_pos)

    W_O_0 = (
        ein.array(lambda i, j: ((i + n_ctx + d_voc) == j) * 1.0, sizes=[d_voc, d_model])
        .float()
        .to(device)
    )
    W_O_0 = W_O_0 + noise(W_O_0)

    W_V_0 = (
        ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc])
        .float()
        .to(device)
    )
    W_V_0 = W_V_0 + noise(W_V_0)

    W_V_1 = (
        ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc])
        .float()
        .to(device)
    )
    W_V_1 = W_V_1 + noise(W_V_1)

    W_O_1 = (
        ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_voc, d_model])
        .float()
        .to(device)
    )
    W_O_1 = W_O_1 + noise(W_O_1)

    W_Q_0 = (
        ein.array(
            lambda i, j: where((i + d_voc + 1) == j, c, 0), sizes=[n_ctx, d_model]
        )
        .float()
        .to(device)
        .T
    )
    W_Q_0 = W_Q_0 + noise(W_Q_0)

    W_Q_1 = (
        ein.array(lambda i, j: where(i == j, d, 0), sizes=[d_voc, d_model])
        .float()
        .T.to(device)
    )
    W_Q_1 = W_Q_1 + noise(W_Q_1)

    W_K_0 = (
        ein.array(lambda i, j: where((i + d_voc) == j, c, 0), sizes=[n_ctx, d_model])
        .float()
        .T
    ).to(device)
    W_K_0 = W_K_0 + noise(W_K_0)

    W_K_1 = (
        ein.array(
            lambda i, j: where((i + n_ctx + d_voc) == j, d, 0), sizes=[d_voc, d_model]
        )
        .float()
        .T
    ).to(device)
    W_K_1 = W_K_1 + noise(W_K_1)

    W_U = ein.array(lambda i, j: i == j, sizes=[d_model, d_voc]).float().to(device)
    W_U = W_U + noise(W_U)
    attn_scale_0 = model.blocks[0].attn.attn_scale
    attn_scale_1 = model.blocks[1].attn.attn_scale

    o = W_O_0
    v = W_V_0
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
                tmp_sm = tmp.softmax(dim=0)
                table[t_q, t_k, p - 2, :] = tmp_sm[-2, :]
    # Table represents post softmax attention paid to t_k, if the final entry is spammed everywhere, and t_q is used as the first entry, at pth poisition

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

    # attn_1=torch.ones(attn_1.shape)
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
            for i_1 in range(2, i_2):
                for j in range(i_2 + 1):
                    if (i_1 < i_2) & (i_1 > 0) & (i_2 + 1 > j):
                        dic = {
                            i_2: a,
                            i_1 - 1: a,
                        }
                        for i in range(8):
                            dic.setdefault(i, torch.arange(8)[torch.arange(8) != a])
                        bound[a, i_2, i_1, j] = least_attention(a, i_1, i_2, j, dic)

    bound_soft = bound.softmax(dim=-1)
    bound_2 = einops.einsum(
        bound_soft,
        "a i_2 i_1 i_1 ->a i_2 i_1",
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

    def loss_diff_1(b, i_1, i_2, n, dic):

        if n == b:
            return 0

        return (term_5[i, :, n] - term_5[i, :, b]).max()

    def loss_diff_2(b, i_1, i_2, n, dic):
        if n == b:
            return 0
        c = (term_6[0, dic[0], n] - term_6[0, dic[0], b]).max()
        for i in range(i_2 - 1):
            c = torch.max(c, (term_6[i, dic[i], n] - term_6[i, dic[i], b]).max())
        c = torch.max(c, (term_6[i_2, dic[i_2], n] - term_6[i_2, dic[i_2], b]).max())

        ld_2 = torch.where(c > 0, (1 - attn_1[dic[i_2], i_2 - 1].min()) * c, 0)

        c = (term_6[i_2 - 1, dic[i_2 - 1], n] - term_6[i_2 - 1, dic[i_2 - 1], b]).max()
        ld_2 = torch.where(
            c > 0, ld_2 + c, ld_2 + (c * attn_1[dic[i_2], i_2 - 1].min())
        )
        return ld_2

    def loss_diff_3(b, i_1, i_2, n, dic):
        if n == b:
            return 0
        c = (term_7[0, dic[0], n] - term_7[0, dic[0], b]).max()
        for i in range(i_1):
            c = torch.max(c, (term_7[i, dic[i], n] - term_7[i, dic[i], b]).max())
        for i in range(i_2, i_1, -1):
            c = torch.max(c, (term_7[i, dic[i], n] - term_7[i, dic[i], b]).max())

        ld_3 = torch.where(c > 0, (1 - bound_2[dic[i_2], i_2, i_1].min()) * c, 0)

        c = (term_7[i_1, dic[i_1], n] - term_7[i_1, dic[i_1], b]).max()
        ld_3 = torch.where(
            c > 0, ld_3 + c, ld_3 + (c * bound_2[dic[i_2], i_2, i_1].min())
        )
        return ld_3

    def loss_diff_4(b, i_1, i_2, n, dic):

        if n == b:
            return 0

        for k in range(i_2 + 1):
            if k != 0 and k != 1:
                c = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()
                for i in range(k - 1):
                    c = torch.max(
                        c, (term_8[i, dic[i], n] - term_8[i, dic[i], b]).max()
                    )
                c = torch.max(c, (term_8[k, dic[k], n] - term_8[k, dic[k], b]).max())
                d = torch.where(c > 0, (1 - attn_1[dic[k], k - 1].min()) * c, 0)
                c = (term_8[k - 1, dic[k - 1], n] - term_8[k - 1, dic[k - 1], b]).max()
                d = torch.where(c > 0, d + c, d + c * attn_1[dic[k], k - 1].min())

            if k == 0:
                d = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()

            if k == 1:
                c = (term_8[1, dic[1], n] - term_8[1, dic[1], b]).max()
                d = torch.where(c > 0, (1 - attn_1[dic[k], k - 1].min()) * c, 0)
                c = (term_8[0, dic[0], n] - term_8[0, dic[0], b]).max()
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

    def total_bound(b, i_1, i_2, n, dic):
        return (
            loss_diff_1(b, i_1, i_2, n, dic)
            + loss_diff_2(b, i_1, i_2, n, dic)
            + loss_diff_3(b, i_1, i_2, n, dic)
            + loss_diff_4(b, i_1, i_2, n, dic)
        )

    out = torch.zeros((d_voc, n_ctx, n_ctx, d_voc)) + torch.inf
    # b i_2 i_1 n

    for b in tqdm(range(e_p.shape[1])):
        for n in tqdm(range(e_p.shape[1])):
            for i_2 in range(e_p.shape[0] - 1):
                for i_1 in range(2, i_2):

                    if (i_1 < i_2) & (i_1 > 0):
                        dic = {
                            i_1: b,
                        }
                        for i in range(8):
                            dic.setdefault(i, torch.arange(8))

                        out[b, i_2, i_1, n] = total_bound(b, i_1, i_2, n, dic)

    out_soft = out.softmax(dim=-1)
    out_2 = einops.einsum(
        out_soft,
        "b i_2 i_1 b ->b i_2 i_1",
    )


# %%
r_W_pos = model.W_pos
r_W_E = model.W_E
r_W_K_1 = model.W_K[1, 0]
r_W_U = model.W_U
r_W_V_1 = model.W_V[1, 0]
r_W_K_0 = model.W_K[0, 0]
r_W_V_0 = model.W_V[0, 0]
r_W_O_0 = model.W_O[0, 0]
r_W_Q_1 = model.W_Q[1, 0]
r_W_Q_0 = model.W_Q[0, 0]
r_W_O_1 = model.W_O[1, 0]
r_W_Q_0 = model.W_Q[0, 0]

r_e_p = r_W_E.unsqueeze(dim=0) + r_W_pos.unsqueeze(dim=1)
r_term_1 = (
    einops.einsum(
        r_e_p,
        r_W_Q_1,
        r_W_K_1,
        r_e_p,
        "q_pos q_val k, k l, m l, k_pos k_val m -> q_pos q_val k_pos k_val",
    )
    / attn_scale_1
)
r_term_2 = (
    einops.einsum(
        r_e_p,
        r_W_V_0,
        r_W_O_0,
        r_W_Q_1,
        r_W_K_1,
        r_e_p,
        "q_pos q_val k, k l, l m, m n, o n, k_pos k_val o -> q_pos q_val k_pos k_val",
    )
    / attn_scale_1
)

r_term_3 = (
    einops.einsum(
        r_e_p,
        r_W_Q_1,
        r_W_K_1,
        r_W_O_0,
        r_W_V_0,
        r_e_p,
        "q_pos q_val k, k l, m l, n m, o n, k_pos k_val o -> q_pos q_val k_pos k_val",
    )
    / attn_scale_1
)

r_term_4 = (
    einops.einsum(
        r_e_p,
        r_W_V_0,
        r_W_O_0,
        r_W_Q_1,
        r_W_K_1,
        r_W_O_0,
        r_W_V_0,
        r_e_p,
        "q_pos q_val k, k l, l m, m n, o n, p o, q p, k_pos k_val q -> q_pos q_val k_pos k_val",
    )
    / attn_scale_1
)


for i in range(10, 40):
    epsilon = i / 100
    error_1 = []
    error_2 = []
    error_3 = []
    for j in range(20):
        W_E = ein.array(lambda i, j: i == j, sizes=[d_voc, d_model]).float().to(device)
        W_E = W_E + noise(W_E)

        W_pos = (
            ein.array(lambda i, j: ((i + d_voc) == j) * 1.0, sizes=[n_ctx, d_model])
            .float()
            .to(device)
        )

        W_pos = W_pos + noise(W_pos)

        W_O_0 = (
            ein.array(
                lambda i, j: ((i + n_ctx + d_voc) == j) * 1.0, sizes=[d_voc, d_model]
            )
            .float()
            .to(device)
        )
        W_O_0 = W_O_0 + noise(W_O_0)

        W_V_0 = (
            ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc])
            .float()
            .to(device)
        )
        W_V_0 = W_V_0 + noise(W_V_0)

        W_V_1 = (
            ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc])
            .float()
            .to(device)
        )
        W_V_1 = W_V_1 + noise(W_V_1)

        W_O_1 = (
            ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_voc, d_model])
            .float()
            .to(device)
        )
        W_O_1 = W_O_1 + noise(W_O_1)

        W_Q_0 = (
            ein.array(
                lambda i, j: where((i + d_voc + 1) == j, c, 0), sizes=[n_ctx, d_model]
            )
            .float()
            .to(device)
            .T
        )
        W_Q_0 = W_Q_0 + noise(W_Q_0)

        W_Q_1 = (
            ein.array(lambda i, j: where(i == j, d, 0), sizes=[d_voc, d_model])
            .float()
            .T.to(device)
        )
        W_Q_1 = W_Q_1 + noise(W_Q_1)

        W_K_0 = (
            ein.array(
                lambda i, j: where((i + d_voc) == j, c, 0), sizes=[n_ctx, d_model]
            )
            .float()
            .T
        ).to(device)
        W_K_0 = W_K_0 + noise(W_K_0)

        W_K_1 = (
            ein.array(
                lambda i, j: where((i + n_ctx + d_voc) == j, d, 0),
                sizes=[d_voc, d_model],
            )
            .float()
            .T
        ).to(device)
        W_K_1 = W_K_1 + noise(W_K_1)

        W_U = ein.array(lambda i, j: i == j, sizes=[d_model, d_voc]).float().to(device)
        W_U = W_U + noise(W_U)
        attn_scale_0 = model.blocks[0].attn.attn_scale
        attn_scale_1 = model.blocks[1].attn.attn_scale

        o = W_O_0
        v = W_V_0
        e_p = W_E.unsqueeze(dim=0) + W_pos.unsqueeze(dim=1)
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

        term_3
