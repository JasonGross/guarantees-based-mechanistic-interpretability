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

"""
def armin(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return ein.apply(f, collect=lambda xs, d: xs.max(d).indices, sizes=sizes)
"""

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


# %%
def noise(M, v):
    return M + (torch.rand_like(M) - 0.5) * 2 * v


W_E = ein.array(lambda i, j: i == j, sizes=[d_voc, d_model]).float().to(device)
W_pos = (
    ein.array(lambda i, j: ((i + d_voc) == j) * 1.0, sizes=[n_ctx, d_model])
    .float()
    .to(device)
)
W_O_0 = (
    ein.array(lambda i, j: ((i + n_ctx + d_voc) == j) * 1.0, sizes=[d_voc, d_model])
    .float()
    .to(device)
)
W_V_0 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
W_V_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
W_O_1 = (
    ein.array(lambda i, j: (i == j) * 100, sizes=[d_voc, d_model]).float().to(device)
)
W_Q_0 = (
    ein.array(lambda i, j: where((i + d_voc + 1) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .to(device)
    .T
)
W_Q_1 = (
    ein.array(lambda i, j: where(i == j, d, 0), sizes=[d_voc, d_model])
    .float()
    .T.to(device)
)
W_K_0 = (
    ein.array(lambda i, j: where((i + d_voc) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .T
).to(device)
W_K_1 = (
    ein.array(
        lambda i, j: where((i + n_ctx + d_voc) == j, d, 0),
        sizes=[d_voc, d_model],
    )
    .float()
    .T
).to(device)
W_U = ein.array(lambda i, j: i == j, sizes=[d_model, d_voc]).float().to(device)

# %%
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale
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


def hand_bound(
    W_E, W_pos, W_V_0, W_V_1, W_O_0, W_O_1, W_Q_0, W_Q_1, W_K_0, W_K_1, W_U, s, v
):

    W_E = noise(W_E, v)

    W_pos = noise(W_pos, v)

    W_O_0 = noise(W_O_0, v)

    W_V_0 = noise(W_V_0, v)

    W_V_1 = noise(W_V_1, v)

    W_O_1 = noise(W_O_1, v)

    W_Q_0 = noise(W_Q_0, v)

    W_Q_1 = noise(W_Q_1, v)

    W_K_0 = noise(W_K_0, v)

    W_K_1 = noise(W_K_1, v)

    W_U = noise(W_U, v)

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

    if s == 1:
        return attn_1

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

    def loss_diff_1(b, i_1, i_2, dic):

        n = torch.arange(d_voc)[torch.arange(d_voc) != b]

        return (
            term_5[i_2, dic[i_2]][..., n] - term_5[i_2, dic[i_2], b].unsqueeze(dim=-1)
        ).max()

    def loss_diff_2(b, i_1, i_2, dic):

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

    def loss_diff_3(b, i_1, i_2, dic):
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

    def loss_diff_4(b, i_1, i_2, dic):

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

    def total_bound(b, i_1, i_2, dic):
        return (
            loss_diff_1(b, i_1, i_2, dic)
            + loss_diff_2(b, i_1, i_2, dic)
            + loss_diff_3(b, i_1, i_2, dic)
            + loss_diff_4(b, i_1, i_2, dic)
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

        t_3 = torch.where(c < 0, t_3 - (1 - attn_1[dic[i_1], i_1 - 1].min()) * c, t_3)

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


# %%
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

for j in range(7):
    print(
        diff_3(
            1,
            1,
            3,
            j,
            {
                0: 1,
                3: 1,
                1: torch.arange(26)[torch.arange(26) != 1],
                2: torch.arange(26)[torch.arange(26) != 1],
                4: torch.arange(26)[torch.arange(26) != 1],
                5: torch.arange(26)[torch.arange(26) != 1],
                6: torch.arange(26)[torch.arange(26) != 1],
                7: torch.arange(26)[torch.arange(26) != 1],
            },
        )
    )
# %%
valid = (
    ein.array(
        lambda i, j, k: where(k > 1, where(j > k, where(j < 7, 1, 0), 0), 0),
        sizes=[d_voc, n_ctx, n_ctx],
    )
    .bool()
    .to(device)
)
bound = []
for v in range(50, 70):
    r = hand_bound(45, v / 1000)
    a = r[4]
    print(a[valid].min())
    bound.append(a[valid].min())

# %%
