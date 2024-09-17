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
for i in range(10, 40):
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
    print(v.shape, "w_v_0")
    print(o.shape, "w_o_0")

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

    def diff_1(a, b, i_1, i_2, j, dic):
        if j == i_1:
            return 0
        else:
            return term_1[i_2, a, j, dic[j]] - term_1[i_2, a, i_1, b]

    def diff_3(a, b, i_1, i_2, j, dic):
        if j == i_1:
            return 0
        if j != 0 and j != 1:
            c = term_3[i_2, dic[i_2], 0, dic[0]].max()
            for i in range(1, j - 1):
                c = torch.max(c, term_3[i_2, dic[i_2], i, dic[i]].max())
            c = torch.max(c, term_3[i_2, dic[i_2], j, dic[j]])

            if c > 0:
                t_3 = (1 - attn_1[dic[j], j - 1]) * c
            else:
                t_3 = 0

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
            c = term_3[i_2, a, j - 1, dic[j - 1]]
            if c > 0:
                t_3 += c
            else:
                t_3 += attn_1[dic[j], j - 1] * c
            # print(t_3)
        if j == 1:
            c = term_3[i_2, a, j, dic[j]]
            if c > 0:
                t_3 = (1 - attn_1[dic[j], j - 1]) * c
            else:
                t_3 = 0
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
            c = term_3[i_2, a, j - 1, dic[j - 1]]
            if c > 0:
                t_3 += c
            else:
                t_3 += attn_1[dic[j], j - 1] * c
            # print(t_3)
        if j == 0:

            t_3 = term_3[i_2, a, j, dic[j]]
            # print(t_3)
        if i_1 != 1:
            c = term_3[i_2, dic[i_2], 0, dic[0]].min()
            for i in range(1, i_1 - 1):
                c = torch.min(c, term_3[i_2, dic[i_2], i, dic[i]].min())
            c = torch.min(c, term_3[i_2, dic[i_2], i_1, dic[i_1]])

            # c = torch.min(term_3[i_2, a, : i_1 - 1, a].min(), term_3[i_2, a, i_1, b].min())
            if c < 0:
                t_3 -= (1 - attn_1[dic[i_1], i_1 - 1]) * c

            c = term_3[i_2, a, i_1 - 1, a]
            if c < 0:
                t_3 -= c
            else:
                t_3 -= attn_1[dic[i_1], i_1 - 1] * c
            # print(t_3)
        if i_1 == 1:
            c = term_3[i_2, a, i_1, b]
            if c < 0:
                t_3 -= (1 - attn_1[dic[i_1], i_1 - 1]) * c

            c = term_3[i_2, a, i_1 - 1, a]
            if c < 0:
                t_3 -= c
            else:
                t_3 -= attn_1[dic[i_1], i_1 - 1] * c
            # print(t_3)
        return t_3

    def diff_2_4(a, b, i_1, i_2, j, dic):
        if j == i_1:
            return 0
        diff = term_2[:, :, j, dic[j]] - term_2[:, :, i_1, b]
        f = []
        for k in range(i_2 + 1):
            if j != 0 and j != 1:

                c = term_4[k, dic[k], 0][..., dic[0]].max()
                for i in range(1, j - 1):

                    c = torch.max(c, term_4[k, dic[k], i][..., dic[i]].max())
                c = torch.max(c, term_4[k, dic[k], j, dic[j]].max())

                if c > 0:
                    d = (1 - attn_1[dic[j], j - 1]) * c
                else:
                    d = 0

                c = term_4[k, dic[k], j - 1, dic[j - 1]].max()

                if c > 0:
                    d += c
                else:
                    d += attn_1[dic[j], j - 1] * c
            if j == 0:

                d = term_4[k, dic[k], j, dic[j]].max()

            if j == 1:
                c = term_4[k, dic[k], j, dic[j]].max()
                if c > 0:
                    d = (1 - attn_1[dic[j], j - 1]) * c
                else:
                    d = 0

                c = term_4[k, dic[k], j - 1, dic[j - 1]].max()

                if c > 0:
                    d += c
                else:
                    d += attn_1[dic[j], j - 1] * c
            # print(d)
            if i_1 != 1:

                c = term_4[k, dic[k], 0][..., dic[0]].min()

                for i in range(1, i_1 - 1):

                    c = torch.min(c, term_4[k, dic[k], i][..., dic[i]].min())
                c = torch.min(c, term_4[k, dic[k], i_1, dic[i_1]].min())

                if c < 0:
                    d -= (1 - attn_1[dic[i_1], i_1 - 1]) * c

                c = term_4[k, dic[k], i_1 - 1, a].min()

                if c < 0:
                    d -= c
                else:
                    d -= attn_1[dic[i_1], i_1 - 1] * c
            if i_1 == 1:
                c = term_4[k, dic[k], i_1, b].min()
                if c < 0:
                    d -= (1 - attn_1[dic[i_1], i_1 - 1]) * c

                c = term_4[k, dic[k], i_1 - 1, a].min()

                if c < 0:
                    d -= c
                else:
                    d -= attn_1[dic[i_1], i_1 - 1] * c
            # print(d)

            d += diff[k, dic[k]].max()

            f.append(d)
        f = torch.tensor(f)
        # print(f)
        c = torch.max(f[: i_2 - 1].max(), f[i_2])
        if c > 0:
            t_4 = (1 - attn_1[dic[i_2], i_2 - 1]) * c
        else:
            t_4 = 0
        c = f[i_2 - 1]
        if c > 0:
            t_4 += c
        else:
            t_4 += c * attn_1[dic[i_2], i_2 - 1]
        return t_4

    def least_attention(a, b, i_1, i_2, j, dic):
        e = diff_2_4(a, b, i_1, i_2, j, dic)
        return diff_1(a, b, i_1, i_2, j, dic) + diff_3(a, b, i_1, i_2, j, dic) + e

    bound = (
        torch.zeros(
            (
                e_p.shape[1],
                e_p.shape[1],
                e_p.shape[1],
                e_p.shape[0],
                e_p.shape[0],
                e_p.shape[0],
            )
        )
        - torch.inf
    )
    for a in [5]:
        for b in [8]:
            for x_j in tqdm(range(e_p.shape[1])):
                for p in tqdm(range(e_p.shape[1])):
                    for n in range(e_p.shape[1]):
                        for i_2 in [6]:
                            for i_1 in [5]:
                                for j in range(i_2 + 1):
                                    if (
                                        (n != a)
                                        & ((i_2 - 1 != i_1) or n == b)
                                        & ((j != i_2) or p == n)
                                        & ((j != i_2 + 1) or p == a)
                                        & ((j != i_1 + 1) or p == b)
                                        & ((j != i_1) or p == a)
                                        & (p != a or (j in [i_1, i_2 + 1]))
                                        & (i_1 < i_2)
                                        & (i_1 > 0)
                                        & (i_2 + 1 > j)
                                        & (a != b)
                                        & ((j != i_2) or x_j == a)
                                        & ((j != i_1) or x_j == b)
                                        & ((j != i_1 - 1) or x_j == a)
                                        & ((j != i_2 - 1) or x_j == n)
                                        & (x_j != a or (j in [i_1 - 1, i_2]))
                                    ):
                                        dic = {
                                            i_2: a,
                                            i_1: b,
                                            i_1 - 1: a,
                                            j: x_j,
                                            j - 1: p,
                                            i_2 - 1: n,
                                        }
                                        for i in range(8):
                                            dic.setdefault(
                                                i, torch.arange(8)[torch.arange(8) != a]
                                            )

                                        bound[x_j, p, n, i_2, i_1, j] = least_attention(
                                            a, b, i_1, i_2, j, dic
                                        )

    bound_soft = (
        bound.max(dim=0).values.max(dim=0).values.max(dim=0).values.softmax(dim=-1)
    )
    bound_2 = einops.einsum(
        bound_soft,
        "i_2 i_1 i_1 -> i_2 i_1",
    )
    mean_bound.append(bound_2[~torch.isnan(bound_2)].mean())
    print(mean_bound[-1])

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
