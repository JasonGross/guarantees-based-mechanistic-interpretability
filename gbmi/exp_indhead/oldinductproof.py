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
from tqdm.auto import tqdm

from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1H
from gbmi.model import train_or_load_model
from gbmi.utils import ein
from gbmi.utils.sequences import generate_all_sequences


def armin(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return ein.apply(f, collect=lambda xs, d: xs.max(d).indices, sizes=sizes)


runtime_model_1, model = train_or_load_model(ABCAB8_1H, force="load")
model.to("cuda")
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

o = model.W_O[0, 0]
v = model.W_V[0, 0]
q_1 = model.W_Q[1, 0]
k_1 = model.W_K[1, 0]
attn_scale_1 = sqrt(128)
"""
everything_1_1 = ein.array(
    lambda a, c, i_2, j, x: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ (k_1.T)
    @ (e_p[j, x].T)
    * (1 / sqrt(32))
)
"""
# %%
everything_1_1 = ein.array(
    lambda a, c, i_2, j, x: torch.where(
        j < i_2,
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o)
        @ q_1
        @ (k_1.T)
        @ (e_p[j, x].T)
        * (1 / attn_scale_1),
        -torch.inf,
    ),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
everything_1_2 = ein.array(
    lambda a, c, i_2, j, y: torch.where(
        torch.logical_and(j >= 1, j < i_2),
        (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ o)
        @ q_1
        @ k_1.T
        @ ((e_p[j - 1, y]) @ v @ o).T
        * (1 / attn_scale_1),
        -torch.inf,
    ),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
"""
everything_1_2 = ein.array(
    lambda a, c, i_2, j, y: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ k_1.T
    @ ((e_p[j - 1, y]) @ v @ z).T
    * (1 / sqrt(32)),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
"""

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
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
"""
everything_1_b = ein.array(
    lambda a, c, i_2, i_1, b: (e_p[i_2, a] + (e_p[i_2 - 1, c]) @ v @ z)
    @ q_1
    @ k_1.T
    @ ((e_p[i_1, b] + (e_p[i_1 - 1, a]) @ v @ z).T)
    * (1 / sqrt(32)),
    sizes=[e_p.shape[1], e_p.shape[1], e_p.shape[0], e_p.shape[0], e_p.shape[1]],
)
"""

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
print(everything_1_2)
# %%

attn = torch.zeros((d_voc, d_voc, d_voc, n_ctx, n_ctx))

for a in tqdm(range(d_voc)):
    for b in range(d_voc):
        for c in range(d_voc):
            for i_2 in range(2, n_ctx):
                for i_1 in range(0, i_2 - 2):
                    vals = []

                    for j in range(i_2):

                        if j != i_1 + 1:
                            if j != 0:

                                # assert everything_1_1[a, c, i_2, j].isfinite().all(), (
                                #     everything_1_1[a, c, i_2, j],
                                #     a,
                                #     c,
                                #     i_2,
                                #     j,
                                # )
                                # assert everything_1_2[a, c, i_2, j].isfinite().all(), (
                                #     everything_1_2[a, c, i_2, j],
                                #     a,
                                #     c,
                                #     i_2,
                                #     j,
                                # )
                                """
                                attn[a, b, c, i_2, i_1] += torch.exp(
                                    everything_1_1[a, c, i_2, j].max(dim=-1).values
                                    + everything_1_1[a, c, i_2, j].max(dim=-1).values
                                )
                                """

                                """
                                x =

                                y = np.argwhere(
                                    (
                                        everything_1_2[a, c, i_2, j].max()
                                        == everything_1_2[a, c, i_2, j]
                                    ).numpy()
                                )[0]
                                """
                                everything_1_1[a, c, i_2, j, a] = -torch.inf
                                everything_1_2[a, c, i_2, j, a] = -torch.inf
                                vals.append(
                                    everything_1_1[a, c, i_2, j].max(dim=-1).values
                                    + everything_1_2[a, c, i_2, j].max(dim=-1).values
                                )

                            if j == 0:
                                # assert everything_1_1[a, c, i_2, j].isfinite().all(), (
                                #     everything_1_1[a, c, i_2, j],
                                #     a,
                                #     c,
                                #     i_2,
                                #     j,
                                # )
                                everything_1_1[a, c, i_2, j, a] = -torch.inf
                                # x = np.argwhere(
                                #     (
                                #         everything_1_1[a, c, i_2, j].max()
                                #         == everything_1_1[a, c, i_2, j]
                                #     ).numpy()
                                # )[0]
                                vals.append(
                                    everything_1_1[a, c, i_2, j].max(dim=-1).values
                                )

                                # attn[a, b, c, i_2, i_1] += torch.exp(
                                #     everything_1_1[a, c, i_2, j].max(dim=-1).values
                                # )
                    assert everything_1_b[a, c, i_2, i_1 + 1, b].isfinite(), (
                        everything_1_b[a, c, i_2, i_1 + 1, b],
                        a,
                        c,
                        i_2,
                        i_1 + 1,
                        b,
                    )
                    vals.append(everything_1_b[a, c, i_2, i_1 + 1, b])
                    vals = torch.tensor(vals)
                    attn[a, b, c, i_2, i_1] = vals.softmax(dim=0)[-1]
print(attn[10, 15, 20, 6, 2])


# %%
