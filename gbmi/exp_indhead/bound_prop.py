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
        "i j k, k l, m l, n o m -> i j n o",
    )
    / model.blocks[0].attn.attn_scale
)
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

# %%
print(table)

# %%
import plotly.express as px

# %%
result, cache = model.run_with_cache(torch.tensor([[9, 9, 9, 9, 9, 25, 2, 25]]))

print(result[:, -1, :])
print(result[:, -1, :].argmax(dim=-1))
# px.line(result.detach().cpu().squeeze()).show('png')


# %%
