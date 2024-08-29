# %%


from math import *

import einops
import pandas as pd
import plotly.express as px
import torch
from torch import tensor
from tqdm.auto import tqdm

from gbmi import utils
from gbmi.exp_f_g.functions import add_sub, max_min, min_max
from gbmi.exp_f_g.train import f_g_config, f_g_TrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils.sequences import generate_all_sequences

device = "cuda"


# %%


"""

def testing(a, b):

    c = torch.tensor([[a, b, 0, 0, 53]])
    z = torch.tensor([[a, b, 0, 0, 53]])
    for i in range(53):
        c[0][2] = i
        for j in range(53):
            c[0][3] = j
            if (i, j) != (0, 0):
                z = torch.cat((z, c), dim=0)
    val = torch.argmax(model_max_min_1(z)[:, -1, :].squeeze(), dim=1).to(device)
    return torch.stack(
        ((val - max(a, b)), (val - torch.min(z[:, 2], z[:, 3]).to(device)))
    )


a = 0
b = 0
for i in range(53):
    for j in range(53):
        s = testing(i, j)
        c = torch.sum(abs(s[0]) >= abs(s[1]))
        d = torch.sum(torch.min(abs(s[0]), abs(s[1])) == 0)
        a += c
        b += d
        print((53 * i + j + 1) * 2809)
        print(a)
        print(b)
        print(c)
        print(d)

"""


def elemental_loss(mod, i, fun):
    logits = mod(i)[:, -1, :].to(torch.float64)
    log_probs = utils.log_softmax(logits, dim=-1)
    correct_log_probs = log_probs.gather(-1, fun(i).unsqueeze(-1))[:, 0]

    return -correct_log_probs


@torch.no_grad()
def loss_scatter(mod, cond, res, fun, avg, lab, fun_lab):

    mod.to(device)

    def cond_index(data):
        out = cond(data)
        return out[0] * 53 + out[1]

    pairs = generate_all_sequences(53, 4).to(device)
    data = torch.cat(
        [
            pairs,
            53 * (torch.ones((len(pairs), 1), device=pairs.device, dtype=torch.long)),
        ],
        dim=1,
    )

    loss_values = []
    if avg:
        out = []

    for sub_data in tqdm(torch.split(data, 53**2)):
        losses = elemental_loss(mod, sub_data, fun)
        loss_values.append(losses)
        if avg:
            value = mod(sub_data)[:, -1, :].argmax(dim=-1).to(torch.float64)
            out.append(value)

    loss_values = torch.cat(loss_values)
    if avg:
        out = torch.cat(out)

    losses = torch.zeros((53**2,), device=loss_values.device, dtype=loss_values.dtype)
    if avg:
        values = torch.zeros(
            (53**2,), device=loss_values.device, dtype=loss_values.dtype
        )
    cond_indices = cond_index(data)
    if res is not None:
        res_mask = res(data)
        cond_indices = cond_indices[res_mask]
        loss_values = loss_values[res_mask]
        if avg:
            out = out[res_mask]
    losses.scatter_reduce_(
        dim=0,
        index=cond_indices,
        src=loss_values,
        reduce="mean",
        include_self=False,
    )
    if avg:
        values.scatter_reduce_(
            dim=0,
            index=cond_indices,
            src=out,
            reduce="mean",
            include_self=False,
        )

    x_coords = torch.arange(53).repeat(53, 1).transpose(0, 1).flatten()
    y_coords = torch.arange(53).repeat(53, 1).flatten()

    fig = px.scatter(
        x=x_coords.numpy(),
        y=y_coords.numpy(),
        color=losses.cpu().numpy(),
        color_continuous_scale="Picnic",
        title="loss" + str(lab) + fun_lab,
        labels={"color": "average loss"},
        width=800,
        height=600,
        color_continuous_midpoint=0,
    )

    fig.update_layout(
        xaxis_title="cond_1",
        yaxis_title="cond_2",
    )

    fig.show()

    if avg:

        fig_2 = px.scatter(
            x=x_coords.numpy(),
            y=y_coords.numpy(),
            color=values.cpu().numpy(),
            color_continuous_scale="Picnic",
            title="average output" + str(lab),
            labels={"color": "average output"},
            width=800,
            height=600,
            color_continuous_midpoint=0,
        )

        fig_2.update_layout(
            xaxis_title="cond_1",
            yaxis_title="cond_2",
        )

        fig_2.show()


def split_min_max(i):
    return ((i[..., :2].min(dim=-1)).values, (i[..., 2:4].max(dim=-1).values))


def res_min_max(i):
    return i[..., 2:4].min(dim=-1).values == i[..., :2].max(dim=-1).values


def max_1(i):
    return i[..., :2].max(dim=-1).values


def min_1(i):
    return i[..., :2].min(dim=-1).values


def min_2(i):
    return i[..., 2:4].min(dim=-1).values


def max_2(i):
    return i[..., 2:4].max(dim=-1).values


def min_1_2(i):
    return i[..., :4].min(dim=-1).values


def max_1_2(i):
    return i[..., :4].max(dim=-1).values


def split_add_sub(i):
    return ((i[..., 0] + i[..., 1]) % 53, (i[..., 2] - i[..., 3]) % 53)


def add_1(i):
    return (i[..., 0] + i[..., 1]) % 53


def sub_2(i):
    return (i[..., 2] - i[..., 3]) % 53


def const_23(i):
    return 23 * torch.ones(len(i)).long().cuda()


for i in range(1200, 2000, 100):
    add_sub_1_head_CONFIG = f_g_config(fun=add_sub(53, 2), n_head=1, elements=2, seed=i)
    runtime_add_sub_1, model_add_sub_1 = train_or_load_model(
        add_sub_1_head_CONFIG, force="train"
    )

    loss_scatter(model_add_sub_1, split_add_sub, None, add_1, True, i, "add_1")
    loss_scatter(model_add_sub_1, split_add_sub, None, sub_2, False, i, "sub_2")

"""
for i in range(700,2000,100):

    min_max_1_head_CONFIG = f_g_config(fun=min_max(53, 2), n_head=1, elements=2, seed=i)
    runtime_min_max_1, model_min_max_1 = train_or_load_model(
        min_max_1_head_CONFIG, force="train"
    )
    loss_scatter(model_min_max_1, split_min_max, None, max_2, True,i,"max_2")
    loss_scatter(model_min_max_1, split_min_max, None, min_1, False,i,"min_1")

"""

"""

def average(mod,max,min):
    set=torch.tensor([])
    for i in range(max+1):
        for j in range(min-1,53):
            a=torch.tensor([[max,i,min,j]])
            b=torch.tensor([[i,max,min,j]])
            c=torch.tensor([[max,i,j,min]])
            d=torch.tensor([[i,max,j,min]])
            if len(set)==0:
                set=torch.cat((a,b,c,d),dim=0)
            else:
                set=torch.cat((set,a,b,c,d),dim=0)
"""
"""
for i in [0,200,500,700,800,1100,1300,1600,1900]:
    max_min_1_head_CONFIG = f_g_config(fun=max_min(53, 2), n_head=1, elements=2, seed=i)
    runtime_max_min_1, model_max_min_1 = train_or_load_model(max_min_1_head_CONFIG)
    loss_scatter(model_max_min_1, min_max, None, min_2, True,i)
    loss_scatter(model_max_min_1, min_max, None, max_1, False,i)
"""

# loss_scatter(model_min_max_1, split_min_max, None, min_2, True, 300)


"""

with torch.no_grad():
    Eeq = model_max_min_1.W_E[-1] + model_max_min_1.W_pos[-1]
    for layer in range(model_max_min_1.W_Q.shape[0]):
        for head in range(model_max_min_1.W_Q.shape[1]):
            EQKE = (
                Eeq
                @ model_max_min_1.W_Q[layer, head]
                @ model_max_min_1.W_K[layer, head].T
                @ (model_max_min_1.W_E[:-1] - model_max_min_1.W_E[-1]).T
            )
            EQKP = (
                Eeq
                @ model_max_min_1.W_Q[layer, head]
                @ model_max_min_1.W_K[layer, head].T
                @ (model_max_min_1.W_pos[:-1] - model_max_min_1.W_pos[-1]).T
            )
            px.line(EQKE.cpu(), title=f"l{layer}h{head} EQKE").show()
            px.scatter(EQKP.cpu(), title=f"l{layer}h{head} EQKP").show()
"""
