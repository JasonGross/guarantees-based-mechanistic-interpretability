# %%

from gbmi.exp_f_g.train import f_g_TrainingWrapper
from gbmi.exp_f_g.functions import add_sub, max_min
from gbmi import utils

from gbmi.exp_f_g.train import f_g_config

from gbmi.model import train_or_load_model

import torch
import einops
from torch import tensor
from math import *
import tqdm
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
import pandas as pd

device = "cuda"

# functions=[("max","min"),("is_sorted","exactly_2_of_3_even"),("add","minus")]

# add_sub_1_head_CONFIG = f_g_config(fun=add_sub(53, 2), n_head=1, elements=2)
# add_sub_2_head_CONFIG = f_g_config(fun=add_sub(53, 2), n_head=2, elements=2)
max_min_1_head_CONFIG = f_g_config(fun=max_min(53, 2), n_head=1, elements=2)
# add_sub_4_head_CONFIG = f_g_config(fun=add_sub(23, 2), n_head=4, elements=2)


# max_min_2_head_CONFIG = f_g_config(fun=max_min(53, 2), n_head=1, elements=2)

# max_min_4_head_CONFIG = f_g_config(fun=max_min(53, 2), n_head=4, elements=2)

# runtime_add_sub_1, model_add_sub_1 = train_or_load_model(add_sub_1_head_CONFIG)
# runtime_add_sub_2, model_add_sub_2 = train_or_load_model(add_sub_2_head_CONFIG)
runtime_max_min_1, model_max_min_1 = train_or_load_model(max_min_1_head_CONFIG)
# runtime_add_sub_4, model_add_sub_4 = train_or_load_model(add_sub_4_head_CONFIG)

# runtime_max_min_2, model_max_min_2 = train_or_load_model(max_min_2_head_CONFIG)


# runtime_max_min_4, model_max_min_4 = train_or_load_model(max_min_4_head_CONFIG)

"""
with torch.no_grad():
    Eeq = model_max_min_4.W_E[-1] + model_max_min_4.W_pos[-1]
    for layer in range(model_max_min_4.W_Q.shape[0]):
        for head in range(model_max_min_4.W_Q.shape[1]):
            EQKE = (
                Eeq
                @ model_max_min_4.W_Q[layer, head]
                @ model_max_min_4.W_K[layer, head].T
                @ (model_max_min_4.W_E[:-1] - model_max_min_4.W_E[-1]).T
            )
            EQKP = (
                Eeq
                @ model_max_min_4.W_Q[layer, head]
                @ model_max_min_4.W_K[layer, head].T
                @ (model_max_min_4.W_pos[:-1] - model_max_min_4.W_pos[-1]).T
            )
            px.line(EQKE.cpu(), title=f"l{layer}h{head} EQKE").show()
            px.scatter(EQKP.cpu(), title=f"l{layer}h{head} EQKP").show()
"""

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


def elemental_loss(mod, i):
    logits = mod(i)[:, -1, :].to(torch.float64)
    log_probs = utils.log_softmax(logits, dim=-1).to(device)
    correct_log_probs = log_probs.gather(
        -1, torch.min(i[:2]).unsqueeze(-1).unsqueeze(-1)
    )[:, 0].to(device)

    return -correct_log_probs.mean()


def loss_scatter(mod, cond, max_is_min):
    pairs = generate_all_sequences(53, 4).to(device)
    data = (
        torch.cat([pairs, 53 * (torch.ones((len(pairs), 1)).to(device))], dim=1)
        .long()
        .to(device)
    )

    loss_values = {}

    for i in data:
        if not (max_is_min and not (torch.max(i[2:4]) == torch.min(i[:2]))):
            if cond(i) in loss_values:
                loss_values[cond(i)] = torch.cat(
                    [loss_values[cond(i)], elemental_loss(mod, i)]
                )
            else:
                loss_values[cond(i)] = torch.tensor([elemental_loss(mod, i)])
    print(loss_values[(torch.tensor(5), torch.tensor(50))])
    x_values = [key[0].item() for key in loss_values.keys()]
    y_values = [key[1].item() for key in loss_values.keys()]
    color_values = [value.mean().item() for value in loss_values.values()]

    # Create a DataFrame
    df = pd.DataFrame({"X": x_values, "Y": y_values, "Color": color_values})

    # Plot scatter diagram using Plotly Express
    fig = px.scatter(df, x="X", y="Y", color="Color", title="Scatter Diagram")
    fig.show()


def min_max(i):
    return (torch.min(i[:2]), torch.max(i[2:4]))


loss_scatter(model_max_min_1, min_max, True)

# print(torch.argmax(model_max_min_4(torch.tensor([0, 0, 17, 20, 53]))[:, -1, :]))

# model_add_sub_1.to(device)
# model_add_sub_2.to(device)
# model_max_min_4.to(device)
# model_max_min_2.to(device)
