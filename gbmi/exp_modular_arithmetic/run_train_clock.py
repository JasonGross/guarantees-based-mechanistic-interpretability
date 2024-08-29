# %%
from gbmi.exp_modular_arithmetic.train import train_or_load_model, CLOCK_CONFIG

runtime, model = train_or_load_model(CLOCK_CONFIG, force="load")
# %%
import plotly.express as px
import torch
import einops
from math import *

n_ctx = 2


def compute_chernoff_bound(x, max_val):
    # if torch.max(x) <= 0:
    #    return torch.tensor(1.0)
    x = x.detach()
    max_val = max_val.detach()
    last_percentage = max_val / ((n_ctx) * torch.max(x))

    # Calculates whether it can get the max_value with n_ctx tokens

    if last_percentage >= 1:
        return (torch.tensor(0.0), torch.zeros(x.shape) / (len(x)))
    elif last_percentage <= 0:
        return (torch.tensor(1.0), torch.ones(x.shape) / (len(x)))
    else:
        lambda_ = torch.log(
            (len(x) - 1) * (last_percentage) / (1 - last_percentage)
        ) / (abs(torch.max(x)))
    if torch.isinf(lambda_):
        return (torch.tensor(1.0), torch.ones(x.shape) / (len(x)))
    lambda_ = torch.tensor(0.0)
    lambda_.requires_grad = True

    optimizer = torch.optim.AdamW([lambda_], lr=5e-3)

    for i in range(300):

        optimizer.zero_grad()
        chernoff_bound = (torch.exp(x * lambda_).mean()) ** (n_ctx) * e ** (
            -lambda_ * max_val
        )
        chernoff_bound.backward()
        optimizer.step()
        print(lambda_)

    return (
        torch.min(torch.tensor(1.0), chernoff_bound),
        torch.exp(x * lambda_) / (torch.exp(x * lambda_).sum()),
    )


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


model.to("cuda")
attn_scale_0 = model.blocks[0].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E
W_U = model.W_U
W_U = W_U - W_U.mean(dim=1, keepdim=True)
W_K = model.W_K[0, :]
W_V_0 = model.W_V[0, 0]
W_O_0 = model.W_O[0, 0]
W_Q = model.W_Q[0, :]

QK = einops.einsum(W_Q, einops.rearrange(W_K, "b i j -> b j i"), "b i j,b j l -> b i l")

EQKP = (W_E @ QK @ W_pos.T) / (attn_scale_0)
PQKP = (W_pos @ QK @ W_pos.T) / (attn_scale_0)
PQKE = (W_pos @ QK @ W_E.T) / (attn_scale_0)
EQKE = (W_E @ QK @ W_E.T) / (attn_scale_0)
PVOU = W_pos @ W_V_0 @ W_O_0 @ W_U
EVOU = W_E @ W_V_0 @ W_O_0 @ W_U
o = W_O_0
v = W_V_0
mlp = model.blocks[0].mlp.W_in
p_mlp = W_pos[-1] @ mlp
e_mlp = W_E[-1] @ mlp
evou_mlp = (W_E @ W_V_0 @ W_O_0 @ mlp) / 3
evou_row = (W_E[-1] @ W_V_0 @ W_O_0 @ mlp) / 6
pvou_mlp = torch.mean(W_pos, dim=0) @ W_V_0 @ W_O_0 @ mlp
init_row = model.blocks[0].mlp.b_in + pvou_mlp + e_mlp + p_mlp + evou_row
# %%
