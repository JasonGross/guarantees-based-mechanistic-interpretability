from math import sqrt

import einops
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch
import tqdm
import transformer_lens
import transformer_lens.utils as utils
from matplotlib import rc
from torch import tensor

from gbmi.exp_group_finetuning.groups import (
    CyclicGroup,
    DihedralGroup,
    GLN_p,
    Group,
    GroupDict,
    PermutedCyclicGroup,
)
from gbmi.exp_group_finetuning.train import (
    DIHEDRAL_100_CLOCK_CONFIG,
    DIHEDRAL_100_PIZZA_CONFIG,
    GL2_P_CLOCK_CONFIG,
    MODULAR_ADDITION_113_CLOCK_CONFIG,
    MODULAR_ADDITION_113_PIZZA_CONFIG,
)
from gbmi.model import train_or_load_model


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show(renderer)


config = DIHEDRAL_100_CLOCK_CONFIG
p = config.experiment.group_index
q = 2 * p
p = q
fourier_basis = []
fourier_basis_names = []
fourier_basis.append(torch.ones(p))
fourier_basis_names.append("Constant")
for freq in range(1, p // 2 + 1):
    fourier_basis.append(torch.sin(torch.arange(p) * 2 * torch.pi * freq / p))
    fourier_basis_names.append(f"Sin {freq}")
    fourier_basis.append(torch.cos(torch.arange(p) * 2 * torch.pi * freq / p))
    fourier_basis_names.append(f"Cos {freq}")
fourier_basis = torch.stack(fourier_basis, dim=0).cuda()
fourier_basis = fourier_basis / fourier_basis.norm(dim=-1, keepdim=True)
device = "cuda"


rundata, model = train_or_load_model(config)
model.to(device)
group = GLN_p(3)

# %%

U, S, Vh = torch.svd(model.embed.W_E)
print(Vh.shape)
scatter(Vh[0], Vh[1])
line(
    U[:, :128].T,
    title="Principal Components of the embedding",
    xaxis="Input Vocabulary",
)
line(S, title="Singular Values")
imshow(U, title="Principal Components on the Input")
