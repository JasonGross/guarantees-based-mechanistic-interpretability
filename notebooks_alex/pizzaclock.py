# %%
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import transformer_lens
import transformer_lens.utils as utils
from matplotlib import rc

from gbmi.exp_group_finetuning.groups import (
    CyclicGroup,
    DihedralGroup,
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

rc("animation", html="jshtml")
fig, ax = plt.subplots()
pio.renderers.default = "notebook_connected"
pio.templates["plotly"].layout.xaxis.title.font.size = 20
pio.templates["plotly"].layout.yaxis.title.font.size = 20
pio.templates["plotly"].layout.title.font.size = 30
from math import sqrt

import einops
import torch
import tqdm
from torch import tensor

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
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


freeze_model = False
config = GL2_P_CLOCK_CONFIG
p = config.experiment.group_index
q = p
print(q, "q")
frac_train = 0.4
seed = 999
num_epochs = 1000
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


if freeze_model:
    for param in model.parameters():
        param.requires_grad = False
    embeddingmatrix = model.embed.W_E.clone()

    model.embed = torch.nn.Identity()


def hook_fn(attnpattern, hook):
    return (
        torch.ones((3, 3)).to(device)
        * (torch.tensor(config.experiment.attention_rate, dtype=torch.float).to(device))
        + (
            1
            - torch.tensor(config.experiment.attention_rate, dtype=torch.float).to(
                device
            )
        )
        * attnpattern
    )


class DifferentModClock(torch.nn.Module):
    def __init__(self):
        super(DifferentModClock, self).__init__()
        self.W_e = torch.nn.Parameter(
            -0.5 / sqrt(p + 1) + torch.rand((q + 1, p + 1)) / sqrt(p + 1)
        )
        self.W_u = torch.nn.Parameter(-0.5 / sqrt(p) + torch.rand((q, p)) / sqrt(p))

    def forward(self, x):
        z = torch.nn.functional.one_hot(x).to(torch.float).to(device)

        y = z @ self.W_e @ embeddingmatrix
        if len(y.shape) != 3:
            y = y.unsqueeze(0)
        return self.W_u @ (
            einops.rearrange(
                model.run_with_hooks(
                    y, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
                ),
                "b d p -> b p d",
            )
        )


def loss_fn(logits, labels):
    print(logits.shape)
    if len(logits.shape) == 3:
        if freeze_model:
            logits = logits[:, :, -1]
        else:
            logits = logits[:, -1, :]
    logits = logits.to(torch.float64)


def loss_fn(logits, labels):
    print(logits.shape)
    if len(logits.shape) == 3:
        if freeze_model:
            logits = logits[:, :, -1]
        else:
            logits = logits[:, -1, :]
    logits = logits.to(torch.float64)


def loss_fn(logits, labels, softmax=True):
    if softmax:
        if len(logits.shape) == 3:
            if freeze_model:
                logits = logits[:, :, -1].squeeze(-1)
            else:
                logits = logits[:, -1, :]

        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]

        return -correct_log_probs.mean()
    else:
        if len(logits.shape) == 3:
            if freeze_model:
                logits = logits[:, :, -1].squeeze(-1)
            else:
                logits = logits[:, -1, :]
        logits = logits.to(torch.float64)
        return torch.linalg.vector_norm(logits - labels) / len(logits)


if freeze_model:
    Clock = DifferentModClock()
    Clock.to(device)
full_model = Clock if freeze_model else model

a_vector = einops.repeat(torch.arange(q), "i -> (i j)", j=q)
b_vector = einops.repeat(torch.arange(q), "j -> (i j)", i=q)
equals_vector = einops.repeat(torch.tensor(q), " -> (i j)", i=q, j=q)
dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
print("here")
labels = PermutedCyclicGroup(q).op(dataset[:, 0].T, dataset[:, 1].T)
print(labels)
optimizer = torch.optim.AdamW(
    full_model.parameters(), lr=1e-3, weight_decay=1, betas=(0.9, 0.98)
)
torch.manual_seed(seed)
indices = torch.randperm(q * q)
cutoff = int(q * q * frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]
train_data = dataset[train_indices]
train_labels = labels[train_indices]
test_data = dataset[test_indices]
test_labels = labels[test_indices]
bases = []

full_model.blocks[0].attn.W_O.requires_grad = False

full_model.blocks[0].attn.W_V.requires_grad = False

for epoch in tqdm.tqdm(range(num_epochs)):
    if freeze_model:
        train_logits = full_model(train_data)
    else:
        train_logits = full_model.run_with_hooks(
            train_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
        )
    if freeze_model:
        train_logits = full_model(train_data)
    else:
        train_logits = full_model.run_with_hooks(
            train_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
        )
    train_logits = full_model.run_with_hooks(
        train_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
    )
    train_loss = loss_fn(train_logits, train_labels)
    train_loss.backward()

    #  print(
    #      torch.mean(torch.abs(model.blocks[0].mlp.W_in.grad))
    #    / torch.mean(torch.abs(model.blocks[0].mlp.W_in))
    # )
    #   print(
    #       torch.mean(torch.abs(full_model.unembed.W_U.grad))
    #       / torch.mean(torch.abs(full_model.unembed.W_U))
    #    )
    optimizer.step()
    optimizer.zero_grad()
    with torch.inference_mode():
        if freeze_model:
            test_logits = full_model(test_data)
        else:
            test_logits = full_model.run_with_hooks(
                test_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
            )
        if freeze_model:
            test_logits = full_model(test_data)
        else:
            test_logits = full_model.run_with_hooks(
                test_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
            )
        test_logits = full_model.run_with_hooks(
            test_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
        )

        test_logits = full_model.run_with_hooks(
            test_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
        )
        test_loss = loss_fn(test_logits, test_labels)

    if ((epoch + 1) % 1) == 0:
        print(
            f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}"
        )


# %%
