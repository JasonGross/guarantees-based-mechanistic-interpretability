from gbmi.exp_group_finetuning.train import MODULAR_ADDITION_113_CLOCK_CONFIG
import transformer_lens
import transformer_lens.utils as utils
from gbmi.exp_group_finetuning.groups import (
    Group,
    GroupDict,
    CyclicGroup,
    DihedralGroup,
    PermutedCyclicGroup,
)
from gbmi.model import train_or_load_model
import torch
from math import sqrt
from torch import tensor
import einops
import tqdm

device = "cuda"
config = MODULAR_ADDITION_113_CLOCK_CONFIG
p = config.experiment.group_index
q = p

rundata, model = train_or_load_model(config)
model.to(device)
modifiedpoint = [1, 1, 5]
a_vector = einops.repeat(torch.arange(q), "i -> (i j)", j=q)
b_vector = einops.repeat(torch.arange(q), "j -> (i j)", i=q)
equals_vector = einops.repeat(torch.tensor(q), " -> (i j)", i=q, j=q)
dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
labels = PermutedCyclicGroup(q).op(dataset[:, 0].T, dataset[:, 1].T)
print(a_vector)
