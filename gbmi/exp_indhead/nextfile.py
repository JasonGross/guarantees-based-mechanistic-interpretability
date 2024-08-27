from gbmi.exp_indhead.train import ABCAB16_1H
from torch import where
from gbmi.model import train_or_load_model
import torch
from torch import tensor
from math import *
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
import copy
from inspect import signature

import plotly.express as px


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

runtime_model_1, model = train_or_load_model(ABCAB16_1H)
model.to(device)