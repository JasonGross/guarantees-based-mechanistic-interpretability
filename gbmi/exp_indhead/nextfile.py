import copy
from inspect import signature
from math import *

import plotly.express as px
import torch
from torch import tensor, where

from gbmi.exp_indhead.train import ABCAB16_1H
from gbmi.model import train_or_load_model
from gbmi.utils.sequences import generate_all_sequences


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

runtime_model_1, model = train_or_load_model(ABCAB16_1H)
model.to(device)
