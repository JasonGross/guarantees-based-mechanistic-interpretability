from gbmi.exp_f_g.train import f_g_TrainingWrapper
from gbmi.exp_f_g.train import f_g_config
from gbmi.exp_f_g.train import (
    add_sub_1_head_CONFIG,
    add_sub_2_head_CONFIG,
    max_min_1_head_CONFIG,
    max_min_2_head_CONFIG,
)
from gbmi.model import train_or_load_model

import torch
import einops
from torch import tensor
from math import *
import tqdm

device = "cuda"

# functions=[("max","min"),("is_sorted","exactly_2_of_3_even"),("add","minus")]

runtime_add_sub_1, model_add_sub_1 = train_or_load_model(add_sub_1_head_CONFIG)
runtime_add_sub_2, model_add_sub_2 = train_or_load_model(add_sub_2_head_CONFIG)
runtime_max_min_1, model_max_min_1 = train_or_load_model(max_min_1_head_CONFIG)
runtime_max_min_2, model_max_min_2 = train_or_load_model(max_min_2_head_CONFIG)

model_add_sub_1.to(device)
model_add_sub_2.to(device)
model_max_min_1.to(device)
model_max_min_2.to(device)
