# from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_CLOCK_CONFIG
# from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_PIZZA_CONFIG
from gbmi.exp_modular_fine_tuning.train import ModularFineTuningTrainingWrapper
from gbmi.exp_modular_fine_tuning.train import modular_addition_config
from gbmi.model import train_or_load_model
import torch
import einops
from torch import tensor
from math import *

device = "cuda"

# %load_ext autoreload
# %autoreload 2

config_clock = modular_addition_config(0)
config_pizza = modular_addition_config(1)
runtime_clock, model_clock = train_or_load_model(config_clock)
runtime_pizza, model_pizza = train_or_load_model(config_pizza)
model_clock.to(device)
model_pizza.to(device)


def loss_fn(logits, labels):
    print(logits.shape)
    if len(logits.shape) == 3:
        logits = logits[:, :, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


p = config_clock.experiment.p

a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
equals_vector = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)
dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
print(dataset.shape)

optim_pizza = torch.optim.AdamW(
    model_pizza.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98)
)
optim_clock = torch.optim.AdamW(
    model_clock.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98)
)


def run_batch(x):
    labels = (x[:, 0] - x[:, 1]) % p

    y_preds_pizza = model_pizza.run_with_hooks(x)
    y_preds_clock = model_clock.run_with_hooks(x)

    loss_pizza = loss_fn(y_preds_pizza, labels)
    loss_clock = loss_fn(y_preds_clock, labels)
    loss_clock.backward()
    loss_pizza.backward()
    optim_pizza.step()
    optim_pizza.zero_grad()
    optim_clock.step()
    optim_clock.zero_grad()
    return f"loss_clock {loss_clock} loss_pizza {loss_pizza}"


run_batch(dataset)
