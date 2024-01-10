# from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_CLOCK_CONFIG
# from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_PIZZA_CONFIG
from gbmi.exp_modular_fine_tuning.train import ModularFineTuningTrainingWrapper
from gbmi.exp_modular_fine_tuning.train import modular_addition_config
from gbmi.model import train_or_load_model

import torch
import einops
from torch import tensor
from math import *
import tqdm

device = "cuda"

# %load_ext autoreload
# %autoreload 2

config_clock = modular_addition_config(0)
config_pizza = modular_addition_config(1)
runtime_clock, model_clock = train_or_load_model(config_clock)
runtime_pizza, model_pizza = train_or_load_model(config_pizza)
model_clock.to(device)
model_pizza.to(device)

frac_train = 0.3
seed = 999
num_epochs = 25000


def loss_fn(logits, labels):
    # print(logits.shape)
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    # print(log_probs)
    # print(log_probs.shape)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    # print(correct_log_probs.numpy())
    return -correct_log_probs.mean()


p = config_clock.experiment.p

a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
equals_vector = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)
dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
# print(dataset.shape)

optim_pizza = torch.optim.AdamW(
    model_pizza.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)
)
optim_clock = torch.optim.AdamW(
    model_clock.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)
)


def hook_function_attention(alpha):
    def hook_function(attnscore, hook):
        return alpha / attnscore.shape[-1] + (1 - alpha) * attnscore

    return hook_function


labels = (dataset[:, 0] - dataset[:, 1]) % p

indices = torch.randperm(p * p)
cutoff = int(p * p * frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]
train_data = dataset[train_indices]
train_labels = labels[train_indices]

test_data = dataset[test_indices]
test_labels = labels[test_indices]

train_losses_clock = []
test_losses_clock = []
train_losses_pizza = []
test_losses_pizza = []
model_checkpoints = []
checkpoint_epochs = []


for epoch in tqdm.tqdm(range(num_epochs)):
    train_logits_clock = model_clock.run_with_hooks(
        train_data,
        fwd_hooks=[("blocks.0.attn.hook_pattern", hook_function_attention(0))],
    )
    train_logits_pizza = model_pizza.run_with_hooks(
        train_data,
        fwd_hooks=[("blocks.0.attn.hook_pattern", hook_function_attention(1))],
    )
    train_loss_clock = loss_fn(train_logits_clock, train_labels)
    train_loss_pizza = loss_fn(train_logits_pizza, train_labels)
    train_loss_clock.backward()
    train_loss_pizza.backward()
    train_losses_clock.append(train_loss_clock.item())
    train_losses_pizza.append(train_loss_pizza.item())
    optim_clock.step()
    optim_pizza.step()
    optim_clock.zero_grad()
    optim_pizza.zero_grad()

    with torch.inference_mode():
        test_logits_clock = model_clock.run_with_hooks(
            test_data,
            fwd_hooks=[("blocks.0.attn.hook_pattern", hook_function_attention(0))],
        )
        test_logits_pizza = model_pizza.run_with_hooks(
            test_data,
            fwd_hooks=[("blocks.0.attn.hook_pattern", hook_function_attention(1))],
        )
        test_loss_clock = loss_fn(test_logits_clock, test_labels)
        test_loss_pizza = loss_fn(test_logits_pizza, test_labels)
        test_losses_clock.append(test_loss_clock.item())
        test_losses_pizza.append(test_loss_pizza.item())
    if ((epoch) % 10) == 0:
        checkpoint_epochs.append(epoch)
        print(
            f"CLOCK: Epoch {epoch} Train Loss {train_loss_clock.item()} Test Loss {test_loss_clock.item()}"
        )
        print(
            f"PIZZA: Epoch {epoch} Train Loss {train_loss_pizza.item()} Test Loss {test_loss_pizza.item()}"
        )
"""
def run_batch(x):
    labels = (x[:, 0] - x[:, 1]) % p

    y_preds_pizza = model_pizza.run_with_hooks(
        x, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_function_attention(1))]
    )
    y_preds_clock = model_clock.run_with_hooks(
        x, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_function_attention(0))]
    )

    loss_pizza = loss_fn(y_preds_pizza, labels)
    loss_clock = loss_fn(y_preds_clock, labels)
    loss_clock.backward()
    loss_pizza.backward()
    optim_pizza.step()
    optim_pizza.zero_grad()
    optim_clock.step()
    optim_clock.zero_grad()
    print(f"loss_clock {loss_clock} loss_pizza {loss_pizza}")


run_batch(dataset)
"""
