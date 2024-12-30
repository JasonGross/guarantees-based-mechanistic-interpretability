# %%
import math
from math import *

import numpy as np
import plotly.express as px
import scipy.misc
import torch
from scipy.stats import binom

from gbmi.exp_max_of_n.train import MAX_OF_4_CONFIG, MAX_OF_10_CONFIG, MAX_OF_20_CONFIG
from gbmi.model import train_or_load_model

# rundata, model = train_or_load_model(MAX_OF_4_CONFIG(123))

rundata, model = train_or_load_model(MAX_OF_4_CONFIG(123))

torch.set_default_device("cuda")
length = model.cfg.n_ctx
model.to("cuda")
model.requires_grad = True
attn_scale_0 = model.blocks[0].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E
W_U = model.W_U
W_U = W_U - W_U.mean(dim=1, keepdim=True)
W_K_0 = model.W_K[0, 0]
W_V_0 = model.W_V[0, 0]
W_O_0 = model.W_O[0, 0]
W_Q_0 = model.W_Q[0, 0]
W_Q_0 = model.W_Q[0, 0]
EQKP = (W_E @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKP = (W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKE = (W_pos @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)
EQKE = (W_E @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)
PVOU = W_pos @ W_V_0 @ W_O_0 @ W_U
EVOU = W_E @ W_V_0 @ W_O_0 @ W_U
o = W_O_0
v = W_V_0
at = W_E @ W_Q_0 @ W_K_0.T @ W_E.T + W_pos[length - 1] @ W_Q_0 @ W_K_0 @ W_E.T


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


#
#
#
#   1, e^(x_1-x_0),...,.
#
#
#
#
#

n_ctx = length - 2


def compute_chernoff_bound(x, max_val):
    if torch.max(abs(x)) < 1e-12:
        return torch.tensor(0.0), torch.tensor(0.0)

    if max_val > 0 and torch.max(x) < 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    last_percentage = max_val / ((n_ctx) * torch.max(x))

    # Calculates whether it can get the max_value with n_ctx tokens

    if last_percentage >= 1:

        return torch.tensor(0.0), torch.tensor(0.0)
    elif last_percentage <= 0:
        return torch.tensor(1.0), torch.tensor(0.0)
    else:
        lambda_ = torch.log(
            (len(x) - 1) * (last_percentage) / (1 - last_percentage)
        ) / (torch.max(x))

    if torch.isinf(lambda_):
        return torch.tensor(1.0), -torch.inf
    chernoff_bound = (torch.exp(x * lambda_).mean()) ** (n_ctx) * e ** (
        -lambda_ * max_val
    )
    return chernoff_bound, lambda_


epochs = 300
torch.set_default_device("cuda")
model = model.to("cuda")
# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# %%


pos_pattern = torch.softmax(PQKP[-1] + EQKP, dim=1)
# %%


# %%


l = 0
bounds = torch.zeros(64, 64)
for M in range(64):
    for last_tok in range(M + 1):
        last_tok_bound = torch.tensor(0.0)
        pattern = pos_pattern[last_tok]
        f = torch.exp(PQKE[-1] + EQKE[last_tok])
        bounding = torch.tensor(0.0)
        for k in range(64):
            C = (W_pos[-1] + W_E[last_tok]) @ (W_U[:, k] - W_U[:, M])
            pvo = torch.min(PVOU[:, k] - PVOU[:, M])
            evo = EVOU[:, k] - EVOU[:, M]
            constants = f * (evo + pvo + C)

            last_tok_val = constants[last_tok] * pattern[-1]
            M_tok = pattern * constants[M]

            if M > 1:
                bound = compute_chernoff_bound(
                    constants[:M] / (n_ctx), max_val=-M_tok[0] - last_tok_val
                )  # chance of violating condition
            else:
                bound = torch.tensor(1.0), torch.tensor(1.0)

            l += 1
            if l % 1000 == 0:
                print(bound)
            if not torch.isnan(bound[0]):
                bounding = bounding + torch.max(bound[0], torch.tensor(0.0))
        bounds[last_tok, M] = 1 - bounding

# %%
length = 4


def prob(row):
    if row > 0:
        return (row / 64) ** (length) - ((row - 1) / 64) ** (length)
    else:
        return (1 / 64) ** length


new_bounds = torch.zeros(64)
for M in range(64):
    new_bounds[M] = torch.nn.ReLU()(bounds)[: (M + 1), M].mean()

matrix = torch.tensor([prob(i) for i in range(1, 65)])
print(matrix @ new_bounds)


# %%


def compute_full_bound(attn):
    iterations = 10000
    sum_ = 0
    for l in range(iterations):
        values = torch.randint(low=0, high=64, size=(length,))
        sequence = torch.tensor(values)

        logits, cache = model.run_with_cache(sequence)
        attention = cache["attn", 0].squeeze()[-1][torch.max(sequence, dim=0).indices]
        if attention > attn:
            sum_ += 1
    return sum_ / iterations


# Cubic proof
# %%
bounds = torch.zeros(64)
mat = EQKE + PQKE[length - 1].unsqueeze(0)

sum_ = 0
max_eqkp_diff = torch.max(EQKP, dim=1).values - torch.min(EQKP, dim=1).values

for maximum in range(64):
    best_diff = torch.exp(torch.max(mat - mat[:, maximum].unsqueeze(1), dim=0).values)[
        : (maximum + 1)
    ]
    currbound = torch.tensor(0.0)
    for first_pos in range(length):
        sume_ = torch.tensor(0.0)

        for current_row in range(64):
            if current_row == maximum:
                continue
            difference_row = (EVOU[:, current_row] - EVOU[:, maximum])[: (maximum + 1)]
            combined_row = difference_row * best_diff
            positional_diff = (W_pos[length - 1] @ W_U)[maximum] - (
                W_pos[length - 1] @ W_U
            )[current_row]
            totnoise = (
                torch.max(
                    (W_E @ W_U)[: (maximum + 1), current_row]
                    - (W_E @ W_U)[: (maximum + 1), maximum],
                )
                + positional_diff
                + torch.max(PVOU[:, current_row] - PVOU[:, maximum])
            )
            if totnoise < 0:
                B = 1
            else:
                B = 3  # n_ctx - 1

            if maximum > 0:
                if (
                    len(combined_row) > 1
                    and torch.max(combined_row[:-1]) * (n_ctx)
                    < -combined_row[-1] - B * totnoise
                ):
                    chernoff_bound = 0.0
                elif (
                    len(combined_row) > 1
                    and torch.min(combined_row[:-1]) * (n_ctx)
                    > -combined_row[-1] - B * totnoise
                ):
                    chernoff_bound = 1.0
                else:
                    chernoff_bound, lambda_ = compute_chernoff_bound(
                        torch.exp(
                            torch.max(PQKP[length - 1] - PQKP[length - 1][first_pos])
                        )
                        * torch.exp(torch.max(max_eqkp_diff[: (maximum + 1)]))
                        * combined_row[:-1],
                        max_val=-combined_row[-1] - B * totnoise,
                    )
                    if length - 1 - first_pos > 0:
                        a_vals = torch.tensor(
                            [
                                chernoff_bound
                                * e ** (combined_row[-1] * i * lambda_)
                                * (1 / 64) ** (i)
                                * ((maximum) / 64) ** (length - 1 - first_pos - i)
                                * math.comb(length - 1 - first_pos, i)
                                for i in range(length - 1 - first_pos)
                            ]
                        ).sum()
                    else:
                        a_vals = chernoff_bound

                    chernoff_bound = min(a_vals, 1.0)

                if maximum > 2:
                    if (
                        torch.max(combined_row[:-2]) * (n_ctx)
                        < -combined_row[-1] - B * totnoise
                    ):
                        chernoff_bound = min(
                            chernoff_bound,
                            1 - torch.tensor(((maximum - 2) / (maximum)) ** (n_ctx)),
                        )
                    if (
                        torch.max(combined_row[:-3]) * (n_ctx)
                        < -combined_row[-1] - B * totnoise
                    ):
                        chernoff_bound = min(
                            chernoff_bound,
                            1
                            - torch.tensor(((maximum - 3) / (maximum - 1)) ** (n_ctx)),
                        )
            else:
                chernoff_bound = torch.tensor(1.0)
            sume_ += chernoff_bound

        currbound += (sume_) * (
            ((maximum) ** (first_pos) * (maximum + 1) ** (length - 1 - first_pos))
            / ((maximum + 1) ** (length) - (maximum) ** (length))
        )
    bounds[maximum] = currbound / (length)

# %%
for i in range(len(bounds)):
    if torch.isnan(bounds[i]):
        bounds[i] = 1.0


def prob(row):
    if row > 0:
        return (row / 64) ** (length) - ((row - 1) / 64) ** (length)
    else:
        return (1 / 64) ** length


mean_accuracy = torch.tensor(bounds)
matrix = torch.tensor([prob(i) for i in range(64)])
print(matrix @ mean_accuracy)
# %%


def compute_full_bound(attn):
    iterations = 10000
    sum_ = 0
    for l in range(iterations):
        values = torch.randint(low=0, high=64, size=(length,))
        sequence = torch.tensor(values)

        logits, cache = model.run_with_cache(sequence)
        attention = cache["attn", 0].squeeze()[-1][torch.max(sequence, dim=0).indices]
        if attention > attn:
            sum_ += 1
    return sum_ / iterations


# %%

bounds = torch.zeros(64)
mat = EQKE + PQKE[length - 1].unsqueeze(0)
iterations = 0
sum_ = 0
max_eqkp_diff = torch.max(EQKP, dim=1).values - torch.min(EQKP, dim=1).values

for maximum in range(25):
    best_diff = torch.exp(torch.max(mat - mat[:, maximum].unsqueeze(1), dim=0).values)[
        : (maximum + 1)
    ]
    currbound = torch.tensor(0.0)
    for first_pos in range(length):

        if maximum == 0:
            chernoff_bound = torch.tensor(1.0)

            currbound += (64 * chernoff_bound) * (
                ((maximum) ** (first_pos) * (maximum + 1) ** (length - 1 - first_pos))
                / ((maximum + 1) ** (length) - (maximum) ** (length))
            )
            continue

        difference_row = (
            torch.max(
                EVOU[: (maximum + 1)] - EVOU[: (maximum + 1), maximum].unsqueeze(1),
                dim=1,
            ).values
            + torch.max(torch.max(PVOU - (PVOU[:, maximum]).unsqueeze(1), dim=1).values)
        )[: (maximum + 1)]
        prov_row = torch.topk(EVOU - EVOU[:, maximum].unsqueeze(1), dim=1, k=2).values[
            maximum
        ]
        if abs(prov_row[0] - 0.0) < 1e-3:
            difference_row[-1] = prov_row[1]
        else:
            difference_row[-1] = prov_row[0]
        combined_row = difference_row * best_diff
        positional_diff = (W_pos[length - 1] @ W_U)[maximum] - torch.max(
            (W_pos[length - 1] @ W_U)[:maximum]
        )

        iterations += 1
        if maximum > 0:
            if (
                len(combined_row) > 1
                and torch.max(combined_row[:-1]) * (n_ctx) < -combined_row[-1]
            ):
                chernoff_bound = 0.0
            elif (
                len(combined_row) > 1
                and torch.min(combined_row[:-1]) * (n_ctx) > -combined_row[-1]
            ):
                chernoff_bound = 1.0
            else:
                chernoff_bound = compute_chernoff_bound(
                    torch.exp(torch.max(PQKP[length - 1] - PQKP[length - 1][first_pos]))
                    * torch.exp(torch.max(max_eqkp_diff[: (maximum + 1)]))
                    * combined_row[:-1],
                    max_val=-combined_row[-1] - positional_diff,
                )
                if length - 1 - first_pos > 0:
                    a_vals = torch.tensor(
                        [
                            chernoff_bound
                            * e ** (combined_row[-1] * i)
                            * (1 / 64) ** (i)
                            * ((maximum) / 64) ** (length - 1 - first_pos - i)
                            * math.comb(length - 1 - first_pos, i)
                            for i in range(length - 1 - first_pos)
                        ]
                    ).sum()
                else:
                    a_vals = chernoff_bound

                chernoff_bound = min(a_vals, 1.0)

            if maximum > 2:
                if torch.max(combined_row[:-2]) * (n_ctx) < -combined_row[-1]:
                    chernoff_bound = min(
                        chernoff_bound,
                        1 - torch.tensor(((maximum - 2) / (maximum)) ** (n_ctx)),
                    )
                if torch.max(combined_row[:-3]) * (n_ctx) < -combined_row[-1]:
                    chernoff_bound = min(
                        chernoff_bound,
                        1 - torch.tensor(((maximum - 3) / (maximum - 1)) ** (n_ctx)),
                    )
        else:
            chernoff_bound = torch.tensor(0.0)

        currbound += (64 * chernoff_bound) * (
            ((maximum) ** (first_pos) * (maximum + 1) ** (length - 1 - first_pos))
            / ((maximum + 1) ** (length) - (maximum) ** (length))
        )
    bounds[maximum] = currbound / (length)


# %%


def prob(row):
    if row > 0:
        return (row / 64) ** (length) - ((row - 1) / 64) ** (length)
    else:
        return (1 / 64) ** length


mean_accuracy = torch.tensor(bounds)
matrix = torch.tensor([prob(i) for i in range(64)])
print(matrix @ mean_accuracy)
# %%
