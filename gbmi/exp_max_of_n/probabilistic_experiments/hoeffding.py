# %%
from math import *

import numpy as np
import plotly.express as px
import torch
from scipy.stats import binom

from gbmi.exp_max_of_n.train import MAX_OF_4_CONFIG, MAX_OF_10_CONFIG
from gbmi.model import train_or_load_model

rundata, model = train_or_load_model(MAX_OF_4_CONFIG(123))

# rundata, model = train_or_load_model(MAX_OF_10_CONFIG)

torch.set_default_device("cuda")
length = 4
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


epochs = 300
for param in model.parameters():
    param.requires_grad = False
torch.set_default_device("cuda")
model = model.to("cuda")


"""
def compute_chernoff_bound(x, max_val):
    lambda_ = torch.tensor(0.0)
    x = x.detach()
    max_val = max_val.detach()
    lambda_.requires_grad = True
    optimizer = torch.optim.AdamW([lambda_], lr=1e-3)
    n_ctx = length - 2
    for l in range(epochs):
        optimizer.zero_grad()
        valuation = ((torch.exp(lambda_ * x) * x).sum()) / (
            torch.exp(lambda_ * x).sum()
        )
        x_s = max_val / (n_ctx)
        loss = torch.abs(valuation - x_s)
        loss.backward()

        optimizer.step()
    print(loss)
    chernoff_bound = (torch.exp(x * lambda_).mean()) ** (n_ctx) * e ** (
        -lambda_ * max_val
    )
    return chernoff_bound
"""
# %%
n_ctx = length - 2


def compute_chernoff_bound(x, max_val):
    last_percentage = max_val / ((n_ctx) * torch.max(x))
    print(last_percentage, "percentage")
    if last_percentage >= 1:
        return torch.tensor(0.0)
    elif last_percentage < 0:
        return torch.tensor(1.0)
    else:
        lambda_ = torch.log(
            (len(x) - 1) * (last_percentage) / (1 - last_percentage)
        ) / (torch.max(x))
    print(lambda_, "lambda")
    chernoff_bound = (torch.exp(x * lambda_).mean()) ** (n_ctx) * e ** (
        -lambda_ * max_val
    )
    print(chernoff_bound, "cher")
    return chernoff_bound


def compute_monte_bound(last, maximum):
    iterations = 10
    sum_ = 0
    for l in range(iterations):
        position = torch.randint(low=0, high=length - 1, size=(1,))
        values = torch.randint(low=0, high=maximum, size=(length - 1,))
        initial = list(values)
        sequence = torch.tensor(initial + [last])
        sequence[position] = maximum

        logits, cache = model.run_with_cache(sequence)
        attention = cache["attn", 0].squeeze()[-1][position]

        if attention > 0.666:
            sum_ += 1
    return sum_ / iterations


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


accuracies = []


def h(x):
    return (1 + x) * torch.log(1 + x) - x


# %%
multiplier = 2
bounds = []
ks = []
bounds = torch.zeros(64, 64)

montecarlobounds = torch.zeros(64, 64)
mat = EQKE + PQKE[length - 1].unsqueeze(0)
mat = mat.to("cuda")

for row in range(
    1, 64
):  # Represents the token in the final position of the sequence, attending to the things before it
    current_mean = 0.0
    current_min = torch.inf
    current_max = 0.0
    min_eqkp = torch.min(torch.exp(EQKP[row]), dim=0).values
    max_eqkp = torch.max(torch.exp(EQKP[row]), dim=0).values
    min_pqkp = torch.min(torch.exp(PQKP[length - 1]), dim=0).values
    max_pqkp = torch.max(torch.exp(PQKP[length - 1]), dim=0).values
    mean_pqkp = torch.mean(torch.exp(PQKP[length - 1]), dim=0)

    for l in range(row):
        current_min = min(
            (
                torch.exp(mat[row][l])
                * torch.min(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0).values
            ).item(),
            current_min,
        )
        current_max = max(
            (
                torch.exp(mat[row][l])
                * torch.max(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0).values
            ).item(),
            current_max,
        )
        current_mean = (
            current_mean * (l)
            + torch.exp(mat[row][l])
            * torch.mean(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0)  # .values
        ) / (l + 1)

        ks.append(
            torch.exp(mat[row][l])
            * torch.mean(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0)  # .values
        )

    for k in range(row, 64):
        currbound = torch.tensor(0.0)
        for pos in range(length - 1):
            exp_sum = torch.tensor(0.0)
            square_sum = torch.tensor(0.0)
            for i in range(length - 1):
                if i == pos:
                    continue
                exp_sum += torch.exp(PQKP[length - 1][i] + EQKP[row][i])
                square_sum += torch.exp(PQKP[length - 1][i] + EQKP[row][i]) ** 2
            c = (
                torch.exp(mat[row][k])
                * torch.exp(PQKP[length - 1][pos])
                * torch.exp(
                    EQKP[row][pos]
                )  # torch.min(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0).values
            )

            t = (
                c * (multiplier)
                - torch.exp(mat[row][row])
                * torch.exp(PQKP[length - 1][length - 1] + EQKP[row][length - 1])
                - current_mean
                * exp_sum
                * (length - 2)
                / (
                    (length - 1)
                    * (torch.mean(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0))
                )
            )
            print(t, "t")
            variance = torch.var(torch.exp(mat[row][:k])) * square_sum
            M = max(
                abs(current_max),
                abs(current_min),
            )
            if t > 0:
                bennet_ineq = e ** (-variance * h(M * t / (variance)) / (M**2))
                bernstein_ineq = e ** (-(t**2) / (2 * M * t / 3 + 2 * variance))
            else:
                bennet_ineq = torch.tensor(1.0)
                bernstein_ineq = torch.tensor(1.0)
            chernoff_bound = compute_chernoff_bound(
                torch.exp(mat[row][:k]) * (exp_sum) / (length - 1),
                t,
            )
            ks.append(
                torch.exp(mat[row][k])
                * torch.mean(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0)  # .values
            )
            print(row)
            current_mean = (
                current_mean * (k)
                + torch.exp(mat[row][k])
                * torch.mean(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0)  # .values
            ) / (
                k + 1
            )  # k represents the current maximum, which varies up to row

            current_min = min(
                (
                    torch.exp(mat[row][k])
                    * torch.min(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0).values
                ).item(),
                current_min,
            )
            current_max = max(
                (
                    torch.exp(mat[row][k])
                    * torch.max(torch.exp(PQKP[length - 1] + EQKP[row]), dim=0).values
                ).item(),
                current_max,
            )
            anomaly_bound = torch.tensor(1.0)
            if k > 2:
                if torch.max(torch.exp(mat[row][: (k - 1)]) * max_pqkp * max_eqkp) * (
                    length - 2
                ) < (
                    c * (multiplier)
                    - torch.exp(mat[row][row])
                    * torch.exp(PQKP[length - 1][length - 1] + EQKP[row][length - 1])
                ):
                    anomaly_bound = torch.tensor(1.0) - ((k - 1) / k) ** (length - 2)

            currbound += min(
                anomaly_bound.item(),
                bennet_ineq.item(),
                chernoff_bound.item(),
                bernstein_ineq.item(),
                1,
            )
        bounds[row][k] = currbound / (length - 1)
# %%
mean_accuracy = []
bounds = torch.tensor(bounds)
for column in range(64):
    mean_accuracy.append(bounds[:column, column].mean())
mean_accuracy[0] = 0.0
mean_accuracy[1] = 0.0
mean_accuracy[-1] = 0.0
mean_accuracy[-2] = 0.0


def prob(row):
    if row > 0:
        return (row / 64) ** (length) - ((row - 1) / 64) ** (length)
    else:
        return (1 / 64) ** length


mean_accuracy = torch.tensor(mean_accuracy)
matrix = torch.tensor([prob(i) for i in range(64)])
print(matrix @ mean_accuracy)
accuracies.append(matrix @ mean_accuracy)
# %%

# %%
"""
multiplier = 1
bounds = []
bounds = torch.zeros(64, 64)
mat = EQKE + PQKE[length - 1]
for row in range(
    64
):  # Represents the token in the final position of the sequence, attending to the things before it
    current_mean = 0.0
    current_min = torch.inf
    current_max = -torch.inf
    min_eqkp = torch.min(EQKP[row])
    max_eqkp = torch.max((EQKP[row]))
    min_pqkp = torch.min((PQKP[length - 1]))
    max_pqkp = torch.max(PQKP[length - 1])
    for l in range(row):
        current_min = min((mat[row][l] + min_eqkp + min_pqkp).item(), current_min)
        current_max = max(((mat[row][l]) + max_eqkp + max_pqkp).item(), current_max)
        current_mean = (
            current_mean * (l)
            + mat[row][l]
            + torch.mean(PQKP[length - 1], dim=0)
            + torch.mean(EQKP[row], dim=0)
        ) / (l + 1)

    for k in range(row, 64):

        c = mat[row][k] + min_pqkp + min_eqkp

        t = c / (multiplier) - current_mean

        hoeffding_inequality = (length - 1) * e ** (
            -2 * (t**2) / ((current_max - current_min) ** 2)
        )  # There are 9 distinct random variables, all of whom who min and max as given
        markov_ineq = (length - 1) * e ** (
            -(t**2)
            / (
                2 * max(abs(current_max), abs(current_min)) * t / 3
                + (k) * torch.var(mat[row][:k])
            )
        )
        print(hoeffding_inequality, markov_ineq)
        current_mean = (
            current_mean * (k)
            + mat[row][k]
            + torch.mean(PQKP[length - 1], dim=0)
            + torch.mean(EQKP[row], dim=0)
        ) / (k + 1)

        current_min = min((mat[row][k] + min_eqkp + min_pqkp).item(), current_min)
        current_max = max(((mat[row][k]) + max_eqkp + max_pqkp).item(), current_max)

        bounds[row][k] = min(markov_ineq, hoeffding_inequality)
print(bounds)
"""

# %%
rowdifferences = torch.max(EVOU, dim=0).values - torch.min(EVOU, dim=0).values
believed_p = 0.05
believed_attn_on_max = 0.5
variance = (
    (rowdifferences) * (length - 1) * min(sqrt(believed_p), 1 / sqrt(length))
) / 2  #
means = []
variances = []
sigmascores = []
for column in range(64):

    attention = EVOU[column, column] * believed_attn_on_max
    sigmascore = 0
    for row in range(column):
        print(attention, "attn")
        print(means[row], "mean")
        print(variances[row], "variance")
        sigmascore += e ** (-((attention - means[row]) ** 2) / (((variances[row]))))
    sigmascores.append(sigmascore)

    row_to_focus = EVOU[:column, column]
    if column > 0:
        variance = (
            (torch.max(row_to_focus) - torch.min(row_to_focus))
            * (length - 1)
            * min(sqrt(believed_p), 1 / sqrt(length))
            / 2
        )
    else:
        variance = 0.1
    p = (64 - (row_to_focus > 0).sum()) / (64)
    p = p.to("cpu")
    dist = binom(p=p, n=length - 1)
    M = torch.tensor(dist.pmf(k=np.arange(10, dtype=np.int32)[:, None])).squeeze()
    things = torch.sqrt(torch.tensor([i for i in range(length)]))
    dotproduct = M.to(torch.double) @ things.to(torch.double)
    x_pos = row_to_focus[row_to_focus > 0]
    mean = (x_pos**2).mean()
    mean = sqrt(believed_p) * dotproduct * torch.sqrt(mean)
    #  print(mean.item(), variance[column].item())  # Looking at contribution from here.
    means.append(mean)
    variances.append(variance)
print(sigmascores)
show(torch.tensor(sigmascores))


def entropy(softmax):
    return


# %%
p = torch.tensor(0.666)

p_norm_bound = torch.tensor(1)
evou_bounds = torch.zeros(64, 64)
for maximum in range(64):
    for col in range(maximum):
        column = (EVOU)[:, col] - (EVOU[:, maximum])
        trunc_ = column[:maximum].to("cuda")
        x_mean = trunc_.mean()
        y = (trunc_ - x_mean) * (1 - p)
        p_val = (y > 0).sum() / (len(y))
        dist = binom(p=p_val.cpu(), n=length - 2)
        M = torch.tensor(
            dist.pmf(k=np.arange(length - 1, dtype=np.int32)[:, None])
        ).squeeze()
        things = torch.sqrt(torch.tensor([i for i in range(length - 1)]))
        dotproduct = M.to(torch.double).to("cuda") @ things.to(torch.double).to("cuda")
        y_pos = y[y > 0].to("cuda")
        mean = p_norm_bound * torch.sqrt(((y_pos) ** 2).mean()) * dotproduct
        variance = (p_norm_bound) ** 2 * ((y) ** 2).mean()
        U = torch.max(y) * (1 - p)
        U = torch.sqrt(torch.max(variance, U**2))
        v_n = 2 * U * mean + variance

        print(mean, variance, col, "col", U)
        believed_evou_on_p = p * (EVOU[maximum, maximum] - EVOU[maximum, col])
        print(believed_evou_on_p)
        t = believed_evou_on_p - (1 - p) * x_mean
        talagrand_bound = e ** (-(t**2) / (2 * t * U / 3 + 2 * v_n))
        other_bound = e ** (-v_n * h(t * U / (v_n)) / (U**2))
        evou_bounds[maximum, col] = other_bound
# %%
