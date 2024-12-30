# %%
<<<<<<< HEAD
||||||| parent of 7287967 (added stuff)
from gbmi.exp_indhead.train import ABCAB8_1HMLP
from torch import where
from gbmi.model import train_or_load_model
import torch
from torch import tensor
from math import *
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
=======
from gbmi.exp_indhead.train import ABCAB8_1HMLP
from gbmi.exp_indhead.data_utils import construct_ngram_counts_table
from gbmi.exp_modular_arithmetic.train import train_or_load_model, CLOCK_CONFIG

from torch import where
from gbmi.model import train_or_load_model
import torch
from torch import tensor
from math import *
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences
>>>>>>> 7287967 (added stuff)
import copy
from inspect import signature
<<<<<<< HEAD
||||||| parent of 7287967 (added stuff)
import plotly.express as px
from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1H
from torch import where
from gbmi.model import train_or_load_model
import torch
import einops
from gbmi.utils import ein
from torch import tensor
=======
import plotly.express as px
from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1HMLP

from torch import where
from gbmi.model import train_or_load_model
import torch
import einops
from gbmi.utils import ein
from torch import tensor
>>>>>>> 7287967 (added stuff)
from math import *

import einops
import plotly.express as px
import torch
from torch import tensor, where

from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1H, ABCAB8_1HMLP
from gbmi.model import train_or_load_model
from gbmi.utils import ein
from gbmi.utils.sequences import generate_all_sequences


def show(matrix):

    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cuda")
runtime_model_1, model = train_or_load_model(ABCAB8_1HMLP, force="load")

# runtime, model = train_or_load_model(CLOCK_CONFIG, force="load")
model.to(device)

# %%
# %%
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E
W_K_1 = model.W_K[1, 0]
W_U = model.W_U
W_U = W_U - W_U.mean(dim=1, keepdim=True)
W_V_1 = model.W_V[1, 0]
W_K_0 = model.W_K[0, 0]
W_V_0 = model.W_V[0, 0]
W_O_0 = model.W_O[0, 0]
W_Q_1 = model.W_Q[1, 0]
W_Q_0 = model.W_Q[0, 0]
W_O_1 = model.W_O[1, 0]
W_Q_0 = model.W_Q[0, 0]
o = W_O_0
v = W_V_0
q_1 = W_Q_1
k_1 = W_K_1
# %%
EQKP = (W_E @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKP = (W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T) / (attn_scale_0)
PQKE = (W_pos @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)
EQKE = (W_E @ W_Q_0 @ W_K_0.T @ W_E.T) / (attn_scale_0)


# We say position of a is at 7 and is a
def get_typical_softmax(a, a_pos):

    # a_error = EQKE[a, :] + PQKE[a_pos, :]

    return torch.softmax(PQKP[a_pos, :] + EQKP[a, :], dim=0)


def jacobian(softmax):

    return torch.diag(softmax) - torch.outer(softmax, softmax)


# %%
PVO = W_pos @ W_V_0 @ W_O_0

EVO = W_E @ W_V_0 @ W_O_0
PVOU = W_pos @ W_V_1 @ W_O_1 @ W_U
PVOVOU = W_pos @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOVOU = W_E @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOU = W_E @ W_V_1 @ W_O_1 @ W_U


# show


# computable_matrices

# W_E@W_U
# W_pos@W_U
# direct path is known factor


# E + pos + EVO + PVO computers the query matrix. Also mostly computable
#
# E+pos+PVO can easily be computed
a = 7
a_pos = 5

vec = W_E[a] + W_pos[5] + get_typical_softmax(a, a_pos) @ PVO

pvo_vec = vec @ W_Q_1 @ W_K_1.T @ PVO.T
e_vec = vec @ W_Q_1 @ W_K_1.T @ W_E.T

pos_vec = vec @ W_Q_1 @ W_K_1.T @ W_pos.T


sup_pattern = EVO @ W_Q_1 @ W_K_1.T @ EVO.T

row_pattern = vec @ W_Q_1 @ W_K_1.T @ EVO.T

col_pattern = EVO @ W_Q_1 @ W_K_1.T @ vec

show(col_pattern.unsqueeze(0).T + row_pattern.unsqueeze(0) + sup_pattern)


n_ctx = 8
# %%
index = 0
pos_pattern = torch.softmax((PQKP[-1] + EQKP), dim=1)

# %%

# (PVO + W_pos + EVO + W_E) @ QK @ (PVO+W_pos+EVO+W_E)

PVO_mat = (PVO) @ model.W_Q[1, 0] @ model.W_K[1, 0].T @ (PVO).T
pos_patterns = torch.zeros(8, 8)
for index in range(n_ctx):
    pos_patterns[index] = torch.tensor(
        torch.softmax((PQKP[index] + EQKP[0])[: (1 + index)], dim=0).tolist()
        + torch.zeros(7 - index).tolist()
    )

#  mat_mul = (W_pos[index]+pos_patterns[index]@PVO[:(1-index)])@model.W_Q[1,0]@model.W_K[1,0].T@(W_pos+pos_patterns)


(W_E) @ (PVO + W_pos)


# %%
# %%


# bounded differences inequality
#
#
#           Given weights on both sides
#
##
#
#          a_ib_1+a_ib_2+....+a_ib_n+a_1b_i+...+a_ib_i
##       row_weight*(difference in rows)
#          +column weight*(difference in columns)
#               +row_weight*column_weight*(diag[n]-diag[m])
#
#
##
#
#
#
###
#
#
def calc_concentrate(mat, row_weights, col_weights, length):
    c_s = torch.zeros(length)
    mean_ = torch.mean(mat) * (1 - row_weights @ col_weights) + (
        row_weights @ col_weights
    ) * (torch.mean(sup_pattern.diag()))

    for index in range(length):
        col = col_weights[index]
        row = row_weights[index]
        maximum_ = torch.tensor(0.0)
        for i in range(length):
            for j in range(length):
                difference = (
                    row * (mat[i, :] - mat[j, :])
                    + col * (mat[:, i] - mat[:, j])
                    + row * col * (mat[i][i] - mat[j][j])
                )
                diff_ = torch.max(abs(difference))
                maximum_ = max(diff_, maximum_)
                print(diff_)
                print(difference, "diff array")

        c_s[index] = maximum_
    variance = (c_s**2).sum()
    return mean_, variance, c_s


pattern = sup_pattern - torch.mean(sup_pattern, dim=0)
diag = pattern.diag()
show(diag)
weight_diff = torch.tensor([0.1, 0.7, -0.8, 0.2, -0.3, 0.1, 0.0, 0.0])
lambda_ = 0.7

row = 5
bounds = torch.zeros(26)
for row in range(26):
    big_mat = weight_diff.unsqueeze(0).T @ pattern[row].unsqueeze(0)
    bound_ = torch.mean(e ** (lambda_ * big_mat), dim=1)
    bounds[row] = torch.prod(bound_)
bound = torch.mean(bounds) * e ** (-lambda_ * 15)
print(bound)
diag_bound = torch.tensor(0.0)
lambda_diag = 0.6

big_mat = weight_diff.unsqueeze(0).T @ diag.unsqueeze(0)
bound_ = torch.mean(e ** (lambda_diag * big_mat), dim=1)
diag_bound = torch.prod(bound_) * e ** (-lambda_diag * 15)

print(diag_bound, "diag")


def monte_carlo_bound(mat, max_val, weight1, weight2):

    sum_ = 0
    iterations = 1000
    for i in range(iterations):
        sequence = torch.randint(low=0, high=26, size=(8,))
        bound_ = (mat[sequence, :])[:, sequence]
        bounding = (bound_ * weight2.unsqueeze(0) * weight1.unsqueeze(1)).sum()

        if bounding > max_val:
            sum_ += 1
    return sum_ / iterations


# %%
EVOQKOVE = W_E @ W


dims = 6
linear = torch.randn(64, dims)
# %%

torch.set_default_device("cuda")
dims = 10
linear = torch.rand(model.embed.W_E.shape[1], dims).to("cuda")
bias = 100.0 * torch.ones(dims).to("cuda")

optim = torch.optim.AdamW([linear, bias], lr=5e-1)
# model.embed.W_E = torch.nn.Parameter(torch.randn(model.embed.W_E.shape)*0.1)


def entr(intermediate):
    intermediate_comp = intermediate[intermediate > 10 ** (-7)]
    intermediate_comp = intermediate_comp / (intermediate_comp.sum())
    return torch.sum(intermediate_comp * torch.log(1 / intermediate_comp))


# %%

model.embed.W_E.requires_grad = False
for param in model.parameters():
    param.requires_grad = False
linear.requires_grad = True
bias.requires_grad = True
epochs = 10000
for epoch in range(epochs):

    optim.zero_grad()
    intermediate = torch.nn.ReLU()((model.embed.W_E) @ linear + bias)
    reconstruction = intermediate @ linear.T
    loss__ = (
        torch.norm(reconstruction - (model.embed.W_E)) + 0.01 * abs(intermediate).sum()
    )

    print(loss__)
    loss__.backward()
    optim.step()
# %%
n_ctx = 8


def compute_chernoff_bound(x, max_val, weighting=[]):
    if len(weighting) > 0:
        seq_length = len(weighting)
    else:
        seq_length = n_ctx
    last_percentage = max_val / (
        (seq_length) * torch.max(x)
    )  # Calculates whether it can get the max_value with n_ctx tokens

    if last_percentage >= 1:
        return torch.tensor(0.0)
    elif last_percentage < 0:
        return torch.tensor(1.0)
    else:
        lambda_ = torch.log(
            (len(x) - 1) * (last_percentage) / (1 - last_percentage)
        ) / (torch.max(x))

    chernoff_bound = (torch.exp(x * lambda_).mean()) ** (seq_length) * e ** (
        -lambda_ * max_val
    )

    return chernoff_bound


# %%
"""

class IndependentSequence:
    def __init__(self,seq):
        self.seq = seq # sequence uses '' to denote independent filler variables
        self.blank_spaces = []
        self.non_blanks = []
        for i in range(len(self.seq)):
            if self.seq[i] == '':
                self.blank_spaces.append(i)
            else:
                self.non_blanks.append(i)


        self.blank_spaces = torch.tensor(self.blank_spaces)
        self.non_blanks = torch.tensor(self.non_blanks)

   def E_propogate(self,mat,max_val,weighting): # propogate matrix that takes embedding as start point
        bounder = torch.tensor(0.0)
        for i in self.non_blanks:
           bounder+=weighting[i]*mat[self.seq[i]]
        compute_chernoff_bound(mat,max_val=bound[],weighting=weighting[self.blank_spaces])

sequence_skeleton = ['',a,b,'','',a,'','']

#class KLSoftMaxBound(sequence_skeleton,noise_terms):



#class KLBound(softmax,eps,column,partial_seq):



"""


# %%
model.remove_all_hook_fns()
model.cfg.use_hook_mlp_in = True

epochs = 20000
batch_size = 64.0


# %%
class AutoEncoder(torch.nn.Module):
    def __init__(self, d_mlp, sae_dim):
        super().__init__()
        self.d_mlp = d_mlp
        self.sae_dim = sae_dim
        self.W_enc = torch.nn.Parameter(torch.zeros((sae_dim, d_mlp)))
        self.W_dec = torch.nn.Parameter(torch.zeros((d_mlp, sae_dim)))

        torch.nn.init.xavier_uniform_(
            self.W_enc, gain=torch.nn.init.calculate_gain("relu")
        )

        torch.nn.init.xavier_uniform_(
            self.W_dec, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.W_dec, gain=torch.nn.init.calculate_gain("relu")
        )

        self.b_enc = torch.nn.Parameter(
            torch.zeros(
                sae_dim,
            )
        )
        self.b_dec = torch.nn.Parameter(
            torch.zeros(
                d_mlp,
            )
        )

    def get_intermediate(self):
        return self.intermediate

    def forward(self, h):

        self.intermediate = torch.nn.ReLU()(
            (h - self.b_dec) @ self.W_enc.T + self.b_enc
        )

        return self.intermediate @ self.W_dec.T + self.b_dec


class AutoEncoderLinear(torch.nn.Module):
    def __init__(self, d_mlp, sae_dim, d_model):
        super().__init__()
        self.d_mlp = d_mlp
        self.sae_dim = sae_dim
        self.W_enc = torch.nn.Parameter(torch.zeros((sae_dim, d_model)))
        self.W_dec = torch.nn.Parameter(torch.zeros((d_model, sae_dim)))
        self.W_dense = torch.nn.Parameter(torch.zeros((d_model, d_model)))
        torch.nn.init.xavier_uniform_(
            self.W_enc, gain=torch.nn.init.calculate_gain("relu")
        )

        torch.nn.init.xavier_uniform_(
            self.W_dec, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.W_dec, gain=torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.xavier_uniform_(
            self.W_dense, gain=torch.nn.init.calculate_gain("relu")
        )
        self.b_enc = torch.nn.Parameter(
            torch.zeros(
                sae_dim,
            )
        )
        self.b_dec = torch.nn.Parameter(
            torch.zeros(
                d_model,
            )
        )
        self.b_f = torch.nn.Parameter(
            torch.zeros(
                d_model,
            )
        )

    def get_intermediate(self):
        return self.intermediate

    def forward(self, h):

        self.intermediate = torch.nn.ReLU()(
            (h - self.b_dec) @ self.W_enc.T + self.b_enc
        )

        return (
            self.intermediate @ self.W_dec.T
            + self.b_f
            + (h - self.b_dec) @ self.W_dense.T
        )


# %%
indices = model.blocks[0].mlp.W_in.abs().sum(dim=0).topk(k=22).indices
mlp_sim = model.blocks[0].mlp.W_in[:, indices]
magnitudes = torch.norm(mlp_sim, dim=[0])
show((mlp_sim.T @ mlp_sim / (magnitudes.unsqueeze(0))) / magnitudes.unsqueeze(1))
# %%
epochs = 10000
batch_size = 128
d_mlp = model.cfg.d_mlp
sae_dim = 128

lambda_ = torch.tensor(0.01)
sae = AutoEncoderLinear(d_mlp, sae_dim, model.cfg.d_model)
pre_auto = AutoEncoder(64, 128)
# %%
optimizer = torch.optim.AdamW(list(sae.parameters()), lr=5e-4)
for epoch in range(epochs):
    f = torch.randint(0, 113, (int(batch_size), 2))
    f = torch.concat((f, (113 * torch.ones(int(batch_size), 1))), dim=-1)
    f = f.to(torch.int32)
    logits, cache = model.run_with_cache(f)
    initial = cache["blocks.0.hook_resid_mid"]
    final = sae(initial)

    optimizer.zero_grad()
    loss = (
        torch.norm((cache["blocks.0.hook_mlp_out"] - final))
        + lambda_ * sae.get_intermediate().abs().sum()
    ) / (batch_size)
    loss.backward()
    optimizer.step()
    print(torch.norm(cache["blocks.0.hook_mlp_out"] - final) / (batch_size))

# %%


# %%


def hook_fn(activation, hook):
    activation = sae(activation)
    return activation


for l in range(f.shape[0]):
    model.add_hook("blocks.0.mlp.hook_post", hook_fn)

    logits, cache = model.run_with_cache(f[l])
    print(torch.argmax(logits, dim=-1).squeeze()[-1], "modified model")
    model.remove_all_hook_fns()
    logits, cache = model.run_with_cache(f[l])
    print(torch.argmax(logits, dim=-1).squeeze()[-1], "original model")
    print(f[l][-2], "ground truth")
# %%
import matplotlib.pyplot as plt

trials = 2000
batch_size = 1
for i in range(trials):
    trial = torch.randint(0, 113, (int(batch_size), 2))
    trial = torch.concat((trial, (113 * torch.ones(int(batch_size), 1))), dim=-1)
    trial = trial.to(torch.int32)
    trial = trial.squeeze()
    logits, cache = model.run_with_cache(trial)
    initial = cache["blocks.0.hook_mlp_out"]
    final = sae(initial)
    if sae.get_intermediate().squeeze()[-2].reshape((2, 4)).sum() > 0:
        print(trial)
        print(sae.get_intermediate().squeeze()[-2].reshape((2, 4)).sum())
        plt.matshow(sae.get_intermediate().squeeze()[-2].reshape((2, 4)).detach().cpu())


# %%
def kl_divergence(p, q):

    return p @ torch.log(p / q)


believed_sequence = torch.tensor(
    [0.0346, 0.1657, 0.0892, 0.1088, 0.0252, 0.5262, 0.0503, 0.0000]
)
n_ctx = 8
index = n_ctx - 2
kl_divergences = []
for i in range(1000):
    sequence = torch.randint(low=0, high=26, size=(8,))
    if len(list(sequence)) != len(set(list(sequence))):
        continue
    logits, cache = model.run_with_cache(sequence)
    softmax = cache["attn", 0].squeeze()[index]
    print(softmax)
    kl = kl_divergence(
        softmax[: (index + 1)],
        believed_sequence[
            : (index + 1)
        ],  # torch.tensor([1 / (index + 1) for i in range(index + 1)]),
    )

    kl_divergences.append(kl)
# %%
kl_clusters = []
distance = 0.01
for i in range(1000):
    kl = kl_divergences[i]
    distinct_cluster = True
    for l in kl_clusters:
        if abs(l - kl) < distance:
            distinct_cluster = False
    if distinct_cluster:
        kl_clusters.append(kl)
# %%
accounted_sequences = []

for i in range(1000):
    sequence = torch.randint(low=0, high=26, size=(8,))
    sequence_list = [i.item() for i in list(sequence)]
    if len(sequence_list) != len(set(sequence_list)):
        continue
    logits, cache = model.run_with_cache(sequence)
    softmax = cache["attn", 0].squeeze()[index]
    print(softmax)
    kl = kl_divergence(
        softmax[: (index + 1)],
        believed_sequence[
            : (index + 1)
        ],  # torch.tensor([1 / (index + 1) for i in range(index + 1)]),
    )

    for f in kl_clusters:
        if f.item() in accounted_sequences:
            continue
        if abs(kl - f) < distance:
            print("HERE")
            accounted_sequences.append(f.item())
            print(sequence)
            print(f)

# %%
