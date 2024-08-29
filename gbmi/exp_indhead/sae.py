# %%
from gbmi.exp_indhead.train import ABCAB8_1HMLP
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
from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1H
from torch import where
from gbmi.model import train_or_load_model
import torch
import einops
from gbmi.utils import ein
from torch import tensor
from math import *


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cuda")
runtime_model_1, model = train_or_load_model(ABCAB8_1HMLP, force="load")
model.to(device)
d_mlp = 256
sae_dim = 256
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
# %%
PVO = (W_pos @ W_V_0 @ W_O_0) / (attn_scale_1)

EVO = (W_E @ W_V_0 @ W_O_0) / (attn_scale_1)
PVOU = (W_pos @ W_V_1 @ W_O_1 @ W_U) / (attn_scale_1)
PVOVOU = (W_pos @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U) / (attn_scale_1)
EVOVOU = (W_E @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U) / (attn_scale_1)
EVOU = (W_E @ W_V_1 @ W_O_1 @ W_U) / (attn_scale_1)

n_ctx = 8


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
lambda_ = torch.tensor(0.03)
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


# %%
indices = model.blocks[0].mlp.W_in.abs().sum(dim=0).topk(k=22).indices
mlp_sim = model.blocks[0].mlp.W_in[:, indices]
magnitudes = torch.norm(mlp_sim, dim=[0])
show((mlp_sim.T @ mlp_sim / (magnitudes.unsqueeze(0))) / magnitudes.unsqueeze(1))
# %%

sae = AutoEncoder(d_mlp, sae_dim)
pre_auto = AutoEncoder(64, 128)
optimizer = torch.optim.AdamW(list(sae.parameters()), lr=3e-3)
for epoch in range(epochs):

    f = torch.randint(0, 26, (int(batch_size), 8))
    f[:, -1] = f[:, -3]

    logits, cache = model.run_with_cache(f)
    initial = cache["blocks.0.mlp.hook_post"]
    final = sae(initial)

    optimizer.zero_grad()
    loss = (
        torch.norm((initial - final)) + lambda_ * sae.get_intermediate().abs().sum()
    ) / (batch_size)
    loss.backward()
    optimizer.step()
    print(torch.norm(initial - final) / (batch_size))

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
trials = 10
for i in range(trials):
    trial = torch.randint(0, 26, (1, 8))

    logits, cache = model.run_with_cache(trial)
    initial = cache["blocks.0.mlp.hook_post"]
    final = sae(initial)
    show(sae.get_intermediate().squeeze()[-2].reshape((16, 16)))


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
