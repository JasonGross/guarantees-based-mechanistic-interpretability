# %%
import torch
import re
from torch import nn
from math import *
from einops import rearrange

d_value = 16
d_vocab = 1000
d_model = 16
n_ctx = 10
import matplotlib.pyplot as plt

torch.set_default_device("cuda")


def upper_right_mask(n_ctx):
    mask = []
    for i in range(n_ctx):
        mask.append([])
        for j in range(n_ctx):
            if j > i:
                mask[i].append(-torch.inf)
            else:
                mask[i].append(0)
    return torch.tensor(mask).to("cuda")


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  # x is d_model x n_ctx
        return self.bias + self.gain * (x - torch.mean(x, dim=0)) / (
            torch.var(x, dim=0, correction=0)
        )


class MLP(nn.Module):
    def __init__(self, d_mlp):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.rel = nn.GELU()

    def forward(self, x):
        return self.W_out(self.rel(self.W_in(x.T))).T


class AttentionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_value)
        self.W_K = nn.Linear(d_model, d_value)
        self.W_V = nn.Linear(d_model, d_value)
        self.W_O = nn.Linear(d_value, d_model)
        self.weights = nn.Parameter(torch.ones(n_ctx, n_ctx).to(torch.float))
        self.mask = upper_right_mask(n_ctx)

    def forward(self, x):  # x is d_model x n_ctx
        attn_scores = self.W_Q(x.T) @ (self.W_K(x.T)).T
        attn_soft = torch.softmax(
            (attn_scores * self.weights / sqrt(d_value) + self.mask),
            dim=1,
        )
        self.attn_soft = attn_soft
        return self.W_O(self.W_V(x.T)).T @ (attn_soft).T  # n_ctx x d_value


class AttentionLayer(nn.Module):
    def __init__(self, numheads):
        super().__init__()
        self.numheads = numheads
        self.heads = nn.ModuleList([AttentionHead() for i in range(self.numheads)])

    def forward(self, x):
        sum_ = 0
        for a in self.heads:
            sum_ += a(x)
        return sum_


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.W_E = nn.Linear(d_vocab, d_model)
        self.W_U = nn.Linear(d_model, d_vocab)
        self.layers = nn.ModuleList(
            [
                AttentionLayer(8),
                LayerNorm(),
                MLP(1000),
                LayerNorm(),
            ]
        )
        self.additivelayers = set([0, 2])
        self.loss_fn = nn.CrossEntropyLoss()

    def loss(self, x, tokens):  # assume that x is p distributions. And
        return self.loss_fn(x, tokens)

    def forward(self, x):
        residual = self.W_E(x).T
        for i in range(len(self.layers)):
            if i in self.additivelayers:
                residual = residual + self.layers[i](residual)
            else:
                residual = self.layers[i](residual)
        return self.W_U(residual.T)


# %%


def one_hot(tokens):  # takes tokens in as list
    return torch.nn.functional.one_hot(torch.tensor(tokens), num_classes=d_vocab).to(
        torch.float
    )


def tokenize(text):
    text = re.sub("\W+", " ", text)
    words = list(set(text.split(" ")))
    tokdict = {}
    tokens = []
    for l in text.split(" "):
        tokdict[words.index(l) + 1] = l
        tokens.append(words.index(l) + 1)
    return tokens, tokdict


# %%
t = Transformer()
load = False
if load:
    t.load_state_dict(torch.load("model2.pt", weights_only=True))
    print("here")

optimizer = torch.optim.AdamW(t.parameters(), lr=1e-3)
t.to("cuda")
# %%
f = open("shakespeare.txt", "r")
lines = f.read()[:100000]

tokens, tokdict = tokenize(lines)
encoding = one_hot(tokens).to("cuda")
chunk_size = n_ctx
# %%
for epoch in range(500):
    for l in range(0, len(tokens) - 1 - chunk_size):
        chunk = encoding[l : l + chunk_size].to("cuda")

        if len(chunk) < n_ctx:
            break
        out = t(chunk)
        s = t.loss(out, torch.tensor(tokens[l + 1 : l + 1 + chunk_size]).to("cuda"))
        optimizer.zero_grad()
        s.backward()
        optimizer.step()
        if l % 1000 == 0:
            print(l, len(tokens))

    print(epoch)
    T = 2
    print(s)
    print(t.layers[0].heads[0].weights)
    if epoch % 10 == 0:
        for token in range(100):
            probs = torch.softmax(t(one_hot(tokens[-n_ctx:]))[-1] / T, dim=0)
            dist = torch.distributions.categorical.Categorical(probs=probs)
            predictedtok = dist.sample()
            if int(predictedtok) in tokdict.keys():
                print(tokdict[int(predictedtok)], end=" ")
            else:
                print("not in dict", end=" ")
            tokens.append(int(predictedtok))
# torch.save(t.state_dict(),'model2.pt')

# %%
for head in range(5):
    show((t.layers[1].heads[head].weights + upper_right_mask(n_ctx)).detach().cpu())
# %%
W_E = t.W_E

# %%
task = induction_head

task = torch.randint(low=0, high=26, size=(n_ctx,))
ab_index = torch.randint(low=0, high=n_ctx - 2, size=(1,))
task[ab_index[0] + 1] = task[ab_index[0]]
task[-1] = task[ab_index[0]]
