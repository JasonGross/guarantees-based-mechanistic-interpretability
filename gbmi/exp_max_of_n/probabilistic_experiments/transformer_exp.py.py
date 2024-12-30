# %%
import torch
import re
from torch import nn
from math import *
from einops import rearrange

torch.set_default_device("cuda")
d_value = 5
d_vocab = 5000
d_model = 32
n_ctx = 100
lambda_ = torch.tensor(2.5)


def upper_right_mask(n_ctx):
    mask = []
    for i in range(n_ctx):
        mask.append([])
        for j in range(n_ctx):
            if j > i:
                mask[i].append(-torch.inf)
            else:
                mask[i].append(0)
    return torch.tensor(mask)


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
        self.W_in = nn.Linear(d_mlp, d_model)
        self.W_out = nn.Linear(d_model, d_mlp)
        self.rel = nn.GELU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        return self.drop(self.W_out(self.rel(self.W_in(x.T))).T)


class AttentionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_value)
        self.W_K = nn.Linear(d_model, d_value)
        self.W_V = nn.Linear(d_model, d_value)
        self.W_O = nn.Linear(d_value, d_model)
        self.mask = upper_right_mask(n_ctx)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):  # x is d_model x n_ctx
        self.attn_scores = self.W_Q(x.T) @ (self.W_K(x.T)).T + self.mask
        self.attn_soft = self.drop1(
            torch.softmax(self.attn_scores / sqrt(d_value), dim=1)
        )
        return self.drop2(
            self.W_O(self.W_V(x.T)).T @ (self.attn_soft).T
        )  # n_ctx x d_value


class AttentionLayer(nn.Module):
    def __init__(self, numheads):
        super().__init__()
        self.numheads = numheads
        self.heads = [AttentionHead() for i in range(self.numheads)]

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
        embed = torch.exp(-log(10000) * 2 / d_model * torch.arange(1, d_model // 2 + 1))
        t_embed = torch.kron(embed, torch.arange(1, n_ctx + 1)).reshape(
            n_ctx, d_model // 2
        )
        sin_embed = torch.sin(t_embed)
        cos_embed = torch.cos(t_embed)
        self.embedding = rearrange([sin_embed, cos_embed], "t h w -> h (w t)").T
        self.layers = [
            AttentionLayer(10),
            LayerNorm(),
            MLP(32),
            LayerNorm(),
            AttentionLayer(10),
            LayerNorm(),
            MLP(32),
            LayerNorm(),
            AttentionLayer(10),
            LayerNorm(),
            MLP(32),
            LayerNorm(),
            AttentionLayer(10),
            LayerNorm(),
            MLP(32),
            LayerNorm(),
        ]
        self.additivelayers = set([0, 2, 4, 6, 8, 10, 12, 14])
        self.attn_layers = set([0, 4, 8, 12])
        self.loss_fn = nn.CrossEntropyLoss()

    def loss(self, x, tokens):  # assume that x is p distributions. And
        l = 0
        for i in self.attn_layers:
            for head in self.layers[i].heads:
                l = l - lambda_ * torch.sum((head.attn_soft) ** 2) / (
                    10 * len(self.attn_layers) * n_ctx
                )
        return self.loss_fn(x[:-1], tokens[1:]) + l

    def forward(self, x):
        residual = self.W_E(x).T
        residual = residual + self.embedding
        for i in range(len(self.layers)):
            if i in self.additivelayers:
                residual = residual + self.layers[i](residual)
            else:
                residual = self.layers[i](residual)
        return self.W_U(residual.T)


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
torch.set_default_device("cuda")

t = Transformer()

load = False
if load:
    t.load_state_dict(torch.load("model.pt", weights_only=True))
optimizer = torch.optim.AdamW(t.parameters())
# %%
f = open("harry_potter.txt", "r")
lines = f.read()[10000:20000]
tokens, tokdict = tokenize(lines)
encoding = one_hot(tokens)
chunk_size = n_ctx

epochs = 1000
for epoch in range(epochs):
    print(epoch)
    for l in range(0, len(tokens), chunk_size):
        print(l)
        chunk = encoding[l : l + chunk_size]
        if len(chunk) < n_ctx:
            break
        out = t(chunk)
        s = t.loss(out, torch.tensor(tokens[l : l + chunk_size]))
        optimizer.zero_grad()
        s.backward()
        optimizer.step()
    print(epoch, s)
    T = 0.5
    for token in range(100):
        probs = torch.softmax(t(one_hot(tokens[-n_ctx:]))[-1] / T, dim=0)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        predictedtok = dist.sample()
        if int(predictedtok) in tokdict.keys():
            print(tokdict[int(predictedtok)], end=" ")
        else:
            print("not in dict", end=" ")
        tokens.append(int(predictedtok))
torch.save(t.state_dict(), "model.pt")
# %%
import plotly.express as px

for i in t.attn_layers:
    for head in t.layers[i].heads:
        print(i)
        px.imshow(head.attn_soft.detach().cpu()).show()

# %%
