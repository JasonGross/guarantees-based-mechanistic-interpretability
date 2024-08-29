# %%
import numpy as np
import torch
import transformer_lens

torch.set_default_device("cuda")
cfg = {}


cfg = transformer_lens.HookedTransformerConfig(
    n_layers=1,
    d_vocab=12,
    d_model=32,
    n_ctx=3,
    d_head=10,
    n_heads=1,
    d_mlp=128,
    act_fn="relu",
)
transformer = transformer_lens.HookedTransformer(cfg)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=0.1)
epoch = 100

and_tokens = [2, 3, 4, 5, 6]
or_tokens = [7, 8, 9, 10, 11]
boolean = [0, 1]

cross_entropy = torch.nn.CrossEntropyLoss()
for epoch in range(10000):
    sequence = torch.randint(low=2, high=12, size=(1,))
    bin_bit = torch.randint(low=0, high=2, size=(2,))
    seq = torch.tensor(list(sequence) + list(bin_bit))
    transformer.run_with_cache(seq)
    things = transformer.run_with_cache(seq)[0].squeeze()[-1]
    if seq[0] in and_tokens:
        correct_tok = seq[0] & seq[1]
    else:
        correct_tok = (not seq[0]) & (not seq[1])
    if int(correct_tok) == 0:
        correct_seq = torch.tensor([1.0, 0.0])
    else:
        correct_seq = torch.tensor([0.0, 1.0])
    optimizer.zero_grad()
    loss = cross_entropy(things[:2], correct_seq)
    print(loss)
    loss.backward()
    optimizer.step()
# %%
d_mlp = 128
sae_dim = 256


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

import plotly.express as px

model = transformer
length = 3
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
mlp = model.blocks[0].mlp
layernorm = model.blocks[0].ln1
o = W_O_0
v = W_V_0


corrupted_input = torch.tensor([0, 0, 5])
logits, activations = model.run_with_cache(corrupted_input)


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


def hook_fn(activation, hook):

    activation = activations[hook.name]

    return activation


def evaluate():
    loss = torch.tensor(0.0)
    for i in range(100):
        sequence = torch.randint(low=2, high=12, size=(1,))
        bin_bit = torch.randint(low=0, high=2, size=(2,))
        seq = torch.tensor(list(sequence) + list(bin_bit))

        things = transformer.run_with_hooks(
            seq, fwd_hooks=[("blocks.0.mlp.hook_pre", hook_fn)]
        )[0].squeeze()[-1]
        if seq[0] in and_tokens:
            correct_tok = seq[0] & seq[1]
        else:
            correct_tok = (not seq[0]) & (not seq[1])
        if int(correct_tok) == 0:
            correct_seq = torch.tensor([1.0, 0.0])
        else:
            correct_seq = torch.tensor([0.0, 1.0])

        loss += cross_entropy(things[:2], correct_seq)
    return loss


# %%


def kldiv(x, y):
    p = torch.softmax(x, dim=0)
    q = torch.softmax(y, dim=0)
    return p @ torch.log(p / q)


batch_size = 128
sae = AutoEncoder(d_mlp, sae_dim)
epochs = 10000
lambda_ = 0.5
optimizer = torch.optim.AdamW(list(sae.parameters()), lr=1e-3)
for epoch in range(epochs):

    sequence = torch.randint(low=2, high=12, size=(batch_size, 1))
    bin_bit = torch.randint(low=0, high=2, size=(batch_size, 2))
    seq = torch.concat((sequence, bin_bit), dim=1)

    logits, cache = model.run_with_cache(sequence)
    initial = cache["blocks.0.mlp.hook_post"].squeeze()[-1]

    final = sae(initial)
    new_log = sae(initial) @ mlp.W_out @ W_U
    old_log = logits.squeeze()[-1]

    optimizer.zero_grad()
    loss = (kldiv(new_log, old_log) + lambda_ * sae.get_intermediate().abs().sum()) / (
        batch_size
    )
    loss.backward()
    optimizer.step()
    print(torch.norm(initial - final) / (batch_size))
# %%
trials = 10


for i in range(trials):
    sequence = torch.randint(low=2, high=12, size=(1,))

    bin_bit = torch.randint(low=0, high=2, size=(2,))
    seq = torch.tensor(list(sequence) + list(bin_bit))

    logits, cache = model.run_with_cache(seq)
    initial = cache["blocks.0.mlp.hook_post"].squeeze()[-1]
    print(initial)
    final = sae(initial)
    show(sae.get_intermediate().reshape((16, 16)))
print(evaluate())


epsilon = 0.1
kl_clusters = []
for epoch in range(epochs):

    sequence = torch.randint(low=2, high=12, size=(batch_size, 1))
    bin_bit = torch.randint(low=0, high=2, size=(batch_size, 2))
    seq = torch.concat((sequence, bin_bit), dim=1)


# %%
import torch
import numpy as np


model = torch.nn.Sequential(
    torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 1)
)


def gen_normal(length, mean, covariance, num_samples):
    return torch.tensor(
        np.random.multivariate_normal(mean=mean, cov=covariance, size=(num_samples))
    )


length = 100
A = torch.rand(10, 10)
covariance = A.T @ A
distance = 0.3
mean = torch.rand(10) * distance
samples = 1000
dist_1 = (
    gen_normal(length, mean.cpu(), covariance.cpu(), samples)
    .to(torch.float32)
    .to("cuda")
)
dist_2 = (
    gen_normal(length, (-mean).cpu(), covariance.cpu(), samples)
    .to(torch.float32)
    .to("cuda")
)
epochs = 20000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32
loss_fn = torch.nn.BCEWithLogitsLoss()
for e in range(epochs):

    optimizer.zero_grad()
    loss = torch.tensor(0.0)
    loss = loss + loss_fn(model(dist_1), torch.ones(samples, 1))

    loss = loss + loss_fn(model(dist_2), torch.zeros(samples, 1))
    loss = loss
    loss.backward()
    print(loss)
    optimizer.step()
# %%
trials = 10
for i in range(trials):
    show(
        torch.stack(
            (
                torch.nn.ReLU()(model[0].weight @ dist_1[i]),
                torch.nn.ReLU()(model[0].weight @ dist_2[i]),
            ),
            dim=0,
        )
    )
# %%
d_mlp = 20
sae_dim = 40
sae = AutoEncoder(d_mlp, sae_dim)
epochs = 100
lambda_ = 1.5
optimizer = torch.optim.AdamW(list(sae.parameters()), lr=5e-4)
for e in range(epochs):
    for i in range(samples):
        if i % 2 == 0:
            initial = model[0].weight @ dist_2[i]
        else:
            initial = model[0].weight @ dist_1[i]

        final = sae(initial)

        optimizer.zero_grad()
        loss = (
            torch.norm(final - initial) + lambda_ * sae.get_intermediate().abs().sum()
        ) / (batch_size)
        loss.backward()
        optimizer.step()
        print(torch.norm(initial - final) / (batch_size))
# %%
for i in range(trials):
    final = sae(model[0].weight @ dist_2[i])
    show(sae.get_intermediate().reshape(8, 5))
    final = sae(model[0].weight @ dist_1[i])
    show(sae.get_intermediate().reshape(8, 5))

# %%
