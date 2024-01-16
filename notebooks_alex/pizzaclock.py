from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_CLOCK_CONFIG
from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_PIZZA_CONFIG
from gbmi.model import train_or_load_model
import torch
from math import *
from torch import tensor
import einops
import tqdm

device = "cuda"
p = 113
q = p
freeze_model = False
config = MODULAR_ADDITION_113_PIZZA_CONFIG
subtracting = True
frac_train = 0.3
seed = 999
num_epochs = 25000

rundata, model = train_or_load_model(config)
model.to(device)
if freeze_model:
    for param in model.parameters():
        param.requires_grad = False
    embeddingmatrix = model.embed.W_E.clone()

    model.embed = torch.nn.Identity()


def hook_fn(attnpattern, hook):
    return (
        torch.ones((3, 3)).to(device)
        * (torch.tensor(config.experiment.attention_rate, dtype=torch.float).to(device))
        + (
            1
            - torch.tensor(config.experiment.attention_rate, dtype=torch.float).to(
                device
            )
        )
        * attnpattern
    )


class DifferentModClock(torch.nn.Module):
    def __init__(self):
        super(DifferentModClock, self).__init__()
        self.W_e = torch.nn.Parameter(
            -0.5 / sqrt(p + 1) + torch.rand((q + 1, p + 1)) / sqrt(p + 1)
        )
        self.W_u = torch.nn.Parameter(
            -0.5 / sqrt(p) + torch.rand((p // 2 + 1 if subtracting else q, p)) / sqrt(p)
        )

    def forward(self, x):
        z = torch.nn.functional.one_hot(x).to(torch.float).to(device)

        y = z @ self.W_e @ embeddingmatrix
        if len(y.shape) != 3:
            y = y.unsqueeze(0)
        return self.W_u @ (
            einops.rearrange(
                model.run_with_hooks(
                    y, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
                ),
                "b d p -> b p d",
            )
        )


def loss_fn(logits, labels):
    print(logits.shape)
    if len(logits.shape) == 3:
        if freeze_model:
            logits = logits[:, :, -1]
        else:
            logits = logits[:, -1, :]
    logits = logits.to(torch.float64)

    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]

    return -correct_log_probs.mean()


if freeze_model:
    Clock = DifferentModClock()
    Clock.to(device)
full_model = Clock if freeze_model else model

a_vector = einops.repeat(torch.arange(q), "i -> (i j)", j=q)
b_vector = einops.repeat(torch.arange(q), "j -> (i j)", i=q)
equals_vector = einops.repeat(torch.tensor(q), " -> (i j)", i=q, j=q)
dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)


if subtracting:
    subtractedset = (dataset[:, 0] - dataset[:, 1]) % q
    if freeze_model:
        labels = (subtractedset - ((subtractedset > (q // 2)) * subtractedset) * 2) % q
    else:
        labels = subtractedset  # Finds either a-b or b-a depending on which one is lower than q//2. Symmetric in a and b.
else:
    labels = (dataset[:, 0] + dataset[:, 1]) % q
optimizer = torch.optim.AdamW(
    full_model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98)
)

torch.manual_seed(seed)
indices = torch.randperm(q * q)
cutoff = int(q * q * frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]
train_data = dataset[train_indices]
train_labels = labels[train_indices]

test_data = dataset[test_indices]
test_labels = labels[test_indices]

train_losses = []
test_losses = []
model_checkpoints = []
checkpoint_epochs = []


for epoch in tqdm.tqdm(range(num_epochs)):
    if freeze_model:
        train_logits = full_model(train_data)
    else:
        train_logits = full_model.run_with_hooks(
            train_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
        )
    train_loss = loss_fn(train_logits, train_labels)
    train_loss.backward()
    train_losses.append(train_loss.item())

    optimizer.step()
    optimizer.zero_grad()

    with torch.inference_mode():
        if freeze_model:
            test_logits = full_model(test_data)
        else:
            test_logits = full_model.run_with_hooks(
                test_data, fwd_hooks=[("blocks.0.attn.hook_pattern", hook_fn)]
            )
        test_loss = loss_fn(test_logits, test_labels)
        test_losses.append(test_loss.item())
    if ((epoch + 1) % 10) == 0:
        checkpoint_epochs.append(epoch)
        print(
            f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}"
        )
