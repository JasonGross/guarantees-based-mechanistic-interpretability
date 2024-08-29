from math import sqrt

import einops
import torch
import tqdm
import transformer_lens
import transformer_lens.utils as utils
from torch import tensor

from gbmi.exp_group_finetuning.groups import (
    CyclicGroup,
    DihedralGroup,
    Group,
    GroupDict,
    PermutedCyclicGroup,
)
from gbmi.exp_group_finetuning.train import (
    MODULAR_ADDITION_113_CLOCK_CONFIG_EPOCH_500,
    modular_addition_config,
)
from gbmi.model import train_or_load_model


def loss_fn(logits, labels, softmax=True):
    if softmax:
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]

        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]

        return -correct_log_probs.mean()
    else:
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        logits = logits.to(torch.float64)
        return torch.linalg.vector_norm(logits - labels) / len(logits)


device = "cuda"
config = MODULAR_ADDITION_113_CLOCK_CONFIG_EPOCH_500
p = config.experiment.group_index
q = p


rundata, model = train_or_load_model(
    modular_addition_config(
        attn_rate=0,
        group=CyclicGroup(101),
        elements=2,
        epochs=25000,
        weight_decay=1.0,
    )
)

model.to(device)

dataset = torch.tensor([[1, 31, q]]).to(device)
labels = torch.tensor([q // 2]).to(device)
checkdata = torch.tensor([[2, 4, q]]).to(device)
checklabel = torch.tensor([6]).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
checkloss = torch.tensor(0)
maxcheckloss = torch.tensor(0)
maxcheckloss.requires_grad = False
checkloss.requires_grad = False
model.embed.W_E.requires_grad = False
model.unembed.W_U.requires_grad = False
differlengths = []
for epoch in range(1000):
    mincheckloss = []
    minindices = []
    differ = []
    if epoch > 30:
        for l in range(1, p, 19):
            mincheckloss.append([])
            minindices.append([])
            for s in range(1, p, 19):
                minindices[l // 19].append(0)
                mincheckloss[l // 19].append(5)
                checkdata = torch.tensor([[l, s, q]]).to(device)
                for k in range(1, p):
                    checkloss = loss_fn(model(checkdata), torch.tensor([k]).to(device))
                    if checkloss < mincheckloss[l // 19][s // 19]:
                        minindices[l // 19][s // 19] = k
                        mincheckloss[l // 19][s // 19] = checkloss
                print(l, s, minindices[l // 19][s // 19], (l + s) % 100)
                if 2 < abs((l + s) % 100 - minindices[l // 19][s // 19]):
                    differ.append((l, s, minindices[l // 19][s // 19]))
    loss = loss_fn(model(dataset), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    differlengths.append(len(differ))

    print(loss, minindices, len(differ))
print(differlengths)
