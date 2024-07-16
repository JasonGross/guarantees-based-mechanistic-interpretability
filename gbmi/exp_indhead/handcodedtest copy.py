# %%
from gbmi.exp_indhead.train import ABCAB8_1H
from torch import where
from gbmi.model import train_or_load_model
import torch
from torch import tensor
from math import *
import plotly.express as px
from gbmi.utils.sequences import generate_all_sequences

from inspect import signature

import plotly.express as px


def show(matrix):
    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
runtime_model_1, model = train_or_load_model(ABCAB8_1H, force="load")
model.to(device)

W_pos = model.W_pos
W_E = model.W_E
n_ctx = W_pos.shape[0]
d_voc = W_E.shape[0]
d_model = W_E.shape[1]


# %%
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E
W_K_1 = model.W_K[1, 0]
W_U = model.W_U
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


attn_matrix = PQKP + EQKP.unsqueeze(1)

typical_softmaxes = []
for b_position in range(8):

    typical_attn_scores = attn_matrix[:, b_position, : (b_position + 1)]

    PQKE_tolerances = PQKE[b_position, :]

    typical_attn_scores[:, b_position] = (
        typical_attn_scores[:, b_position] + PQKE_tolerances + EQKE.diag()[:]
    )

    EQKE_max = torch.max(EQKE, dim=1)
    EQKE_min = torch.min(EQKE, dim=1)

    index_to_max_min = b_position - 1

    min_attn_scores = torch.zeros(typical_attn_scores.shape)
    max_attn_scores = torch.zeros(typical_attn_scores.shape)

    for index in range(b_position):
        if index == index_to_max_min:
            min_attn_scores[:, index] = EQKE_min[0][:] + typical_attn_scores[:, index]
            max_attn_scores[:, index] = EQKE_max[0][:] + typical_attn_scores[:, index]
        else:
            min_attn_scores[:, index] = EQKE_max[0][:] + typical_attn_scores[:, index]
            max_attn_scores[:, index] = EQKE_min[0][:] + typical_attn_scores[:, index]

    min_attn_scores[:, b_position] = typical_attn_scores[:, b_position]
    max_attn_scores[:, b_position] = typical_attn_scores[:, b_position]

    minimum_softmax = min_attn_scores.softmax(dim=-1)

    maximum_softmax = max_attn_scores.softmax(dim=-1)

    typical_softmax = typical_attn_scores.softmax(dim=-1)

    typical_softmaxes.append(typical_softmax)

# %%

# %%


# Specify E values
evo_pos = (W_E @ v @ o) @ q_1 @ k_1.T @ (W_pos.T) / (attn_scale_1)

# Specify E values including b

evo_e = (W_E @ v @ o) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)
e_ove = (W_E) @ q_1 @ k_1.T @ (o.T @ v.T @ W_E.T) / (attn_scale_1)  # NEED LOOKING AT


pos_ove = (W_pos @ q_1 @ k_1.T @ o.T @ v.T @ W_E.T) / (
    attn_scale_1
)  # Needs attention from b as according to typical_softmaxes[6]


posvo_pos = (typical_softmaxes[7] @ W_pos @ v @ o @ q_1 @ k_1.T @ W_pos.T) / (
    attn_scale_1
)  # Due to positional info


pos_pos = (W_pos @ q_1 @ k_1.T @ W_pos.T) / (attn_scale_1)  # 0.1 max
e_e = (W_E) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)

evo_ove = (W_E @ v @ o) @ q_1 @ k_1.T @ (o.T @ v.T @ W_E.T) / (attn_scale_1)

# %%
index = 5
evo_ovpos = (
    (W_E @ v @ o)
    @ q_1
    @ k_1.T
    @ (o.T @ v.T @ W_pos[: (index + 1)].T @ typical_softmaxes[index].T)
    / (attn_scale_1)
)
posvo_ovpos = (
    typical_softmaxes[7]
    @ W_pos
    @ v
    @ o
    @ q_1
    @ k_1.T
    @ o.T
    @ v.T
    @ W_pos[: (index + 1)].T
    @ typical_softmaxes[index].T
) / (attn_scale_1)
posvo_ove = (typical_softmaxes[7] @ W_pos @ v @ o @ q_1 @ k_1.T @ o.T @ v.T @ W_E.T) / (
    attn_scale_1
)
pos_ovpos = (
    W_pos
    @ q_1
    @ k_1.T
    @ o.T
    @ v.T
    @ W_pos[: (index + 1)].T
    @ typical_softmaxes[index].T
) / (  # attention paid to b due to positional
    attn_scale_1
)

for a in range(10, 12):
    for b in range(10, 12):
        posvo_e = (typical_softmaxes[7] @ W_pos @ v @ o @ q_1 @ k_1.T @ W_E[b].T) / (
            attn_scale_1
        )
        posvo_e = posvo_e.mean()
        e_ovpos = (
            (W_E[a])
            @ q_1
            @ k_1.T
            @ (o.T @ v.T @ W_pos[: (index + 1)].T @ typical_softmaxes[index].T)
            / (attn_scale_1)
        ).mean(dim=0)
        e_pos = (W_E[a]) @ q_1 @ k_1.T @ (W_pos[index].T) / (attn_scale_1)

        e_e = (W_E[a]) @ q_1 @ k_1.T @ (W_E[b].T) / (attn_scale_1)
        pos_e = (W_pos[7] @ q_1 @ k_1.T @ W_E[b].T) / (attn_scale_1)

        b_pattern = e_ove[a] + pos_ove[7] + posvo_ove.mean(dim=0)
        print(b_pattern)
        show(b_pattern.unsqueeze(0))

        a_pattern = evo_e[:, b] + evo_pos[:, index] + evo_ovpos.mean(dim=-1)
        show(a_pattern.unsqueeze(0))

        pos_pattern = (
            posvo_pos[a, index]
            + posvo_ovpos[a][b]
            + pos_ovpos[7, b]
            + pos_pos[7][index]
        )

        print(pos_pattern + pos_e + e_e + e_pos + e_ovpos + posvo_e + posvo_e)


# %%
# ------------------------


# ------------------------------------------

# -------------------------------


# Tell me a


# ------------------------

# Needs attention from b as according to typical_softmaxes[6].


# NEED LOOKING AT

# ------------------------------


# Tell me what b is and done


# -------------------------------------------


PVOU = W_pos @ W_V_1 @ W_O_1 @ W_U
PVOVOU = W_pos @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOVOU = W_E @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOU = W_E @ W_V_1 @ W_O_1 @ W_U
# %%
px.imshow(PVOU.detach().cpu())
px.imshow(PVOVOU.detach().cpu()).show()

px.imshow(EVOVOU.detach().cpu())
px.imshow(EVOU.detach().cpu())


# %%
def compute_worst_case_scenario_pos(M, b, attn_on_b):

    attn_to_others = torch.tensor(M.max(dim=0).values) * (1.0 - attn_on_b)
    attn_to_others[b] = attn_on_b * M.min(dim=0).values[b]
    return attn_to_others


def compute_worst_case_scenario_evou(b, attn_on_b):
    attn_to_others = torch.tensor(EVOU.max(dim=0).values) * (1.0 - attn_on_b)
    attn_to_others[b] = attn_on_b * EVOU[b][b]

    return attn_to_others


attn_to_b = 0.65
pre_softmax = (
    compute_worst_case_scenario_pos(PVOU, 1, attn_to_b)
    + compute_worst_case_scenario_pos(PVOVOU, 1, attn_to_b)
    + compute_worst_case_scenario_pos(EVOVOU, 1, attn_to_b)
    + compute_worst_case_scenario_evou(1, attn_to_b)
)
print(pre_softmax.softmax(dim=0))


# %%
