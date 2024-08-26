# %%
from gbmi.exp_indhead.train import ABCAB8_1H
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


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
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
minimum_softmaxes = []
maximum_softmaxes = []
for b_position in range(8):

    typical_attn_scores = attn_matrix[:, b_position, : (b_position + 1)]

    PQKE_tolerances = PQKE[b_position, :]

    typical_attn_scores[:, b_position] = (
        typical_attn_scores[:, b_position] + PQKE_tolerances + EQKE.diag()[:]
    )

    EQKE_max = torch.max(EQKE + PQKE[b_position].unsqueeze(0).T, dim=1)
    print(EQKE_max)
    EQKE_min = torch.min(EQKE + PQKE[b_position].unsqueeze(0).T, dim=1)

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
    minimum_softmaxes.append(minimum_softmax)
    maximum_softmaxes.append(maximum_softmax)
# %%
print(minimum_softmaxes)
# %%

evo_pos = (W_E @ v @ o) @ q_1 @ k_1.T @ (W_pos.T) / (attn_scale_1)

evo_e = (W_E @ v @ o) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)
e_ove = (W_E) @ q_1 @ k_1.T @ (o.T @ v.T @ W_E.T) / (attn_scale_1)  # NEED LOOKING AT

pos_ove = (W_pos @ q_1 @ k_1.T @ o.T @ v.T @ W_E.T) / (attn_scale_1)

posvo_pos = (typical_softmaxes[7] @ W_pos @ v @ o @ q_1 @ k_1.T @ W_pos.T) / (
    attn_scale_1
)

pos_pos = (W_pos @ q_1 @ k_1.T @ W_pos.T) / (attn_scale_1)  # 0.1 max
e_e = (W_E) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)

evo_ove = (W_E @ v @ o) @ q_1 @ k_1.T @ (o.T @ v.T @ W_E.T) / (attn_scale_1)


# %%
def compute_worst_case_scenario_pos(M, b, attn_on_b):

    attn_to_others = torch.tensor(M.max(dim=0).values) * (maximum_other_attn)
    s = set(attn_to_others.topk(8)[1].tolist())
    attn_to_others = torch.zeros(attn_to_others.shape)

    for i in range(len(attn_to_others)):
        if i in s:

            attn_to_others[i] = torch.tensor(M.max(dim=0).values)[i] * (
                maximum_other_attn
            )
        else:

            attn_to_others[i] = torch.tensor(M[:, i].topk(2, dim=0)[0][-1]) * (
                maximum_other_attn
            )

    attn_to_others[b] = attn_on_b * M.min(dim=0).values[b]

    return attn_to_others


def compute_worst_case_scenario_evou(M, b, attn_on_b):
    attn_to_others = torch.tensor(M.max(dim=0).values) * (maximum_other_attn)
    s = set(attn_to_others.topk(7)[1].tolist())
    attn_to_others = torch.zeros(attn_to_others.shape)

    for i in range(len(attn_to_others)):
        if i in s:

            attn_to_others[i] = torch.tensor(M.max(dim=0).values)[i] * (
                maximum_other_attn
            )
        else:

            attn_to_others[i] = torch.tensor(M[:, i].topk(2, dim=0)[0][-1]) * (
                maximum_other_attn
            )
    attn_to_others[b] = attn_on_b * M[b][b]

    return attn_to_others


PVOU = W_pos @ W_V_1 @ W_O_1 @ W_U
PVOVOU = W_pos @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOVOU = W_E @ W_V_0 @ W_O_0 @ W_V_1 @ W_O_1 @ W_U
EVOU = W_E @ W_V_1 @ W_O_1 @ W_U
# %%
bounds = torch.zeros(d_voc, d_voc)
for a in range(1, 25):
    t = [i[a].tolist() + [0 for i in range(8 - len(i[a]))] for i in typical_softmaxes]
    t = torch.tensor(t)
    full_computed = t @ PVOVOU + PVOU

    for b in range(1, a):
        super_pattern = []
        b_patterns = []
        a_patterns = []
        partialscore = torch.zeros(7)
        for index in range(7):
            believed_attention = a if index != 6 else b  # Which attention is being
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
                @ q_1  ##
                @ k_1.T
                @ o.T
                @ v.T
                @ W_pos[: (index + 1)].T
                @ typical_softmaxes[index].T
            ) / (attn_scale_1)
            posvo_ove = (
                typical_softmaxes[7]
                @ W_pos
                @ v
                @ o
                @ q_1
                @ k_1.T
                @ o.T
                @ v.T
                @ W_E.T  ###
            ) / (attn_scale_1)
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
            posvo_e = (
                typical_softmaxes[7][a]
                @ W_pos
                @ v
                @ o
                @ q_1
                @ k_1.T
                @ W_E[believed_attention].T
            ) / (attn_scale_1)

            e_ovpos = (
                (W_E[a])
                @ q_1
                @ k_1.T
                @ (o.T @ v.T @ W_pos[: (index + 1)].T @ typical_softmaxes[index].T)
                / (attn_scale_1)
            )[believed_attention]

            e_pos = (W_E[a]) @ q_1 @ k_1.T @ (W_pos[index].T) / (attn_scale_1)

            e_e = (
                (W_E[a]) @ q_1 @ k_1.T @ (W_E[believed_attention].T) / (attn_scale_1)
            )  # 0.2?                   ## KEY THING BECAUSE IT CHANGED BASED ON THE ATTENTION
            print(e_e)
            pos_e = (W_pos[7] @ q_1 @ k_1.T @ W_E[believed_attention].T) / (
                attn_scale_1
            )

            b_pattern = e_ove[a] + pos_ove[7] + posvo_ove[a]
            b_patterns.append(b_pattern)
            a_pattern = (
                evo_e[:, believed_attention]  # ALSO KIND OF KEY
                + evo_pos[:, index]
                + evo_ovpos[:, believed_attention]
            )
            a_patterns.append(a_pattern)
            pos_pattern = (
                posvo_pos[a, index]
                + posvo_ovpos[a, believed_attention]
                + pos_ovpos[7, believed_attention]
                + pos_pos[7, index]
            )

            partialscore[index] = pos_pattern + pos_e + e_e + e_pos + e_ovpos + posvo_e

            super_pattern.append(
                a_pattern.unsqueeze(0).T + b_pattern.unsqueeze(0) + evo_ove
            )

        indexdiffs = []
        for index in range(6):

            if index != 0:
                worst = (
                    super_pattern[6][:, a] * typical_softmaxes[6][b][-2]
                ).unsqueeze(0).T - super_pattern[index] * typical_softmaxes[index][a][
                    -2
                ]
            else:
                worst = (
                    super_pattern[6][:, a] * typical_softmaxes[6][b][-2]
                ).unsqueeze(0).T - super_pattern[index] * typical_softmaxes[index][a][0]

            worst_removed = torch.concat((worst[:, :a], worst[:, (a + 1) :]), dim=1)
            worst_removed_col = worst_removed.min(dim=1)[0]

            worst_difference = 0
            sequence2 = []
            for i in range(5):
                if i == (index - 1) or i == 5:
                    continue

                if i < index:
                    competing_pattern = (
                        typical_softmaxes[index][a][i] * super_pattern[index]
                    )
                else:
                    competing_pattern = 0.0 * super_pattern[index]

                difference = (
                    typical_softmaxes[6][b][i] * super_pattern[6] - competing_pattern
                )

                worst_cols = (
                    #  4 * torch.min(difference, dim=1)[0]
                    +5 * difference.diag()
                    + 2 * difference[a, :]
                    + difference[b, :]
                ) / 8

                worst_cols = worst_cols + worst_removed_col / 8

                worst_cols[a] = torch.inf
                worst_difference += worst_cols.min(dim=0)[0]

            worst_index_1 = torch.min(torch.min(worst_removed, dim=0)[0], dim=0)[1]
            worst_index = worst_index_1 if worst_index_1 < a else worst_index_1 - 1

            totdiff = (
                (
                    worst_removed_col[a] * 2
                    + worst_removed_col[b]
                    + worst_removed_col[worst_index_1]
                )
                / 8
                + partialscore[6]
                - partialscore[index]
                + worst_difference
            )

            indexdiffs.append(-totdiff.item())
        vals = torch.tensor(indexdiffs + [0, indexdiffs[0]])

        valnos = torch.tensor(indexdiffs + [indexdiffs[0]])
        attn_to_b = torch.softmax(vals, dim=0)[-2]

        maximum_other_attn = torch.exp(torch.max(valnos))

        pre_softmax = compute_worst_case_scenario_pos(
            EVOVOU, b, attn_to_b
        ) + compute_worst_case_scenario_evou(EVOU + full_computed[-1], b, attn_to_b)

        bounds[a][b] = pre_softmax.softmax(dim=0)[b]
    print(a)
# %%

t = [i[a].tolist() + [0 for i in range(8 - len(i[a]))] for i in typical_softmaxes]
t = torch.tensor(t)
typical_6 = torch.tensor(typical_softmaxes[6][b].tolist() + [0.0])

typical_4 = torch.tensor(typical_softmaxes[4][a].tolist() + [0.0, 0.0, 0.0])

typical_5 = torch.tensor(typical_softmaxes[5][a].tolist() + [0.0, 0.0])
e_e = (W_E) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)
pos_e = (W_pos[7] @ q_1 @ k_1.T @ W_E.T) / (attn_scale_1)
posvo_e = (typical_softmaxes[7][b] @ W_pos @ v @ o @ q_1 @ k_1.T @ W_E.T) / (
    attn_scale_1
)
total = 0
for i in range(6):
    totmatrix = (
        typical_6.unsqueeze(0).T @ b_patterns[6].unsqueeze(0)
        + typical_softmaxes[7][b].unsqueeze(0).T @ a_patterns[4].unsqueeze(0)
        - typical_softmaxes[7][a].unsqueeze(0).T @ (a_patterns[i]).unsqueeze(0)
        - t[i].unsqueeze(0).T @ b_patterns[i].unsqueeze(0)
    )
    show(totmatrix)

    totmatrix[:, a] = torch.inf
    minima = torch.min(totmatrix, dim=1)
    fullsequence = torch.tensor(minima[1][:5].tolist() + [a, b, a])

    for j in range(7):
        total = total + totmatrix[j, fullsequence[j]]
    print(total + partialscore[6] - partialscore[i])
    logits, cache = model.run_with_cache(fullsequence)
    print(cache["attn_scores", 1].squeeze()[-1])
    print(fullsequence)


# show(
#   typical_6.unsqueeze(0).T @ b_patterns[6].unsqueeze(0)
#  + typical_softmaxes[7][b].unsqueeze(0).T @ a_patterns[6].unsqueeze(0)
# - typical_softmaxes[7][a].unsqueeze(0).T @ a_patterns[5].unsqueeze(0)
#  - typical_5.unsqueeze(0).T @ b_patterns[5].unsqueeze(0)
# )


# %%
