# %%
import copy
from inspect import signature
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
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cuda")
runtime_model_1, model = train_or_load_model(ABCAB8_1HMLP, force="load")
model.to(device)
c = 10
d = 10
W_pos = model.W_pos
W_E = model.W_E
epsilon = 0.5
n_ctx = W_pos.shape[0]
d_voc = W_E.shape[0]
d_model = W_E.shape[1]


def noise(M):
    return epsilon * (torch.rand_like(M) - 0.5)


def add_noise(*ms):
    for m in ms:
        m += noise(m.shape)


# %%
W_E = ein.array(lambda i, j: i == j, sizes=[d_voc, d_model]).float().to(device) + noise(
    W_E
)
# %%
W_pos = (
    ein.array(lambda i, j: ((i + d_voc) == j) * 1.0, sizes=[n_ctx, d_model])
    .float()
    .to(device)
) + noise(W_pos)

# %%


W_O_0 = model.W_O[0, 0]
W_O_0 = (
    ein.array(lambda i, j: ((i + n_ctx + d_voc) == j) * 1.0, sizes=[d_voc, d_model])
    .float()
    .to(device)
)
W_O_0 = W_O_0 + noise(W_O_0)
W_V_0 = model.W_V[0, 0]
W_V_0 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
W_V_0 = W_V_0 + noise(W_V_0)
W_V_1 = model.W_V[1, 0]
W_V_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_model, d_voc]).float().to(device)
)
W_V_1 = W_V_1 + noise(W_V_1)
W_O_1 = model.W_O[1, 0]
W_O_1 = (
    ein.array(lambda i, j: (i == j) * 1.0, sizes=[d_voc, d_model]).float().to(device)
)
W_O_1 = W_O_1 + noise(W_O_1)
W_Q_0 = model.W_Q[0, 0]
W_Q_0 = (
    ein.array(lambda i, j: where((i + d_voc + 1) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .to(device)
    .T
)
W_Q_0 = W_Q_0 + noise(W_Q_0)

W_Q_1 = (
    ein.array(lambda i, j: where(i == j, d, 0), sizes=[d_voc, d_model])
    .float()
    .T.to(device)
)
W_Q_1 = W_Q_1 + noise(W_Q_1)

W_K_0 = (
    ein.array(lambda i, j: where((i + d_voc) == j, c, 0), sizes=[n_ctx, d_model])
    .float()
    .T
).to(device)
W_K_0 = W_K_0 + noise(W_K_0)

W_K_1 = (
    ein.array(
        lambda i, j: where((i + n_ctx + d_voc) == j, d, 0), sizes=[d_voc, d_model]
    )
    .float()
    .T
).to(device)
W_K_1 = W_K_1 + noise(W_K_1)
o = W_O_0
v = W_V_0
q_1 = W_Q_1
k_1 = W_K_1
W_U = ein.array(lambda i, j: i == j, sizes=[d_model, d_voc]).float().to(device)
W_U = W_U + noise(W_U)
attn_scale_0 = model.blocks[0].attn.attn_scale
attn_scale_1 = model.blocks[1].attn.attn_scale
# %%
# px.imshow((W_pos @ W_Q_0 @ W_K_0.T @ W_pos.T).cpu())
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

for i in range(n_ctx):
    for j in range(n_ctx):
        if j > i:
            PQKP[i][j] = -torch.inf

attn_matrix = PQKP + EQKP.unsqueeze(1)
print(attn_matrix.shape)
typical_softmaxes = []
minimum_softmaxes = []
maximum_softmaxes = []


for b_position in range(1, n_ctx):
    minimum_softmaxes.append([])
    maximum_softmaxes.append([])
    for index_to_minmax in range(b_position):
        minimum_softmaxes[b_position - 1].append([])
        maximum_softmaxes[b_position - 1].append([])
        for b in range(d_voc):
            min_attn_scores = []
            max_attn_scores = []
            for index in range(n_ctx):
                min_attn_scores.append(PQKP[b_position, index] + EQKP[b, index])
                max_attn_scores.append(PQKP[b_position, index] + EQKP[b, index])

            minimum = torch.min(PQKE[b_position, :] + EQKE[b, :])
            maximum = torch.max(PQKE[b_position, :] + EQKE[b:])
            for index in range(n_ctx):
                if index == b_position:
                    min_attn_scores[index] += PQKE[b_position, b] + EQKE[b, b]
                    max_attn_scores[index] += PQKE[b_position, b] + EQKE[b, b]
                elif index == index_to_minmax:
                    min_attn_scores[index] += minimum
                    max_attn_scores[index] += maximum
                else:
                    min_attn_scores[index] += maximum
                    max_attn_scores[index] += minimum
            print(minimum)
            minimum_softmaxes[b_position - 1][index_to_minmax].append(
                torch.tensor(min_attn_scores[: (b_position + 1)]).softmax(dim=0)
            )
            maximum_softmaxes[b_position - 1][index_to_minmax].append(
                torch.tensor(max_attn_scores[: (b_position + 1)]).softmax(dim=0)
            )
# %%
# %%
typical_grads = []
for b_position in range(n_ctx):

    typical_attn_scores = attn_matrix[:, b_position, : b_position + 1]

    PQKE_tolerances = PQKE[b_position, :]

    typical_attn_scores[:, b_position] = (
        typical_attn_scores[:, b_position] + PQKE_tolerances + EQKE.diag()[:]
    )
    typical_attn_scores[:, :b_position] = typical_attn_scores[
        :, :b_position
    ]  # + PQKE_tolerances.mean(dim=-1)
    typical_attn_scores[:, (b_position + 1) :] = typical_attn_scores[
        :, (b_position + 1) :
    ]  # + PQKE_tolerances.mean(dim=-1)
    EQKE_max = torch.max(EQKE + PQKE[b_position].unsqueeze(0).T, dim=1)
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

    typical_grad = (
        typical_softmax.T / (torch.sum(torch.exp(typical_attn_scores), dim=-1))
    ).T
    typical_softmaxes.append(typical_softmax)
    typical_grads.append(typical_grad)
    minimum_softmaxes.append(minimum_softmax)
    maximum_softmaxes.append(maximum_softmax)


# %%
def soft_grad(original):  #    (b,h,n)
    return torch.diag(original) - torch.outer(original, original)


# %%
def get_full_score(pattern, sequence, partial, index):
    logits, cache = model.run_with_cache(torch.tensor(sequence))
    attn = cache["attn", 0].squeeze()
    f = torch.nn.functional.one_hot(sequence, 26).to(torch.float)
    show(attn[-1] @ f)
    return attn[-1] @ f @ pattern[index] @ f.T @ attn[index].T + partial[index]


def get_evo_ove(sequence, index):
    logits, cache = model.run_with_cache(torch.tensor(sequence))
    attn = cache["attn", 0].squeeze()
    f = torch.nn.functional.one_hot(sequence, 26).to(torch.float)
    show(attn[-1] @ f)
    return attn[-1] @ f @ evo_ove @ f.T @ attn[index].T


# %%

evo_pos = (W_E @ v @ o) @ q_1 @ k_1.T @ (W_pos.T) / (attn_scale_1)

evo_e = (W_E @ v @ o) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)
e_ove = (W_E) @ q_1 @ k_1.T @ (o.T @ v.T @ W_E.T) / (attn_scale_1)  # NEED LOOKING AT

pos_ove = (W_pos @ q_1 @ k_1.T @ o.T @ v.T @ W_E.T) / (attn_scale_1)

posvo_pos = (typical_softmaxes[n_ctx - 1] @ W_pos @ v @ o @ q_1 @ k_1.T @ W_pos.T) / (
    attn_scale_1
)

pos_pos = (W_pos @ q_1 @ k_1.T @ W_pos.T) / (attn_scale_1)  # 0.1 max
e_e = (W_E) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)

evo_ove = (W_E @ v @ o) @ q_1 @ k_1.T @ (o.T @ v.T @ W_E.T) / (attn_scale_1)


# %%


def compute_worst_case_scenario_pos(M, a, b, attn_on_b):

    attn_to_others = torch.tensor(M.max(dim=0).values) * (maximum_other_attn)
    s = set(attn_to_others.topk(n_ctx)[1].tolist())
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
    s = set(attn_to_others.topk(n_ctx)[1].tolist())

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


bounds = torch.zeros(d_voc, d_voc)
for a in range(0, 26):

    t = [
        i[a].tolist() + [0 for i in range(n_ctx - len(i[a]))] for i in typical_softmaxes
    ]
    t = torch.tensor(t)
    full_computed = t @ PVOVOU + PVOU

    super_pattern = []
    b_patterns = []
    a_patterns = []
    partialscore = torch.zeros(n_ctx - 1)
    for index in range(n_ctx - 1):
        believed_attention = (
            a if index != (n_ctx - 2) else a
        )  # Which attention is being
        evo_ovpos = (
            (W_E @ v @ o)
            @ q_1
            @ k_1.T
            @ (o.T @ v.T @ W_pos[: (index + 1)].T @ typical_softmaxes[index].T)
            / (attn_scale_1)
        )
        posvo_ovpos = (
            typical_softmaxes[n_ctx - 1]
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
            typical_softmaxes[n_ctx - 1]
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
            typical_softmaxes[n_ctx - 1][a]
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

        pos_e = (W_pos[n_ctx - 1] @ q_1 @ k_1.T @ W_E[believed_attention].T) / (
            attn_scale_1
        )

        b_pattern = e_ove[a] + pos_ove[n_ctx - 1] + posvo_ove[a]
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
            + pos_ovpos[n_ctx - 1, believed_attention]
            + pos_pos[n_ctx - 1, index]
        )

        partialscore[index] = pos_pattern + pos_e + e_e + e_pos + e_ovpos + posvo_e

        super_pattern.append(
            a_pattern.unsqueeze(0).T + b_pattern.unsqueeze(0) + evo_ove
        )

    indexdiffs = []
    t = torch.tensor(
        [
            i[a].tolist() + [0 for i in range(n_ctx - len(i[a]))]
            for i in typical_softmaxes
        ]
    )
    typical_6 = torch.tensor(typical_softmaxes[n_ctx - 2][b].tolist() + [0.0])

    e_e = (W_E) @ q_1 @ k_1.T @ (W_E.T) / (attn_scale_1)
    pos_e = (W_pos[n_ctx - 1] @ q_1 @ k_1.T @ W_E.T) / (attn_scale_1)
    posvo_e = (
        typical_softmaxes[n_ctx - 1][a] @ W_pos @ v @ o @ q_1 @ k_1.T @ W_E.T
    ) / (attn_scale_1)
    badindices = [25]
    for i in range(n_ctx - 3):
        e_ovpos = (
            (W_E[a])
            @ q_1
            @ k_1.T
            @ (o.T @ v.T @ W_pos[: (i + 1)].T @ typical_softmaxes[i].T)
            / (attn_scale_1)
        )

        totmatrix = (typical_6 - t[i]).unsqueeze(0).T @ b_patterns[n_ctx - 2].unsqueeze(
            0
        ) + typical_softmaxes[n_ctx - 1][a].unsqueeze(0).T @ (
            evo_pos[:, n_ctx - 2] - evo_pos[:, i]
        ).unsqueeze(
            0
        )

        totmatrix[i, :] = totmatrix[i, :] + (
            posvo_e[a]
            - posvo_e[:]
            + pos_e[a]
            + e_e[a, a]
            - e_e[a, :]
            - pos_e[:]
            + e_ovpos[a]
            - e_ovpos[:]
            + posvo_ovpos[a, a]
            - posvo_ovpos[a, :]
            + pos_ovpos[n_ctx - 1, a]
            - pos_ovpos[n_ctx - 1, :]
        )

        totmatrix[:, a] = torch.inf
        for index in badindices:
            totmatrix[:, index] = torch.inf
        minima = torch.min(totmatrix, dim=1)

        fullsequence = torch.tensor(minima[1][: (n_ctx - 3)].tolist() + [a, b, a])

        t = torch.tensor(
            [
                m[fullsequence[i]].tolist()
                + [0 for s in range(n_ctx - len(m[fullsequence[i]]))]
                for m in typical_softmaxes
            ]
        )
        totmatrix = (
            typical_6.unsqueeze(0).T @ b_patterns[n_ctx - 2].unsqueeze(0)
            + typical_softmaxes[n_ctx - 1][a].unsqueeze(0).T
            @ a_patterns[n_ctx - 2].unsqueeze(0)
            - typical_softmaxes[n_ctx - 1][a].unsqueeze(0).T
            @ (
                a_patterns[i]
                - evo_e[:, a]
                + evo_e[:, fullsequence[i]]
                - evo_ovpos[:, a]
                + evo_ovpos[:, fullsequence[i]]
            ).unsqueeze(0)
            - t[i].unsqueeze(0).T @ b_patterns[i].unsqueeze(0)
        )
        totmatrix[i, :] = totmatrix[i, :] + (
            posvo_e[a]
            - posvo_e[:]
            + pos_e[a]
            + e_e[a, a]
            - e_e[a, :]
            - pos_e[:]
            + e_ovpos[a]
            - e_ovpos[:]
            + posvo_ovpos[a, a]
            - posvo_ovpos[a, :]
            + pos_ovpos[n_ctx - 1, a]
            - pos_ovpos[n_ctx - 1, :]
        )
        total = totmatrix[n_ctx - 3, a] + totmatrix[n_ctx - 1, a]
        for j in range(n_ctx - 3):
            total = total + totmatrix[j, fullsequence[j]]

        answer = total + partialscore[n_ctx - 2] - partialscore[i]

        indexdiffs.append(-answer)

    indexdiffs = torch.tensor(indexdiffs + [0.0, indexdiffs[0]])

    valnos = torch.tensor(indexdiffs)
    valnos[6] = -torch.inf
    for b in range(26):
        attn_to_b = torch.softmax(indexdiffs, dim=0)[-2]

        maximum_other_attn = 1 - attn_to_b
        attns = torch.exp(valnos) * attn_to_b
        print(attn_to_b)
        pre_softmax = compute_worst_case_scenario_pos(
            EVOVOU, a, b, attn_to_b
        ) + compute_worst_case_scenario_evou(
            EVOU + full_computed[-1],
            b,
            attn_to_b,
        )

        bounds[a][b] = pre_softmax.softmax(dim=0)[b]
    print(a)
# %%
#   typical_6.unsqueeze(0).T @ b_patterns[6].unsqueeze(0)
#  + typical_softmaxes[7][b].unsqueeze(0).T @ a_patterns[6].unsqueeze(0)
# - typical_softmaxes[7][a].unsqueeze(0).T @ a_patterns[5].unsqueeze(0)
#  - typical_5.unsqueeze(0).T @ b_patterns[5].unsqueeze(0)
# )
#
#
#
#
""" fullchange = (posvo_e.unsqueeze(0).T- posvo_e.unsqueeze(0)
                + pos_e.unsqueeze(0).T
                + e_e.diag().unsqueeze(0)
                - e_e
                - pos_e[:].unsqueeze(0)
                + e_ovpos.unsqueeze(0).T
                - e_ovpos.unsqueeze(0)
               + posvo_ovpos.diag().unsqueeze(0) - posvo_ovpos
                + pos_ovpos[7].unsqueeze(0).T
                - pos_ovpos[7].unsqueeze(0))
"""
# %%
