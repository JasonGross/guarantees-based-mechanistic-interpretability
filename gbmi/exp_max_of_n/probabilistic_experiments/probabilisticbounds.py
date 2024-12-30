# %%
import torch
from gbmi.exp_max_of_n.train import MAX_OF_4_CONFIG
import plotly.express as px
from gbmi.model import train_or_load_model

rundata, model = train_or_load_model(MAX_OF_4_CONFIG(123))


torch.set_default_device("cuda")
length = model.cfg.n_ctx
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
o = W_O_0
v = W_V_0
at = W_E @ W_Q_0 @ W_K_0.T @ W_E.T + W_pos[length - 1] @ W_Q_0 @ W_K_0 @ W_E.T


def show(matrix):
    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)
    px.imshow(matrix.detach().cpu()).show()


# g_(x_(i)) = torch.exp((PQKE))
