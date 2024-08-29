from math import *

import einops
import pandas as pd
import plotly.express as px
import torch
from torch import tensor
from tqdm.auto import tqdm

from gbmi import utils
from gbmi.exp_indhead.train import ABCAB8_1H
from gbmi.model import train_or_load_model
from gbmi.utils.sequences import generate_all_sequences

device = "cuda"


class check_induction:
    def __init__(self, model):
        self.model = model
        self.p = self.model.W_pos
        self.e = self.model.W_E
        self.q = self.model.blocks[0].attn.W_Q.squeeze(dim=0)
        self.k = self.model.blocks[0].attn.W_K.squeeze(dim=0)
        self.q_1 = self.model.blocks[1].attn.W_Q.squeeze(dim=0)
        self.k_1 = self.model.blocks[1].attn.W_K.squeeze(dim=0)
        self.O = self.model.blocks[0].attn.W_O.squeeze(dim=0)
        self.V = self.model.blocks[0].attn.W_V.squeeze(dim=0)
        self.O_1 = self.model.blocks[1].attn.W_O.squeeze(dim=0)
        self.V_1 = self.model.blocks[1].attn.W_V.squeeze(dim=0)
        self.e_p = self.e.unsqueeze(dim=0) + self.p.unsqueeze(dim=1)
        self.d_voc = self.e.shape[0]
        self.n_ctx = self.p.shape[0]

    def off_diagonal_att(self):
        print(self.e_p.shape)
        print(self.q.shape)

        # tensor[d_vocab,d_vocab,n_ctx] stores minimum attention paid to thing before it

        att = einops.einsum(
            self.e_p, self.q, self.k, self.e_p, "i j k, k l, m l, n o m -> i j n o"
        )
        out = []
        for i in range(1, self.n_ctx):
            run = []
            for j in range(i - 2):
                diff = att[i, :, i - 1, :] - att[i, :, j, :]
                run.append(diff.amin(dim=(0, 1)))
            diff = att[i, :, i - 1, :] - att[i, :, i, :]
            run.append(diff.amin(dim=(0, 1)))
            run = torch.tensor(run)
            out.append(run.min())
        out = torch.tensor(out)
        return out

    def diagonal_att(self):
        ov = einops.einsum(
            self.e_p,
            self.q_1,
            self.k_1,
            self.O,
            self.V,
            self.e_p,
            "i j k, k l, m l, n m, o n, p q o -> i j p q",
        )
        out = []
        for i in range(self.n_ctx):
            for j in range(i):
                curr = ov[i, :, j, :]
                diff = curr.diag().unsqueeze(dim=-1) - curr
                diff.diagonal().fill_(float("inf"))
                run = diff.min(dim=-1).values.unsqueeze(0)
                out.append(run)
        return torch.cat(out, dim=0).min(dim=0).values

    # def diagonal_val(self):


runtime_model_1, model_1 = train_or_load_model(ABCAB8_1H, force="load")
a = check_induction(model_1)
print(a.off_diagonal_att())
print(a.diagonal_att())
