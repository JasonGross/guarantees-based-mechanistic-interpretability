# %%
import torch
from gbmi.exp_indhead.train import TRIGRAM4, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
print(TRIGRAM4)
print(IndHeadTrainingWrapper.build_model(TRIGRAM4).cfg)
_, model = train_or_load_model(TRIGRAM4)  # , force="train")
