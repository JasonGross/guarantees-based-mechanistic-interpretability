# %%
import torch
from gbmi.exp_bigram_stats.train import TRIGRAM4, BigramTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
print(TRIGRAM4)
print(BigramTrainingWrapper.build_model(TRIGRAM4).cfg)
_, model = train_or_load_model(TRIGRAM4)  # , force="train")
