# %%
import torch
from gbmi.exp_bigram_stats.train import TRIGRAM4, BigramTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
cfg = set_params(
    TRIGRAM4,
    {
        ("experiment", "seq_length"): 6,
    },
)

print(cfg)
print(BigramTrainingWrapper.build_model(cfg).cfg)
_, model = train_or_load_model(cfg)  # , force="train")
