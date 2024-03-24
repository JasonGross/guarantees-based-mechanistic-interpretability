# %%
import torch
from gbmi.exp_indhead.train import TRIGRAM4, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
cfg = set_params(
    TRIGRAM4,
    {
        ("experiment", "seq_length"): 5,
    },
)

print(cfg)
print(IndHeadTrainingWrapper.build_model(cfg).cfg)
_, model = train_or_load_model(cfg)  # , force="train")
