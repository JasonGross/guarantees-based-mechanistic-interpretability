# %%
import torch
from gbmi.exp_indhead.train import TRIGRAM4, main
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
print(TRIGRAM4)
print(main.build_model(TRIGRAM4).cfg)
_, model = main(default=TRIGRAM4)  # , default_force="train")
