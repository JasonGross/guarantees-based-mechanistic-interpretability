# %%
import torch
from gbmi.exp_indhead.train import TRIGRAM4, main
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
cfg = set_params(
    TRIGRAM4,
    {
        ("experiment", "seq_length"): 7,
    },
    post_init=True,
)

print(cfg)
print(main.build_model(cfg).cfg)
_, model = main(default=cfg)  # , default_force="train")
