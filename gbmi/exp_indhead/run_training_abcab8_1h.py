# %%
from gbmi.exp_indhead.train import ABCAB8_1H, main

# %%
_, model = main(default=ABCAB8_1H, default_force="train")
