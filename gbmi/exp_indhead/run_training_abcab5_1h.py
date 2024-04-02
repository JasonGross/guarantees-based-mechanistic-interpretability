# %%
from gbmi.exp_indhead.train import ABCAB5_1H, main

# %%
_, model = main(default=ABCAB5_1H, default_force="train")
