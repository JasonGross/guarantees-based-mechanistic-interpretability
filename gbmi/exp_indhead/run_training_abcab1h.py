# %%
from gbmi.exp_indhead.train import ABCAB_1H, main

# %%
_, model = main(default=ABCAB_1H, default_force="train")
