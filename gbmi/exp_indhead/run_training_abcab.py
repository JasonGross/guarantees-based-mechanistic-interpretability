# %%
from gbmi.exp_indhead.train import ABCAB, main

# %%
_, model = main(default=ABCAB, default_force="train")
