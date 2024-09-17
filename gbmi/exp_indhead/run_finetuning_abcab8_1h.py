# %%
from gbmi.exp_indhead.finetune import main, make_default_finetune
from gbmi.exp_indhead.train import ABCAB8_1H

# %%
_, model = main(default=make_default_finetune(ABCAB8_1H), default_force="train")
