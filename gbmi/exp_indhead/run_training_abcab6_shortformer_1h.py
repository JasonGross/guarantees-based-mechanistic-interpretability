# %%
from gbmi.exp_indhead.train import ABCAB6_SHORTFORMER_1H, main
from gbmi.model import train_or_load_model

# %%
print(ABCAB6_SHORTFORMER_1H)
print(main.build_model(ABCAB6_SHORTFORMER_1H).cfg)
_, model = main(default=ABCAB6_SHORTFORMER_1H, default_force="train")
