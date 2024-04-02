# %%
from gbmi.exp_indhead.train import ABCAB7_1H, main
from gbmi.model import train_or_load_model

# %%
print(ABCAB7_1H)
print(main.build_model(ABCAB7_1H).cfg)
_, model = main(default=ABCAB7_1H, default_force="train")
