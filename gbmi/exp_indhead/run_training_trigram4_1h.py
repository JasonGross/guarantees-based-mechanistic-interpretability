# %%
from gbmi.exp_indhead.train import TRIGRAM4, main

# %%
print(TRIGRAM4)
print(main.build_model(TRIGRAM4).cfg)
_, model = main(default=TRIGRAM4)  # , default_force="train")
