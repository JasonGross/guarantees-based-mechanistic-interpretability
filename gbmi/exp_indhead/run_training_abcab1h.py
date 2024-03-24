# %%
from gbmi.exp_indhead.train import ABCAB_1H, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model

# %%
print(ABCAB_1H)
print(IndHeadTrainingWrapper.build_model(ABCAB_1H).cfg)
_, model = train_or_load_model(ABCAB_1H, force="train")
