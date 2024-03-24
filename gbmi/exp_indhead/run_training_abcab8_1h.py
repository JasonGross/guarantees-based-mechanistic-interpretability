# %%
from gbmi.exp_indhead.train import ABCAB8_1H, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model

# %%
print(ABCAB8_1H)
print(IndHeadTrainingWrapper.build_model(ABCAB8_1H).cfg)
_, model = train_or_load_model(ABCAB8_1H, force="train")
