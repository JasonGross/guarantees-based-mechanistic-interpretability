# %%
from gbmi.exp_indhead.train import ABCAB7_1H, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model

# %%
print(ABCAB7_1H)
print(IndHeadTrainingWrapper.build_model(ABCAB7_1H).cfg)
_, model = train_or_load_model(ABCAB7_1H, force="train")
