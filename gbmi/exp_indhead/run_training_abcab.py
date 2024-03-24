# %%
from gbmi.exp_indhead.train import ABCAB, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model

# %%
print(ABCAB)
print(IndHeadTrainingWrapper.build_model(ABCAB).cfg)
_, model = train_or_load_model(ABCAB, force="train")
