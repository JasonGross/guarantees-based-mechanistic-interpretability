# %%
from gbmi.exp_indhead.train import (
    ABCAB6_SMALL_HIDDEN_1H,
    IndHeadTrainingWrapper,
)
from gbmi.model import train_or_load_model

# %%
print(ABCAB6_SMALL_HIDDEN_1H)
print(IndHeadTrainingWrapper.build_model(ABCAB6_SMALL_HIDDEN_1H).cfg)
_, model = train_or_load_model(ABCAB6_SMALL_HIDDEN_1H, force="train")
