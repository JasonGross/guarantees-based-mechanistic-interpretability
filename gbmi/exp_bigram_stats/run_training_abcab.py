# %%
from gbmi.exp_bigram_stats.train import ABCAB_BIGRAM, BigramTrainingWrapper
from gbmi.model import train_or_load_model

# %%
print(ABCAB_BIGRAM)
print(BigramTrainingWrapper.build_model(ABCAB_BIGRAM).cfg)
_, model = train_or_load_model(ABCAB_BIGRAM, force="train")
