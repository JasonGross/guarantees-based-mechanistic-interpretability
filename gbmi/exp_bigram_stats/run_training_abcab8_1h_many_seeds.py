# %%
import torch
from gbmi.exp_bigram_stats.train import ABCAB8_BIGRAM1H, BigramTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
torch.manual_seed(123)
for seed in torch.randint(0, 2**32 - 1, (20,)):
    cfg = set_params(
        ABCAB8_BIGRAM1H,
        {
            "seed": seed.item(),
            "train_for": (20000, "epochs"),
            "validate_every": (100, "epochs"),
            ("experiment", "summary_slug_extra"): "manseed",
        },
    )

    print(cfg)
    print(BigramTrainingWrapper.build_model(cfg).cfg)
    _, model = train_or_load_model(cfg)  # , force="train")
