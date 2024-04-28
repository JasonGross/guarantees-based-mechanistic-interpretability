# %%
import torch
from gbmi.exp_indhead.train import ABCAB5_1H, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
torch.manual_seed(123)
for seed in torch.randint(0, 2**32 - 1, (20,)):
    cfg = set_params(
        ABCAB5_1H,
        {
            "seed": seed.item(),
            "train_for": (20000, "epochs"),
            "validate_every": (100, "epochs"),
            ("experiment", "summary_slug_extra"): "manseed",
        },
        post_init=True,
    )

    print(cfg)
    print(IndHeadTrainingWrapper.build_model(cfg).cfg)
    _, model = train_or_load_model(cfg)  # , force="train")
