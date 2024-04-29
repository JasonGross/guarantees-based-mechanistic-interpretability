# %%
import torch
import numpy as np
from gbmi.exp_indhead.train import ABCAB8_1H, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
torch.manual_seed(123)
for seed in torch.randint(0, 2**32 - 1, (20,)):
    for num_tokens in range(5, 27):
        for task in ("abcab", "exact-ngram"):
            for ngram in (2, 3):
                for alpha_mix_uniform in np.linspace(start=0, stop=1, num=4):
                    cfg = set_params(
                        ABCAB8_1H,
                        {
                            "seed": seed.item(),
                            "train_for": (500, "epochs"),
                            "validate_every": (10, "epochs"),
                            ("experiment", "summary_slug_extra"): "sweep",
                            ("experiment", "num_tokens"): num_tokens,
                            ("experiment", "ngram"): ngram,
                            ("experiment", "alpha_mix_uniform"): alpha_mix_uniform,
                            ("experiment", "task"): task,
                        },
                        post_init=True,
                    )

                    print(cfg)
                    print(IndHeadTrainingWrapper.build_model(cfg).cfg)
                    _, model = train_or_load_model(cfg, force="train")

# _, model = main(default=ABCAB8_1H, default_force="train")
