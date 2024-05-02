# %%
import torch
import numpy as np
from gbmi.exp_indhead.train import ABCABC8, IndHeadTrainingWrapper
from gbmi.model import train_or_load_model
from gbmi.utils import set_params

# %%
torch.manual_seed(123)
for seed in torch.randint(0, 2**32 - 1, (20,)):
    for n_train_samples in (4860, 48600):
        for num_tokens in (26, 6, 16):
            for ngram in (3, 2):
                for alpha_mix_uniform in np.linspace(start=0, stop=1, num=4):
                    cfg = set_params(
                        ABCABC8,
                        {
                            "seed": seed.item(),
                            "train_for": (500, "epochs"),
                            "validate_every": (50, "epochs"),
                            "batch_size": n_train_samples,
                            ("experiment", "summary_slug_extra"): "sweep",
                            ("experiment", "ngram"): ngram,
                            ("experiment", "alpha_mix_uniform"): alpha_mix_uniform,
                            ("experiment", "n_train_samples"): n_train_samples,
                            ("experiment", "num_tokens"): num_tokens,
                        },
                        post_init=True,
                    )

                    print(cfg)
                    print(IndHeadTrainingWrapper.build_model(cfg).cfg)
                    _, model = train_or_load_model(cfg, force="train")

    # _, model = main(default=ABCAB8_1H, default_force="train")
