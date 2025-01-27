# %%
from tqdm.auto import tqdm

from gbmi.exp_modular_arithmetic import SEEDS
from gbmi.exp_modular_arithmetic.train import CLOCK_CONFIG, train_or_load_model
from gbmi.training_tools.logging import ModelMatrixLoggingOptions
from gbmi.utils import set_params

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix({"seed": seed})
        runtime, model = train_or_load_model(
            set_params(
                CLOCK_CONFIG,
                {
                    "seed": seed,
                    "train_for": (10000, "epochs"),
                    ("experiment", "logging_options"): ModelMatrixLoggingOptions.none(),
                },
                post_init=True,
            ),
            # force="load",
            # force="train",
        )


# %%
import shutil
from pathlib import Path

import torch

base = Path(".").resolve()
wandbs = (base / "artifacts").glob("*/*.pth")
total = len(list((base / "artifacts").glob("*/*.pth")))
model_base = base / "models"
model_base.mkdir(exist_ok=True, parents=True)
# %%
with tqdm(wandbs, total=total) as pbar:
    for p in pbar:
        total -= 1
        cache = torch.load(p, map_location="cpu")
        pbar.set_postfix(
            {
                "seed": cache["run_config"]["seed"],
                "orig_name": p.name,
                "suffix_drop": "-".join(p.name.split("-")[-6:]),
            }
        )
        seed = cache["run_config"]["seed"]
        shutil.copy(
            p,
            model_base / f"{'-'.join(p.name.split('-')[:-6])}-{seed}{p.suffix}",
        )
    # break
# %%
