# %%
from tqdm.auto import tqdm

from gbmi.exp_modular_arithmetic import SEEDS
from gbmi.exp_modular_arithmetic.train import (
    CLOCK_CONFIG,
    PIZZA_CONFIG,
    train_or_load_model,
)
from gbmi.utils import set_params

with (
    tqdm(total=len(SEEDS), desc="Seed", position=0, leave=True) as seed_pbar,
    tqdm(total=2, desc="p", position=1, leave=True) as p_pbar,
    tqdm(total=2, position=2, leave=True) as cfg_pbar,
    tqdm(total=2, desc="use eos", position=3, leave=True) as eos_pbar,
):
    for seed in SEEDS:
        seed_pbar.update(1)
        seed_pbar.set_postfix({"seed": seed})
        p_pbar.reset()
        for p in (7, 12):
            p_pbar.update(1)
            p_pbar.set_postfix({"p": p})
            cfg_pbar.reset()
            for cfg in (PIZZA_CONFIG, CLOCK_CONFIG):
                cfg_pbar.update(1)
                cfg_pbar.set_postfix({"cfg": cfg})
                eos_pbar.reset()
                for use_eos in (True, False):
                    eos_pbar.update(1)
                    eos_pbar.set_postfix({"use_eos": use_eos})
                    runtime, model = train_or_load_model(
                        set_params(
                            cfg,
                            {
                                "seed": seed,
                                "train_for": (3000, "epochs"),
                                ("experiment", "training_ratio"): 0.9,
                                ("experiment", "p"): p,
                                (
                                    "experiment",
                                    "use_end_of_sequence",
                                ): use_eos,
                            },
                            post_init=True,
                        ),
                        # force="train",
                        # force="load",
                    )
    eos_pbar.close()
    cfg_pbar.close()
    p_pbar.close()
    seed_pbar.close()

# %%
import shutil
from pathlib import Path

import torch

base = Path(".").resolve()
wandbs = list((base / "artifacts").glob("*/ModularAdd-7-*.pth")) + list(
    (base / "artifacts").glob("*/ModularAdd-12-*.pth")
)
model_base = base / "models"
model_base.mkdir(exist_ok=True, parents=True)

with tqdm(wandbs) as pbar:
    for path in pbar:
        cache = torch.load(path, map_location="cpu")
        if cache["run_config"]["experiment"]["p"] not in (7, 12):
            continue
        pbar.set_postfix(
            {
                "seed": cache["run_config"]["seed"],
                "orig_name": path.name,
                "suffix_drop": "-".join(path.name.split("-")[-6:]),
            }
        )
        seed = cache["run_config"]["seed"]
        shutil.copy(
            path,
            model_base / f"{'-'.join(path.name.split('-')[:-6])}-{seed}{path.suffix}",
        )
    # break
# %%
# gtar --transform='s|.*/||' --owner=0 --group=0 --numeric-owner -czf modular-add-7,12-pizza-no-eos-partial.tar.gz models/ModularAdd-{7,12}*attention-rate-1*no-eos*.pth
# gtar --transform='s|.*/||' --owner=0 --group=0 --numeric-owner -czf modular-add-7,12-pizza-partial.tar.gz $(ls models/ModularAdd-{7,12}*attention-rate-1*.pth | grep -v no-eos)
