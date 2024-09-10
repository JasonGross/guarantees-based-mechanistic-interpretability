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
    tqdm(SEEDS, desc="Seed", position=0, leave=False) as seed_pbar,
    tqdm((7, 12), desc="p", position=1, leave=False) as p_pbar,
    tqdm((PIZZA_CONFIG, CLOCK_CONFIG), position=2, leave=False) as cfg_pbar,
    tqdm((True, False), desc="use eos", position=3, leave=False) as eos_pbar,
):
    for seed in seed_pbar:
        for p in p_pbar:
            for cfg in cfg_pbar:
                for use_eos in eos_pbar:
                    p_pbar.set_postfix({"p": p})
                    cfg_pbar.set_postfix({"cfg": cfg})
                    eos_pbar.set_postfix({"use_eos": use_eos})
                    seed_pbar.set_postfix({"seed": seed})
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
                        )
                    )  # , force="train")
