from tqdm.auto import tqdm
from gbmi.exp_modular_arithmetic.train import train_or_load_model, PIZZA_CONFIG
from gbmi.exp_modular_arithmetic import SEEDS
from gbmi.utils import set_params

runtime, model = train_or_load_model(PIZZA_CONFIG, force="train")


with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix({"seed": seed})
        runtime, model = train_or_load_model(
            set_params(
                PIZZA_CONFIG,
                {"seed": seed},
                post_init=True,
            )
        )  # , force="train"
