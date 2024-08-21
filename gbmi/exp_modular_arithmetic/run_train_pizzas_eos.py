# %%
from tqdm.auto import tqdm
from gbmi.exp_modular_arithmetic.train import train_or_load_model, PIZZA_CONFIG
from gbmi.exp_modular_arithmetic import SEEDS
from gbmi.utils import set_params

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix({"seed": seed})
        runtime, model = train_or_load_model(
            set_params(
                PIZZA_CONFIG,
                {"seed": seed, ("experiment", "use_end_of_sequence"): True},
                post_init=True,
            )
        )  # , force="train"

# # %%
# from pathlib import Path
# import torch
# import shutil

# base = Path(__file__).parent
# wandbs = (base / "artifacts").glob("*/*.pth")
# model_base = base / "models"
# model_base.mkdir(exist_ok=True, parents=True)
# # %%
# with tqdm(wandbs) as pbar:
#     for p in pbar:
#         cache = torch.load(p, map_location="cpu")
#         pbar.set_postfix(
#             {
#                 "seed": cache["run_config"]["seed"],
#                 "orig_name": p.name,
#                 "suffix_drop": "-".join(p.name.split("-")[-6:]),
#             }
#         )
#         shutil.copy(
#             p,
#             model_base
#             / f"{'-'.join(p.name.split('-')[:-6])}-{cache['run_config']['seed']}{p.suffix}",
#         )
#     # break
# # %%
