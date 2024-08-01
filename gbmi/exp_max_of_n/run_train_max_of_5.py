from tqdm.auto import tqdm
from gbmi.exp_max_of_n.train import train_or_load_model, SEEDS, MAX_OF_5_CONFIG

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        cfg = MAX_OF_5_CONFIG(seed, deterministic=False)
        pbar.set_postfix({"seed": seed, "cfg": cfg})
        runtime, model = train_or_load_model(cfg)  # , force="train"
