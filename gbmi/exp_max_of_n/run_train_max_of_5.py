from tqdm.auto import tqdm

from gbmi.exp_max_of_n.train import MAX_OF_5_CONFIG, SEEDS, train_or_load_model

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        cfg = MAX_OF_5_CONFIG(seed, deterministic=False)
        pbar.set_postfix({"seed": seed, "cfg": cfg})
        runtime, model = train_or_load_model(cfg)  # , force="train"
