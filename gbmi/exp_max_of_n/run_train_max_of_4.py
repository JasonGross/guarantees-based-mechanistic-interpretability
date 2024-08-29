from tqdm.auto import tqdm

from gbmi.exp_max_of_n.train import MAX_OF_4_CONFIG, SEEDS, train_or_load_model

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix({"seed": seed})
        runtime, model = train_or_load_model(MAX_OF_4_CONFIG(seed))  # , force="train"
