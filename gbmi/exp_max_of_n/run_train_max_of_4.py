from tqdm.auto import tqdm
from gbmi.exp_max_of_n.train import train_or_load_model, SEEDS, MAX_OF_4_CONFIG

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix_str(f"Seed {seed}")
        runtime, model = train_or_load_model(MAX_OF_4_CONFIG(seed))  # , force="train"
