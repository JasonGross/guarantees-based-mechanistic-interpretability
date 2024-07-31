from tqdm.auto import tqdm
from gbmi.exp_max_of_n.train import train_or_load_model, SEEDS, MAX_OF_4_CONFIG
from gbmi.utils import set_params

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix_str(f"Seed {seed}")
        runtime, model = train_or_load_model(
            set_params(
                MAX_OF_4_CONFIG(seed), {("experiment", "seq_len"): 5}, post_init=True
            )
        )  # , force="train"
