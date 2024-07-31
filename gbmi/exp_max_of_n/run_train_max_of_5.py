from tqdm.auto import tqdm
from gbmi.exp_max_of_n.train import train_or_load_model, SEEDS, MAX_OF_4_CONFIG
from gbmi.utils import set_params

with tqdm(SEEDS, desc="Seed") as pbar:
    for seed in pbar:
        cfg = set_params(
            MAX_OF_4_CONFIG(seed),
            {("deterministic",): False, ("experiment", "seq_len"): 5},
            post_init=True,
        )
        pbar.set_postfix({"seed": seed, "cfg": cfg})
        runtime, model = train_or_load_model(cfg)  # , force="train"