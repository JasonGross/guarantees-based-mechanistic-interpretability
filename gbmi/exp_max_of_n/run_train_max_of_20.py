from tqdm.auto import tqdm
from gbmi.exp_max_of_n.train import train_or_load_model, SEEDS, MAX_OF_20_CONFIG

for d_vocab_out in tqdm((64, 512), desc="d_vocab"):
    with tqdm(SEEDS, desc="Seed", leave=False) as pbar:
        for seed in pbar:
            cfg = MAX_OF_20_CONFIG(seed, d_vocab_out=d_vocab_out, deterministic=False)
            pbar.set_postfix({"seed": seed, "d_vocab": d_vocab_out, "cfg": cfg})
            runtime, model = train_or_load_model(cfg)  # , force="train"
