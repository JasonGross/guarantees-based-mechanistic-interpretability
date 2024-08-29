from tqdm.auto import tqdm

from gbmi.exp_max_of_n.train import MAX_OF_10_CONFIG, SEEDS, train_or_load_model

for d_vocab_out in tqdm((64, 128), desc="d_vocab"):
    with tqdm(SEEDS, desc="Seed", leave=False) as pbar:
        for seed in pbar:
            cfg = MAX_OF_10_CONFIG(seed, d_vocab_out=d_vocab_out, deterministic=False)
            pbar.set_postfix({"seed": seed, "d_vocab": d_vocab_out, "cfg": cfg})
            runtime, model = train_or_load_model(cfg)  # , force="train"
