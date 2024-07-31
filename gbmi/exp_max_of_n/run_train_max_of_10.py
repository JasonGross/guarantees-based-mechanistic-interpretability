from tqdm.auto import tqdm
from gbmi.exp_max_of_n.train import train_or_load_model, SEEDS, MAX_OF_4_CONFIG
from gbmi.utils import set_params

for d_vocab_out in tqdm((64, 128), desc="d_vocab"):
    with tqdm(SEEDS, desc="Seed", leave=False) as pbar:
        for seed in pbar:
            cfg = set_params(
                MAX_OF_4_CONFIG(seed),
                {
                    ("deterministic",): False,
                    ("experiment", "seq_len"): 10,
                    ("experiment", "d_vocab_out"): d_vocab_out,
                },
                post_init=True,
            )
            pbar.set_postfix({"seed": seed, "d_vocab": d_vocab_out, "cfg": cfg})
            runtime, model = train_or_load_model(cfg)  # , force="train"
