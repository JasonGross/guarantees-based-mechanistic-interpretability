from argparse import ArgumentParser, BooleanOptionalAction

from tqdm.auto import tqdm

from gbmi.exp_argmax_of_n.train import ARGMAX_OF_20_CONFIG, SEEDS, train_or_load_model

parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default=",".join(sorted(map(str, SEEDS))),
    help="Comma-separated list of seeds to use",
)
args = parser.parse_args()

for d_vocab in tqdm((64, 512), desc="d_vocab"):
    with tqdm(map(int, args.seeds.split(",")), desc="Seed", leave=False) as pbar:
        for seed in pbar:
            cfg = ARGMAX_OF_20_CONFIG(seed, d_vocab=d_vocab, deterministic=False)
            pbar.set_postfix({"seed": seed, "d_vocab": d_vocab, "cfg": cfg})
            runtime, model = train_or_load_model(cfg)  # , force="train"
