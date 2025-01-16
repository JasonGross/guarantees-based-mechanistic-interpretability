from argparse import ArgumentParser, BooleanOptionalAction

from tqdm.auto import tqdm

from gbmi.exp_argmax_of_n.train import ARGMAX_OF_5_CONFIG, SEEDS, train_or_load_model

parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default=",".join(sorted(map(str, SEEDS))),
    help="Comma-separated list of seeds to use",
)
parser.add_argument(
    "--force",
    choices=["train", "load", "none"],
    default="train",
    help="Force training or loading",
)
args = parser.parse_args()

with tqdm(map(int, args.seeds.split(",")), desc="Seed") as pbar:
    for seed in pbar:
        cfg = ARGMAX_OF_5_CONFIG(seed, deterministic=False)
        pbar.set_postfix({"seed": seed, "cfg": cfg})
        runtime, model = train_or_load_model(cfg, force=args.force)
