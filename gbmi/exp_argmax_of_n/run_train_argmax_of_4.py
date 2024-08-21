from argparse import ArgumentParser, BooleanOptionalAction

from tqdm.auto import tqdm

from gbmi.exp_argmax_of_n.train import ARGMAX_OF_4_CONFIG, SEEDS, train_or_load_model

parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default=",".join(sorted(map(str, SEEDS))),
    help="Comma-separated list of seeds to use",
)
args = parser.parse_args()

with tqdm(map(int, args.seeds.split(",")), desc="Seed") as pbar:
    for seed in pbar:
        pbar.set_postfix({"seed": seed})
        runtime, model = train_or_load_model(
            ARGMAX_OF_4_CONFIG(seed)
        )  # , force="train"
