from argparse import ArgumentParser

from tqdm.auto import tqdm

from gbmi.exp_multifun.train import MULTIFUN_OF_5_CONFIG, SEEDS, train_or_load_model

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
if args.force == "none":
    args.force = None

with (
    tqdm((True, False), desc="eos", position=0) as eos_pbar,
    tqdm(
        sorted(map(int, args.seeds.split(","))), desc="Seed", position=1, leave=False
    ) as pbar,
    tqdm((2, 1), desc="n_heads", position=2, leave=False) as n_heads_pbar,
):
    for eos in eos_pbar:
        for seed in pbar:
            for n_heads in n_heads_pbar:
                cfg = MULTIFUN_OF_5_CONFIG(
                    seed, use_end_of_sequence=eos, n_heads=n_heads, deterministic=False
                )
                pbar.set_postfix(
                    {"seed": seed, "eos": eos, "n_heads": n_heads, "cfg": cfg}
                )
                runtime, model = train_or_load_model(cfg, force=args.force)
