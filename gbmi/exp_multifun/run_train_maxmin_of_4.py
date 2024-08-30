from argparse import ArgumentParser, BooleanOptionalAction

from tqdm.auto import tqdm

from gbmi.exp_multifun.train import MULTIFUN_OF_4_CONFIG, SEEDS, train_or_load_model

parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default=",".join(sorted(map(str, SEEDS))),
    help="Comma-separated list of seeds to use",
)
args = parser.parse_args()

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
                cfg = MULTIFUN_OF_4_CONFIG(
                    seed,
                    use_end_of_sequence=eos,
                    n_heads=n_heads,
                )
                pbar.set_postfix(
                    {"seed": seed, "eos": eos, "n_heads": n_heads, "cfg": cfg}
                )
                runtime, model = train_or_load_model(cfg)  # , force="train"
