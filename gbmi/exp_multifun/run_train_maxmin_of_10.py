from argparse import ArgumentParser, BooleanOptionalAction

from tqdm.auto import tqdm

from gbmi.exp_multifun.train import MULTIFUN_OF_10_CONFIG, SEEDS, train_or_load_model

parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default=",".join(sorted(map(str, SEEDS))),
    help="Comma-separated list of seeds to use",
)
args = parser.parse_args()

for d_vocab in tqdm((64, 128), desc="d_vocab"):
    with (
        tqdm((True, False), desc="eos", position=0) as eos_pbar,
        tqdm(
            sorted(map(int, args.seeds.split(","))),
            desc="Seed",
            position=1,
            leave=False,
        ) as pbar,
        tqdm((2, 1), desc="n_heads", position=2, leave=False) as n_heads_pbar,
    ):
        for eos in eos_pbar:
            for seed in pbar:
                for n_heads in n_heads_pbar:
                    cfg = MULTIFUN_OF_10_CONFIG(
                        seed,
                        d_vocab=d_vocab,
                        n_heads=n_heads,
                        use_end_of_sequence=eos,
                        deterministic=False,
                    )
                    pbar.set_postfix(
                        {
                            "seed": seed,
                            "d_vocab": d_vocab,
                            "eos": eos,
                            "n_heads": n_heads,
                            "cfg": cfg,
                        }
                    )
                    runtime, model = train_or_load_model(cfg)  # , force="train"
