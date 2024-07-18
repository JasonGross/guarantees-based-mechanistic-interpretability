# %%
from __future__ import annotations
import argparse
import sys
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from functools import cache, partial

from wandb.sdk.wandb_run import Run
from gbmi.utils.hashing import _EXCLUDE
from gbmi.training_tools.logging import ModelMatrixLoggingOptions, log_tensor
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Literal,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
import torch
from jaxtyping import Float, Integer, Bool
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

from gbmi.model import (
    TrainingWrapper,
    Config,
    ExperimentConfig,
    add_HookedTransformerConfig_arguments,
    train_or_load_model,
    DataModule,
    add_force_argument,
    add_no_save_argument,
    update_HookedTransformerConfig_from_args,
)
import gbmi.utils as utils
from gbmi.utils import (
    shuffle_data,
    SingleTensorDataset,
    TupleCollectionDataset,
    reseed,
    set_params,
    ein,
)
from gbmi.utils.sequences import generate_all_sequences

SEEDS = tuple(
    sorted(
        set(
            map(
                int,
                "50,104,123,519,742,913,1185,1283,1412,1490,1681,1696,1895,1951,2236,2306,2345,2549,2743,2773,3175,3254,3284,4157,4305,4430,4647,4729,4800,4810,5358,5615,5781,5928,6082,6155,6159,6204,6532,6549,6589,6910,7098,7238,7310,7467,7790,7884,8048,8299,8721,8745,8840,8893,9132,9134,9504,9816,10248,11124,11130,11498,11598,11611,12141,12287,12457,12493,12552,12561,13036,13293,13468,13654,13716,14095,14929,15043,15399,15622,15662,16069,16149,16197,16284,17080,17096,17194,17197,18146,18289,18668,19004,19093,19451,19488,19538,19917,20013,20294,20338,20415,20539,20751,20754,20976,21317,21598,22261,22286,22401,22545,23241,23367,23447,23633,23696,24144,24173,24202,24262,24438,24566,25516,26278,26374,26829,26932,27300,27484,27584,27671,27714,28090,28716,28778,29022,29052,29110,29195,29565,29725,29726,30371,30463,30684,30899,31308,32103,32374,32382".split(
                    ","
                ),
            )
        )
    )
)

SELECTED_SEED = 123


@dataclass
class IterableDatasetCfg:
    n_samples: Optional[int] = None
    pick_max_first: bool = False


@dataclass
class FullDatasetCfg:
    force_adjacent: Sequence[int] = tuple()
    # only for n_ctx=2: for all i in force_adjacent, force all sequences (n, n±i) to be in training set
    # bounds: Optional[Tuple[int, int]] = None
    # range of vocab tokens within which to sample
    training_ratio: float = 0.7


DatasetCfg = Union[IterableDatasetCfg, FullDatasetCfg]


@dataclass
class MaxOfN(ExperimentConfig):
    # Model config
    model_config: HookedTransformerConfig = field(
        default_factory=lambda: HookedTransformerConfig(
            n_layers=1,
            n_heads=1,
            d_model=32,
            d_head=32,
            d_vocab=64,
            attn_only=True,
            normalization_type=None,
            n_ctx=2,
        )
    )
    zero_biases: bool = True
    use_log1p: bool = False
    # TODO(Euan, from Jason): Should this go in DatasetCfg?  In some shared dataset cfg?
    use_end_of_sequence: bool = False  # if set, is final token
    seq_len: int = 64
    nth_max: int = 1
    use_kaiming_init: bool = False
    summary_slug_extra: str = ""
    log_matrix_on_run_batch_prefixes: set[
        Optional[Literal["", "periodic_test_", "test_"]]
    ] = field(default_factory=lambda: {"periodic_test_", "test_"})
    logging_options: ModelMatrixLoggingOptions = field(
        default_factory=ModelMatrixLoggingOptions
    )

    train_dataset_cfg: DatasetCfg = field(
        default_factory=lambda: IterableDatasetCfg(n_samples=None)
    )
    test_dataset_cfg: DatasetCfg = field(
        default_factory=lambda: IterableDatasetCfg(n_samples=1024)
    )
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999)}
    )
    optimizer: Literal["Adam", "AdamW", "SGD"] = "Adam"

    def __post_init__(self):
        self.model_config.n_ctx = self.seq_len
        self.logging_options.qpos = -1
        self.model_config.init_weights = not self.use_kaiming_init
        if self.use_end_of_sequence:
            self.model_config.n_ctx = self.seq_len + 1
            self.model_config.d_vocab = self.model_config.d_vocab_out + 1
            self.logging_options.qtok = -1
        exclude = ("logging_options", "log_matrix_on_run_batch_prefixes")
        if not self.use_kaiming_init:
            exclude += ("use_kaiming_init",)
        setattr(self, _EXCLUDE, exclude)
        self.model_config.__post_init__()
        self.logging_options.__post_init__()

    def config_post_init(self, config: Config[MaxOfN]) -> None:
        self.model_config.seed = reseed(config.seed, "model")

    def get_eos_token(self) -> Optional[int]:
        return self.model_config.d_vocab - 1 if self.use_end_of_sequence else None

    def get_training_wrapper(self):
        return MaxOfNTrainingWrapper

    def get_datamodule(self):
        return MaxOfNDataModule

    def get_summary_slug(self, config: Config[MaxOfN]) -> str:
        n_layers = config.experiment.model_config.n_layers
        if isinstance(config.experiment.train_dataset_cfg, FullDatasetCfg):
            force_adjacent = ",".join(
                map(str, config.experiment.train_dataset_cfg.force_adjacent)
            )
            training_ratio = config.experiment.train_dataset_cfg.training_ratio
            max_first = False
        else:
            force_adjacent = tuple()
            training_ratio = None
            max_first = config.experiment.train_dataset_cfg.pick_max_first
        return (
            f"MaxOf{config.experiment.seq_len}-{config.train_for[0]}-{config.train_for[1]}"
            f"{f'-{n_layers}L' if n_layers != 1 else ''}"
            f"{f'-adj-{force_adjacent}' if force_adjacent else ''}"
            f"{'-max-first' if max_first else ''}"
            f"{f'-training-ratio-{training_ratio:.3f}' if training_ratio is not None else ''}"
            f"{'-with-eos' if config.experiment.use_end_of_sequence else ''}"
            f"{'-' + config.experiment.summary_slug_extra if config.experiment.summary_slug_extra else ''}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )

    def get_ground_truth(
        self, x: Integer[Tensor, "... n"]  # noqa: F722
    ) -> Integer[Tensor, "..."]:  # noqa: F722
        if self.use_end_of_sequence:
            x = x[..., :-1]
        if self.nth_max == 1:
            return x.max(dim=-1).values
        else:
            return torch.topk(
                x, self.nth_max, dim=-1, sorted=True, largest=True
            ).values[..., self.nth_max - 1]


MAX_OF_2_CONFIG: Config[MaxOfN] = Config(
    experiment=MaxOfN(
        train_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        test_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        seq_len=2,
    ),
    validate_every=None,
)
MAX_OF_10_CONFIG: Config[MaxOfN] = Config(
    experiment=MaxOfN(
        train_dataset_cfg=IterableDatasetCfg(n_samples=None, pick_max_first=False),
        test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        seq_len=10,
    ),
    validate_every=None,
    train_for=(50000, "steps"),
)


def MAX_OF_4_CONFIG(seed: int) -> Config[MaxOfN]:
    return Config(
        experiment=MaxOfN(
            model_config=HookedTransformerConfig(
                act_fn=None,
                attn_only=True,
                d_head=32,
                d_mlp=None,
                d_model=32,
                d_vocab=64,
                device="cpu",
                n_ctx=4,
                n_heads=1,
                n_layers=1,
                normalization_type=None,
            ),
            zero_biases=True,
            use_log1p=True,
            use_end_of_sequence=False,
            seq_len=4,
            optimizer="AdamW",
            optimizer_kwargs={"lr": 0.001, "betas": (0.9, 0.999)},
            train_dataset_cfg=IterableDatasetCfg(pick_max_first=False),
            test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        ),
        deterministic=True,
        seed=seed,
        batch_size=128,
        train_for=(3000, "steps"),
    )


class MaxOfNTrainingWrapper(TrainingWrapper[MaxOfN]):
    def __init__(self, config: Config[MaxOfN], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @property
    def log_softmax(self):
        return (
            F.log_softmax if not self.config.experiment.use_log1p else utils.log_softmax
        )

    @staticmethod
    def build_model(config: Config[MaxOfN]) -> HookedTransformer:
        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.use_kaiming_init:
            if model.cfg.seed is not None:
                torch.manual_seed(model.cfg.seed)

            for name, param in model.named_parameters():
                if "W_" in name:
                    torch.nn.init.kaiming_uniform_(param)

        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    def loss_fn(
        self,
        logits: Float[Tensor, "batch n_ctx d_vocab"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
        log_softmax: Optional[Callable] = None,
    ) -> Float[Tensor, ""]:  # noqa F722
        if log_softmax is None:
            log_softmax = self.log_softmax
        log_probs = log_softmax(logits[:, -1, :], dim=-1)
        correct_log_probs = log_probs.gather(-1, labels.unsqueeze(-1))
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn_per_seq(
        logits: Float[Tensor, "batch n_ctx d_vocab"],  # noqa: F722, F821
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> Bool[Tensor, "batch"]:  # noqa: F821
        pred_tokens = logits[:, -1, :].argmax(dim=-1)
        return pred_tokens == labels

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch n_ctx d_vocab"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> float:
        return (
            MaxOfNTrainingWrapper.acc_fn_per_seq(logits, labels).float().mean().item()
        )

    def compute_batch(
        self,
        x_y: Tuple[Integer[Tensor, "batch pos"], Integer[Tensor, "batch"]],  # noqa F722
        *,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[
        Integer[Tensor, "batch pos"],  # noqa F722
        Integer[Tensor, "batch"],  # noqa F821
        Float[Tensor, "batch pos d_vocab_out"],  # noqa F722
    ]:
        xs, ys = x_y

        if device is not None:
            xs = xs.to(device)
            ys = ys.to(device)
        self.model.to(xs.device, print_details=False)

        y_preds = self.model(xs)
        return xs, ys, y_preds

    def run_batch(
        self,
        x_y: Tuple[Integer[Tensor, "batch pos"], Integer[Tensor, "batch"]],  # noqa F722
        prefix: Optional[str] = None,
        *,
        log_output: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[Float[Tensor, ""], float]:  # noqa F722
        assert prefix is not None or not log_output, "Must not log if prefix is None"
        xs, ys, y_preds = self.compute_batch(x_y, device=device)
        loss = self.loss_fn(y_preds, ys)
        acc = self.acc_fn(y_preds, ys)

        if log_output:
            self.log(f"{prefix}loss", loss, prog_bar=True)
            self.log(f"{prefix}acc", acc, prog_bar=True)
            if prefix in self.config.experiment.log_matrix_on_run_batch_prefixes:
                assert self.logger is not None
                self.config.experiment.logging_options.log_matrices(
                    self.logger.experiment,  # type: ignore
                    self.model,
                )
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.run_batch(batch, prefix="")
        return loss

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def configure_optimizers(self):
        optimizer = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }[self.config.experiment.optimizer]
        return optimizer(self.parameters(), **self.config.experiment.optimizer_kwargs)


class MaxOfNIterableDataset(
    IterableDataset[
        Tuple[Integer[Tensor, "seq_length"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
):
    def __init__(
        self,
        seed: int,
        config: Config[MaxOfN],
        max_length: Optional[int] = None,
        pick_max_first: bool = False,
    ):
        self.config = config
        self.seed = seed
        self.pick_max_first = pick_max_first
        if max_length is None:
            n, unit = config.train_for
            assert unit == "steps"
            self.max_length = n * config.batch_size
        else:
            self.max_length = max_length

    def __len__(self):
        return self.max_length

    def __iter__(self):
        def generator():
            g = torch.Generator()
            g.manual_seed(self.seed)
            n_samples = 0
            while True:
                max_val = (
                    int(
                        torch.randint(
                            0, self.config.experiment.model_config.d_vocab_out, tuple()
                        ).item()
                    )
                    if self.pick_max_first
                    else self.config.experiment.model_config.d_vocab_out - 1
                )
                val = torch.randint(
                    0,
                    max_val + 1,
                    (self.config.experiment.seq_len,),
                    generator=g,
                )
                if self.pick_max_first and max_val not in val:
                    val[
                        torch.randint(
                            0, self.config.experiment.seq_len, (1,), generator=g
                        )
                    ] = max_val

                # Process output
                eos_token = self.config.experiment.get_eos_token()
                if eos_token is not None:
                    val = utils.add_eos(val, eos_token)
                yield val, self.config.experiment.get_ground_truth(val)

                n_samples += 1
                if self.max_length is not None and n_samples >= self.max_length:
                    return
                # TODO: add adversarial generation

        return iter(generator())


class MaxOfNDataModule(DataModule):
    data_train: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
    data_test: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
    batch_size: Optional[int]
    seq_len: int
    dataset_seed: int

    def __init__(self, config: Config[MaxOfN]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = config.experiment.seq_len
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    @cache
    def get_full_dataset(self, force_adjacent: Sequence[int], training_ratio: float):
        rng = np.random.default_rng(self.dataset_seed)
        data = generate_all_sequences(self.model_config.d_vocab_out, self.seq_len)
        data = shuffle_data(data, rng)

        if force_adjacent:
            assert self.seq_len == 2
            idxs = torch.zeros_like(data[:, 0], dtype=torch.bool)
            for k in force_adjacent:
                idxs |= (data[:, 0] - data[:, 1]).abs() == k
            data, extra_data = data[~idxs], data[idxs]
            data = torch.cat([extra_data, data], dim=0)

        split_idx = int(len(data) * training_ratio)

        data_train = shuffle_data(data[:split_idx], rng)
        data_test = shuffle_data(data[split_idx:], rng)
        # concatenate on a tensor of self.mode_config.d_vocab-1, if needed
        return data_train, data_test

    def build_dataset(
        self, cfg: DatasetCfg, mode: Literal["train", "test"]
    ) -> Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]:
        # TODO: factor these out into the classes
        base_dataset: Dataset[
            Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
        ]
        if isinstance(cfg, IterableDatasetCfg):
            base_dataset = MaxOfNIterableDataset(
                reseed(self.dataset_seed, mode),
                self.config,
                max_length=cfg.n_samples,
                pick_max_first=cfg.pick_max_first,
            )
        elif isinstance(cfg, FullDatasetCfg):
            data_train, data_test = self.get_full_dataset(
                cfg.force_adjacent, cfg.training_ratio
            )
            eos_token = self.config.experiment.get_eos_token()
            if eos_token is not None:
                data_train = utils.add_eos(data_train, eos_token)
                data_test = utils.add_eos(data_test, eos_token)

            base_dataset = {
                "train": TensorDataset(
                    data_train, self.config.experiment.get_ground_truth(data_train)
                ),
                "test": TensorDataset(
                    data_test, self.config.experiment.get_ground_truth(data_test)
                ),
            }[mode]
        else:
            raise NotImplementedError
        return base_dataset

    def setup(self, stage: str):
        self.data_train = self.build_dataset(
            self.config.experiment.train_dataset_cfg, "train"
        )
        self.data_test = self.build_dataset(
            self.config.experiment.test_dataset_cfg, "test"
        )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.validation_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size)


def config_of_argv(argv=sys.argv) -> tuple[Config[MaxOfN], dict]:
    parser = argparse.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    add_force_argument(parser)
    add_no_save_argument(parser)
    # add --max-of N argument accepting 2 and 10
    parser.add_argument(
        "--max-of",
        metavar="N",
        type=int,
        default=10,
        help="The length of the list to take the maximum of.",
    )
    parser.add_argument(
        "--force-adjacent-gap",
        metavar="K",
        type=str,
        action="append",
        help="For --max-of 2, include all sequences (n, n±K) in training set. Accepts int and comma-separated-list.",
    )
    parser.add_argument(
        "--training-ratio",
        type=float,
        default=0.7,
        help="For --max-of 2, the fraction of sequences to include in training.",
    )
    parser.add_argument(
        "--use-log1p",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a more accurate implementation of log_softmax.",
    )
    parser.add_argument(
        "--use-end-of-sequence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an end-of-sequence token so the query-side attention vector is fixed.",
    )
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument(
        "--optimizer",
        choices=["Adam", "AdamW", "SGD"],
        default="Adam",
        help="The optimizer to use.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.999),
        help="coefficients used for computing running averages of gradient and its square",
    )
    parser.add_argument(
        "--summary-slug-extra", type=str, default="", help="Extra model description"
    )
    parser.add_argument(
        "--pick-max-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pick the maximum value first, then fill in the rest of the sequence. Only meaningful for --max-of N > 2.",
    )
    parser.add_argument(
        "--use-kaiming-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.nn.init.kaiming_uniform_, rather than HookedTransformer's init.",
    )
    parser.add_argument(
        "--log-matrix-interp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log matrices every train step",
    )
    parser.add_argument(
        "--checkpoint-matrix-interp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log matrices for checkpointing",
    )
    parser.add_argument(
        "--log-final-matrix-interp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log matrices after training",
    )
    HOOKED_TRANSFORMER_CONFIG_ARGS = set(
        (
            "normalization_type",
            "d_model",
            "d_head",
            "n_layers",
            "n_heads",
            "d_vocab",
            "dtype",
            "eps",
        )
    )
    Config.add_arguments(parser)
    add_HookedTransformerConfig_arguments(parser, HOOKED_TRANSFORMER_CONFIG_ARGS)
    args = parser.parse_args(argv[1:])

    config = set_params(
        (MAX_OF_2_CONFIG if args.max_of <= 2 else MAX_OF_10_CONFIG),
        {
            ("experiment", "seq_len"): args.max_of,
            ("experiment", "use_end_of_sequence"): args.use_end_of_sequence,
            ("experiment", "use_log1p"): args.use_log1p,
            ("experiment", "optimizer"): args.optimizer,
            ("experiment", "summary_slug_extra"): args.summary_slug_extra,
            ("experiment", "train_dataset_cfg", "pick_max_first"): args.pick_max_first,
            ("experiment", "logging_options"): ModelMatrixLoggingOptions.all(),
            ("experiment", "log_matrix_on_run_batch_prefixes"): set()
            | ({"test_"} if args.log_final_matrix_interp else set())
            | ({"periodic_test_"} if args.checkpoint_matrix_interp else set())
            | ({""} if args.log_matrix_interp else set()),
        },
    ).update_from_args(args)
    config.experiment.model_config = update_HookedTransformerConfig_from_args(
        config, config.experiment.model_config, args, HOOKED_TRANSFORMER_CONFIG_ARGS
    )
    config.experiment.__post_init__()  # for seq_len, d_vocab
    if args.weight_decay is not None:
        config.experiment.optimizer_kwargs["weight_decay"] = args.weight_decay
    config.experiment.optimizer_kwargs.update(
        {"lr": args.lr, "betas": tuple(args.betas)}
    )
    if args.max_of <= 2:
        if args.force_adjacent_gap:
            force_adjacent = tuple(
                sorted(
                    set(
                        int(k.strip())
                        for s in args.force_adjacent_gap
                        for k in s.split(",")
                    )
                )
            )
            config = set_params(
                config,
                {
                    (
                        "experiment",
                        "train_dataset_cfg",
                        "force_adjacent",
                    ): force_adjacent,
                    (
                        "experiment",
                        "test_dataset_cfg",
                        "force_adjacent",
                    ): force_adjacent,
                },
            )
        config = set_params(
            config,
            {
                (
                    "experiment",
                    "train_dataset_cfg",
                    "training_ratio",
                ): args.training_ratio,
                (
                    "experiment",
                    "test_dataset_cfg",
                    "training_ratio",
                ): args.training_ratio,
            },
        )
    return config, dict(force=args.force, save_to=args.save_to)


def main(argv=sys.argv):
    config, kwargs = config_of_argv(argv)
    print("Training model:", config)
    return train_or_load_model(config, **kwargs)


# %%
if __name__ == "__main__":
    main()

# # %%
# runtime, model = main([i for i in "train  --max-of 2 --non-deterministic --train-for-epochs 3000 --validate-every-epochs 20 --force-adjacent-gap 0,1,2 --use-log1p --training-ratio 0.095 --weight-decay 1.0 --betas 0.9 0.98 --optimizer AdamW --use-end-of-sequence --force load".strip().split(" ") if i])
# # %%
# from gbmi.analysis_tools.plot import imshow, line
# # %%
# line((model.W_E[-1] + model.W_pos[-1]) @ model.W_Q[0, 0, :, :] @ model.W_K[0, 0, :, :].T @ (model.W_E[:-1] + model.W_pos[:-1].mean(dim=0)).T, renderer='png')
# # %%
# res = model(torch.tensor([[39, 40, model.cfg.d_vocab - 1]]))[:, -1, :]
# print(res.softmax(dim=-1))
# print(res[:, 39:41])
# print(res.argmax(dim=-1))
# # %%
