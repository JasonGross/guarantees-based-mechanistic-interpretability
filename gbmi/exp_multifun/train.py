# %%
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

import gbmi.utils as utils
from gbmi.exp_multifun import SEEDS, SELECTED_SEED
from gbmi.model import (
    Config,
    DataModule,
    ExperimentConfig,
    TrainingWrapper,
    add_force_argument,
    add_HookedTransformerConfig_arguments,
    add_no_save_argument,
    train_or_load_model,
    update_HookedTransformerConfig_from_args,
)
from gbmi.training_tools.logging import ModelMatrixLoggingOptions
from gbmi.utils import reseed, set_params, shuffle_data
from gbmi.utils.hashing import _EXCLUDE
from gbmi.utils.sequences import generate_all_sequences


@dataclass
class IterableDatasetCfg:
    n_samples: Optional[int] = None


@dataclass
class FullDatasetCfg:
    force_adjacent: Sequence[int] = tuple()
    # only for n_ctx=2: for all i in force_adjacent, force all sequences (n, n±i) to be in training set
    # bounds: Optional[Tuple[int, int]] = None
    # range of vocab tokens within which to sample
    training_ratio: float = 0.7


DatasetCfg = Union[IterableDatasetCfg, FullDatasetCfg]

FUNCS = Sequence[Literal["max", "min", "sum", "argmax", "argmin", "summod"]]


@dataclass
class Multifun(ExperimentConfig):
    # Model config
    n_layers: int = 1
    n_heads: int = 2
    d_model: int = 32
    d_vocab: int = 64
    attn_only: bool = True
    d_mlp: Optional[int] = None
    normalization_type: Optional[Literal["LN", "LNPre"]] = None
    zero_biases: bool = True
    use_log1p: bool = False
    use_end_of_sequence: bool = False  # if set, is final token
    seq_len: int = 64
    funcs: FUNCS = ("max", "min")
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
        self.logging_options.qpos = -1
        if self.use_end_of_sequence:
            self.logging_options.qtok = -1
        exclude = ("logging_options", "log_matrix_on_run_batch_prefixes")
        setattr(self, _EXCLUDE, exclude)
        self.logging_options.__post_init__()

    def get_eos_token(self) -> Optional[int]:
        return self.d_vocab + len(self.funcs) if self.use_end_of_sequence else None

    def get_training_wrapper(self):
        return MultifunTrainingWrapper

    def get_datamodule(self):
        return MultifunDataModule

    def get_summary_slug(self, config: Config[Multifun]) -> str:
        n_layers = config.experiment.n_layers
        d_vocab = config.experiment.d_vocab
        d_model = config.experiment.d_model
        n_heads = config.experiment.n_heads
        funcs = "-".join(config.experiment.funcs)
        if isinstance(config.experiment.train_dataset_cfg, FullDatasetCfg):
            force_adjacent = ",".join(
                map(str, config.experiment.train_dataset_cfg.force_adjacent)
            )
            training_ratio = config.experiment.train_dataset_cfg.training_ratio
        else:
            force_adjacent = tuple()
            training_ratio = None
        return (
            f"Multifun-{funcs}-Of{config.experiment.seq_len}-{config.train_for[0]}-{config.train_for[1]}"
            f"{f'-{n_layers}L' if n_layers != 1 else ''}"
            f"{f'-{n_heads}h' if n_heads != 1 else ''}"
            f"{f'-{d_vocab}v' if d_vocab != 64 else ''}"
            f"{f'-{d_model}m' if d_model != 32 else ''}"
            f"{f'-adj-{force_adjacent}' if force_adjacent else ''}"
            f"{f'-training-ratio-{training_ratio:.3f}' if training_ratio is not None else ''}"
            f"{'-with-eos' if config.experiment.use_end_of_sequence else ''}"
            f"{'-' + config.experiment.summary_slug_extra if config.experiment.summary_slug_extra else ''}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )

    @property
    def d_vocab_out(self) -> int:
        d_vocab_out = 0
        if "max" in self.funcs or "min" in self.funcs or "summod" in self.funcs:
            d_vocab_out = max(d_vocab_out, self.d_vocab)
        if "argmax" in self.funcs or "argmin" in self.funcs:
            d_vocab_out = max(d_vocab_out, self.seq_len)
        if "sum" in self.funcs:
            d_vocab_out = max(d_vocab_out, 1 + (self.d_vocab - 1) * self.seq_len)
        return d_vocab_out

    def get_ground_truth(
        self, x: Integer[Tensor, "... n"]  # noqa: F722
    ) -> Integer[Tensor, "..."]:  # noqa: F722
        if self.use_end_of_sequence:
            x = x[..., :-1]
        kind, x = x[..., 0].long() - self.d_vocab, x[..., 1:]
        # Create masks for each function type
        funcs = np.array(self.funcs)
        max_mask = funcs[kind] == "max"
        min_mask = funcs[kind] == "min"
        sum_mask = funcs[kind] == "sum"
        argmax_mask = funcs[kind] == "argmax"
        argmin_mask = funcs[kind] == "argmin"
        summod_mask = funcs[kind] == "summod"

        results = torch.empty(x.shape[:-1], dtype=x.dtype)

        if max_mask.any():
            results[max_mask] = x[max_mask].max(dim=-1).values
        if min_mask.any():
            results[min_mask] = x[min_mask].min(dim=-1).values
        if sum_mask.any():
            results[sum_mask] = x[sum_mask].sum(dim=-1)
        if argmax_mask.any():
            results[argmax_mask] = x[argmax_mask].argmax(dim=-1)
        if argmin_mask.any():
            results[argmin_mask] = x[argmin_mask].argmin(dim=-1)
        if summod_mask.any():
            results[summod_mask] = x[summod_mask].sum(dim=-1) % self.d_vocab

        assert (
            max_mask | min_mask | sum_mask | argmax_mask | argmin_mask | summod_mask
        ).all(), funcs[
            kind[
                ~(
                    max_mask
                    | min_mask
                    | sum_mask
                    | argmax_mask
                    | argmin_mask
                    | summod_mask
                )
            ]
        ]

        return results


MULTIFUN_OF_2_CONFIG: Config[Multifun] = Config(
    experiment=Multifun(
        train_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        test_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        seq_len=2,
    ),
    validate_every=None,
)
MULTIFUN_OF_10_SINGLE_CONFIG: Config[Multifun] = Config(
    experiment=Multifun(
        train_dataset_cfg=IterableDatasetCfg(n_samples=None),
        test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        seq_len=10,
    ),
    validate_every=None,
    train_for=(50000, "steps"),
)


def MULTIFUN_OF_4_CONFIG(
    seed: int,
    funcs: FUNCS = ("max", "min"),
    *,
    n_heads: int = 2,
    use_end_of_sequence: bool = False,
) -> Config[Multifun]:
    return Config(
        experiment=Multifun(
            # act_fn=None,
            attn_only=True,
            # d_head=32,
            d_mlp=None,
            d_model=32,
            d_vocab=64,
            # device="cpu",
            # n_ctx= 4,
            n_heads=n_heads,
            n_layers=1,
            normalization_type=None,
            zero_biases=True,
            use_log1p=True,
            use_end_of_sequence=use_end_of_sequence,
            funcs=funcs,
            seq_len=4,
            optimizer="AdamW",
            optimizer_kwargs={"lr": 0.001, "betas": (0.9, 0.999)},
            train_dataset_cfg=IterableDatasetCfg(),
            test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        ),
        deterministic=True,
        seed=seed,
        batch_size=128,
        train_for=(3000, "steps"),
    )


def MULTIFUN_OF_5_CONFIG(
    seed: int,
    funcs: FUNCS = ("max", "min"),
    *,
    n_heads: int = 2,
    use_end_of_sequence: bool = False,
    deterministic: bool = False,
) -> Config[Multifun]:
    return set_params(
        MULTIFUN_OF_4_CONFIG(
            seed, funcs=funcs, n_heads=n_heads, use_end_of_sequence=use_end_of_sequence
        ),
        {("deterministic",): deterministic, ("experiment", "seq_len"): 5},
        post_init=True,
    )


def MULTIFUN_OF_10_CONFIG(
    seed: int,
    d_vocab: int = 64,
    funcs: FUNCS = ("max", "min"),
    *,
    n_heads: int = 2,
    use_end_of_sequence: bool = False,
    deterministic: bool = False,
) -> Config[Multifun]:
    return set_params(
        MULTIFUN_OF_4_CONFIG(
            seed, funcs=funcs, n_heads=n_heads, use_end_of_sequence=use_end_of_sequence
        ),
        {
            ("deterministic",): deterministic,
            ("experiment", "seq_len"): 10,
            ("experiment", "d_vocab"): d_vocab,
        },
        post_init=True,
    )


def MULTIFUN_OF_20_CONFIG(
    seed: int,
    d_vocab: int = 64,
    funcs: FUNCS = ("max", "min"),
    *,
    n_heads: int = 2,
    use_end_of_sequence: bool = False,
    deterministic: bool = False,
) -> Config[Multifun]:
    return set_params(
        MULTIFUN_OF_4_CONFIG(
            seed, funcs=funcs, n_heads=n_heads, use_end_of_sequence=use_end_of_sequence
        ),
        {
            ("deterministic",): deterministic,
            ("experiment", "seq_len"): 20,
            ("experiment", "d_vocab"): d_vocab,
        },
        post_init=True,
    )


class MultifunTrainingWrapper(TrainingWrapper[Multifun]):
    def __init__(self, config: Config[Multifun], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @property
    def log_softmax(self):
        return (
            F.log_softmax if not self.config.experiment.use_log1p else utils.log_softmax
        )

    @staticmethod
    def build_model_config(config: Config[Multifun]) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers=config.experiment.n_layers,
            n_heads=config.experiment.n_heads,
            d_model=config.experiment.d_model,
            d_head=config.experiment.d_model // config.experiment.n_heads,
            d_vocab=config.experiment.d_vocab
            + len(config.experiment.funcs)
            + (1 if config.experiment.use_end_of_sequence else 0),
            n_ctx=config.experiment.seq_len
            + 1  # for the selector token
            + (1 if config.experiment.use_end_of_sequence else 0),
            d_vocab_out=config.experiment.d_vocab_out,
            attn_only=config.experiment.attn_only,
            normalization_type=config.experiment.normalization_type,
            d_mlp=config.experiment.d_mlp,
            seed=reseed(config.seed, "model"),
            init_weights=not config.experiment.use_kaiming_init,
            device="cpu" if config.deterministic else None,
        )

    @staticmethod
    def update_config_from_model_config(
        experiment: Multifun, config: HookedTransformerConfig
    ) -> Multifun:
        experiment.n_layers = config.n_layers
        experiment.n_heads = config.n_heads
        experiment.d_model = config.d_model
        experiment.d_vocab = (
            config.d_vocab
            - (1 if experiment.use_end_of_sequence else 0)
            - len(experiment.funcs)
        )
        experiment.attn_only = config.attn_only
        experiment.normalization_type = config.normalization_type
        experiment.d_mlp = config.d_mlp
        return experiment

    @staticmethod
    def build_model(config: Config[Multifun]) -> HookedTransformer:
        model = HookedTransformer(MultifunTrainingWrapper.build_model_config(config))
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
        logits: Float[Tensor, "batch n_ctx n_ctx_out"],  # noqa: F722
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
        logits: Float[Tensor, "batch n_ctx n_ctx_out"],  # noqa: F722, F821
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> Bool[Tensor, "batch"]:  # noqa: F821
        pred_tokens = logits[:, -1, :].argmax(dim=-1)
        return pred_tokens == labels

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch n_ctx n_ctx_out"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> float:
        return (
            MultifunTrainingWrapper.acc_fn_per_seq(logits, labels).float().mean().item()
        )

    def compute_batch(
        self,
        x_y: Tuple[Integer[Tensor, "batch pos"], Integer[Tensor, "batch"]],  # noqa F722
        *,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[
        Integer[Tensor, "batch pos"],  # noqa F722
        Integer[Tensor, "batch"],  # noqa F821
        Float[Tensor, "batch pos n_ctx_out"],  # noqa F722
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


class MultifunIterableDataset(
    IterableDataset[
        Tuple[Integer[Tensor, "seq_length"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
):
    def __init__(
        self,
        seed: int,
        config: Config[Multifun],
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
                func = (
                    torch.randint(
                        0, len(self.config.experiment.funcs), (1,), generator=g
                    )
                    + self.config.experiment.d_vocab
                )
                max_val = (
                    int(
                        torch.randint(0, self.config.experiment.d_vocab, tuple()).item()
                    )
                    if self.pick_max_first
                    else self.config.experiment.d_vocab - 1
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
                val = torch.cat([func, val], dim=0)
                yield val, self.config.experiment.get_ground_truth(val)

                n_samples += 1
                if self.max_length is not None and n_samples >= self.max_length:
                    return
                # TODO: add adversarial generation

        return iter(generator())


class MultifunDataModule(DataModule):
    data_train: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
    data_test: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
    batch_size: Optional[int]
    seq_len: int
    dataset_seed: int

    def __init__(self, config: Config[Multifun]):
        super().__init__(config)
        self.config = config
        # self.model_config = MultifunTrainingWrapper.build_model_config(config)
        self.seq_len = config.experiment.seq_len
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    @cache
    def get_full_dataset(self, force_adjacent: Sequence[int], training_ratio: float):
        rng = np.random.default_rng(self.dataset_seed)
        data = generate_all_sequences(self.config.experiment.d_vocab, self.seq_len)
        data = torch.cat(
            [
                utils.add_bos(func + self.config.experiment.d_vocab, data)
                for func in range(len(self.config.experiment.funcs))
            ],
            dim=0,
        )
        data = shuffle_data(data, rng)

        if force_adjacent:
            assert self.seq_len == 2
            idxs = torch.zeros_like(data[:, 1], dtype=torch.bool)
            for k in force_adjacent:
                idxs |= (data[:, 1] - data[:, 2]).abs() == k
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
            base_dataset = MultifunIterableDataset(
                reseed(self.dataset_seed, mode),
                self.config,
                max_length=cfg.n_samples,
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


def config_of_argv(argv=sys.argv) -> tuple[Config[Multifun], dict]:
    parser = argparse.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    add_force_argument(parser)
    add_no_save_argument(parser)
    # add --K N argument accepting 2 and 10
    parser.add_argument(
        "--K",
        metavar="K",
        type=int,
        default=10,
        help="The length of the list to take the reduction of.",
    )
    parser.add_argument(
        "--func",
        metavar="FUNC",
        type=str,
        nargs="+",
        default=["max", "min"],
        help="The functions to apply to the list.",
    )
    parser.add_argument(
        "--force-adjacent-gap",
        metavar="K",
        type=str,
        action="append",
        help="For --K 2, include all sequences (n, n±K) in training set. Accepts int and comma-separated-list.",
    )
    parser.add_argument(
        "--training-ratio",
        type=float,
        default=0.7,
        help="For --K 2, the fraction of sequences to include in training.",
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
        help="Pick the maximum value first, then fill in the rest of the sequence. Only meaningful for --K N > 2.",
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
        (MULTIFUN_OF_2_CONFIG if args.K <= 2 else MULTIFUN_OF_10_SINGLE_CONFIG),
        {
            ("experiment", "seq_len"): args.K,
            ("experiment", "funcs"): tuple(args.func),
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
    config.experiment = MultifunTrainingWrapper.update_config_from_model_config(
        config.experiment,
        update_HookedTransformerConfig_from_args(
            config,
            MultifunTrainingWrapper.build_model_config(config),
            args,
            HOOKED_TRANSFORMER_CONFIG_ARGS,
        ),
    )
    config.experiment.__post_init__()  # for seq_len, d_vocab
    if args.weight_decay is not None:
        config.experiment.optimizer_kwargs["weight_decay"] = args.weight_decay
    config.experiment.optimizer_kwargs.update(
        {"lr": args.lr, "betas": tuple(args.betas)}
    )
    if args.K <= 2:
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
# runtime, model = main([i for i in "train  --K 2 --non-deterministic --train-for-epochs 3000 --validate-every-epochs 20 --force-adjacent-gap 0,1,2 --use-log1p --training-ratio 0.095 --weight-decay 1.0 --betas 0.9 0.98 --optimizer AdamW --use-end-of-sequence --force load".strip().split(" ") if i])
# # %%
# from gbmi.analysis_tools.plot import  imshow, line
# # %%
# line((model.W_E[-1] + model.W_pos[-1]) @ model.W_Q[0, 0, :, :] @ model.W_K[0, 0, :, :].T @ (model.W_E[:-1] + model.W_pos[:-1].mean(dim=0)).T, renderer='png')
# # %%
# res = model(torch.tensor([[39, 40, model.cfg.d_vocab - 1]]))[:, -1, :]
# print(res.softmax(dim=-1))
# print(res[:, 39:41])
# print(res.argmax(dim=-1))
# # %%
