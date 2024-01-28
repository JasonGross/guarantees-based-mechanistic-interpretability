# %%
from __future__ import annotations
import argparse
import sys

from dataclasses import dataclass, field
from functools import cache
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
from jaxtyping import Float, Integer
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
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
    reseed,
    set_params,
)
from gbmi.utils.sequences import generate_all_sequences


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


DatasetCfg = IterableDatasetCfg | FullDatasetCfg


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
    use_end_of_sequence: bool = False
    seq_len: int = 64
    summary_slug_extra: str = ""

    train_dataset_cfg: DatasetCfg = field(
        default_factory=lambda: IterableDatasetCfg(n_samples=None)
    )
    test_dataset_cfg: DatasetCfg = field(
        default_factory=lambda: IterableDatasetCfg(n_samples=1024)
    )
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999)}
    )
    optimizer: Literal["Adam", "AdamW"] = "Adam"

    def __post_init__(self):
        self.model_config.n_ctx = self.seq_len
        if self.use_end_of_sequence:
            self.model_config.n_ctx = self.seq_len + 1
            self.model_config.d_vocab = self.model_config.d_vocab_out + 1
        self.model_config.__post_init__()

    def config_post_init(self, config: Config[MaxOfN]) -> None:
        self.model_config.seed = reseed(config.seed, "model")

    def get_training_wrapper(self):
        return MaxOfNTrainingWrapper

    def get_datamodule(self):
        return MaxOfNDataModule

    def get_summary_slug(self, config: Config[MaxOfN]) -> str:
        if isinstance(config.experiment.train_dataset_cfg, FullDatasetCfg):
            force_adjacent = ",".join(
                map(str, config.experiment.train_dataset_cfg.force_adjacent)
            )
            training_ratio = config.experiment.train_dataset_cfg.training_ratio
        else:
            force_adjacent = tuple()
            training_ratio = None
        return (
            f"MaxOf{config.experiment.seq_len}-{config.train_for[0]}-{config.train_for[1]}"
            f"{f'-adj-{force_adjacent}' if force_adjacent else ''}"
            f"{f'-training-ratio-{training_ratio:.3f}' if training_ratio is not None else ''}"
            f"{'-with-eos' if config.experiment.use_end_of_sequence else ''}"
            f"{'-' + config.experiment.summary_slug_extra if config.experiment.summary_slug_extra else ''}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )


MAX_OF_2_CONFIG = Config(
    experiment=MaxOfN(
        train_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        test_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        seq_len=2,
    ),
    validate_every=None,
)
MAX_OF_10_CONFIG = Config(
    experiment=MaxOfN(
        train_dataset_cfg=IterableDatasetCfg(n_samples=None),
        test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        seq_len=10,
    ),
    validate_every=None,
    train_for=(50000, "steps"),
)


class MaxOfNTrainingWrapper(TrainingWrapper[MaxOfN]):
    def __init__(self, config: Config[MaxOfN], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[MaxOfN]) -> HookedTransformer:
        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch d_vocab"],  # noqa: F821, F722
        tokens: Integer[Tensor, "batch seq_len"],  # noqa: F821, F722
        log_softmax: Callable = F.log_softmax,
    ) -> Float[Tensor, ""]:  # noqa F722
        true_maximum = torch.max(tokens, dim=1)[0]
        log_probs = log_softmax(logits, dim=-1)
        correct_log_probs = log_probs.gather(-1, true_maximum.unsqueeze(-1))
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch d_vocab"],  # noqa: F821, F722
        tokens: Integer[Tensor, "batch seq_len"],  # noqa: F821, F722
    ) -> float:
        pred_tokens = torch.argmax(logits, dim=1)
        true_maximum = torch.max(tokens, dim=1)[0]
        return (pred_tokens == true_maximum).float().mean().item()

    @overload
    def run_batch(
        self,
        x: Float[Tensor, "batch pos"],  # noqa F722
        prefix: Literal[None] = None,
        *,
        return_accuracy: Literal[True],
        log_output: Literal[False],
    ) -> Tuple[Float[Tensor, ""], float]:  # noqa F722
        ...

    @overload
    def run_batch(
        self,
        x: Float[Tensor, "batch pos"],  # noqa F722
        prefix: Literal[None] = None,
        *,
        return_accuracy: Literal[False] = False,
        log_output: Literal[False],
    ) -> Float[Tensor, ""]:  # noqa F722
        ...

    @overload
    def run_batch(
        self,
        x: Float[Tensor, "batch pos"],  # noqa F722
        prefix: str,
        *,
        return_accuracy: Literal[True],
        log_output: bool = True,
    ) -> Tuple[Float[Tensor, ""], float]:  # noqa F722
        ...

    @overload
    def run_batch(
        self,
        x: Float[Tensor, "batch pos"],  # noqa F722
        prefix: str,
        *,
        return_accuracy: Literal[False] = False,
        log_output: bool = True,
    ) -> Float[Tensor, ""]:  # noqa F722
        ...

    def run_batch(
        self,
        x: Float[Tensor, "batch pos"],  # noqa F722
        prefix: Optional[str] = None,
        *,
        return_accuracy: bool = False,
        log_output: bool = True,
    ) -> Union[Float[Tensor, ""], Tuple[Float[Tensor, ""], float]]:  # noqa F722
        assert prefix is not None or not log_output, "Must not log if prefix is None"
        log_softmax = (
            F.log_softmax if not self.config.experiment.use_log1p else utils.log_softmax
        )
        self.model.to(x.device, print_details=False)
        y_preds = self.model(x)[:, -1, :]
        if self.config.experiment.use_end_of_sequence:
            x = x[:, :-1]
        loss = self.loss_fn(
            y_preds,
            x,
            log_softmax=log_softmax,
        )
        if log_output:
            self.log(f"{prefix}loss", loss, prog_bar=True)
        acc = self.acc_fn(y_preds, x)
        if log_output:
            self.log(f"{prefix}acc", acc, prog_bar=True)
        if return_accuracy:
            return loss, acc
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="")

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def configure_optimizers(self):
        optimizer = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[
            self.config.experiment.optimizer
        ]
        return optimizer(self.parameters(), **self.config.experiment.optimizer_kwargs)


class MaxOfNDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    data_test: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    batch_size: Optional[int]
    seq_len: int
    use_end_of_sequence: bool
    dataset_seed: int

    def __init__(self, config: Config[MaxOfN]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = config.experiment.seq_len
        self.use_end_of_sequence = config.experiment.use_end_of_sequence
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    def cat_eos(
        self,
        data: Integer[Tensor, "... seq_len"],  # noqa: F722
    ) -> Union[
        Integer[Tensor, "... seq_len+1"], Integer[Tensor, "... seq_len"]  # noqa: F722
    ]:
        if not self.use_end_of_sequence:
            return data
        return torch.cat(
            [
                data,
                torch.full(
                    (len(data), 1),
                    self.model_config.d_vocab - 1,
                    dtype=torch.long,
                    device=data.device,
                ),
            ],
            dim=1,
        )

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
        data_train = self.cat_eos(data_train)
        data_test = self.cat_eos(data_test)
        return data_train, data_test

    def build_dataset(
        self, cfg: DatasetCfg, mode: Literal["train", "test"]
    ) -> Dataset[Tensor]:
        # TODO: factor these out into the classes
        if isinstance(cfg, IterableDatasetCfg):
            return MaxOfNDataset(
                reseed(self.dataset_seed, mode),
                self.config,
                cfg.n_samples,
                pick_max_first=cfg.pick_max_first,
            )
        elif isinstance(cfg, FullDatasetCfg):
            data_train, data_test = self.get_full_dataset(
                cfg.force_adjacent, cfg.training_ratio
            )
            return {
                "train": SingleTensorDataset(data_train),
                "test": SingleTensorDataset(data_test),
            }[mode]
        else:
            raise NotImplementedError

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
        return DataLoader(self.data_test, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size)


class MaxOfNDataset(IterableDataset[Integer[Tensor, "seq_length"]]):
    def __init__(
        self,
        seed: int,
        config: Config[MaxOfN],
        max_length: Optional[int] = None,
        pick_max_first: bool = False,
    ):
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = config.experiment.seq_len
        self.use_end_of_sequence = config.experiment.use_end_of_sequence
        self.seed = seed
        self.pick_max_first = pick_max_first
        if max_length is None:
            n, unit = config.train_for
            assert unit == "steps"
            self.max_length = n * config.batch_size
        else:
            self.max_length = max_length

    def cat_eos(
        self,
        data: Integer[Tensor, "... seq_len"],  # noqa: F722
    ) -> Union[
        Integer[Tensor, "... seq_len+1"], Integer[Tensor, "... seq_len"]  # noqa: F722
    ]:
        if not self.use_end_of_sequence:
            return data
        return torch.cat(
            [
                data,
                torch.full(
                    (len(data), 1),
                    self.model_config.d_vocab - 1,
                    dtype=torch.long,
                    device=data.device,
                ),
            ],
            dim=1,
        )

    def __len__(self):
        return self.max_length

    def __iter__(self):
        def generator():
            g = torch.Generator()
            g.manual_seed(self.seed)
            n_samples = 0
            while True:
                max_val = (
                    int(torch.randint(0, self.model_config.d_vocab_out, tuple()).item())
                    if self.pick_max_first
                    else self.model_config.d_vocab_out
                )
                val = torch.randint(
                    0,
                    max_val + 1,
                    (self.seq_len,),
                    generator=g,
                )
                if self.pick_max_first and max_val not in val:
                    val[torch.randint(0, self.seq_len, (1,), generator=g)] = max_val
                yield self.cat_eos(val)
                n_samples += 1
                if self.max_length is not None and n_samples >= self.max_length:
                    return
                # TODO: add adversarial generation

        return iter(generator())


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
        choices=["Adam", "AdamW"],
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
