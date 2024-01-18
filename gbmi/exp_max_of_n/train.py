from __future__ import annotations
import argparse

from dataclasses import dataclass, field
from functools import cache
from typing import Any, Dict, Optional, Literal, Sequence

import numpy as np
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

from gbmi.model import (
    TrainingWrapper,
    Config,
    ExperimentConfig,
    train_or_load_model,
    DataModule,
    add_force_argument,
    add_no_save_argument,
)
from gbmi.utils import (
    generate_all_sequences,
    shuffle_data,
    SingleTensorDataset,
    reseed,
    set_params,
)


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

    train_dataset_cfg: DatasetCfg = field(
        default_factory=lambda: IterableDatasetCfg(n_samples=None)
    )
    test_dataset_cfg: DatasetCfg = field(
        default_factory=lambda: IterableDatasetCfg(n_samples=1024)
    )
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999)}
    )

    def get_training_wrapper(self):
        return MaxOfNTrainingWrapper

    def get_datamodule(self):
        return MaxOfNDataModule

    def get_summary_slug(self, config: Config[MaxOfN]) -> str:
        if isinstance(config.experiment.train_dataset_cfg, FullDatasetCfg):
            force_adjacent = config.experiment.train_dataset_cfg.force_adjacent
        else:
            force_adjacent = tuple()
        return (
            f"MaxOf{config.experiment.model_config.n_ctx}-{config.train_for[0]}-{config.train_for[1]}"
            f"{f'-adj-{force_adjacent}' if force_adjacent else ''}{'-nondeterministic' if not config.deterministic else ''}"
        )


MAX_OF_2_CONFIG = set_params(
    Config(
        experiment=MaxOfN(
            train_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
            test_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1), training_ratio=0.7),
        ),
        validate_every=None,
    ),
    {("experiment", "model_config", "n_ctx"): 2},
)
MAX_OF_10_CONFIG = set_params(
    Config(
        experiment=MaxOfN(
            train_dataset_cfg=IterableDatasetCfg(n_samples=None),
            test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        ),
        validate_every=None,
        train_for=(50000, "steps"),
    ),
    {("experiment", "model_config", "n_ctx"): 10},
)


class MaxOfNTrainingWrapper(TrainingWrapper[MaxOfN]):
    def __init__(self, config: Config[MaxOfN], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[MaxOfN]) -> HookedTransformer:
        set_params(
            config.experiment.model_config,
            {"seed": reseed(config.seed, "model")},
            warn_if_not_default=True,
        )
        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos d_vocab"],  # noqa: F821, F722
        tokens: Integer[Tensor, "batch pos"],  # noqa: F821, F722
    ) -> Float[Tensor, ""]:  # noqa F722
        logits = logits[:, -1, :]
        true_maximum = torch.max(tokens, dim=1)[0]
        log_probs = logits.log_softmax(-1)
        correct_log_probs = log_probs.gather(-1, true_maximum.unsqueeze(-1))
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch pos d_vocab"],  # noqa: F821, F722
        tokens: Integer[Tensor, "batch pos"],  # noqa: F821, F722
    ) -> float:
        pred_logits = logits[:, -1, :]
        pred_tokens = torch.argmax(pred_logits, dim=1)
        true_maximum = torch.max(tokens, dim=1)[0]
        return (pred_tokens == true_maximum).float().mean().item()

    def run_batch(
        self, x: Float[Tensor, "batch pos"], prefix: str  # noqa F722
    ) -> Float[Tensor, ""]:  # noqa F722
        self.model.to(x.device, print_details=False)
        # print(self.model.)
        # print(x.device)
        y_preds = self.model(x)
        loss = self.loss_fn(y_preds, x)
        self.log(f"{prefix}loss", loss, prog_bar=True)
        acc = self.acc_fn(y_preds, x)
        self.log(f"{prefix}acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="")

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )


class MaxOfNDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    data_test: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    batch_size: Optional[int]

    def __init__(self, config: Config[MaxOfN]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = config.experiment.model_config.n_ctx
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    @cache
    def get_full_dataset(self, force_adjacent: bool, training_ratio: float):
        rng = np.random.default_rng(self.dataset_seed)
        data = generate_all_sequences(
            self.model_config.d_vocab, self.model_config.n_ctx
        )
        data = shuffle_data(data, rng)

        if force_adjacent:
            assert self.model_config.n_ctx == 2
            idxs = (data[:, 0] - data[:, 1]).abs() == 1
            data, extra_data = data[~idxs], data[idxs]
            data = torch.cat([extra_data, data], dim=0)

        split_idx = int(len(data) * training_ratio)

        data_train = shuffle_data(data[:split_idx], rng)
        data_test = shuffle_data(data[split_idx:], rng)
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
        self, seed: int, config: Config[MaxOfN], max_length: Optional[int] = None
    ):
        self.config = config
        self.model_config = config.experiment.model_config
        self.seed = seed
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
                yield torch.randint(
                    0,
                    self.model_config.d_vocab,
                    (self.model_config.n_ctx,),
                    generator=g,
                )
                n_samples += 1
                if self.max_length is not None and n_samples >= self.max_length:
                    return
                # TODO: add adversarial generation

        return iter(generator())


if __name__ == "__main__":
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
    Config.add_arguments(parser)
    args = parser.parse_args()

    config = set_params(
        (MAX_OF_2_CONFIG if args.max_of <= 2 else MAX_OF_10_CONFIG),
        {("experiment", "model_config", "n_ctx"): args.max_of},
    ).update_from_args(args)
    print("Training model:", config)
    train_or_load_model(config, force=args.force, save_to=args.save_to)
