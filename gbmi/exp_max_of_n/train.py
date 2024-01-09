from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader, IterableDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

from gbmi.model import (
    TrainingWrapper,
    Config,
    ExperimentConfig,
    train_or_load_model,
    DataModule,
)
from gbmi.utils import (
    generate_all_sequences,
    shuffle_data,
    default_device,
    SingleTensorDataset,
    reseed,
    set_params,
)


@dataclass
class MaxOfN(ExperimentConfig):
    # Model config
    model_config: HookedTransformerConfig = HookedTransformerConfig(
        n_layers=1,
        n_heads=1,
        d_model=32,
        d_head=32,
        d_vocab=64,
        attn_only=True,
        normalization_type=None,
        n_ctx=2,
    )
    zero_biases: bool = True

    # Max of n (iterable dataset)
    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024

    # Max of 2 (full dataset)
    force_adjacent: bool = (
        False  # whether to force all adjacent-pair inputs to be in training set
    )
    training_ratio: float = 0.7  # fraction of dataset to use for training
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999)}
    )

    def get_training_wrapper(self):
        return MaxOfNTrainingWrapper

    def get_datamodule(self):
        return MaxOfNDataModule

    def get_summary_slug(self, config: Config[MaxOfN]) -> str:
        return f"MaxOf{config.experiment.model_config.n_ctx}-{config.train_for[0]}-{config.train_for[1]}{'-adj' if self.force_adjacent else ''}{'-nondeterministic' if not config.deterministic else ''}"


MAX_OF_2_CONFIG = set_params(
    Config(
        experiment=MaxOfN(
            force_adjacent=True,
        ),
    ),
    {("experiment", "model_config", "n_ctx"): 2},
)
MAX_OF_10_CONFIG = set_params(
    Config(
        experiment=MaxOfN(
            n_train_samples=None,
            n_test_samples=1024,
        ),
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
        logits: Float[Tensor, "batch pos d_vocab"],
        tokens: Integer[Tensor, "batch pos"],
    ) -> Float[Tensor, ""]:
        logits = logits[:, -1, :]
        true_maximum = torch.max(tokens, dim=1)[0]
        log_probs = logits.log_softmax(-1)
        correct_log_probs = log_probs.gather(-1, true_maximum.unsqueeze(-1))
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch pos d_vocab"],
        tokens: Integer[Tensor, "batch pos"],
    ) -> float:
        pred_logits = logits[:, -1, :]
        pred_tokens = torch.argmax(pred_logits, dim=1)
        true_maximum = torch.max(tokens, dim=1)[0]
        return (pred_tokens == true_maximum).float().mean().item()

    def run_batch(self, x: Float[Tensor, "batch pos"], prefix: str):
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

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )


class MaxOfNDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]
    data_test: Dataset[Integer[Tensor, "seq_len"]]
    batch_size: Optional[int]

    def __init__(self, config: Config[MaxOfN]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = config.experiment.model_config.n_ctx
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    def setup(self, stage: str):
        if self.model_config.n_ctx == 2:
            # Full dataset
            rng = np.random.default_rng(self.dataset_seed)
            data = generate_all_sequences(
                self.model_config.d_vocab, self.model_config.n_ctx
            )
            data = shuffle_data(data, rng)
            if self.config.experiment.force_adjacent:
                idxs = (data[:, 0] - data[:, 1]).abs() == 1
                data, extra_data = data[~idxs], data[idxs]
                data = torch.cat([extra_data, data], dim=0)

            split_idx = int(len(data) * self.config.experiment.training_ratio)

            data_train = data[:split_idx]
            data_test = data[split_idx:]

            if self.config.experiment.force_adjacent:
                data_train = shuffle_data(data_train, rng)
                data_test = shuffle_data(data_test, rng)

            self.data_train = cast(Dataset[Tensor], SingleTensorDataset(data_train))
            self.data_test = cast(Dataset[Tensor], SingleTensorDataset(data_test))
        else:
            # Sampled dataset
            self.data_train = MaxOfNDataset(
                self.dataset_seed * 2,
                self.config,
                self.config.experiment.n_train_samples,
            )
            self.data_test = MaxOfNDataset(
                self.dataset_seed * 2 + 1,
                self.config,
                self.config.experiment.n_test_samples,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.config.batch_size)

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
    print("Training model:", MAX_OF_10_CONFIG)
    train_or_load_model(MAX_OF_10_CONFIG, force="train")
