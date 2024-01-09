from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field

import sys
from typing import Any, Dict, Optional, Sequence, cast, Literal
from gbmi import utils

import numpy as np
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader, IterableDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
import argparse

import einops
import torch.nn.functional as F

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
class SortedList(ExperimentConfig):
    model_config: HookedTransformerConfig = field(
        default_factory=lambda: HookedTransformerConfig(
            n_layers=1,
            n_heads=2,
            d_mlp=None,
            d_model=96,
            d_head=48,
            # Layernorm makes things way more accurate, even though it makes
            # mech interp a little more annoying!
            normalization_type="LN",
            n_ctx=field(init=False),
            # it's a small transformer so may as well use these hooks
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_tokens=True,
            attn_only=True,
            act_fn="relu",
        )
    )

    list_len: int = 10
    max_value: int = 50
    lr_end: float = 1e-4
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "weight_decay": 0.005}
    )

    zero_biases: bool = False
    n_train_samples: int = 150_000
    n_test_samples: int = 10_000
    version_number: int = 0

    def __post_init__(self):
        # sequences are [a_1, ..., a_n, SEP, b_1, ..., b_n]
        self.model_config.n_ctx = 2 * self.list_len + 1
        # We have all digits from zero to max_value, plus SEP
        self.model_config.d_vocab = self.max_value + 2
        self.model_config.d_vocab_out = self.max_value + 1

    def config_post_init(self, config: Config[SortedList]) -> None:
        self.model_config.seed = reseed(config.seed, "model")

    def get_training_wrapper(self):
        return SortedListTrainingWrapper

    def get_datamodule(self):
        return SortedListDataModule

    def get_summary_slug(self, config: Config[SortedList]) -> str:
        return (
            f"SortedList-{config.experiment.list_len}-{config.experiment.max_value}-tokens"
            f"-{config.train_for[0]}-{config.train_for[1]}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )


SORTED_LIST_CONFIG = Config(
    experiment=SortedList(),
    seed=42,
    deterministic=True,
    batch_size=512,
    train_for=(25, "epochs"),
    log_every_n_steps=1,
    validate_every=(1, "epochs"),
)


class SortedListTrainingWrapper(TrainingWrapper[SortedList]):
    def __init__(self, config: Config[SortedList], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[SortedList]) -> HookedTransformer:
        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch list_len d_vocab"],
        labels: Integer[Tensor, "batch list_len"],
    ) -> Float[Tensor, ""]:
        log_probs = utils.log_softmax(logits, dim=-1)
        loss = F.cross_entropy(
            einops.rearrange(log_probs, "batch seq vocab_out -> (batch seq) vocab_out"),
            einops.rearrange(labels, "batch seq -> (batch seq)"),
        )

        return loss

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch list_len d_vocab"],
        labels: Integer[Tensor, "batch list_len"],
        per_token: bool = True,
    ) -> float:
        predictions = logits.argmax(dim=-1)
        correct = predictions == labels
        if not per_token:
            correct = correct.all(dim=-1)
        return correct.float().mean().item()

    def run_batch(self, x: Float[Tensor, "batch pos"], prefix: str):
        self.model.to(x.device, print_details=False)
        logits = self.model(x)[:, self.config.experiment.list_len : -1, :]
        labels = x[:, self.config.experiment.list_len + 1 :]
        loss = self.loss_fn(logits, labels)
        self.log(f"{prefix}loss", loss, prog_bar=True)
        acc = self.acc_fn(logits, labels)
        self.log(f"{prefix}acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="")

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )


class SortedListDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]
    data_test: Dataset[Integer[Tensor, "seq_len"]]
    data_train_str: Sequence[str]
    data_test_str: Sequence[str]

    def __init__(self, config: Config[SortedList]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.dataset_seed = reseed(self.config.seed, "dataset_seed")

        self.list_len = self.config.experiment.list_len
        self.seq_len = self.model_config.n_ctx
        self.max_value = self.config.experiment.max_value
        self.vocab = tuple([str(i) for i in range(self.max_value + 1)] + ["SEP"])

        assert self.list_len <= self.max_value, "list_len must be <= max_value"

    def setup(self, stage: str):
        torch.manual_seed(self.dataset_seed)
        size = (
            self.config.experiment.n_train_samples
            + self.config.experiment.n_test_samples
        )

        # Create list, by concatenating sorted & unsorted lists with SEP in the middle
        sep_toks = torch.full(size=(size, 1), fill_value=self.vocab.index("SEP"))
        unsorted_list = torch.argsort(torch.rand(size, self.max_value + 1), dim=-1)[
            :, : self.list_len
        ]
        sorted_list = torch.sort(unsorted_list, dim=-1).values
        toks = torch.concat([unsorted_list, sep_toks, sorted_list], dim=-1)
        data_train = toks[: self.config.experiment.n_train_samples]
        data_test = toks[self.config.experiment.n_test_samples :]
        self.data_train_str = tuple(
            tuple(self.vocab[i] for i in toks) for toks in data_train.tolist()
        )
        self.data_test_str = tuple(
            tuple(self.vocab[i] for i in toks) for toks in data_test.tolist()
        )

        self.data_train = cast(Dataset[Tensor], SingleTensorDataset(data_train))
        self.data_test = cast(Dataset[Tensor], SingleTensorDataset(data_test))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sorted list model.")
    parser.add_argument(
        "--force-train", action="store_true", help="Force training the model."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable saving the model."
    )
    args = parser.parse_args()

    config = SORTED_LIST_CONFIG
    print("Training model:", config)

    force_train: Optional[Literal["train"]] = "train" if args.force_train else None
    save_to: Optional[Literal["disk_and_wandb"]] = (
        None if args.no_save else "disk_and_wandb"
    )

    train_or_load_model(config, force=force_train, save_to=save_to)
