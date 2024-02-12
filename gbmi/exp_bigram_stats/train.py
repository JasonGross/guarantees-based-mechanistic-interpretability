from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field
from collections.abc import Callable

from gbmi.exp_group_finetuning.groups import (
    Group,
    GroupDict,
    CyclicGroup,
    DihedralGroup,
    GLN_p,
)
import sys
from typing import Any, Dict, List, Optional, cast, Literal, Generic, TypeVar, Type
from gbmi import utils

import numpy as np
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader, IterableDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
import argparse
import einops
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
from gbmi.utils import (
    shuffle_data,
    default_device,
    SingleTensorDataset,
    reseed,
    set_params,
)
from gbmi.utils.sequences import generate_all_sequences

torch.set_default_device("cuda")


@dataclass
class Bigram(ExperimentConfig):
    model_config: HookedTransformerConfig

    # using int instead of abstract class because i'm clueless what's going on with typing

    zero_biases: bool = False

    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024
    training_ratio: float = 0.5  # fraction of dataset to use for training
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 1.0}
    )
    version_number: int = 1

    def get_training_wrapper(self):
        return BigramTrainingWrapper

    def get_datamodule(self):
        return BigramDataModule

    def get_summary_slug(self, config: Config[Bigram]) -> str:
        return (
            f"InductionHead-{config.experiment.model_config.n_ctx}-{config.train_for[0]}-"
            f"{config.train_for[1]}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )


def modular_addition_config(
    epochs: int,
    weight_decay: float = 1.0,
    train_ratio: float = 0.5,
    n_ctx: int = 6,
):
    return Config(
        experiment=Bigram(
            model_config=HookedTransformerConfig(
                d_vocab=3,
                d_vocab_out=3,
                n_ctx=n_ctx,
                d_model=10,
                d_head=10,
                n_layers=2,
                n_heads=4,
                act_fn="relu",
                init_weights=True,
                attn_only=True,
                normalization_type=None,
            ),
            zero_biases=False,
            training_ratio=train_ratio,
            optimizer_kwargs={
                "lr": 1e-3,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.98),
            },
        ),
        seed=999,
        deterministic=False,
        batch_size=int(3000),
        train_for=(epochs, "epochs"),
        log_every_n_steps=1,
        validate_every=(10, "epochs"),
    )


DEFAULT_BIGRAM = modular_addition_config(10000, 1, 0.7, 10)


def calculate_batch_probabilities(batch_input, num_tokens):
    # Convert batch input to a PyTorch tensor
    batch_tensor = torch.tensor(batch_input, dtype=torch.long)

    # Get the shape of the batch tensor
    batch_size, seq_length = batch_tensor.shape

    # Initialize a tensor to store the probability distributions
    # Starting with a uniform distribution for the first position
    probability_distributions = (
        torch.ones((batch_size, seq_length, num_tokens), dtype=torch.float) / num_tokens
    )

    # Create tensors to count occurrences and calculate cumulative probabilities
    for i in range(1, seq_length):
        # Count occurrences of each token in positions before the current one
        for token in range(num_tokens):
            token_occurrences = (batch_tensor[:, :i] == token).float().sum(dim=1)
            probability_distributions[:, i, token] = token_occurrences

        # Normalize to get probabilities for positions from the second onwards
        sums = probability_distributions[:, i].sum(dim=1, keepdim=True)
        probability_distributions[:, i] /= sums

    return probability_distributions


class BigramTrainingWrapper(TrainingWrapper[Bigram]):
    def __init__(self, config: Config[Bigram], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[Bigram]) -> HookedTransformer:
        model_config = config.experiment.model_config
        set_params(
            model_config,
            {
                "seed": reseed(config.seed, "model"),
                "d_vocab": 3,
                "d_vocab_out": 3,
            },
            warn_if_not_default=False,
        )

        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos"],  # noqa: F722
        labels: Integer[Tensor, "batch pos"],  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        print(labels.shape)
        print(logits.shape)

        logits = einops.rearrange(logits, "b p d-> b d p")
        logits = torch.softmax(logits, dim=1)
        print(logits[0, :, -1])
        labels = einops.rearrange(labels, "b p d -> b d p")
        loss = torch.nn.functional.cross_entropy(logits[:, :, -1], labels[:, :, -1])

        return loss

    def run_batch(
        self, x: Float[Tensor, "batch pos"], prefix: str  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        self.model.to(x.device, print_details=False)
        labels = calculate_batch_probabilities(x, 3)
        y_preds = self.model(x)
        loss = self.loss_fn(y_preds, labels)

        self.log(f"{prefix}loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="")

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )


class BigramDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    data_test: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    batch_size: Optional[int]

    def __init__(self, config: Config[Bigram]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = self.model_config.n_ctx
        self.dataset_seed = reseed(self.config.seed, "dataset_seed")

    def setup(self, stage: str):
        # Full dataset
        rng = np.random.default_rng(self.dataset_seed)
        pairs = generate_all_sequences(
            3,
            self.model_config.n_ctx,
        )
        data = shuffle_data(pairs, rng)

        split_idx = int(len(data) * self.config.experiment.training_ratio)

        data_train = data[:split_idx]
        data_test = data[split_idx:]
        print(
            f"data_train.shape: {data_train.shape}, data_test.shape: {data_test.shape}"
        )

        self.data_train = cast(Dataset[Tensor], SingleTensorDataset(data_train))
        self.data_test = cast(Dataset[Tensor], SingleTensorDataset(data_test))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.config.batch_size)


# class ModularFineTuningDataset(IterableDataset[Integer[Tensor, "seq_length"]]):
#     def __init__(
#         self, seed: int, config: Config[ModularFineTuning], max_length: Optional[int] = None
#     ):
#         self.config = config
#         self.seed = seed
#         if max_length is None:
#             n, unit = config.train_for
#             assert unit == "steps"
#             self.max_length = n * config.batch_size
#         else:
#             self.max_length = max_length

#     def __len__(self):
#         return self.max_length

#     def __iter__(self):
#         def generator():
#             g = torch.Generator()
#             g.manual_seed(self.seed)
#             n_samples = 0
#             while True:
#                 yield torch.randint(
#                     0,
#                     self.config.d_vocab,
#                     (self.config.n_ctx,),
#                     generator=g,
#                 )
#                 n_samples += 1
#                 if self.max_length is not None and n_samples >= self.max_length:
#                     return

#         return iter(generator())
"""

def main(argv: List[str] = sys.argv):
    parser = argparse.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_argument(
        "--group", type=str, default="Cyclic", help="The family of group to use."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=113,
        help="The index of the group among the specified family.",
    )
    parser.add_argument(
        "--sequence-length",
        type=float,
        default=2,
        help="The number of elements to reduce.",
    )
    parser.add_argument(
        "--attention-rate", type=float, default=0, help="Attention rate for the model."
    )

    add_force_argument(parser)
    add_no_save_argument(parser)
    HOOKED_TRANSFORMER_CONFIG_EXCLUDE_ARGS = set(("d_vocab", "d_vocab_out", "group"))
    Config.add_arguments(parser)
    add_HookedTransformerConfig_arguments(
        parser, exclude_arguments=HOOKED_TRANSFORMER_CONFIG_EXCLUDE_ARGS
    )
    args = parser.parse_args(argv[1:])

    config = modular_addition_config(
        attn_rate=args.attention_rate,
        group=GroupDict[args.group](args.index),
        elements=args.sequence_length,
    )
    config.experiment.model_config = update_HookedTransformerConfig_from_args(
        config,
        config.experiment.model_config,
        args,
        exclude_arguments=HOOKED_TRANSFORMER_CONFIG_EXCLUDE_ARGS,
    )
    config = config.update_from_args(args)
    print("Training model:", config)

    train_or_load_model(config, force=args.force, save_to=args.save_to)


if __name__ == "__main__":
    main()

    """
