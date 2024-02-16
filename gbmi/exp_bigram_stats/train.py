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
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
    Literal,
    Generic,
    TypeVar,
    Type,
    Tuple,
)
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
    bos: bool = True
    seq_length: int = 30

    n_test_samples: int = 1024

    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 1.0}
    )
    version_number: int = 1

    def __post_init__(self):
        self.model_config.n_ctx = self.seq_length
        if self.bos:
            self.model_config.n_ctx = self.seq_length + 1
            self.model_config.d_vocab = self.model_config.d_vocab_out + 1
        self.model_config.__post_init__()

    def config_post_init(self, config: Config[Bigram]) -> None:
        self.model_config.seed = reseed(config.seed, "model")

    def get_training_wrapper(self):
        return BigramTrainingWrapper

    def get_datamodule(self):
        return BigramDataModule

    def get_summary_slug(self, config: Config[Bigram]) -> str:
        return (
            f"InductionHead-{config.experiment.seq_length}-{config.train_for[0]}-"
            f"{config.train_for[1]}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )


def bigram_config(
    samples: int,
    weight_decay: float = 1.0,
    seq_length: int = 5,
    bos: bool = True,
    d_vocab_out=3,
    batch_size=512,
):
    return Config(
        experiment=Bigram(
            model_config=HookedTransformerConfig(
                d_vocab=d_vocab_out + bos,
                d_vocab_out=d_vocab_out,
                n_ctx=seq_length + bos,
                d_model=32,
                d_head=32,
                n_layers=2,
                n_heads=1,
                init_weights=True,
                attn_only=True,
                normalization_type=None,
            ),
            zero_biases=False,
            seq_length=seq_length,
            bos=bos,
            optimizer_kwargs={
                "lr": 1e-3,
                "weight_decay": weight_decay,
                "betas": (0.5, 0.5),
            },
        ),
        seed=999,
        deterministic=False,
        batch_size=batch_size,
        train_for=(samples // batch_size, "steps"),
        log_every_n_steps=1,
        validate_every=(10, "steps"),
    )


DEFAULT_BIGRAM = bigram_config(125000, 1.0, seq_length=15)


def calculate_batch_probabilities(
    batch_input: Integer[Tensor, "... seq_length"], num_tokens: int  # noqa: F821, F722
) -> Float[Tensor, "... seq_length num_tokens"]:  # noqa: F821, F722
    # Convert batch input to a PyTorch tensor
    batch_tensor = torch.tensor(batch_input, dtype=torch.long)

    # Get the shape of the batch tensor
    batch_dims, seq_length = batch_tensor.shape[:-1], batch_tensor.shape[-1]

    # Initialize a tensor to store the probability distributions
    # Starting with a uniform distribution for the first position
    probability_distributions = (
        torch.ones(batch_dims + (seq_length, num_tokens), dtype=torch.float)
        / num_tokens
    )

    # Create tensors to count occurrences and calculate cumulative probabilities
    for i in range(1, seq_length):
        # Count occurrences of each token in positions before the current one
        for token in range(num_tokens):
            token_occurrences = (batch_tensor[..., :i] == token).float().sum(dim=-1)
            probability_distributions[..., i, token] = token_occurrences

        # Normalize to get probabilities for positions from the second onwards

        sums = probability_distributions[..., i, :].sum(
            dim=len(probability_distributions.shape) - 2, keepdim=True
        )
        probability_distributions[..., i, :] /= sums

    return probability_distributions


class BigramBaseIterableDataset(IterableDataset[Integer[Tensor, "seq_length"]]):
    def __init__(
        self,
        seed: int,
        config: Config[Bigram],
        max_length: Optional[int] = None,
    ):
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_length = config.experiment.seq_length
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
            default_device = torch.tensor([]).device

            g = torch.Generator(device=default_device)
            g.manual_seed(self.seed)
            n_samples = 0
            while True:
                val = torch.randint(
                    0,
                    self.model_config.d_vocab_out,
                    (self.seq_length,),
                    generator=g,
                )

                yield val
                n_samples += 1
                if self.max_length is not None and n_samples >= self.max_length:
                    return
                # TODO: add adversarial generation

        return iter(generator())


class BigramLabeledDataset(
    IterableDataset[
        Tuple[
            Integer[Tensor, "seq_length"],  # noqa F821 F722
            Float[Tensor, "seq_length num_tokens"],  # noqa F821 F722
        ]
    ]
):
    def __init__(
        self,
        unlabeled_dataset: IterableDataset[Integer[Tensor, "seq_length"]],  # noqa: F821
        num_tokens: int,
    ):
        self.dataset = unlabeled_dataset
        self.num_tokens = num_tokens

    def __len__(self):
        return len(self.dataset)

    def label(self, val: Integer[Tensor, "seq_length"]) -> Tuple[  # noqa: F821 F722
        Integer[Tensor, "seq_length"],  # noqa: F821, F722
        Float[Tensor, "seq_length num_tokens"],  # noqa: F821, F722
    ]:
        return val, calculate_batch_probabilities(val, self.num_tokens)

    def __iter__(self):
        for val in self.dataset:
            yield self.label(val)

    def __getitem__(self, index):
        return self.label(self.dataset[index])


class BigramCatBOSLabeledDataset(
    IterableDataset[
        Tuple[
            Integer[Tensor, "n_ctx"],  # noqa: F821, F722
            Float[Tensor, "n_ctx num_tokens"],  # noqa: F821, F722
        ]
    ]
):
    def __init__(
        self,
        labeled_dataset: IterableDataset[
            Tuple[
                Integer[Tensor, "seq_length"],  # noqa: F821, F722
                Float[Tensor, "seq_length num_tokens"],  # noqa: F821, F722
            ]
        ],
        bos: Optional[int] = None,
    ):
        self.dataset = labeled_dataset
        self.bos = bos

    def __len__(self):
        return len(self.dataset)

    def cat_bos(
        self,
        val: Tuple[
            Integer[Tensor, "... seq_length"],  # noqa: F821, F722
            Float[Tensor, "seq_length num_tokens"],  # noqa: F821, F722
        ],
    ) -> Tuple[
        Integer[Tensor, "... n_ctx"],  # noqa: F821, F722
        Float[Tensor, "seq_length num_tokens"],  # noqa: F821, F722
    ]:
        x, y = val
        num_tokens = y.shape[-1]
        if self.bos is None:
            return x, y
        return (
            torch.cat(
                [
                    torch.full(
                        x.shape[:-1] + (1,),
                        self.bos,
                        dtype=torch.long,
                        device=x.device,
                    ),
                    x,
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    torch.full(
                        y.shape[:-2] + (1, num_tokens),
                        1 / num_tokens,
                        dtype=y.dtype,
                        device=y.device,
                    ),
                    y,
                ],
                dim=-2,
            ),
        )

    def __iter__(self):
        for val in self.dataset:
            yield self.cat_bos(val)

    def __getitem__(self, index):
        return self.cat_bos(self.dataset[index])


class BigramTrainingWrapper(TrainingWrapper[Bigram]):
    def __init__(self, config: Config[Bigram], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[Bigram]) -> HookedTransformer:
        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        labels: Integer[Tensor, "batch pos num_tokens"],  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        logits = torch.softmax(logits, dim=-1)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss

    def run_batch(
        self,
        x_y: Tuple[
            Integer[Tensor, "batch pos"],  # noqa F722
            Float[Tensor, "batch pos num_tokens"],  # noqa F722
        ],
        prefix: Optional[str] = None,
        *,
        log_output: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Float[Tensor, ""]:  # noqa F722
        assert prefix is not None or not log_output, "Must not log if prefix is None"
        xs, ys = x_y
        if device is not None:
            xs = xs.to(device)
        ys = ys.to(xs.device)
        self.model.to(xs.device, print_details=False)
        y_preds = self.model(xs)
        loss = self.loss_fn(y_preds, ys)
        if log_output:
            self.log(f"{prefix}loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="")

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )


class BigramDataModule(DataModule):
    data_train: Dataset[
        Tuple[
            Integer[Tensor, "n_ctx"],  # noqa: F821
            Float[Tensor, "n_ctx num_tokens"],  # noqa: F821, F722
        ]
    ]
    data_test: Dataset[
        Tuple[
            Integer[Tensor, "n_ctx"],  # noqa: F821, F722
            Float[Tensor, "n_ctx num_tokens"],  # noqa: F821, F722
        ]
    ]
    batch_size: Optional[int]
    seq_length: int
    bos: Optional[int]
    dataset_seed: int
    num_tokens: int

    def __init__(self, config: Config[Bigram]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config

        self.num_tokens = self.model_config.d_vocab_out
        self.seq_length = config.experiment.seq_length
        self.bos = (
            config.experiment.model_config.d_vocab - 1
            if config.experiment.bos
            else None
        )
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    def build_dataset(
        self, mode: Literal["train", "test"]
    ) -> Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]:
        base_dataset = BigramBaseIterableDataset(
            reseed(self.dataset_seed, mode),
            self.config,
            max_length=(
                self.config.experiment.n_test_samples if mode == "test" else None
            ),
        )

        return BigramCatBOSLabeledDataset(
            BigramLabeledDataset(base_dataset, num_tokens=self.num_tokens),
            bos=self.bos,
        )

    def setup(self, stage: str):
        self.data_train = self.build_dataset("train")
        self.data_test = self.build_dataset("test")

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
