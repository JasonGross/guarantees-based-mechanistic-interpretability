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
    generate_all_sequences,
    shuffle_data,
    default_device,
    SingleTensorDataset,
    reseed,
    set_params,
)


@dataclass
class ModularFineTuning(ExperimentConfig):
    model_config: HookedTransformerConfig
    # using int instead of abstract class because i'm clueless what's going on with typing
    group_family: str
    group_index: int
    group_size: int
    group_name: str
    zero_biases: bool = True
    attention_rate: float = 0  # 0 is use attention, 1 is uniformly constant attention
    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024
    training_ratio: float = 0.4  # fraction of dataset to use for training
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999)}
    )
    version_number: int = 1

    def get_training_wrapper(self):
        return ModularFineTuningTrainingWrapper

    def get_datamodule(self):
        return ModularFineTuningDataModule

    def get_summary_slug(self, config: Config[ModularFineTuning]) -> str:
        return (
            f"GroupFineTuning-{config.experiment.group_family+str(config.experiment.group_index)}-{config.experiment.model_config.n_ctx}-{config.train_for[0]}-"
            f"{config.train_for[1]}-attention-rate-{config.experiment.attention_rate}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )


def modular_addition_config(attn_rate: float, group: Group, elements: int):
    return Config(
        experiment=ModularFineTuning(
            model_config=HookedTransformerConfig(
                n_ctx=elements + 1,
                d_model=128,
                d_mlp=512,
                d_head=32,
                n_layers=1,
                n_heads=4,
                act_fn="relu",
                init_weights=True,
                attn_only=False,
                normalization_type=None,
            ),
            group_family=type(group).__name__,
            group_index=group.index(),
            group_size=group.size(),
            group_name=group.name(),
            zero_biases=True,
            attention_rate=attn_rate,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1.0, "betas": (0.9, 0.98)},
        ),
        seed=999,
        deterministic=False,
        batch_size=int((group.size()) ** (elements + 1) * 0.4),
        train_for=(25000, "epochs"),
        log_every_n_steps=1,
        validate_every=(10, "epochs"),
    )


MODULAR_ADDITION_113_CLOCK_CONFIG = modular_addition_config(
    attn_rate=0, group=CyclicGroup(113), elements=2
)

MODULAR_ADDITION_113_PIZZA_CONFIG = modular_addition_config(
    attn_rate=1, group=CyclicGroup(113), elements=2
)
DIHEDRAL_100_CLOCK_CONFIG = modular_addition_config(
    attn_rate=0, group=DihedralGroup(104), elements=2
)
DIHEDRAL_100_PIZZA_CONFIG = modular_addition_config(
    attn_rate=1, group=DihedralGroup(104), elements=2
)


class ModularFineTuningTrainingWrapper(TrainingWrapper[ModularFineTuning]):
    def __init__(self, config: Config[ModularFineTuning], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[ModularFineTuning]) -> HookedTransformer:
        model_config = config.experiment.model_config
        set_params(
            model_config,
            {
                "seed": reseed(config.seed, "model"),
                "d_vocab": config.experiment.group_size + 1,
                "d_vocab_out": config.experiment.group_size,
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
        logits: Float[Tensor, "batch pos d_vocab"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> Float[Tensor, ""]:  # noqa: F722
        logits = logits
        labels = labels
        logits = logits[:, -1, :].to(torch.float64)

        log_probs = utils.log_softmax(logits, dim=-1)

        correct_log_probs = log_probs.gather(-1, labels.unsqueeze(-1))[:, 0]

        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch pos d_vocab"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> float:
        logits = logits[:, -1, :]
        predictions = logits.argmax(dim=-1)
        return (predictions == labels).float().mean().item()

    def attention_hook(self, attnscore, hook):
        alpha = self.config.experiment.attention_rate
        # note that this is different from the paper, which does not do the division to enforce the constraint
        # that the attention scores add up to 1
        return alpha / attnscore.shape[-1] + (1 - alpha) * attnscore

    def run_batch(
        self, x: Float[Tensor, "batch pos"], prefix: str  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        self.model.to(x.device, print_details=False)

        labels = GroupDict[self.config.experiment.group_family](
            self.config.experiment.group_index
        ).reduce(list(x[:, :-1].T))
        assert (
            len(labels.shape) == 1
        ), f"labels.shape == {labels.shape} != 1 (from x.shape == {x.shape})"
        y_preds = self.model.run_with_hooks(
            x, fwd_hooks=[("blocks.0.attn.hook_pattern", self.attention_hook)]
        )
        loss = self.loss_fn(y_preds, labels)

        self.log(f"{prefix}loss", loss, prog_bar=True)
        acc = self.acc_fn(y_preds, labels)
        self.log(f"{prefix}acc", acc, prog_bar=True)
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


class ModularFineTuningDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    data_test: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    batch_size: Optional[int]

    def __init__(self, config: Config[ModularFineTuning]):
        super().__init__(config)
        self.config = config
        self.model_config = config.experiment.model_config
        self.seq_len = self.model_config.n_ctx
        self.dataset_seed = reseed(self.config.seed, "dataset_seed")

    def setup(self, stage: str):
        # Full dataset
        rng = np.random.default_rng(self.dataset_seed)
        pairs = generate_all_sequences(
            self.config.experiment.group_size,
            self.model_config.n_ctx - 1,
        )
        # concat a special token of value self.config.experiment.p to the end of each sequence for '='
        equals_token = self.config.experiment.group_size
        data = torch.cat(
            [pairs, equals_token * torch.ones((len(pairs), 1))], dim=1
        ).long()
        data = shuffle_data(data, rng)

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
