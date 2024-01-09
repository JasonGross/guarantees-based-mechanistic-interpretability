from __future__ import annotations
from dataclasses import dataclass, field


import sys
from typing import Optional, cast, Literal
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
class ModularFineTuning(ExperimentConfig):
    model_config: HookedTransformerConfig
    p: int  # the prime
    zero_biases: bool = True
    attention_rate: float = 0  # 0 is use attention, 1 is uniformly constant attention
    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024
    training_ratio: float = 0.3  # fraction of dataset to use for training
    version_number: int = 0

    def get_training_wrapper(self):
        return ModularFineTuningTrainingWrapper

    def get_datamodule(self):
        return ModularFineTuningDataModule

    def get_summary_slug(self, config: Config[ModularFineTuning]) -> str:
        return (
            f"ModularFineTuning-{config.experiment.model_config.n_ctx}-{config.train_for[0]}-"
            f"{config.train_for[1]}-attention-rate-{config.experiment.attention_rate}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )


def modular_addition_config(attn_rate: float, p: int = 113):
    return Config(
        experiment=ModularFineTuning(
            model_config=HookedTransformerConfig(
                n_ctx=3,
                d_model=128,
                d_mlp=512,
                d_head=32,
                n_layers=1,
                n_heads=4,
                act_fn="relu",
                attn_only=False,
                normalization_type=None,
            ),
            p=p,
            zero_biases=True,
            attention_rate=attn_rate,
        ),
        seed=999,
        deterministic=False,
        batch_size=113**2,
        train_for=(25000, "epochs"),
        log_every_n_steps=1,
        validate_every=(10, "epochs"),
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 1.0, "betas": (0.9, 0.98)},
    )


MODULAR_ADDITION_113_CLOCK_CONFIG = modular_addition_config(attn_rate=0)

MODULAR_ADDITION_113_PIZZA_CONFIG = modular_addition_config(attn_rate=1)


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
                "d_vocab": config.experiment.p + 1,
                "d_vocab_out": config.experiment.p,
            },
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
        labels: Integer[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        logits = logits[:, -1, :].to(torch.float64)
        log_probs = utils.log_softmax(logits, dim=-1)
        correct_log_probs = log_probs.gather(-1, labels.unsqueeze(-1))[:, 0]
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch pos d_vocab"],
        labels: Integer[Tensor, "batch"],
    ) -> float:
        logits = logits[:, -1, :]
        predictions = logits.argmax(dim=-1)
        return (predictions == labels).float().mean().item()

    def attention_hook(self, attnscore, hook):
        alpha = self.config.experiment.attention_rate
        # note that this is different from the paper, which does not do the division to enforce the constraint
        # that the attention scores add up to 1
        return alpha / attnscore.shape[-1] + (1 - alpha) * attnscore

    def run_batch(self, x: Float[Tensor, "batch pos"], prefix: str):
        self.model.to(x.device, print_details=False)
        labels = (x[:, 0] + x[:, 1]) % self.config.experiment.p
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
        return torch.optim.Adam(
            self.parameters(),
            **self.config.optimizer_kwargs,
        )


class ModularFineTuningDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]
    data_test: Dataset[Integer[Tensor, "seq_len"]]
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
            self.config.experiment.p, self.model_config.n_ctx - 1
        )
        # concat a special token of value self.config.experiment.p to the end of each sequence for '='
        equals_token = self.config.experiment.p
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_argument(
        "--attention-rate", type=float, default=0, help="Attention rate for the model."
    )
    parser.add_argument(
        "--force-train", action="store_true", help="Force training the model."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable saving the model."
    )
    args = parser.parse_args()

    config = modular_addition_config(args.attention_rate)
    print("Training model:", config)

    force_train: Optional[Literal["train"]] = "train" if args.force_train else None
    save_to: Optional[Literal["disk_and_wandb"]] = (
        None if args.no_save else "disk_and_wandb"
    )

    train_or_load_model(config, force=force_train, save_to=save_to)
