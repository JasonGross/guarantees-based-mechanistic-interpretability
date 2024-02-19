from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field

import sys
from typing import Any, Callable, Dict, List, Optional, cast, Literal
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
    shuffle_data,
    default_device,
    SingleTensorDataset,
    reseed,
    set_params,
)
from gbmi.utils.sequences import generate_all_sequences


@dataclass
class ModularArithmetic(ExperimentConfig):
    p: int = 113  # the prime
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_mlp: int = field(init=False)
    act_fn: Literal["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"] = "relu"
    normalization_type: Optional[str] = None
    seq_len: int = 2
    attention_rate: float = 0  # 0 is use attention, 1 is uniformly constant attention
    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024
    zero_biases: bool = False
    fn_name: Literal["add", "subtract", "x2xyy2"] = "add"
    training_ratio: float = 0.3  # fraction of dataset to use for training
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "weight_decay": 1.0, "betas": (0.9, 0.98)}
    )
    version_number: int = 1

    def get_training_wrapper(self):
        return ModularArithmeticTrainingWrapper

    def get_datamodule(self):
        return ModularArithmeticDataModule

    def get_summary_slug(self, config: Config[ModularArithmetic]) -> str:
        return (
            f"Modular{config.experiment.fn_name.capitalize()}"
            f"-{config.experiment.seq_len}-{config.train_for[0]}-"
            f"{config.train_for[1]}-attention-rate-{config.experiment.attention_rate}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )

    def fn(self, x: int, y: int) -> int:
        p = self.p
        match self.fn_name:
            case "add":
                return (x + y) % p
            case "subtract":
                return (x - y) % p
            case "x2xyy2":
                return (x**2 + x * y + y**2) % p

    @property
    def model_config(self) -> HookedTransformerConfig:
        assert (
            self.d_model % self.n_heads == 0
        ), f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        return HookedTransformerConfig(
            d_vocab=self.p + 1,
            d_vocab_out=self.p,
            n_ctx=self.seq_len + 1,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            d_head=self.d_model // self.n_heads,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            act_fn=self.act_fn,
            normalization_type=self.normalization_type,
        )

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model

    def config_post_init(self, config: Config[ModularArithmetic]) -> None:
        self.model_config.seed = reseed(config.seed, "model")
        config.batch_size = int(self.p**self.seq_len * self.training_ratio)


CLOCK_CONFIG = Config(
    experiment=ModularArithmetic(p=113, training_ratio=0.3),
    seed=0,
    deterministic=False,
    train_for=(50000, "epochs"),
    log_every_n_steps=1,
    validate_every=(1, "epochs"),
    checkpoint_every=(500, "epochs"),
)

PIZZA_CONFIG = Config(
    experiment=ModularArithmetic(p=113, training_ratio=0.8),
    seed=0,
    deterministic=False,
    train_for=(10000, "epochs"),
    log_every_n_steps=1,
    validate_every=(1, "epochs"),
    checkpoint_every=(500, "epochs"),
)


class ModularArithmeticTrainingWrapper(TrainingWrapper[ModularArithmetic]):
    def __init__(self, config: Config[ModularArithmetic], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[ModularArithmetic]) -> HookedTransformer:
        model = HookedTransformer(config.experiment.model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def log_softmax(x: Tensor, **kwargs) -> Tensor:
        x = x.to(torch.float64)
        return utils.log_softmax(x, **kwargs)

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch n_ctx d_vocab_out"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
        log_softmax: Optional[Callable] = None,
    ) -> Float[Tensor, ""]:  # noqa F722
        if log_softmax is None:
            log_softmax = ModularArithmeticTrainingWrapper.log_softmax
        log_probs = log_softmax(logits[:, -1, :], dim=-1)
        correct_log_probs = log_probs.gather(-1, labels.unsqueeze(-1))
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch pos d_vocab_out"],  # noqa: F722
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
        return torch.optim.AdamW(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )


class ModularArithmeticDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    data_test: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    batch_size: Optional[int]

    def __init__(self, config: Config[ModularArithmetic]):
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


def main(argv: List[str] = sys.argv):
    parser = argparse.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_argument("--p", type=int, default=113, help="The prime to use.")
    parser.add_argument(
        "--attention-rate", type=float, default=0, help="Attention rate for the model."
    )
    add_force_argument(parser)
    add_no_save_argument(parser)
    HOOKED_TRANSFORMER_CONFIG_EXCLUDE_ARGS = set(
        (
            "d_vocab",
            "d_vocab_out",
            "seed",
        )
    )
    Config.add_arguments(parser)
    add_HookedTransformerConfig_arguments(
        parser, exclude_arguments=HOOKED_TRANSFORMER_CONFIG_EXCLUDE_ARGS
    )
    args = parser.parse_args(argv[1:])

    config = Config(ModularArithmetic(attn_rate=args.attention_rate, p=args.p))
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
