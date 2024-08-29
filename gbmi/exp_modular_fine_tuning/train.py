from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
import simple_parsing
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

from gbmi import utils
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
from gbmi.utils import (
    SingleTensorDataset,
    default_device,
    reseed,
    set_params,
    shuffle_data,
)
from gbmi.utils.sequences import generate_all_sequences


@dataclass
class ModularFineTuning(ExperimentConfig):
    p: int  # the prime
    zero_biases: bool = True
    seq_len: int = 2
    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 1
    n_heads: int = 4
    act_fn: Literal["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"] = "relu"
    normalization_type: Optional[str] = None
    attention_rate: float = 0  # 0 is use attention, 1 is uniformly constant attention
    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024
    training_ratio: float = 0.4  # fraction of dataset to use for training
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999)}
    )
    version_number: int = 1

    @property
    def n_ctx(self) -> int:
        return self.seq_len + 1

    def get_training_wrapper(self):
        return ModularFineTuningTrainingWrapper

    def get_datamodule(self):
        return ModularFineTuningDataModule

    def get_summary_slug(self, config: Config[ModularFineTuning]) -> str:
        return (
            f"ModularFineTuning-{config.experiment.n_ctx}-{config.train_for[0]}-"
            f"{config.train_for[1]}-attention-rate-{config.experiment.attention_rate}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )

    # def __post_init__(self):
    #     self.model_config.d_vocab = self.p + 1
    #     self.model_config.d_vocab_out = self.p
    #     self.model_config.__post_init__()

    def config_post_init(self, config: Config[ModularFineTuning]) -> None:
        # self.model_config.seed = reseed(config.seed, "model")
        config.batch_size = int(self.p**self.seq_len * self.training_ratio)


def modular_addition_config(attn_rate: float, p: int = 113):
    return Config(
        experiment=ModularFineTuning(
            seq_len=2,
            d_model=128,
            d_mlp=512,
            n_layers=1,
            n_heads=4,
            act_fn="relu",
            p=p,
            zero_biases=True,
            attention_rate=attn_rate,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1.0, "betas": (0.9, 0.98)},
        ),
        seed=999,
        deterministic=False,
        batch_size=int(113**2 * 0.4),
        train_for=(25000, "epochs"),
        log_every_n_steps=1,
        validate_every=(10, "epochs"),
    )


MODULAR_ADDITION_113_CLOCK_CONFIG = modular_addition_config(attn_rate=0)

MODULAR_ADDITION_113_PIZZA_CONFIG = modular_addition_config(attn_rate=1)


class ModularFineTuningTrainingWrapper(TrainingWrapper[ModularFineTuning]):
    def __init__(self, config: Config[ModularFineTuning], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model_config(
        config: Config[ModularFineTuning],
    ) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            d_vocab=config.experiment.p + 1,
            d_vocab_out=config.experiment.p,
            n_ctx=config.experiment.seq_len + 1,
            d_model=config.experiment.d_model,
            d_mlp=config.experiment.d_mlp,
            d_head=config.experiment.d_model // config.experiment.n_heads,
            n_layers=config.experiment.n_layers,
            n_heads=config.experiment.n_heads,
            act_fn=config.experiment.act_fn,
            normalization_type=config.experiment.normalization_type,
            init_weights=True,
            attn_only=False,
            seed=reseed(config.seed, "model"),
        )

    @staticmethod
    def build_model(config: Config[ModularFineTuning]) -> HookedTransformer:
        model = HookedTransformer(
            ModularFineTuningTrainingWrapper.build_model_config(config)
        )
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


class ModularFineTuningDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    data_test: Dataset[Integer[Tensor, "seq_len"]]  # noqa: F821
    batch_size: Optional[int]

    def __init__(self, config: Config[ModularFineTuning]):
        super().__init__(config)
        self.config = config
        self.dataset_seed = reseed(self.config.seed, "dataset_seed")

    def setup(self, stage: str):
        # Full dataset
        rng = np.random.default_rng(self.dataset_seed)
        pairs = generate_all_sequences(
            self.config.experiment.p, self.config.experiment.seq_len
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
    parser = simple_parsing.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_arguments(
        ModularFineTuning,
        dest="experiment_config",
        default=MODULAR_ADDITION_113_CLOCK_CONFIG.experiment,
    )
    add_force_argument(parser)
    add_no_save_argument(parser)
    Config.add_arguments(parser, default=MODULAR_ADDITION_113_CLOCK_CONFIG)

    args = parser.parse_args(argv[1:])

    config = Config(args.experiment_config)
    config = config.update_from_args(args)
    print("Model config:", ModularFineTuningTrainingWrapper.build_model_config(config))
    print("Training model:", config)
    train_or_load_model(config, force=args.force, save_to=args.save_to)


if __name__ == "__main__":
    main()
