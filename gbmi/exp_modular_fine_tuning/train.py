from __future__ import annotations
from dataclasses import dataclass
import sys
from typing import Optional, cast
from gbmi import utils

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
from gbmi.utils import generate_all_sequences, shuffle_data, default_device


@dataclass
class ModularFineTuning(ExperimentConfig):
    p: int # the prime
    attention_rate: float = 0 # 0 is use attention, 1 is uniformly constant attention
    n_train_samples: Optional[int] = None  # if none, infinite dataset
    n_test_samples: int = 1024
    training_ratio: float = 0.3  # fraction of dataset to use for training

    def get_training_wrapper(self):
        return ModularFineTuningTrainingWrapper

    def get_datamodule(self):
        return ModularFineTuningDataModule

    def get_summary_slug(self, config: Config[ModularFineTuning]) -> str:
        return f"ModularFineTuning-{config.n_ctx}-{config.train_for[0]}-{config.train_for[1]}-attention-rate-{config.experiment.attention_rate}"



MODULAR_ADDITION_113_CLOCK_CONFIG = Config(
    experiment=ModularFineTuning(
        attention_rate=0,
        p=113,
    ),
    n_ctx=3,
    d_model=128,
    d_mlp=512,
    d_head=32,
    n_layers=1,
    n_heads=4,
    seed=999,
    zero_biases=True,
    deterministic=False,
    batch_size=113**2,
    train_for=(25000, "epochs"),
    log_every_n_steps=1,
    act_fn="relu",
)

class ModularFineTuningTrainingWrapper(TrainingWrapper[ModularFineTuning]):
    def __init__(self, config: Config[ModularFineTuning], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[ModularFineTuning]) -> HookedTransformer:
        simpler_cfg = HookedTransformerConfig(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_head=config.d_head,
            n_ctx=config.n_ctx,
            d_vocab=config.experiment.p+1,
            d_vocab_out=config.experiment.p,
            seed=config.seed,
            attn_only=False,
            normalization_type=None,
            act_fn=config.act_fn,
            # device=default_device(deterministic=config.deterministic),
        )
        model = HookedTransformer(simpler_cfg)
        if config.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos d_vocab"],
        labels: Integer[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        logits = logits[:, -1, :]
        # TODO: FIXME
        # ../aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144: operator(): block: [7,0,0], thread: [40,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.

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
        # TODO(Euan, From Jason): Why is x a singleton list of a tensor???
        assert type(x) == list # remove when this bug is fixed
        assert len(x) == 1 # remove when bug is fixed
        x = x[0] # remove when bug is fixed
        self.model.to(x.device, print_details=False)
        labels = (x[:, 0] + x[:, 1]) % self.config.experiment.p
        assert len(labels.shape) == 1, f"labels.shape == {labels.shape} != 1 (from x.shape == {x.shape})"
        y_preds = self.model.run_with_hooks(
            x,
            fwd_hooks=[("blocks.0.attn.hook_pattern", self.attention_hook)]
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

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer_kwargs["lr"],
            betas=self.config.optimizer_kwargs["betas"],
        )


class ModularFineTuningDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]
    data_test: Dataset[Integer[Tensor, "seq_len"]]
    batch_size: Optional[int]

    def __init__(self, config: Config[ModularFineTuning]):
        super().__init__(config)
        self.config = config
        self.seq_len = config.n_ctx
        self.dataset_seed = config.seed * 10 + 1

    def setup(self, stage: str):
        # Full dataset
        rng = np.random.default_rng(self.dataset_seed)
        pairs = generate_all_sequences(self.config.experiment.p, self.config.n_ctx - 1)
        # concat a special token of value self.config.experiment.p to the end of each sequence for '='
        equals_token = self.config.experiment.p
        data = torch.cat([pairs, equals_token * torch.ones((len(pairs), 1))], dim=1).long()
        data = shuffle_data(data, rng)

        split_idx = int(len(data) * self.config.experiment.training_ratio)

        data_train = data[:split_idx]
        data_test = data[split_idx:]
        print(f"data_train.shape: {data_train.shape}, data_test.shape: {data_test.shape}")

        self.data_train = cast(Dataset[Tensor], TensorDataset(data_train))
        self.data_test = cast(Dataset[Tensor], TensorDataset(data_test))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.config.batch_size)

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
    print("Training model:", MODULAR_ADDITION_113_CLOCK_CONFIG)
    train_or_load_model(MODULAR_ADDITION_113_CLOCK_CONFIG, force="train" if "--force-train" in sys.argv[1:] else None)
