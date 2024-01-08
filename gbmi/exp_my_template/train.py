from __future__ import annotations
from dataclasses import dataclass

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig

from gbmi.model import (
    TrainingWrapper,
    Config,
    ExperimentConfig,
    train_or_load_model,
    DataModule,
)


@dataclass
class MyTemplate(ExperimentConfig):
    # Experiment config dataclass. Add experiment-specific settings here.
    some_setting: int = 1

    def get_training_wrapper(self):
        return MyTemplateTrainingWrapper

    def get_datamodule(self):
        return MyTemplateDataModule

    def get_summary_slug(self, config: Config[MyTemplate]) -> str:
        # Returns a brief summary of config settings.
        # e.g. "MyTemplate-1000-steps-setting-1"
        return f"MyTemplate-{config.train_for[0]}-{config.train_for[1]}{'-nondeterministic' if not config.deterministic else ''}"


# Put any 'default' configs for your experiment type here.
MY_TEMPLATE_CONFIG_V1 = Config(experiment=MyTemplate(some_setting=1))


class MyTemplateTrainingWrapper(TrainingWrapper[MyTemplate]):
    def __init__(self, config: Config[MyTemplate], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[MyTemplate]) -> HookedTransformer:
        # Given a config, returns an untrained HookedTransformer.
        simpler_cfg = HookedTransformerConfig(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_head=config.d_head,
            n_ctx=config.n_ctx,
            d_vocab=config.d_vocab,
            seed=config.seed,
            attn_only=True,
            normalization_type=None,
        )
        model = HookedTransformer(simpler_cfg)
        if config.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    def run_batch(
        self, x: Float[Tensor, "batch pos"], prefix: str
    ) -> Float[Tensor, ""]:
        # Given a batch of inputs, returns the model loss on those inputs (and logs appropriate values).
        raise NotImplementedError

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


class MyTemplateDataModule(DataModule):
    data_train: Dataset[Integer[Tensor, "seq_len"]]
    data_test: Dataset[Integer[Tensor, "seq_len"]]

    def __init__(self, config: Config[MyTemplate]):
        super(MyTemplateDataModule, self).__init__(config)
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        # Setup anything required for your dataloaders.
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[Float[Tensor, "seq_length"]]:
        # Return a (batched) dataloader for training data.
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader[Float[Tensor, "seq_length"]]:
        # return a (batched) dataloader for test data.
        raise NotImplementedError


if __name__ == "__main__":
    print("Training model:", MY_TEMPLATE_CONFIG_V1)
    train_or_load_model(MY_TEMPLATE_CONFIG_V1, force="train")
