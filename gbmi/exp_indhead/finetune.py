from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import simple_parsing
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

from gbmi.exp_indhead.data_utils import (
    ABCABCEnglishTask,
    ABCABCExhaustiveTask,
    ABCBCEnglishTask,
    ABCBCTask,
    EnglishExactNgramTask,
    ExactBigramTask,
    ExhaustiveTask,
    calculate_batch_probabilities,
    cat_bos_token,
    cat_bos_uniform_labels,
)
from gbmi.exp_indhead.train import (
    ABCAB8_1H,
    IndHead,
    IndHeadDataModule,
    IndHeadTrainingWrapper,
)
from gbmi.model import (
    Config,
    DataModule,
    ExperimentConfig,
    TrainingWrapper,
    add_force_argument,
    add_no_save_argument,
    train_or_load_model,
)
from gbmi.training_tools.logging import ModelMatrixLoggingOptions
from gbmi.utils import reseed, set_params
from gbmi.utils.hashing import _EXCLUDE


@dataclass
class IndHeadOnlyFineTune:
    ngram: int = 3
    task: Literal["exact-bigram", "exact-ngram", "abcab", "abcabc", "exhaustive"] = (
        "exact-bigram"
    )
    corpus: Optional[str] = None
    only_last_tokens: Optional[int] = None
    only_strong_signal: bool = True
    random_tokens_at_end: bool = False
    other_tokens_distinct_from_predicted_token: bool = False
    alpha_mix_uniform: Optional[float] = None
    high_precision: bool = True
    n_train_samples: int = 4096
    n_test_samples: int = 1
    n_validate_samples: int = 1024
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 1.0}
    )
    summary_slug_extra: str = ""
    version_number: int = 1
    datagen_version_number: Optional[int] = None
    logging_options: ModelMatrixLoggingOptions = field(
        default_factory=ModelMatrixLoggingOptions
    )

    @staticmethod
    def of_IndHead(config: IndHead) -> IndHeadOnlyFineTune:
        return IndHeadOnlyFineTune(
            ngram=config.ngram,
            task=config.task,
            corpus=config.corpus,
            only_last_tokens=config.only_last_tokens,
            only_strong_signal=config.only_strong_signal,
            random_tokens_at_end=config.random_tokens_at_end,
            other_tokens_distinct_from_predicted_token=config.other_tokens_distinct_from_predicted_token,
            alpha_mix_uniform=config.alpha_mix_uniform,
            high_precision=config.high_precision,
            n_train_samples=config.n_train_samples,
            n_test_samples=config.n_test_samples,
            n_validate_samples=config.n_validate_samples,
            optimizer_kwargs=config.optimizer_kwargs,
            summary_slug_extra=config.summary_slug_extra,
            logging_options=config.logging_options,
        )

    def update_IndHead(self, other: IndHead) -> IndHead:
        return set_params(
            other,
            {
                k: v
                for k, v in self.__dict__.items()
                if k not in ("version_number", "datagen_version_number")
            },
            post_init=True,
        )

    @property
    def corpus_relevant(self):
        return self.task in ("abcab", "abcabc", "exact-ngram")

    @property
    def ngram_relevant(self):
        return self.corpus_relevant and self.corpus is not None

    def __post_init__(self):
        match self.task:
            case "exact-bigram" | "exact-ngram" | "abcab" | "exhaustive" | "abcabc":
                self.datagen_version_number = None
        exclude: set[str] = set(getattr(self, _EXCLUDE, ()))
        for field, should_ignore in [
            ("logging_options", True),
            ("corpus", self.corpus_relevant),
            ("ngram", self.ngram == 3 or not self.ngram_relevant),
            ("datagen_version_number", self.datagen_version_number is None),
        ]:
            if should_ignore:
                exclude.add(field)
            else:
                exclude.discard(field)
        setattr(self, _EXCLUDE, tuple(sorted(exclude)))

    def get_summary_slug(self, config: Config) -> str:
        return (
            f"IndHead"
            f"{f'{self.ngram}gram' if self.ngram != 3 and self.ngram_relevant else ''}"
            f"{f'-{self.task}' if self.task != 'exact-bigram' else ''}"
            f"-{config.train_for[0]}-{config.train_for[1]}"
            f"{'-randend' if self.random_tokens_at_end else ''}"
            f"{f'-{self.corpus}' if self.corpus and self.task != 'exact-bigram' else ''}"
            f"{f'-{self.alpha_mix_uniform:.2f}U' if self.alpha_mix_uniform and self.corpus and self.task != 'exact-bigram' else ''}"
            f"{'-' + self.summary_slug_extra if self.summary_slug_extra else ''}"
            f"{'-nondet' if not config.deterministic else ''}"
        )


@dataclass
class IndHeadFineTune(ExperimentConfig):
    train: Config[IndHead]
    finetune: IndHeadOnlyFineTune
    base_model_force: Literal["train", "load", None] = "load"
    version_number: int = 1

    def __post_init__(self):
        exclude: set[str] = set(getattr(self, _EXCLUDE, ()))
        for field, should_ignore in [
            ("base_model_force", True),
        ]:
            if should_ignore:
                exclude.add(field)
            else:
                exclude.discard(field)
        setattr(self, _EXCLUDE, tuple(sorted(exclude)))
        self.train.__post_init__()
        self.finetune.__post_init__()

    def get_training_wrapper(self):
        return IndHeadFineTuneTrainingWrapper

    def get_datamodule(self):
        return IndHeadFineTuneDataModule

    def get_summary_slug(self, config: Config[IndHeadFineTune]) -> str:
        return (
            f"{self.train.experiment.get_summary_slug(self.train)}"
            "-FineTune-"
            f"{self.finetune.get_summary_slug(config)}"
        )

    @property
    def bos(self) -> bool:
        return self.train.experiment.bos

    @property
    def num_tokens(self) -> int:
        return self.train.experiment.num_tokens

    @property
    def bos_token(self) -> Optional[int]:
        return self.train.experiment.bos_token

    def get_ground_truth(
        self,
        x: Integer[Tensor, "... n"],  # noqa: F722
        readoff: Optional[Bool[Tensor, "... n"]] = None,  # noqa: F722
    ) -> Integer[Tensor, "..."]:  # noqa: F722
        return self.train.experiment.get_ground_truth(x, readoff)

    @staticmethod
    def from_IndHead(
        config: Config[IndHead],
        config_finetune: IndHead,
        base_model_force: Literal["train", "load", None] = "load",
    ) -> IndHeadFineTune:
        return IndHeadFineTune(
            train=config,
            finetune=IndHeadOnlyFineTune.of_IndHead(config_finetune),
            base_model_force=base_model_force,
        )

    @staticmethod
    def from_IndHeadConfig(
        config: Config[IndHead],
        config_finetune: Config[IndHead],
        base_model_force: Literal["train", "load", None] = "load",
    ) -> Config[IndHeadFineTune]:
        return cast(
            Config[IndHeadFineTune],
            set_params(
                cast(Config, config_finetune),
                {
                    "experiment": IndHeadFineTune.from_IndHead(
                        config,
                        config_finetune.experiment,
                        base_model_force=base_model_force,
                    ),
                },
                post_init=True,
            ),
        )

    @staticmethod
    def to_IndHeadFineTuneConfig(
        config: Config[IndHeadFineTune],
    ) -> Config[IndHead]:
        return cast(
            Config[IndHead],
            set_params(
                cast(Config, config),
                {
                    "experiment": config.experiment.finetune.update_IndHead(
                        config.experiment.train.experiment
                    ),
                },
                post_init=True,
            ),
        )


class IndHeadFineTuneTrainingWrapper(
    TrainingWrapper[IndHeadFineTune], IndHeadTrainingWrapper
):
    def __init__(self, config: Config[IndHeadFineTune], model: HookedTransformer):
        # super().__init__(config, model)
        # self.config = config
        finetune_config = IndHeadFineTune.to_IndHeadFineTuneConfig(config)
        finetune_training_wrapper_module = (
            finetune_config.experiment.get_training_wrapper()
        )
        finetune_training_wrapper_module.__init__(self, finetune_config, model)

    @staticmethod
    def build_model(config: Config[IndHeadFineTune]) -> HookedTransformer:
        _runtime, model = train_or_load_model(
            config.experiment.train,
            force=config.experiment.base_model_force,
        )
        return model

    loss_fn = IndHeadTrainingWrapper.loss_fn
    run_batch = IndHeadTrainingWrapper.run_batch
    training_step = IndHeadTrainingWrapper.training_step
    validation_step = IndHeadTrainingWrapper.validation_step
    test_step = IndHeadTrainingWrapper.test_step
    configure_optimizers = IndHeadTrainingWrapper.configure_optimizers


class IndHeadFineTuneDataModule(DataModule):
    config: Config[IndHeadFineTune]
    finetune_config: Config[IndHead]
    finetune_data_module: IndHeadDataModule

    def __init__(self, config: Config[IndHeadFineTune]):
        super().__init__(config)
        self.config = config
        self.finetune_config = IndHeadFineTune.to_IndHeadFineTuneConfig(config)
        self.finetune_data_module = self.finetune_config.experiment.get_datamodule()(
            self.finetune_config
        )

    def build_dataset(
        self, mode: Literal["train", "test", "validate"]
    ) -> Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]:
        return self.finetune_data_module.build_dataset(mode)

    def setup(self, stage: str):
        self.finetune_data_module.setup(stage)

    def train_dataloader(self):
        return self.finetune_data_module.train_dataloader()

    def val_dataloader(self):
        return self.finetune_data_module.val_dataloader()

    def test_dataloader(self):
        return self.finetune_data_module.test_dataloader()


def make_default_finetune(
    config: Config[IndHead],
    alpha_mix_uniform: float = 1,
    base_model_force: Literal["train", "load", None] = "load",
) -> Config[IndHeadFineTune]:
    return IndHeadFineTune.from_IndHeadConfig(
        config,
        set_params(
            config,
            {
                ("experiment", "alpha_mix_uniform"): alpha_mix_uniform,
            },
            post_init=True,
        ),
        base_model_force=base_model_force,
    )


ABCAB8_1H_FINETUNE = make_default_finetune(ABCAB8_1H)


def main(
    argv: List[str] = sys.argv,
    default: Config[IndHeadFineTune] = ABCAB8_1H_FINETUNE,
    default_force: Literal["train", "load", None] = None,
):
    parser = simple_parsing.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_arguments(
        IndHeadFineTune, dest="experiment_config", default=default.experiment
    )
    add_force_argument(parser, default=default_force)
    add_no_save_argument(parser)
    Config.add_arguments(parser, default=default)

    args = parser.parse_args(argv[1:])

    config = Config(args.experiment_config)
    config = config.update_from_args(args)
    print(
        "Model config:", IndHeadTrainingWrapper.build_model(config.experiment.train).cfg
    )
    print("Training model:", config)
    return train_or_load_model(config, force=args.force, save_to=args.save_to)


if __name__ == "__main__":
    main()
