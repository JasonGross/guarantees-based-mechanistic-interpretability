from __future__ import annotations
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass
from dataclasses import field
from collections.abc import Callable
from gbmi.exp_bigram_stats.data_utils import (
    ExactBigramTask,
    ABCBCBigramTask,
    calculate_batch_probabilities,
    cat_bos_token,
    cat_bos_uniform_labels,
)

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

from gbmi.utils.hashing import _EXCLUDE
from gbmi.training_tools.logging import (
    ModelMatrixLoggingOptions,
)


@dataclass
class Bigram(ExperimentConfig):
    # using int instead of abstract class because i'm clueless what's going on with typing
    zero_biases: bool = False
    bos: bool = True
    seq_length: int = 5
    num_tokens: int = 3
    d_model: int = 8
    task: Literal["exact-bigram", "abcab"] = "exact-bigram"
    only_last_tokens: Optional[int] = None
    only_strong_signal: bool = True
    random_tokens_at_end: bool = False
    n_heads: int = 1

    n_train_samples: int = 4096
    n_test_samples: int = 1
    n_validate_samples: int = 1024

    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 1.0}
    )
    version_number: int = 2
    logging_options: ModelMatrixLoggingOptions = field(
        default_factory=ModelMatrixLoggingOptions
    )

    def __post_init__(self):
        setattr(self, _EXCLUDE, ("logging_options",))

    def get_training_wrapper(self):
        return BigramTrainingWrapper

    def get_datamodule(self):
        return BigramDataModule

    def get_summary_slug(self, config: Config[Bigram]) -> str:
        return (
            f"IndHead-Len{config.experiment.seq_length}"
            f"{f'-{config.experiment.task}' if config.experiment.task != 'exact-bigram' else ''}"
            f"-d_model{config.experiment.d_model}"
            f"-ntok{config.experiment.num_tokens}"
            f"{f'-nhead{config.experiment.n_heads}' if config.experiment.n_heads > 1 else ''}"
            f"-{config.train_for[0]}-{config.train_for[1]}"
            f"{'-random-at-end' if config.experiment.random_tokens_at_end else ''}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )

    @property
    def bos_token(self) -> Optional[int]:
        return self.num_tokens if self.bos else None

    def get_ground_truth(
        self, x: Integer[Tensor, "... n"]  # noqa: F722
    ) -> Integer[Tensor, "..."]:  # noqa: F722
        x = x[..., 1:] if self.bos else x
        return cat_bos_uniform_labels(
            calculate_batch_probabilities(x, self.num_tokens), bos=self.bos_token
        )


DEFAULT_BIGRAM = Config(
    experiment=Bigram(
        seq_length=6,
        n_train_samples=4096,
        only_strong_signal=False,
        random_tokens_at_end=False,
        logging_options=ModelMatrixLoggingOptions.all(add_mean_pos_to_tok=False),
    ),
    seed=999,
    deterministic=False,
    batch_size=4096,
    train_for=(10000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)

ABCAB_BIGRAM1H = Config(
    experiment=Bigram(
        seq_length=4,
        num_tokens=26,
        n_heads=1,
        d_model=128,
        task="abcab",
        bos=False,
        only_strong_signal=True,
        random_tokens_at_end=False,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=True, add_mean_pos_to_tok=False
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=512,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)

ABCAB5_BIGRAM1H = Config(
    experiment=Bigram(
        seq_length=5,
        num_tokens=26,
        n_heads=1,
        d_model=128,
        task="abcab",
        bos=False,
        only_strong_signal=True,
        random_tokens_at_end=False,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=True, add_mean_pos_to_tok=False
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=512,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)

ABCAB6_BIGRAM1H = Config(
    experiment=Bigram(
        seq_length=6,
        num_tokens=26,
        n_heads=1,
        d_model=128,
        task="abcab",
        bos=False,
        only_strong_signal=True,
        random_tokens_at_end=True,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=True, add_mean_pos_to_tok=False
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=512,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)


ABCAB6_SMALL_HIDDEN_BIGRAM1H = Config(
    experiment=Bigram(
        seq_length=6,
        num_tokens=26,
        n_heads=1,
        d_model=16,
        task="abcab",
        bos=False,
        only_strong_signal=True,
        random_tokens_at_end=True,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=True, add_mean_pos_to_tok=False
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=512,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)

ABCAB_BIGRAM = Config(
    experiment=Bigram(
        seq_length=4,
        num_tokens=26,
        n_heads=4,
        d_model=128,
        task="abcab",
        bos=False,
        only_strong_signal=True,
        random_tokens_at_end=False,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=False, add_mean_pos_to_tok=False
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=512,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(100, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)


class BigramTrainingWrapper(TrainingWrapper[Bigram]):
    def __init__(self, config: Config[Bigram], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[Bigram]) -> HookedTransformer:
        cfg = config.experiment
        model_config = HookedTransformerConfig(
            d_vocab=cfg.num_tokens + cfg.bos,
            d_vocab_out=cfg.num_tokens,
            n_ctx=cfg.seq_length + cfg.bos,
            d_model=cfg.d_model,
            d_head=cfg.d_model // cfg.n_heads,
            n_layers=2,
            n_heads=cfg.n_heads,
            init_weights=True,
            attn_only=True,
            normalization_type=None,
        )
        model = HookedTransformer(model_config)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    def loss_fn(
        self,
        logits: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        labels: Integer[Tensor, "batch pos num_tokens"],  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        return ExactBigramTask.loss_fn(
            logits,
            labels,
            use_bos=self.config.experiment.bos,
            only_eos=self.config.experiment.only_last_tokens,
            only_strong_signal=self.config.experiment.only_strong_signal,
        )

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

        if log_output and prefix is not None and prefix != "":
            assert self.logger is not None
            self.config.experiment.logging_options.log_matrices(
                self.logger.experiment,  # type: ignore
                self.model,
            )
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
    data_validate: Dataset[
        Tuple[
            Integer[Tensor, "n_ctx"],  # noqa: F821, F722
            Float[Tensor, "n_ctx num_tokens"],  # noqa: F821, F722
        ]
    ]
    n_train_samples: int
    n_test_samples: int
    n_validate_samples: int
    task: Literal["exact-bigram", "abcab"]
    seq_length: int
    bos: Optional[int]
    dataset_seed: int
    num_tokens: int

    def __init__(self, config: Config[Bigram]):
        super().__init__(config)
        self.config = config
        self.n_train_samples = config.experiment.n_train_samples
        self.n_test_samples = config.experiment.n_test_samples
        self.n_validate_samples = config.experiment.n_validate_samples
        self.num_tokens = config.experiment.num_tokens
        self.seq_length = config.experiment.seq_length
        self.bos = config.experiment.num_tokens if config.experiment.bos else None
        self.task = config.experiment.task
        self.random_tokens_at_end = config.experiment.random_tokens_at_end
        self.dataset_seed = reseed(config.seed, "dataset_seed")

    def build_dataset(
        self, mode: Literal["train", "test", "validate"]
    ) -> Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]:
        seed = reseed(self.dataset_seed, mode)
        n_samples = getattr(self, f"n_{mode}_samples")
        match self.task:
            case "exact-bigram":
                generator = ExactBigramTask.generator
            case "abcab":
                generator = partial(
                    ABCBCBigramTask.generator, skip_end=not self.random_tokens_at_end
                )
        data = torch.stack(
            tuple(
                generator(
                    seed=seed,
                    num_tokens=self.num_tokens,
                    seq_length=self.seq_length,
                    max_length=n_samples,
                )
            )
        )
        data = cat_bos_token(data, bos=self.bos)
        dataset = TensorDataset(data, self.config.experiment.get_ground_truth(data))
        return dataset  # type: ignore

    def setup(self, stage: str):
        self.data_train = self.build_dataset("train")
        self.data_test = self.build_dataset("test")
        self.data_validate = self.build_dataset("validate")

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=min(
                self.config.validation_batch_size or self.n_validate_samples,
                self.n_validate_samples,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=min(self.config.batch_size, self.n_test_samples)
        )
