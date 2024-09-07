from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import simple_parsing
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

import gbmi.utils as utils
from gbmi import utils
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
from gbmi.utils import reseed, shuffle_data, zero_biases_of_HookedTransformer
from gbmi.utils.dataclass import DataclassMapping
from gbmi.utils.hashing import _EXCLUDE
from gbmi.utils.sequences import generate_all_sequences


@dataclass
class OptimizerConfig(DataclassMapping[Any]):
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.98)


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
    use_end_of_sequence: bool = True
    use_log1p: bool = True
    use_float64_log: bool = True
    num_workers: int = 0
    summary_slug_extra: str = ""
    init_mode: Literal[
        "gpt2",
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
        "muP",
    ] = "gpt2"
    zero_biases: list[
        Literal["Embed", "Unembed", "PosEmbed", "LayerNorm", "Attention", "MLP"]
    ] = field(
        default_factory=[
            "Embed",
            "Unembed",
            "PosEmbed",
            "Attention",
        ].copy  # type: ignore
    )
    fn_name: Literal["add", "subtract", "x2xyy2"] = "add"
    training_ratio: float = 0.3  # fraction of dataset to use for training
    validation_max_samples: Optional[int] = None
    optimizer_kwargs: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_scheduler: bool = True
    logging_options: ModelMatrixLoggingOptions = field(
        default_factory=ModelMatrixLoggingOptions
    )
    version_number: int = 2

    def get_training_wrapper(self):
        return ModularArithmeticTrainingWrapper

    def get_datamodule(self):
        return ModularArithmeticDataModule

    def get_summary_slug(self, config: Config[ModularArithmetic]) -> str:
        return (
            f"Modular{config.experiment.fn_name.capitalize()}"
            f"-{config.experiment.p}"
            f"{f'-{config.experiment.seq_len}' if config.experiment.seq_len != 2 else ''}"
            f"-{config.train_for[0]}-{config.train_for[1]}"
            f"{f'-attention-rate-{config.experiment.attention_rate}' if config.experiment.attention_rate != 0 else ''}"
            f"{'-no-eos' if not config.experiment.use_end_of_sequence else ''}"
            f"{'-' + config.experiment.summary_slug_extra if config.experiment.summary_slug_extra else ''}"
            f"{'-nondeterministic' if not config.deterministic else ''}"
        )

    def __post_init__(self):
        if not hasattr(self, "d_mlp"):
            self.d_mlp = self.n_heads * self.d_model
        self.zero_biases = sorted(set(self.zero_biases))
        setattr(
            self, _EXCLUDE, ("validation_max_samples", "num_workers", "logging_options")
        )
        for attrname, default_val in (("init_mode", "gpt2"),):
            if getattr(self, attrname, default_val) == default_val:
                setattr(self, _EXCLUDE, getattr(self, _EXCLUDE) + (attrname,))
        assert (
            self.d_model % self.n_heads == 0
        ), f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"

    def config_post_init(self, config: Config[ModularArithmetic]) -> None:
        config.batch_size = int(self.p**self.seq_len * self.training_ratio)

    def get_eos_token(self) -> Optional[int]:
        return self.p if self.use_end_of_sequence else None

    def add_eos(
        self, x: Integer[Tensor, "... n"]  # noqa F722
    ) -> Integer[Tensor, "... n+1"]:  # noqa F722
        match self.get_eos_token():
            case None:
                return x
            case tok:
                return utils.add_eos(x, tok)

    def strip_eos(
        self, x: Integer[Tensor, "... n+1"]  # noqa F722
    ) -> Integer[Tensor, "... n"]:  # noqa F722
        match self.get_eos_token():
            case None:
                return x
            case tok:
                return x[..., :-1]

    def fn(
        self, x: Integer[Tensor, "... n"]  # noqa: F722
    ) -> Integer[Tensor, "..."]:  # noqa: F722
        p = self.p
        if not isinstance(x, Tensor):
            x = torch.tensor(x)
        match self.fn_name:
            case "add":
                return x.sum(dim=-1) % p
            case "subtract":
                return (x[..., 0] - x[..., 1:].sum(dim=-1)) % p
            case "x2xyy2":
                return (x[..., :, None] * x[..., None, :]).sum(dim=-1).sum(dim=-1) % p

    def get_ground_truth(
        self, x: Integer[Tensor, "... n"]  # noqa: F722
    ) -> Integer[Tensor, "..."]:  # noqa: F722
        x = self.strip_eos(x)
        return self.fn(x)

    def log_softmax(self, x: Tensor, **kwargs) -> Tensor:
        if self.use_float64_log:
            x = x.to(torch.float64)
        if self.use_log1p:
            return utils.log_softmax(x, **kwargs)
        else:
            return F.log_softmax(x, **kwargs)


CLOCK_CONFIG = Config(
    experiment=ModularArithmetic(
        p=113,
        training_ratio=0.3,
        logging_options=ModelMatrixLoggingOptions.all(
            EVOU=False,
            PVOU=False,
        ),
    ),
    seed=0,
    deterministic=False,
    train_for=(50000, "epochs"),
    log_every_n_steps=1,
    validate_every=(100, "epochs"),
    checkpoint_every=(500, "epochs"),
)

PIZZA_CONFIG = Config(
    experiment=ModularArithmetic(
        p=59,
        training_ratio=0.8,
        use_end_of_sequence=False,
        attention_rate=1,
        logging_options=ModelMatrixLoggingOptions.none(
            EVOU=False,
            PVOU=False,
        ),
    ),
    seed=0,
    deterministic=False,
    train_for=(10000, "epochs"),
    log_every_n_steps=1,
    validate_every=(100, "epochs"),
    checkpoint_every=(500, "epochs"),
)


class ModularArithmeticTrainingWrapper(TrainingWrapper[ModularArithmetic]):
    def __init__(self, config: Config[ModularArithmetic], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[ModularArithmetic]) -> HookedTransformer:
        model_config = HookedTransformerConfig(
            d_vocab=config.experiment.p
            + (1 if config.experiment.use_end_of_sequence else 0),
            d_vocab_out=config.experiment.p,
            n_ctx=config.experiment.seq_len
            + (1 if config.experiment.use_end_of_sequence else 0),
            d_model=config.experiment.d_model,
            d_mlp=config.experiment.d_mlp,
            d_head=config.experiment.d_model // config.experiment.n_heads,
            n_layers=config.experiment.n_layers,
            n_heads=config.experiment.n_heads,
            act_fn=config.experiment.act_fn,
            normalization_type=config.experiment.normalization_type,
            init_mode=config.experiment.init_mode,
            seed=reseed(config.seed, "model"),
        )
        model = HookedTransformer(model_config)
        zero_biases_of_HookedTransformer(model, config.experiment.zero_biases)
        return model

    def loss_fn(
        self,
        logits: Float[Tensor, "batch n_ctx d_vocab_out"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
        log_softmax: Optional[Callable] = None,
    ) -> Float[Tensor, ""]:  # noqa F722
        if log_softmax is None:
            log_softmax = self.config.experiment.log_softmax
        log_probs = log_softmax(logits[:, -1, :], dim=-1)
        correct_log_probs = log_probs.gather(-1, labels.unsqueeze(-1))
        return -correct_log_probs.mean()

    @staticmethod
    def acc_fn_per_seq(
        logits: Float[Tensor, "batch n_ctx d_vocab_out"],  # noqa: F722, F821
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> Bool[Tensor, "batch"]:  # noqa: F821
        pred_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        return pred_tokens == labels

    @staticmethod
    def acc_fn(
        logits: Float[Tensor, "batch n_ctx d_vocab"],  # noqa: F722
        labels: Integer[Tensor, "batch"],  # noqa: F821
    ) -> float:
        return (
            ModularArithmeticTrainingWrapper.acc_fn_per_seq(logits, labels)
            .float()
            .mean()
            .item()
        )

    def attention_hook(self, attnscore, hook):
        alpha = self.config.experiment.attention_rate
        # note that this is different from the paper, which does not do the division to enforce the constraint
        # that the attention scores add up to 1
        return alpha / attnscore.shape[-1] + (1 - alpha) * attnscore

    def compute_batch(
        self,
        x_y: Tuple[Integer[Tensor, "batch pos"], Integer[Tensor, "batch"]],  # noqa F722
        *,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[
        Integer[Tensor, "batch pos"],  # noqa F722
        Integer[Tensor, "batch"],  # noqa F821
        Float[Tensor, "batch pos d_vocab_out"],  # noqa F722
    ]:
        xs, ys = x_y

        if device is not None:
            xs = xs.to(device)
            ys = ys.to(device)
        self.model.to(xs.device, print_details=False)

        y_preds = self.model.run_with_hooks(
            xs, fwd_hooks=[("blocks.0.attn.hook_pattern", self.attention_hook)]
        )
        return xs, ys, y_preds

    def run_batch(
        self,
        x_y: Tuple[Integer[Tensor, "batch pos"], Integer[Tensor, "batch"]],  # noqa F722
        prefix: Optional[str] = None,
        *,
        log_output: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[Float[Tensor, ""], float]:  # noqa F722
        assert prefix is not None or not log_output, "Must not log if prefix is None"
        xs, ys, y_preds = self.compute_batch(x_y, device=device)
        loss = self.loss_fn(y_preds, ys)
        acc = self.acc_fn(y_preds, ys)

        if log_output:
            self.log(f"{prefix}loss", loss, prog_bar=True)
            self.log(f"{prefix}acc", acc, prog_bar=True)

        if log_output and prefix is not None and prefix != "":
            assert self.logger is not None
            self.config.experiment.logging_options.log_matrices(
                self.logger.experiment,  # type: ignore
                self.model,
            )

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.run_batch(batch, prefix="")
        return loss

    def test_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="test_")

    def validation_step(self, batch, batch_idx):
        self.run_batch(batch, prefix="periodic_test_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), **self.config.experiment.optimizer_kwargs
        )
        if self.config.experiment.use_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: min(epoch / 10, 1)
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer


class ModularArithmeticDataModule(DataModule):
    data_train: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
    data_validate: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]
    data_test: Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]

    def __init__(self, config: Config[ModularArithmetic]):
        super().__init__(config)
        self.config = config
        self.p = config.experiment.p
        self.seq_len = config.experiment.seq_len
        self.training_ratio = config.experiment.training_ratio
        self.num_workers = config.experiment.num_workers
        self.dataset_seed = reseed(self.config.seed, "dataset_seed")

    @cache
    def get_full_dataset(self) -> Tuple[
        Integer[Tensor, "batch_train seq_len"],  # noqa: F722
        Integer[Tensor, "batch_validate seq_len"],  # noqa: F722
        Integer[Tensor, "batch_full seq_len"],  # noqa: F722
    ]:
        rng = np.random.default_rng(self.dataset_seed)
        data = generate_all_sequences(self.p, self.seq_len)
        data_full = self.config.experiment.add_eos(data)

        data = shuffle_data(data_full, rng)

        split_idx = int(len(data) * self.training_ratio)
        data_train, data_validate = data[:split_idx], data[split_idx:]
        if self.config.experiment.validation_max_samples is not None:
            data_validate = data_validate[
                : self.config.experiment.validation_max_samples
            ]
        data_train = shuffle_data(data_train, rng)
        data_validate = shuffle_data(data_validate, rng)
        return data_train, data_validate, data_full

    def build_dataset(
        self, mode: Literal["train", "test", "validate"]
    ) -> Dataset[
        Tuple[Integer[Tensor, "n_ctx"], Integer[Tensor, ""]]  # noqa: F821, F722
    ]:
        data_train, data_validate, data_test = self.get_full_dataset()
        data = {"train": data_train, "test": data_test, "validate": data_validate}[mode]
        return TensorDataset(data, self.config.experiment.get_ground_truth(data))  # type: ignore

    def setup(self, stage: str):
        self.data_train = self.build_dataset("train")
        self.data_test = self.build_dataset("test")
        self.data_validate = self.build_dataset("validate")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_validate,
            batch_size=self.config.validation_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )


def main(argv: List[str] = sys.argv):
    parser = simple_parsing.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_arguments(
        ModularArithmetic, dest="experiment_config", default=CLOCK_CONFIG.experiment
    )
    add_force_argument(parser)
    add_no_save_argument(parser)
    Config.add_arguments(parser, default=CLOCK_CONFIG)

    args = parser.parse_args(argv[1:])

    config = Config(args.experiment_config)
    config = config.update_from_args(args)
    print("Model config:", ModularArithmeticTrainingWrapper.build_model(config).cfg)
    print("Training model:", config)
    train_or_load_model(config, force=args.force, save_to=args.save_to)


if __name__ == "__main__":
    main()
