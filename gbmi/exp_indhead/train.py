from __future__ import annotations
from functools import partial
from dataclasses import dataclass
from dataclasses import field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Literal,
    Tuple,
)
import sys
import torch
from jaxtyping import Float, Integer
from torch import Tensor
import simple_parsing
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
from gbmi.exp_indhead.data_utils import (
    ExactBigramTask,
    ABCBCTask,
    EnglishExactNgramTask,
    ABCBCEnglishTask,
    calculate_batch_probabilities,
    cat_bos_token,
    cat_bos_uniform_labels,
)
from gbmi.model import (
    TrainingWrapper,
    Config,
    ExperimentConfig,
    DataModule,
    train_or_load_model,
    add_force_argument,
    add_no_save_argument,
)
from gbmi.utils import (
    reseed,
    set_params,
)

from gbmi.utils.hashing import _EXCLUDE
from gbmi.training_tools.logging import (
    ModelMatrixLoggingOptions,
)


@dataclass
class IndHead(ExperimentConfig):
    # using int instead of abstract class because i'm clueless what's going on with typing
    zero_biases: bool = True
    bos: bool = True
    seq_length: int = 5
    num_tokens: int = 3
    d_model: int = 8
    ngram: int = 3
    task: Literal["exact-bigram", "exact-ngram", "abcab"] = "exact-bigram"
    corpus: Optional[str] = None
    only_last_tokens: Optional[int] = None
    only_strong_signal: bool = True
    random_tokens_at_end: bool = False
    n_heads: int = 1
    positional_embedding_type: Literal["standard", "rotary", "shortformer"] = "standard"
    other_tokens_distinct_from_predicted_token: bool = False
    alpha_mix_uniform: Optional[float] = None
    high_precision: bool = True
    use_kaiming_init: bool = True

    n_train_samples: int = 4096
    n_test_samples: int = 1
    n_validate_samples: int = 1024

    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 1.0}
    )
    summary_slug_extra: str = ""
    version_number: int = 6
    logging_options: ModelMatrixLoggingOptions = field(
        default_factory=ModelMatrixLoggingOptions
    )

    @property
    def corpus_relevant(self):
        return self.task in ("abcab", "exact-ngram")

    @property
    def ngram_relevant(self):
        return self.corpus_relevant and self.corpus is not None

    def __post_init__(self):
        exclude: set[str] = set(getattr(self, _EXCLUDE, ()))
        for field, should_ignore in [
            ("logging_options", True),
            ("corpus", self.corpus_relevant),
            ("ngram", self.ngram == 3 or not self.ngram_relevant),
            ("use_kaiming_init", not self.use_kaiming_init),
        ]:
            if should_ignore:
                exclude.add(field)
            else:
                exclude.discard(field)
        setattr(self, _EXCLUDE, tuple(sorted(exclude)))
        self.logging_options.shortformer = (
            self.positional_embedding_type == "shortformer"
        )
        self.logging_options.__post_init__()

    def get_training_wrapper(self):
        return IndHeadTrainingWrapper

    def get_datamodule(self):
        return IndHeadDataModule

    def get_summary_slug(self, config: Config[IndHead]) -> str:
        return (
            f"IndHead"
            f"{f'{config.experiment.ngram}gram' if self.ngram != 3 and self.ngram_relevant else ''}"
            f"-Len{config.experiment.seq_length}"
            f"{f'-{config.experiment.task}' if config.experiment.task != 'exact-bigram' else ''}"
            f"-d_model{config.experiment.d_model}"
            f"-ntok{config.experiment.num_tokens}"
            f"{f'-pos-{config.experiment.positional_embedding_type}' if config.experiment.positional_embedding_type != 'standard' else ''}"
            f"{f'-nhead{config.experiment.n_heads}' if config.experiment.n_heads > 1 else ''}"
            f"-{config.train_for[0]}-{config.train_for[1]}"
            f"{'-randend' if config.experiment.random_tokens_at_end else ''}"
            f"{f'-{config.experiment.corpus}' if config.experiment.corpus and config.experiment.task != 'exact-bigram' else ''}"
            f"{f'-{config.experiment.alpha_mix_uniform:.2f}U' if config.experiment.alpha_mix_uniform and config.experiment.corpus and config.experiment.task != 'exact-bigram' else ''}"
            f"{'-' + config.experiment.summary_slug_extra if config.experiment.summary_slug_extra else ''}"
            f"{'-nondet' if not config.deterministic else ''}"
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


DEFAULT_INDHEAD = Config(
    experiment=IndHead(
        seq_length=6,
        n_train_samples=4096,
        only_strong_signal=False,
        random_tokens_at_end=False,
        logging_options=ModelMatrixLoggingOptions.all(
            add_mean={-1: None, 0: "tok_to_pos", 1: None}
        ),
    ),
    seed=999,
    deterministic=False,
    batch_size=10240,
    train_for=(10000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)

ABCAB_1H = Config(
    experiment=IndHead(
        seq_length=4,
        alpha_mix_uniform=None,
        num_tokens=26,
        n_heads=1,
        d_model=128,
        task="abcab",
        corpus="webtext",
        bos=False,
        only_strong_signal=True,
        random_tokens_at_end=False,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=True, add_mean={-1: None, 0: "tok_to_pos", 1: None}
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=10240,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)

ABCAB5_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 5,
    },
    post_init=True,
)

ABCAB6_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 6,
    },
    post_init=True,
)

ABCAB7_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 7,
    },
    post_init=True,
)


ABCAB8_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 8,
    },
    post_init=True,
)

ABCAB6_SHORTFORMER_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 6,
        ("experiment", "positional_embedding_type"): "shortformer",
    },
    post_init=True,
)

ABCAB8_SHORTFORMER_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 8,
        ("experiment", "positional_embedding_type"): "shortformer",
    },
    post_init=True,
)

ABCAB6_SMALL_HIDDEN_1H = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 6,
        ("experiment", "d_model"): 16,
    },
    post_init=True,
)

ABCAB = set_params(
    ABCAB_1H,
    {
        ("experiment", "seq_length"): 4,
        ("experiment", "n_heads"): 4,
    },
    post_init=True,
)

TRIGRAM4 = Config(
    experiment=IndHead(
        seq_length=4,
        num_tokens=26,
        n_heads=1,
        d_model=128,
        task="exact-ngram",
        corpus="webtext",
        bos=False,
        only_strong_signal=True,
        n_train_samples=10240,
        logging_options=ModelMatrixLoggingOptions.all(
            use_subplots=True, add_mean={-1: None, 0: "tok_to_pos", 1: None}
        ),
        optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 1.0},
    ),
    seed=999,
    deterministic=False,
    batch_size=10240,
    train_for=(5000, "epochs"),
    log_every_n_steps=1,
    validate_every=(10, "epochs"),
    validation_batch_size=1,  # we want validation right now only to log the plots
)


class IndHeadTrainingWrapper(TrainingWrapper[IndHead]):
    def __init__(self, config: Config[IndHead], model: HookedTransformer):
        super().__init__(config, model)
        self.model = model
        self.config = config

    @staticmethod
    def build_model(config: Config[IndHead]) -> HookedTransformer:
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
            seed=reseed(config.seed, "model"),
        )
        model = HookedTransformer(model_config)
        if config.experiment.use_kaiming_init:
            if model.cfg.seed is not None:
                torch.manual_seed(model.cfg.seed)

            for name, param in model.named_parameters():
                if "W_" in name:
                    torch.nn.init.kaiming_uniform_(param)
        if config.experiment.zero_biases:
            for name, param in model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False
        return model

    def loss_fn(
        self,
        logits: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        labels: Integer[Tensor, "batch pos num_tokens"],  # noqa: F722
        *,
        # xs only for logging purposes
        _xs: Optional[Integer[Tensor, "batch pos"]] = None,  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        return ExactBigramTask.loss_fn(
            logits,
            labels,
            use_bos=self.config.experiment.bos,
            only_eos=self.config.experiment.only_last_tokens,
            only_strong_signal=self.config.experiment.only_strong_signal,
            high_precision=self.config.experiment.high_precision,
            _xs=_xs,
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
        loss = self.loss_fn(y_preds, ys, _xs=xs)
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


class IndHeadDataModule(DataModule):
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
    task: Literal["exact-bigram", "exact-ngram", "abcab"]
    corpus: Optional[str] = None
    alpha_mix_uniform: Optional[float] = None
    ngram: int = 3
    seq_length: int
    bos: Optional[int]
    dataset_seed: int
    num_tokens: int

    def __init__(self, config: Config[IndHead]):
        super().__init__(config)
        self.config = config
        self.n_train_samples = config.experiment.n_train_samples
        self.n_test_samples = config.experiment.n_test_samples
        self.n_validate_samples = config.experiment.n_validate_samples
        self.num_tokens = config.experiment.num_tokens
        self.seq_length = config.experiment.seq_length
        self.bos = config.experiment.num_tokens if config.experiment.bos else None
        self.task = config.experiment.task
        self.corpus = config.experiment.corpus
        self.alpha_mix_uniform = config.experiment.alpha_mix_uniform
        self.ngram = config.experiment.ngram
        self.random_tokens_at_end = config.experiment.random_tokens_at_end
        self.force_strong_signal = config.experiment.only_strong_signal
        self.other_tokens_distinct_from_predicted_token = (
            config.experiment.other_tokens_distinct_from_predicted_token
        )
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
            case "exact-ngram":
                if self.corpus is None:
                    raise ValueError("Corpus must be provided for exact trigram task")
                generator = partial(
                    EnglishExactNgramTask.generator,
                    force_strong_signal=self.force_strong_signal,
                    corpus=self.corpus,
                    ngram=self.ngram,
                    alpha_mix_uniform=self.alpha_mix_uniform,
                )
            case "abcab":
                generator = (
                    ABCBCTask.generator
                    if self.corpus is None
                    else partial(
                        ABCBCEnglishTask.generator,
                        corpus=self.corpus,
                        ngram=self.ngram,
                        alpha_mix_uniform=self.alpha_mix_uniform,
                    )
                )
                generator = partial(
                    generator,
                    skip_end=not self.random_tokens_at_end,
                    b_unique=self.other_tokens_distinct_from_predicted_token,
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


def main(
    argv: List[str] = sys.argv,
    default: Config[IndHead] = ABCAB8_1H,
    default_force: Optional[Literal["train", "load"]] = None,
):
    parser = simple_parsing.ArgumentParser(
        description="Train a model with configurable attention rate."
    )
    parser.add_arguments(IndHead, dest="experiment_config", default=default.experiment)
    add_force_argument(parser, default=default_force)
    add_no_save_argument(parser)
    Config.add_arguments(parser, default=default)

    args = parser.parse_args(argv[1:])

    config = Config(args.experiment_config)
    config = config.update_from_args(args)
    print("Model config:", IndHeadTrainingWrapper.build_model(config).cfg)
    print("Training model:", config)
    return train_or_load_model(config, force=args.force, save_to=args.save_to)


if __name__ == "__main__":
    main()
