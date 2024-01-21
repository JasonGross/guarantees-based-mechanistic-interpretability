from __future__ import annotations
from argparse import ArgumentParser, Namespace, BooleanOptionalAction

import datetime
import logging
import json
import os
import re
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
import re
from transformer_lens import HookedTransformerConfig
from transformer_lens.HookedTransformerConfig import SUPPORTED_ACTIVATIONS
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    TypeVar,
    Generic,
    Optional,
    Literal,
    Tuple,
    Type,
    Dict,
    Mapping,
    Sequence,
    Union,
)

import torch
import wandb
import wandb.apis.public.runs
import wandb.apis.public.artifacts
from wandb.sdk.lib.paths import FilePathStr
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformer_lens import HookedTransformer
from gbmi.utils.lazy import lazy
from gbmi.utils import (
    get_trained_model_dir,
    DEFAULT_WANDB_ENTITY,
    MetricsCallback,
    handle_size_warnings_and_prompts,
)
from gbmi.utils.hashing import get_hash, _json_dumps, _EXCLUDE

ConfigT = TypeVar("ConfigT")
ExpT = TypeVar("ExpT", bound="ExperimentConfig")


class TrainingWrapper(ABC, LightningModule, Generic[ExpT]):
    @abstractmethod
    def __init__(
        self,
        config: Config[ExpT],
        model: HookedTransformer,
    ):
        super(TrainingWrapper, self).__init__()

    @staticmethod
    @abstractmethod
    def build_model(config: Config[ExpT]) -> HookedTransformer:
        ...


class DataModule(LightningDataModule, Generic[ExpT], ABC):
    @abstractmethod
    def __init__(self, config: Config[ExpT]):
        super(DataModule, self).__init__()


@dataclass
class ExperimentConfig(ABC):
    @abstractmethod
    def get_training_wrapper(
        self: ExpT,
    ) -> Type[TrainingWrapper[ExpT]]:
        ...

    @abstractmethod
    def get_datamodule(self: ExpT) -> Type[DataModule[ExpT]]:
        ...

    def get_summary_slug(self: ExpT, config: Config[ExpT]) -> str:
        return self.__class__.__name__

    def config_post_init(self: ExpT, config: Config[ExpT]) -> None:
        """This function gets called on the post_init of the Config object."""
        pass


@dataclass
class Config(Generic[ExpT]):
    experiment: ExpT
    # Training
    deterministic: bool = True
    seed: int = 123
    batch_size: int = 128
    train_for: Tuple[int, Literal["steps", "epochs"]] = (15000, "steps")
    log_every_n_steps: int = 10
    validate_every: Optional[Tuple[int, Literal["steps", "epochs"]]] = (10, "steps")
    checkpoint_every: Optional[Tuple[int, Literal["steps", "epochs"]]] = None

    def __post_init__(self):
        setattr(
            self, _EXCLUDE, ("log_every_n_steps", "validate_every", "checkpoint_every")
        )
        self.experiment.config_post_init(self)

    def get_summary_slug(self):
        return self.experiment.get_summary_slug(self)

    def get_id(self):
        config_summary_slug = self.get_summary_slug()
        config_hash = get_hash(
            self,
            exclude_filter=(
                lambda obj: (
                    ["device"] if isinstance(obj, HookedTransformerConfig) else None
                )
            ),
        ).hex()
        return f"{config_summary_slug}-{config_hash}"

    def to_dict(self) -> Dict:
        # TODO: hack
        return json.loads(_json_dumps(self))

    @classmethod
    def add_arguments(
        cls: Type[Config[ExpT]], parser: ArgumentParser
    ) -> ArgumentParser:
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="Force training on the CPU for avoiding non-deterministic floating point behavior",
        )
        parser.add_argument(
            "--non-deterministic",
            action="store_false",
            dest="deterministic",
            help="Allow training on the GPU for faster training",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed for random number generators",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Batch size",
        )
        parser.add_argument(
            "--train-for-steps",
            type=int,
            default=None,
            help="Number of steps to train for.",
        )
        parser.add_argument(
            "--train-for-epochs",
            type=int,
            default=None,
            help="Number of epochs to train for.",
        )
        parser.add_argument(
            "--log-every-n-steps",
            type=int,
            metavar="N",
            default=None,
            help="Log every N steps",
        )
        parser.add_argument(
            "--validate-every-steps",
            type=int,
            metavar="N",
            default=None,
            help="Validate every N steps",
        )
        parser.add_argument(
            "--validate-every-epochs",
            type=int,
            metavar="N",
            default=None,
            help="Validate every N epochs",
        )
        parser.add_argument(
            "--checkpoint-every-steps",
            type=int,
            metavar="N",
            default=None,
            help="Checkpoint every N steps",
        )
        parser.add_argument(
            "--checkpoint-every-epochs",
            type=int,
            metavar="N",
            default=None,
            help="Checkpoint every N epochs",
        )
        return parser

    def update_from_args(self: Config[ExpT], parsed: Namespace) -> Config[ExpT]:
        parsed = vars(parsed)
        cfg = replace(
            self,
            **{
                k: v
                for k, v in parsed.items()
                if k in self.__dataclass_fields__ and v is not None
            },
        )
        for field_name in ("train_for", "validate_every", "checkpoint_every"):
            if parsed[f"{field_name}_epochs"] is not None:
                setattr(cfg, field_name, (parsed[f"{field_name}_epochs"], "epochs"))
            elif parsed[f"{field_name}_steps"] is not None:
                setattr(cfg, field_name, (parsed[f"{field_name}_steps"], "steps"))
        return cfg


@dataclass
class RunData:
    wandb_id: Optional[str]
    train_metrics: Optional[Sequence[Mapping[str, float]]]
    test_metrics: Optional[Sequence[Mapping[str, float]]]
    epoch: Optional[int] = None
    global_step: Optional[int] = None

    def artifact(self) -> Optional[wandb.Artifact]:
        if self.wandb_id is None:
            return None
        return wandb.Api().artifact(self.wandb_id)

    def run(self) -> Optional[wandb.apis.public.runs.Run]:
        artifact = self.artifact()
        if artifact is None:
            return None
        return artifact.logged_by()

    def logged_artifacts(self) -> Optional[wandb.apis.public.artifacts.RunArtifacts]:
        run = self.run()
        if run is None:
            return None
        return run.logged_artifacts()

    def model_versions(
        self,
        config: Config,
        max_count: Optional[int] = None,
        step: int = 1,
        tqdm: Optional[Callable] = tqdm,
        types: Collection[str] = ("model",),
    ) -> Optional[
        Iterable[
            Tuple[str, Optional[Tuple[RunData, HookedTransformer]], wandb.Artifact]
        ]
    ]:
        if not hasattr(self, "_lazy_model_versions"):
            logged_artifacts = self.logged_artifacts()
            if logged_artifacts is None:
                return None
            self._lazy_model_versions: Sequence[
                Tuple[wandb.Artifact, lazy[FilePathStr]]
            ] = tuple(
                (artifact, lazy(artifact.download)) for artifact in logged_artifacts
            )
        relevant_model_versions = [
            (artifact, download)
            for artifact, download in self._lazy_model_versions
            if artifact.type in types
        ]
        if max_count is None:
            max_count = len(relevant_model_versions)
        relevant_model_versions = relevant_model_versions[:max_count:step]
        if tqdm is not None:
            relevant_model_versions = tqdm(relevant_model_versions)
        for artifact, download in relevant_model_versions:
            yield (
                artifact.version,
                try_load_model_from_wandb_download(config, download.force()),
                artifact,
            )


# TODO(Euan or Jason): figure out why we need this for .ckpt state_dicts and write documentation or remove
def _adjust_statedict_to_model(state_dict: Optional[dict]) -> Optional[dict]:
    """removes 'model.' prefixes from the keys of state_dict; I have no idea why this is necessary"""
    if state_dict is None:
        return None

    return dict(
        ((key[len("model.") :] if key.startswith("model.") else key), value)
        for key, value in state_dict.items()
    )


def _load_model(
    config: Config, model_pth_path: Path, wandb_id: Optional[str] = None
) -> Tuple[RunData, HookedTransformer]:
    model = config.experiment.get_training_wrapper().build_model(config)
    try:
        cached_data = torch.load(str(model_pth_path))
        model.load_state_dict(
            cached_data.get(
                "model", _adjust_statedict_to_model(cached_data.get("state_dict"))
            )
        )
        wandb_id = cached_data.get("wandb_id", wandb_id)
        # model_checkpoints = cached_data["checkpoints"]
        # checkpoint_epochs = cached_data["checkpoint_epochs"]
        # test_losses = cached_data['test_losses']
        # train_losses = cached_data["train_losses"]
        # train_indices = cached_data["train_indices"]
        # test_indices = cached_data["test_indices"]
        return (
            RunData(
                wandb_id=wandb_id,
                train_metrics=cached_data.get("train_metrics"),
                test_metrics=cached_data.get("test_metrics"),
                epoch=cached_data.get("epoch"),
                global_step=cached_data.get("global_step"),
            ),
            model,
        )
    except Exception as e:
        logging.warning(f"Could not load model from {model_pth_path}:\n{e}")
        raise RuntimeError(f"Could not load model from {model_pth_path}:\n", e)


def try_load_model_from_wandb_download(
    config: Config, model_dir: Union[str, Path]
) -> Optional[Tuple[RunData, HookedTransformer]]:
    model_dir = Path(model_dir)
    for model_path in list(model_dir.glob("*.pth")) + list(model_dir.glob("*.ckpt")):
        res = _load_model(config, model_path)
        if res is not None:
            return res


def try_load_model_from_wandb(
    config: Config, wandb_model_path: str
) -> Optional[Tuple[RunData, HookedTransformer]]:
    # Try loading the model from wandb
    model_dir = None
    try:
        api = wandb.Api()
        model_at = api.artifact(wandb_model_path)
        model_dir = Path(model_at.download())
    except Exception as e:
        logging.warning(f"Could not download model {wandb_model_path} from wandb:\n{e}")
    if model_dir is not None:
        return try_load_model_from_wandb_download(config, model_dir)


def train_or_load_model(
    config: Config,
    force: Optional[Literal["train", "load"]] = None,
    save_to: Optional[Literal["disk", "disk_and_wandb"]] = "disk_and_wandb",
    overwrite_existing_ckpt: bool = False,
    model_ckpt_path: Optional[Path] = None,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_project: Optional[str] = None,  # otherwise default name
    model_description: str = "trained model",  # uploaded to wandba
    accelerator: str = "auto",
    model_version: str = "latest",
) -> Tuple[RunData, HookedTransformer]:
    """
    Train model, or load from disk / wandb.
    @param config: Config specifying model
    @param force: If 'train', will train model regardless of whether a saved copy exists;
    if 'load', will fail if unable to load model from disk.
    @param save_to: Where to save model
    @param overwrite_existing_ckpt: Whether to overwrite existing checkpoint saved to disk
    @param model_ckpt_path: Path to save model to (defaults to PROJECT_ROOT / trained_models)
    @param wandb_entity: WandB entity to log to (defaults to DEFAULT_WANDB_ENTITY)
    @param wandb_project: WandB project to log to; if not provided, defaults to `config`'s class name
    @param model_description: Description to provide for WandB project
    @param accelerator: Accelerator to use (cpu or auto)
    @param model_version: Version of model to load from wandb (must be "latest" if force != "load")
    @return:
    """
    # Seed everything
    seed_everything(config.seed)

    # Compute model name
    model_name = config.get_id()
    # Artifact name may only contain alphanumeric characters, dashes, underscores, and dots.
    # replace all other characters with _ using re.sub
    model_name = re.sub(r"[^a-zA-Z0-9\-_.]", "_", model_name)

    # Set model save path if not provided
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}-{datetime_str}"
    if model_ckpt_path is None:
        model_ckpt_path = get_trained_model_dir(create=True) / f"{run_name}.pth"
    model_ckpt_dir_path = Path(model_ckpt_path).parent

    # Set wandb project if not provided
    if wandb_project is None:
        wandb_project = config.get_summary_slug()
    if force != "load":
        assert (
            model_version == "latest"
        ), f"model_version must be 'latest' (not {model_version}) if force != 'load' (force == {force})"
    wandb_model_path = f"{wandb_entity}/{wandb_project}/{model_name}:{model_version}"

    # If we aren't forcing a re-train:
    if force != "train":
        # Try loading the model locally
        if os.path.exists(model_ckpt_path):
            res = _load_model(config, model_ckpt_path)
            if res is not None:
                return res

        res = try_load_model_from_wandb(config, wandb_model_path)
        if res is not None:
            return res

    # Fail if we couldn't load model and we forced a load.
    if force == "load":
        raise FileNotFoundError(
            f"Couldn't load model from {model_ckpt_path}{f' or wandb ({wandb_model_path})' if wandb_model_path is not None else ''}, and force is set to 'load'"
        )

    # Otherwise train the model...

    # Warn if using MPS
    if (
        torch.backends.mps.is_available()
        and accelerator != "cpu"
        and not config.deterministic
    ):
        input(
            f"WARNING: currently training with MPS on Mac (accelerator={accelerator}, cfg.deterministic={config.deterministic}) -- here be bugs!\n"
            "Disable MPS training by calling train_or_load with accelerator=cpu, or press ENTER to continue."
        )

    # Build model, wrapper and datamodule
    ExpWrapper = config.experiment.get_training_wrapper()
    wrapped_model = ExpWrapper(config, ExpWrapper.build_model(config))
    datamodule = config.experiment.get_datamodule()(config)

    trainer_args = {}

    # How long should we train for?
    n, unit = config.train_for
    if unit == "steps":
        trainer_args["max_steps"] = n
    elif unit == "epochs":
        trainer_args["max_epochs"] = n
    else:
        raise ValueError

    # How often should we validate?
    if config.validate_every is not None:
        n, unit = config.validate_every
        if unit == "steps":
            trainer_args["val_check_interval"] = n
        elif unit == "epochs":
            trainer_args["check_val_every_n_epoch"] = n
        else:
            raise ValueError
    else:
        trainer_args["limit_val_batches"] = 0
        trainer_args["num_sanity_val_steps"] = 0

    # Initialise a wandb run if necessary
    loggers = []
    if save_to == "disk_and_wandb":
        handle_size_warnings_and_prompts()
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config.to_dict(),
            job_type="train",
            log_model=("all" if config.checkpoint_every is not None else False),
        )
        loggers.append(wandb_logger)
        run = wandb_logger.experiment
    else:
        run = None

    # Set up model checkpointing
    # TODO(Euan or Jason, low-ish priority): fix model checkpointing, it doesn't seem to work
    checkpoint_callback = None
    if config.checkpoint_every is not None:
        if config.checkpoint_every[1] == "epochs":
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_ckpt_dir_path,
                filename=run_name + "-{epoch}-{step}",
                every_n_epochs=config.checkpoint_every[0],
                save_top_k=-1,  # Set to -1 to save all checkpoints
            )
        elif config.checkpoint_every[1] == "steps":
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_ckpt_dir_path,
                filename=run_name + "-{epoch}-{step}",
                every_n_train_steps=config.checkpoint_every[0],
                save_top_k=-1,  # Set to -1 to save all checkpoints
            )

    # Fit model
    train_metric_callback = MetricsCallback()
    callbacks = [train_metric_callback, RichProgressBar()]
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    trainer = Trainer(
        accelerator="cpu" if config.deterministic else accelerator,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        logger=loggers,
        deterministic=config.deterministic or "warn",
        **trainer_args,  # type: ignore
    )
    trainer.fit(wrapped_model, datamodule)
    test_metrics = trainer.test(wrapped_model, datamodule)

    if save_to is not None:
        data = {
            "model": wrapped_model.model.state_dict(),
            "model_config": wrapped_model.model.cfg,
            "run_config": config.to_dict(),
            "train_metrics": train_metric_callback.metrics,
            "test_metrics": test_metrics,
            "wandb_id": wandb_model_path,
        }
        if overwrite_existing_ckpt or not os.path.exists(model_ckpt_path):
            print("Saving to disk...")
            torch.save(data, model_ckpt_path)

        if run is not None:
            print("Saving to WandB...")
            trained_model_artifact = wandb.Artifact(
                model_name,
                type="model",
                description=model_description,
                metadata=wrapped_model.model.cfg.to_dict(),
            )
            trained_model_artifact.add_file(str(model_ckpt_path))
            run.log_artifact(trained_model_artifact)

    if run is not None:
        run.finish()

    return (
        RunData(
            wandb_id=wandb_model_path,
            train_metrics=train_metric_callback.metrics,
            test_metrics=test_metrics,
        ),
        wrapped_model.model,
    )


def add_force_argument(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        choices=[None, "train", "load"],
        default=None,
        help="Force action: None (default), 'train', or 'load'.",
    )
    return parser


def add_no_save_argument(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--no-save",
        dest="save_to",
        action="store_const",
        const=None,
        default="disk_and_wandb",
        help="Disable saving the model.",
    )
    return parser


def _parse_HookedTransformerConfig_arguments():
    """parses HookedTransformerConfig.__doc__ for various simple arguments"""
    spaces = " " * 8
    doc = HookedTransformerConfig.__doc__
    assert doc is not None
    assert "Args:" in doc, f"HookedTransformerConfig.__doc__ is missing 'Args:'\n{doc}"
    simple_args: List[Tuple[str, Union[dict, type, Any], str]] = [
        (
            "act_fn",
            dict(choices=SUPPORTED_ACTIVATIONS),
            "The activation function to use.",
        ),
        (
            "normalization_type",
            dict(choices=[None, "LN", "LNPre"]),
            "the type of normalization to use. Options are None (no normalization), 'LN' (use LayerNorm, including weights & biases) and 'LNPre' (use LayerNorm, but no weights & biases). Defaults to LN (optional)",
        ),
        (
            "attention_dir",
            dict(choices=["causal", "bidirectional"]),
            "Whether to use causal (aka unidirectional aka GPT-2 style) or bidirectional attention. Defaults to 'causal'",
        ),
        (
            "positional_embedding_type",
            dict(choices=["standard", "rotary", "shortformer"]),
            "The positional embedding used. Options are 'standard' (ie GPT-2 style, absolute, randomly initialized learned positional embeddings, directly added to the residual stream), 'rotary' (described here: https://blog.eleuther.ai/rotary-embeddings/ ) and 'shortformer' (GPT-2 style absolute & learned, but rather than being added to the residual stream they're only added to the inputs to the keys and the queries (ie key = W_K(res_stream + pos_embed), but values and MLPs don't get any positional info)). Sinusoidal are not currently supported. Defaults to 'standard'.",
        ),
        (
            "attn_types",
            dict(choices=["global", "local"], action="append"),
            "the types of attention to use for local attention",
        ),
        (
            "weight_init_mode",
            dict(choices=["gpt2"]),
            "the initialization mode to use for the weights. Only relevant for custom models, ignored for pre-trained. Currently the only supported mode is 'gpt2', where biases are initialized to 0 and weights are standard normals of range initializer_range.",
        ),
    ]
    valid_dtypes = [torch.float32, torch.float64, torch.bool]
    valid_dtypes += [
        getattr(torch, attr)
        for attr in dir(torch)
        if isinstance(getattr(torch, attr), torch.dtype)
        and getattr(torch, attr) not in valid_dtypes
    ]
    known_names = set(name for name, _, _ in simple_args)
    known_types = {
        "int": int,
        "bool": BooleanOptionalAction,
        "float": float,
        "torch.dtype": dict(type=torch.dtype, choices=valid_dtypes),
    }
    for name, ty, description in re.findall(
        rf"\n{spaces}([^ ]+) \(([^\)]*)\): ([^\n]*(?:\n{spaces} [^\n]*)*)",
        doc[doc.index("Args:") + len("Args:") :],
    ):
        if name in known_names:
            continue
        description = re.sub(rf"\n{spaces}\s*", " ", description).strip()
        if ty.endswith(", *optional*"):
            ty = ty[: -len(", *optional*")]
            description += " (optional)"
        elif ty.endswith(", optional"):
            ty = ty[: -len(", optional")]
            description += " (optional)"
        if ty in known_types.keys():
            simple_args.append((name, known_types[ty], description))
            continue
        if ty == "str" and name in ("tokenizer_name", "model_name"):
            simple_args.append((name, str, description))
            continue
        if name in ("original_architecture", "checkpoint_label_type"):
            # deliberately not yet handled, because it's not clear we'd want to set these
            continue
        logging.warning(
            f"Unknown type {ty} for unrecognized {name} in HookedTransformerConfig.__doc__ while parsing doc for CLI arguments ({description})"
        )
    return simple_args


def add_HookedTransformerConfig_arguments(
    parser: ArgumentParser,
    arguments: Optional[Collection[str]] = None,
    exclude_arguments: Optional[Collection[str]] = None,
    underscores_to_dashes: bool = True,
    prefix: str = "",
) -> ArgumentParser:
    """Adds arguments from HookedTransformerConfig.__doc__ to parser

    Args:
        parser: parser to add arguments to
        arguments: if provided, only add arguments with names in this collection
        exclude_arguments: if provided, exclude arguments with names in this collection
    """
    for name, ty, description in _parse_HookedTransformerConfig_arguments():
        if arguments is not None and name not in arguments:
            continue
        if exclude_arguments is not None and name in exclude_arguments:
            continue
        dest = name
        if underscores_to_dashes:
            name = name.replace("_", "-")
        kwargs = ty if isinstance(ty, dict) else dict(type=ty)
        parser.add_argument(
            f"--{prefix}{name}", dest=dest, default=None, help=description, **kwargs
        )
    return parser


def update_HookedTransformerConfig_from_args(
    cfg: HookedTransformerConfig,
    parsed: Namespace,
    arguments: Optional[Collection[str]] = None,
    exclude_arguments: Optional[Collection[str]] = None,
) -> HookedTransformerConfig:
    parsed_vars = vars(parsed)
    htc_names = set(name for name, _, _ in _parse_HookedTransformerConfig_arguments())
    cfg = replace(
        cfg,
        **{
            k: v
            for k, v in parsed_vars.items()
            if k in htc_names
            and (arguments is None or k in arguments)
            and (exclude_arguments is None or k not in exclude_arguments)
            and v is not None
            and k in cfg.__dataclass_fields__
        },
    )
    return cfg
