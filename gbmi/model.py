from __future__ import annotations

import datetime
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from transformer_lens.HookedTransformerConfig import SUPPORTED_ACTIVATIONS
from typing import (
    Any,
    TypeVar,
    Generic,
    Optional,
    Literal,
    Tuple,
    Type,
    Dict,
    List,
    Mapping,
    Sequence,
)

import torch
import wandb
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from transformer_lens import HookedTransformer
from gbmi.utils import (
    get_trained_model_dir,
    DEFAULT_WANDB_ENTITY,
    MetricsCallback,
    handle_size_warnings_and_prompts,
)
from gbmi.utils.hashing import get_hash, _json_dumps

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
    validate_every: Tuple[int, Literal["steps", "epochs"]] = (10, "steps")

    def __post_init__(self):
        self.experiment.config_post_init(self)

    def get_summary_slug(self):
        return self.experiment.get_summary_slug(self)

    def get_id(self):
        config_summary_slug = self.get_summary_slug()
        config_hash = get_hash(self).hex()
        return f"{config_summary_slug}-{config_hash}"

    def to_dict(self) -> Dict:
        # TODO: hack
        return json.loads(_json_dumps(self))


@dataclass
class RunData:
    wandb_id: Optional[str]
    train_metrics: Sequence[Mapping[str, float]]
    test_metrics: Sequence[Mapping[str, float]]


def _load_model(
    config: Config, model_pth_path: Path
) -> Tuple[RunData, HookedTransformer]:
    model = config.experiment.get_training_wrapper().build_model(config)
    try:
        cached_data = torch.load(str(model_pth_path))
        model.load_state_dict(cached_data["model"])
        wandb_id = cached_data["wandb_id"] if "wandb_id" in cached_data else None
        train_metrics = cached_data["train_metrics"]
        test_metrics = cached_data["test_metrics"]
        # model_checkpoints = cached_data["checkpoints"]
        # checkpoint_epochs = cached_data["checkpoint_epochs"]
        # test_losses = cached_data['test_losses']
        # train_losses = cached_data["train_losses"]
        # train_indices = cached_data["train_indices"]
        # test_indices = cached_data["test_indices"]
        return (
            RunData(
                wandb_id=wandb_id,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            ),
            model,
        )
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_pth_path}:\n", e)


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
    @return:
    """
    # Seed everything
    seed_everything(config.seed)

    # Compute model name
    model_name = config.get_id()

    # Set model save path if not provided
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}-{datetime_str}"
    if model_ckpt_path is None:
        model_ckpt_path = get_trained_model_dir(create=True) / f"{run_name}.pth"

    # Set wandb project if not provided
    if wandb_project is None:
        wandb_project = config.get_summary_slug()
    wandb_model_path = f"{wandb_entity}/{wandb_project}/{model_name}:latest"

    # If we aren't forcing a re-train:
    if force != "train":
        # Try loading the model locally
        if os.path.exists(model_ckpt_path):
            res = _load_model(config, model_ckpt_path)
            if res is not None:
                return res

        # Try loading the model from wandb
        model_dir = None
        try:
            api = wandb.Api()
            model_at = api.artifact(wandb_model_path)
            model_dir = Path(model_at.download())
        except Exception as e:
            print(f"Could not load model {wandb_model_path} from wandb:\n", e)
        if model_dir is not None:
            for model_path in model_dir.glob("*.pth"):
                res = _load_model(config, model_path)
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
    n, unit = config.validate_every
    if unit == "steps":
        trainer_args["val_check_interval"] = n
    elif unit == "epochs":
        trainer_args["check_val_every_n_epoch"] = n
    else:
        raise ValueError

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
        )
        loggers.append(wandb_logger)
        run = wandb_logger.experiment
    else:
        run = None

    # Fit model
    train_metric_callback = MetricsCallback()
    trainer = Trainer(
        accelerator="cpu" if config.deterministic else accelerator,
        callbacks=[
            train_metric_callback,
            RichProgressBar(),
        ],
        log_every_n_steps=config.log_every_n_steps,
        logger=loggers,
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
