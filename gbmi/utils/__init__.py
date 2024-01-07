from __future__ import annotations

import copy
import itertools
import os
import subprocess
import sys

from pathlib import Path
from typing import Optional, TypeVar, List, Dict
from transformer_lens import HookedTransformer

import numpy as np
import torch
from jaxtyping import Float
from lightning import Callback
from numpy.random import Generator
from torch import Tensor

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_WANDB_ENTITY = "gbmi"


def get_trained_model_dir(create: bool = True) -> Path:
    """
    Returns the base path for saving models. If `save_in_google_drive` is True, returns the path to the Google Drive
    folder where models are saved. Otherwise, returns the path to the local folder where models are saved.
    """
    pth_base_path = PROJECT_ROOT / "trained-models"

    if create and not os.path.exists(pth_base_path):
        os.makedirs(pth_base_path)

    return pth_base_path


def default_device(deterministic: bool = False) -> str:
    return "cuda" if torch.cuda.is_available() and not deterministic else "cpu"


T = TypeVar("T")


def shuffle_data(data, rng: Generator):
    indices = np.array(range(len(data)))
    rng.shuffle(indices)
    data = data[indices]
    return data


def generate_all_sequences(
    n_digits: int, sequence_length: int = 2
) -> Float[Tensor, "n_seqs sequence_length"]:
    data = list(itertools.product(range(n_digits), repeat=sequence_length))
    return torch.tensor(data)


def generate_all_sequences_for_model(
    model: HookedTransformer,
) -> Float[Tensor, "n_seqs sequence_length"]:
    return generate_all_sequences(
        n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx
    )


class MetricsCallback(Callback):
    """PyTorch Lightning callback to save metrics in a Python object."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: List[Dict[str, float]] = []
        self.steps = 0

    def log_metrics(self, trainer):
        metrics = copy.deepcopy(trainer.callback_metrics)
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = self.steps
        self.metrics.append(metrics)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps += 1

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer)

    def on_validation_epoch_end(self, trainer, module):
        self.log_metrics(trainer)


# Function to check if current directory is a Git repository
def is_git_repo():
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"])
        return True
    except subprocess.CalledProcessError:
        return False


# Function to handle the warning, prompt, and error logic
def handle_size_warnings_and_prompts(
    error_on_diff_larger_than=None, prompt_if_diff_larger_than=1024
):
    if not is_git_repo():
        return
    size_kb = len(subprocess.check_output(["git", "diff"])) / 1024

    print(
        f"Warning: Code-saving on wandb may upload a diff/file of size {size_kb} KiB",
        file=sys.stderr,
    )

    if error_on_diff_larger_than is not None and size_kb > error_on_diff_larger_than:
        raise Exception(
            f"The size ({size_kb} KiB) is larger than the specified limit ({error_on_diff_larger_than} KB)."
        )

    if prompt_if_diff_larger_than is not None and size_kb > prompt_if_diff_larger_than:
        input("Press enter to continue or Ctrl+C to break")

def log_softmax(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    log_softmax is only precise to around 2e-7, cf https://github.com/pytorch/pytorch/issues/113708
    we can get better precision by using log1p
    """
    x_max_idxs = x.argmax(dim=dim, keepdim=True)
    x_centered = x - x.gather(dim=dim, index=x_max_idxs)
    x_exp = x_centered.exp()
    # x_exp[max] will be 1, so we can zero it and use log1p(x) = log(1 + x)
    x_exp.scatter_(dim=dim, index=x_max_idxs, src=torch.zeros_like(x_max_idxs, device=x.device, dtype=x.dtype))
    return x_centered - x_exp.sum(dim=dim, keepdim=True).log1p()