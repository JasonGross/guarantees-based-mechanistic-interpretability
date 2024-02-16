from __future__ import annotations
from matplotlib import pyplot as plt
import torch
from transformer_lens import HookedTransformer
from dataclasses import dataclass
from typing import Any, Callable, Optional
from lightning.pytorch.loggers.wandb import WandbLogger
import logging


@dataclass
class ModelMatrixLoggingOptions:
    EQKE: bool = False
    EQKP: bool = False
    PQKE: bool = False
    PQKP: bool = False
    EU: bool = False
    PU: bool = False
    EVOU: bool = False
    PVOU: bool = False

    @staticmethod
    def all() -> ModelMatrixLoggingOptions:
        return ModelMatrixLoggingOptions(
            EQKE=True,
            EQKP=True,
            EU=True,
            PU=True,
            EVOU=True,
            PVOU=True,
            PQKP=True,
            PQKE=True,
        )

    def assert_model_supported(self, model: HookedTransformer, unsafe: bool = False):
        def error_unless(test: bool, message: str):
            if unsafe:
                if not test:
                    logging.warning(message)
            else:
                assert test, message

        error_unless(
            (model.cfg.normalization_type is None),
            f"Automatic logging for normalization type {model.cfg.normalization_type} is not yet implemented",
        )
        max_supported_layers = 1
        error_unless(
            (model.cfg.n_layers == 1),
            f"Automatic logging for {model.cfg.n_layers} layers is not yet implemented (max is {max_supported_layers})",
        )
        error_unless(
            (model.cfg.attn_only),
            "Automatic logging is only supported for attention-only models",
        )

    @torch.no_grad()
    def log_matrices(
        self,
        log: Callable[[str, Any], None],
        model: HookedTransformer,
        unsafe: bool = False,
        **kwargs,
    ):
        self.assert_model_supported(model, unsafe=unsafe)
        W_E, W_pos, W_U, W_Q, W_K, W_V, W_O = (
            model.W_E,
            model.W_pos,
            model.W_U,
            model.W_Q,
            model.W_K,
            model.W_V,
            model.W_O,
        )
        b_U, b_Q, b_K, b_V, b_O = (
            model.b_U,
            model.b_Q,
            model.b_K,
            model.b_V,
            model.b_O,
        )
        for h in range(W_Q.shape[1]):
            if self.EQKE:
                log(
                    f"EQKE.{h}",
                    (W_E @ W_Q[0, h, :, :] + b_Q[0, h, None, :])
                    @ (W_E @ W_K[0, h, :, :] + b_K[0, h, None, :]).transpose(-1, -2),
                    **kwargs,
                )
            if self.EQKP:
                log(
                    f"EQKP.{h}",
                    (W_E @ W_Q[0, h, :, :] + b_Q[0, h, None, :])
                    @ (W_pos @ W_K[0, h, :, :] + b_K[0, h, None, :]).transpose(-1, -2),
                    **kwargs,
                )
            if self.EVOU:
                log(
                    f"EVOU.{h}",
                    (
                        (W_E @ W_V[0, h, :, :] + b_V[0, h, None, :]) @ W_O[0, h, :, :]
                        + b_O[0, None, None, :]
                    )
                    @ W_U
                    + b_U,
                    **kwargs,
                )
            if self.PVOU:
                log(
                    f"PVOU.{h}",
                    (
                        (W_pos @ W_V[0, h, :, :] + b_V[0, h, None, :]) @ W_O[0, h, :, :]
                        + b_O[0, None, None, :]
                    )
                    @ W_U
                    + b_U,
                    **kwargs,
                )
        if self.EU:
            log(f"EU", W_E @ W_U, **kwargs)
        if self.PU:
            log(f"EU", W_E @ W_U, **kwargs)


@torch.no_grad()
def log_tensor(logger: WandbLogger, name, matrix, **kwargs):
    # Check the number of dimensions in the matrix to determine the plot type
    if len(matrix.shape) == 1:
        # For 1D tensors, create a line plot
        fig, ax = plt.subplots()
        # Ensure matrix is on CPU and converted to numpy for plotting
        ax.plot(matrix.cpu().numpy())
        ax.set_title(name)
    elif len(matrix.shape) == 2:
        # For 2D tensors, use imshow to create a heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(
            matrix.cpu().numpy(), **kwargs
        )  # Ensure matrix is on CPU and converted to numpy for plotting
        fig.colorbar(cax)
        ax.set_title(name)
        # Optional: Customize the plot further, e.g., adjust the aspect ratio, add labels, etc.
    else:
        raise ValueError(f"Cannot plot tensor of shape {matrix.shape} ({name})")
    logger.log_image(name, [fig], **kwargs)
    # I'd like to do https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots but am not sure how cf https://github.com/JasonGross/guarantees-based-mechanistic-interpretability/issues/33 cc Euan
    # self.log(name, fig, **kwargs)
    plt.close(fig)
