from __future__ import annotations
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from dataclasses import dataclass
from typing import Any, Callable, Optional
from jaxtyping import Float
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
        max_supported_layers = 2
        error_unless(
            (model.cfg.n_layers <= max_supported_layers),
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
        W_E: Float[Tensor, "d_vocab d_model"]  # noqa: F722
        W_pos: Float[Tensor, "n_ctx d_model"]  # noqa: F722
        W_U: Float[Tensor, "d_model d_vocab_out"]  # noqa: F722
        W_Q: Float[Tensor, "n_layers n_heads d_model d_head"]  # noqa: F722
        W_K: Float[Tensor, "n_layers n_heads d_model d_head"]  # noqa: F722
        W_V: Float[Tensor, "n_layers n_heads d_model d_head"]  # noqa: F722
        W_O: Float[Tensor, "n_layers n_heads d_head d_model"]  # noqa: F722
        b_U: Float[Tensor, "d_vocab_out"]  # noqa: F821
        b_Q: Float[Tensor, "n_layers n_heads d_head"]  # noqa: F722
        b_K: Float[Tensor, "n_layers n_heads d_head"]  # noqa: F722
        b_V: Float[Tensor, "n_layers n_heads d_head"]  # noqa: F722
        b_O: Float[Tensor, "n_layers d_model"]  # noqa: F722
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
        if self.EU:
            log(f"EU", W_E @ W_U, **kwargs)
        if self.PU:
            log(f"PU", W_pos @ W_U, **kwargs)
        if self.EQKE or self.EQKP or self.PQKE or self.PQKP or self.EVOU or self.PVOU:
            EVO: Float[Tensor, "n_layers n_heads d_vocab d_model"] = (  # noqa: F722
                W_E @ W_V + b_V[:, :, None, :]
            ) @ W_O[:, :, :, :] + b_O[:, None, None, :]
            PVO: Float[Tensor, "n_layers n_heads n_ctx d_model"] = (  # noqa: F722
                W_pos @ W_V + b_V[:, :, None, :]
            ) @ W_O[:, :, :, :] + b_O[:, None, None, :]

            def apply_VO(
                x: Float[Tensor, "... a d_model"], l: int, h: int  # noqa: F722
            ) -> Float[Tensor, "... a d_model"]:  # noqa: F722
                return (x @ W_V[l, h, :, :] + b_V[l, h, None, :]) @ W_O[
                    l, h, :, :
                ] + b_O[l, None, None, :]

            def apply_Q(
                x: Float[Tensor, "... a d_model"], l: int, h: int  # noqa: F722
            ) -> Float[Tensor, "... a d_head"]:  # noqa: F722
                return x @ W_Q[l, h, :, :] + b_Q[l, h, None, :]

            def apply_KT(
                x: Float[Tensor, "... a d_model"], l: int, h: int  # noqa: F722
            ) -> Float[Tensor, "... d_head a"]:  # noqa: F722
                return (x @ W_K[l, h, :, :] + b_K[l, h, None, :]).transpose(-1, -2)

            for l in range(W_Q.shape[0]):
                for h in range(W_Q.shape[1]):
                    if self.EQKE:
                        log(
                            f"EQKE.l{l}h{h}",
                            apply_Q(W_E, l, h) @ apply_KT(W_E, l, h),
                            **kwargs,
                        )
                    if self.EQKP:
                        log(
                            f"EQKP.l{l}h{h}",
                            apply_Q(W_E, l, h) @ apply_KT(W_pos, l, h),
                            **kwargs,
                        )
                    if self.PQKE:
                        log(
                            f"PQKE.l{l}h{h}",
                            apply_Q(W_pos, l, h) @ apply_KT(W_E, l, h),
                            **kwargs,
                        )
                    if self.PQKP:
                        log(
                            f"PQKP.l{l}h{h}",
                            apply_Q(W_pos, l, h) @ apply_KT(W_pos, l, h),
                            **kwargs,
                        )
                    if self.EVOU:
                        log(
                            f"EVOU.l{l}h{h}",
                            EVO[l, h] @ W_U + b_U,
                            **kwargs,
                        )
                    if self.PVOU:
                        log(
                            f"PVOU.l{l}h{h}",
                            PVO[l, h] @ W_U + b_U,
                            **kwargs,
                        )
            for l in range(1, W_Q.shape[0]):
                for h0 in range(W_Q.shape[1]):
                    W_E_possibilities = ((W_E, "E"), (EVO[0, h0], "EVO"))
                    W_pos_possibilities = ((W_pos, "P"), (PVO[0, h0], "PVO"))
                    for h in range(W_Q.shape[1]):
                        if self.EQKE:
                            for lhs, lhs_name in W_E_possibilities:
                                for rhs, rhs_name in W_E_possibilities:
                                    if "VO" not in lhs_name and "VO" not in rhs_name:
                                        continue
                                    log(
                                        f"{lhs_name}QK{rhs_name[::-1]}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                        if self.EQKP:
                            for lhs, lhs_name in W_E_possibilities:
                                for rhs, rhs_name in W_pos_possibilities:
                                    if "VO" not in lhs_name and "VO" not in rhs_name:
                                        continue
                                    log(
                                        f"{lhs_name}QK{rhs_name[::-1]}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                        if self.PQKE:
                            for lhs, lhs_name in W_pos_possibilities:
                                for rhs, rhs_name in W_E_possibilities:
                                    if "VO" not in lhs_name and "VO" not in rhs_name:
                                        continue
                                    log(
                                        f"{lhs_name}QK{rhs_name[::-1]}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                        if self.PQKP:
                            for lhs, lhs_name in W_pos_possibilities:
                                for rhs, rhs_name in W_pos_possibilities:
                                    if "VO" not in lhs_name and "VO" not in rhs_name:
                                        continue
                                    log(
                                        f"{lhs_name}QK{rhs_name[::-1]}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                        if self.EVOU:
                            log(
                                f"EVOU.l{l}h{h}h₀{h0}",
                                (apply_VO(EVO[0, h0], l, h)) @ W_U + b_U,
                                **kwargs,
                            )
                        if self.PVOU:
                            log(
                                f"PVOU.l{l}h{h}h₀{h0}",
                                (apply_VO(PVO[0, h0], l, h)) @ W_U + b_U,
                                **kwargs,
                            )


@torch.no_grad()
def log_tensor(logger: WandbLogger, name, matrix, **kwargs):
    matrix = matrix.squeeze()
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
