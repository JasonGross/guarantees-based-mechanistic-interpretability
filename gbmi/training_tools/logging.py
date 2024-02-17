from __future__ import annotations
from functools import partial
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional
from jaxtyping import Float
from lightning.pytorch.loggers.wandb import WandbLogger
import logging


@torch.no_grad()
def log_tensor(
    logger: WandbLogger,
    name,
    matrix,
    plot_1D_kind: Literal["line", "scatter"] = "line",
    **kwargs,
):
    # Ensure matrix is on CPU and converted to numpy for plotting
    matrix = matrix.squeeze().cpu().numpy()
    # Check the number of dimensions in the matrix to determine the plot type
    if len(matrix.shape) == 1:
        # For 1D tensors, create a line plot
        fig, ax = plt.subplots()
        match plot_1D_kind:
            case "line":
                ax.plot(matrix)
            case "scatter":
                ax.scatter(range(len(matrix)), matrix)
        ax.set_title(name)
    elif len(matrix.shape) == 2:
        # For 2D tensors, use imshow to create a heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(
            matrix, **kwargs
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
    log_zeros: bool = False
    qpos: Optional[int] = None
    qtok: Optional[int] = None
    add_mean_pos_to_tok: bool = True
    plot_1D_kind: Literal["line", "scatter"] = "line"

    @staticmethod
    def all(**kwargs) -> ModelMatrixLoggingOptions:
        return ModelMatrixLoggingOptions(
            EQKE=True,
            EQKP=True,
            EU=True,
            PU=True,
            EVOU=True,
            PVOU=True,
            PQKP=True,
            PQKE=True,
            **kwargs,
        )

    def __post_init__(self):
        pass

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
    def _log_matrices(
        self,
        log: Callable[[str, Any], None],
        model: HookedTransformer,
        *,
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
        if (
            self.EQKE
            or self.EQKP
            or self.PQKE
            or self.PQKP
            or self.EVOU
            or self.PVOU
            or self.EU
            or self.PU
        ):

            def apply_U(
                x: Float[Tensor, "... d_model"]  # noqa: F722
            ) -> Float[Tensor, "... d_vocab_out"]:  # noqa: F722
                return x @ W_U + b_U

            d_vocab = W_E.shape[0]
            n_ctx = W_pos.shape[0]
            if self.qtok is not None:
                sEq = f"E[{self.qtok}]"
                W_E_q: Float[Tensor, "d_model"]  # noqa: F821
                W_E_q = W_E[self.qtok]
                W_E_k: Float[Tensor, "d_vocab-1 d_model"]  # noqa: F722
                if self.qtok % d_vocab == -1 % d_vocab:
                    sEk = f"(E[:-1] - E[-1])"
                    W_E_k = W_E[: self.qtok] - W_E_q
                elif self.qtok == 0:
                    sEk = f"(E[1:] - E[0])"
                    W_E_k = W_E[self.qtok + 1 :] - W_E_q
                else:
                    sEk = f"(E[:{self.qtok}] + E[{self.qtok+1}:] - E[{self.qtok}])"
                    W_E_k = (
                        torch.cat([W_E[: self.qtok], W_E[self.qtok + 1 :]], dim=0)
                        - W_E_q
                    )
            else:
                sEq = f"E"
                W_E_q: Float[Tensor, "d_vocab d_model"]  # noqa: F722
                W_E_q = W_E
                sEk = f"E"
                W_E_k: Float[Tensor, "d_vocab d_model"]  # noqa: F722
                W_E_k = W_E
            if self.qpos is not None:
                sPq = f"P[{self.qpos}]"
                W_pos_q: Float[Tensor, "d_model"]  # noqa: F821
                W_pos_q = W_pos[self.qpos]
                match self.qpos, self.add_mean_pos_to_tok:
                    case -1, False:
                        sPk = f"(P[:-1] - P[-1])"
                    case 0, False:
                        sPk = f"(P[1:] - P[0])"
                    case _, False:
                        sPk = f"(P[:{self.qpos}] + P[{self.qpos+1}:] - P[{self.qpos}])"
                    case -1, True:
                        sEk = f"({sEk} + mean(P[:-1] - P[-1]))"
                        sPk = f"(P[:-1] - mean(P[:-1]))"
                    case 0, True:
                        sEk = f"({sEk} + mean(P[1:] - P[0]))"
                        sPk = f"(P[1:] - mean(P[1:]))"
                    case _, True:
                        sEk = f"({sEk} + mean(P[:{self.qpos}] + P[{self.qpos+1}:] - P[{self.qpos}]))"
                        sPk = f"(P[:{self.qpos}] + P[{self.qpos+1}:] - mean(P[:{self.qpos}] + P[{self.qpos+1}:]))"
                W_pos_k: Float[Tensor, "n_ctx-1 d_model"]  # noqa: F722
                if self.qpos % n_ctx == -1 % n_ctx:
                    W_pos_k = W_pos[: self.qpos] - W_pos_q
                elif self.qpos == 0:
                    W_pos_k = W_pos[self.qpos + 1 :] - W_pos_q
                else:
                    W_pos_k = (
                        torch.cat([W_pos[: self.qpos], W_pos[self.qpos + 1 :]], dim=0)
                        - W_pos_q
                    )
                if self.add_mean_pos_to_tok:
                    W_E_q = W_E_q + W_pos_q
                    W_pos_q = W_pos_q - W_pos_q
                    W_pos_k_avg = W_pos_k.mean(dim=0)
                    W_E_k = W_E_k + W_pos_k_avg
                    W_pos_k = W_pos_k - W_pos_k_avg
                    sEq = f"({sEq} + {sPq})"
                    sPq = f"0"
            else:
                W_pos_q: Float[Tensor, "n_ctx d_model"]  # noqa: F722
                W_pos_q = W_pos
                sPq = f"P"
                W_pos_k: Float[Tensor, "n_ctx d_model"]  # noqa: F722
                W_pos_k = W_pos
                sPk = f"P"
                if self.add_mean_pos_to_tok:
                    W_pos_k_avg = W_pos_k.mean(dim=0)
                    W_pos_q_avg = W_pos_q.mean(dim=0)
                    W_E_q = W_E_q + W_pos_q_avg
                    W_pos_q = W_pos_q - W_pos_q_avg
                    W_E_k = W_E_k + W_pos_k_avg
                    W_pos_k = W_pos_k - W_pos_k_avg
                    sEq = f"({sEq} + mean({sPq}))"
                    sPq = f"({sPq} - mean({sPq}))"
                    sEk = f"({sEk} + mean({sPk}))"
                    sPk = f"({sPk} - mean({sPk}))"
            W_E_v: Float[Tensor, "d_vocab d_model"]  # noqa: F722
            W_pos_v: Float[Tensor, "n_ctx d_model"]  # noqa: F722
            W_E_v = W_E
            W_pos_v = W_pos
            sEv = f"E"
            sPv = f"P"
            if self.add_mean_pos_to_tok:
                W_E_v = W_E_v + W_pos_v.mean(dim=0)
                W_pos_v = W_pos_v - W_pos_v.mean(dim=0)
                sEv = f"({sEv} + mean({sPv}))"
                sPv = f"({sPv} - mean({sPv}))"
        sPk = f"{sPk}ᵀ"
        sEk = f"{sEk}ᵀ"
        if self.EU:
            log(f"{sEq}U", apply_U(W_E_q), **kwargs)
        if self.PU and (sPq != "0" or self.log_zeros):
            log(f"{sPq}U", apply_U(W_pos_q), **kwargs)

        def apply_VO(
            x: Float[Tensor, "... a d_model"], l: int, h: int  # noqa: F722
        ) -> Float[Tensor, "... a d_model"]:  # noqa: F722
            return (x @ W_V[l, h, :, :] + b_V[l, h, None, :]) @ W_O[l, h, :, :] + b_O[
                l, None, None, :
            ]

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
                        f"{sEq}QKᵀ{sEk}.l{l}h{h}",
                        apply_Q(W_E_q, l, h) @ apply_KT(W_E_k, l, h),
                        **kwargs,
                    )
                if self.EQKP:
                    log(
                        f"{sEq}QKᵀ{sPk}.l{l}h{h}",
                        apply_Q(W_E_q, l, h) @ apply_KT(W_pos_k, l, h),
                        **kwargs,
                    )
                if self.PQKE and (sPq != "0" or self.log_zeros):
                    log(
                        f"{sPq}QKᵀ{sEk}.l{l}h{h}",
                        apply_Q(W_pos_q, l, h) @ apply_KT(W_E_k, l, h),
                        **kwargs,
                    )
                if self.PQKP and (sPq != "0" or self.log_zeros):
                    log(
                        f"{sPq}QKᵀ{sPk}.l{l}h{h}",
                        apply_Q(W_pos_q, l, h) @ apply_KT(W_pos_k, l, h),
                        **kwargs,
                    )
                if self.EVOU:
                    log(
                        f"{sEv}VOU.l{l}h{h}",
                        apply_U(apply_VO(W_E_v, l, h)),
                        **kwargs,
                    )
                if self.PVOU:
                    log(
                        f"{sPv}VOU.l{l}h{h}",
                        apply_U(apply_VO(W_pos_v, l, h)),
                        **kwargs,
                    )
        for l in range(1, W_Q.shape[0]):
            for h0 in range(W_Q.shape[1]):
                W_E_q_possibilities = (
                    (W_E_q, f"{sEq}"),
                    (apply_VO(W_E_v, 0, h0), "{sEv}VO"),
                )
                W_pos_q_possibilities = (
                    (W_pos_q, f"{sPq}"),
                    (apply_VO(W_pos_v, 0, h0), "{sPv}VO"),
                )
                W_E_k_possibilities = (
                    (W_E_k, f"{sEk}"),
                    (apply_VO(W_E_v, 0, h0), "OᵀVᵀ{sEv}ᵀ"),
                )
                W_pos_k_possibilities = (
                    (W_pos_k, f"{sPk}"),
                    (apply_VO(W_pos_v, 0, h0), "OᵀVᵀ{sPv}ᵀ"),
                )
                for h in range(W_Q.shape[1]):
                    if self.EQKE:
                        for li, (lhs, lhs_name) in enumerate(W_E_q_possibilities):
                            for ri, (rhs, rhs_name) in enumerate(W_E_k_possibilities):
                                if li == 0 and ri == 0:
                                    continue
                                if lhs_name != "0" or self.log_zeros:
                                    log(
                                        f"{lhs_name}QKᵀ{rhs_name}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                    if self.EQKP:
                        for li, (lhs, lhs_name) in enumerate(W_E_q_possibilities):
                            for ri, (rhs, rhs_name) in enumerate(W_pos_k_possibilities):
                                if li == 0 and ri == 0:
                                    continue
                                if lhs_name != "0" or self.log_zeros:
                                    log(
                                        f"{lhs_name}QKᵀ{rhs_name}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                    if self.PQKE:
                        for li, (lhs, lhs_name) in enumerate(W_pos_q_possibilities):
                            for ri, (rhs, rhs_name) in enumerate(W_E_k_possibilities):
                                if li == 0 and ri == 0:
                                    continue
                                if lhs_name != "0" or self.log_zeros:
                                    log(
                                        f"{lhs_name}QKᵀ{rhs_name}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                    if self.PQKP:
                        for li, (lhs, lhs_name) in enumerate(W_pos_q_possibilities):
                            for ri, (rhs, rhs_name) in enumerate(W_pos_k_possibilities):
                                if li == 0 and ri == 0:
                                    continue
                                if lhs_name != "0" or self.log_zeros:
                                    log(
                                        f"{lhs_name}QKᵀ{rhs_name}.l{l}h{h}h₀{h0}",
                                        apply_Q(lhs, l, h) @ apply_KT(rhs, l, h),
                                        **kwargs,
                                    )
                    if self.EVOU:
                        log(
                            f"{sEv}VOVOU.l{l}h{h}h₀{h0}",
                            apply_U(apply_VO(apply_VO(W_E_v, 0, h0), l, h)),
                            **kwargs,
                        )
                    if self.PVOU:
                        log(
                            f"{sPv}VOVOU.l{l}h{h}h₀{h0}",
                            apply_U(apply_VO(apply_VO(W_pos_v, 0, h0), l, h)),
                            **kwargs,
                        )

    @torch.no_grad()
    def log_matrices(
        self,
        logger: WandbLogger,
        model: HookedTransformer,
        *,
        unsafe: bool = False,
        **kwargs,
    ):
        self._log_matrices(
            partial(log_tensor, logger, plot_1D_kind=self.plot_1D_kind, **kwargs),
            model,
            unsafe=unsafe,
        )
