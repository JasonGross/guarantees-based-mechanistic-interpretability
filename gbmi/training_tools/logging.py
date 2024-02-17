from __future__ import annotations
from functools import partial
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Tuple
from jaxtyping import Float
from lightning.pytorch.loggers.wandb import WandbLogger
import logging

from gbmi.utils import subscript


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
        error_unless(
            (model.cfg.attn_only),
            "Automatic logging is only supported for attention-only models",
        )

    @staticmethod
    @torch.no_grad()
    def _compute_paths(
        apply_VO: Callable[
            [Float[Tensor, "... a d_model"], int, int],  # noqa: F722
            Float[Tensor, "... a d_model"],  # noqa: F722
        ],  # x, layer, head
        n_heads: int,
        x: Float[Tensor, "... a d_model"],  # noqa: F722
        sx: str,
        l: int,  # layer that we're computing input paths to
        reverse_strs: bool = False,
    ) -> Iterable[Tuple[str, str, Float[Tensor, "... a d_model"]]]:  # noqa: F722
        """Returns an iterable of ("VO"*, "lₙ{l}hₙ{h}"*, value) tuples of what x transforms to under repeated applications of apply_VO to layers up to and inlcuding l"""
        if l < 0:
            return
        if l == 0:
            for h in range(n_heads):
                cur_vo = f"{sx}VO" if not reverse_strs else f"OᵀVᵀ{sx}"
                yield cur_vo, f"h{subscript(str(l))}{h}", apply_VO(x, l, h)
            return
        vo2 = "VO" if not reverse_strs else "OᵀVᵀ"
        for vo, lh, value in ModelMatrixLoggingOptions._compute_paths(
            apply_VO, n_heads, x, sx, l - 1, reverse_strs=reverse_strs
        ):
            for h in range(n_heads):
                lh2 = f"h{subscript(str(l-1))}{h}"
                cur_vo, cur_lh = (
                    (f"{vo}{vo2}", f"{lh}{lh2}")
                    if not reverse_strs
                    else (f"{vo2}{vo}", f"{lh2}{lh}")
                )
                yield cur_vo, cur_lh, apply_VO(value, l, h)

    @staticmethod
    def compute_paths(
        apply_VO: Callable[
            [Float[Tensor, "... a d_model"], int, int],  # noqa: F722
            Float[Tensor, "... a d_model"],  # noqa: F722
        ],  # x, layer, head
        n_heads: int,
        x: Float[Tensor, "... a d_model"],  # noqa: F722
        x_direct: Float[Tensor, "... a d_model"],  # noqa: F722
        sx: str,
        sx_direct: str,
        l: int,  # layer that we're computing input paths to
        reverse_strs: bool = False,
    ) -> Iterable[Tuple[str, str, Float[Tensor, "... a d_model"]]]:  # noqa: F722
        """Returns an iterable of ("VO"*, "lₙ{l}hₙ{h}"*, value) tuples of what x transforms to under repeated applications of apply_VO to layers strictly before l"""
        yield sx_direct, "", x_direct
        yield from ModelMatrixLoggingOptions._compute_paths(
            apply_VO, n_heads, x, sx, l - 1, reverse_strs=reverse_strs
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

        def apply_U(
            x: Float[Tensor, "... d_model"]  # noqa: F722
        ) -> Float[Tensor, "... d_vocab_out"]:  # noqa: F722
            return x @ W_U + b_U

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

        if self.EU:
            log(f"{sEq}U", apply_U(W_E_q), **kwargs)
        if self.PU and (sPq != "0" or self.log_zeros):
            log(f"{sPq}U", apply_U(W_pos_q), **kwargs)

        for l in range(W_Q.shape[0]):
            for h in range(W_Q.shape[1]):
                for (
                    (qx, qx_direct, qsx, qsx_direct),
                    (kx, kx_direct, ksx, ksx_direct),
                    test,
                ) in (
                    ((W_E_v, W_E_q, sEv, sEq), (W_E_v, W_E_k, sEv, sEk), self.EQKE),
                    (
                        (W_E_v, W_pos_q, sEv, sPq),
                        (W_pos_v, W_pos_k, sPv, sPk),
                        self.EQKP,
                    ),
                    (
                        (W_pos_v, W_pos_q, sPv, sPq),
                        (W_E_v, W_E_k, sEv, sEk),
                        self.PQKE,
                    ),
                    (
                        (W_pos_v, W_pos_q, sPv, sPq),
                        (W_pos_v, W_pos_k, sPv, sPk),
                        self.PQKP,
                    ),
                ):
                    if test:
                        for sq, lh_q, v_q in ModelMatrixLoggingOptions.compute_paths(
                            apply_VO,
                            model.cfg.n_heads,
                            x=qx,
                            x_direct=qx_direct,
                            sx=qsx,
                            sx_direct=qsx_direct,
                            l=l,
                            reverse_strs=False,
                        ):
                            for (
                                sk,
                                lh_k,
                                v_k,
                            ) in ModelMatrixLoggingOptions.compute_paths(
                                apply_VO,
                                model.cfg.n_heads,
                                x=kx,
                                x_direct=kx_direct,
                                sx=f"{ksx}ᵀ",
                                sx_direct=ksx_direct,
                                l=l,
                                reverse_strs=True,
                            ):
                                if sq != "0" or self.log_zeros:
                                    log(
                                        f"{sq}QKᵀ{sk}.{lh_q}l{l}h{h}{lh_k}",
                                        apply_Q(v_q, l, h) @ apply_KT(v_k, l, h),
                                        **kwargs,
                                    )
                if self.EVOU:
                    for sv, lh_v, v in ModelMatrixLoggingOptions.compute_paths(
                        apply_VO,
                        model.cfg.n_heads,
                        x=W_E_v,
                        x_direct=W_E_v,
                        sx=sEv,
                        sx_direct=sEv,
                        l=l,
                        reverse_strs=False,
                    ):
                        log(
                            f"{sv}VOU.{lh_v}l{l}h{h}",
                            apply_U(apply_VO(v, l, h)),
                            **kwargs,
                        )
                if self.PVOU:
                    for sv, lh_v, v in ModelMatrixLoggingOptions.compute_paths(
                        apply_VO,
                        model.cfg.n_heads,
                        x=W_pos_v,
                        x_direct=W_pos_v,
                        sx=sPv,
                        sx_direct=sPv,
                        l=l,
                        reverse_strs=False,
                    ):
                        log(
                            f"{sv}VOU.{lh_v}l{l}h{h}",
                            apply_U(apply_VO(v, l, h)),
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
