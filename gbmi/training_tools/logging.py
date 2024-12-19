from __future__ import annotations

import logging
import re
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from jaxtyping import Float
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from torch import Tensor
from transformer_lens import HookedTransformer
from wandb.sdk.wandb_run import Run

from gbmi.utils import subscript


def encode_4_byte_unicode(text: str) -> str:
    encoded_parts = []
    for char in text:
        # Check if the character is a 4-byte Unicode character
        if ord(char) > 0xFFFF:
            # URL-encode the 4-byte character
            encoded_parts.append(urllib.parse.quote(char))
        else:
            # Add the normal character to the result
            encoded_parts.append(char)
    return "".join(encoded_parts)


def str_mean(s: str) -> str:
    if len(s) == 1 or all(c.isalnum() for c in s):
        return f"𝔼{s}"
    return f"𝔼({s})"


def calculate_zmax_zmin_args(
    matrices: Iterable[Tuple[str, Tensor]],
    groups: Optional[
        Union[Collection[Collection[str]], dict[Collection[str], dict[str, Any]]]
    ] = None,
) -> dict[str, dict[str, Any]]:
    """Computes zmax and zmin by grouping matrices"""
    if groups is None:
        return {}
    groups_map: dict[str, int] = {}
    groups_extra_args: dict[Optional[int], dict[str, Any]] = {}
    for i, group in enumerate(groups):
        for name in group:
            groups_map[name] = i
        if isinstance(groups, dict):
            groups_extra_args[i] = groups[group]
    group_to_matrix_map: dict[Optional[int], list[Tensor]] = defaultdict(list)
    matrices = list(matrices)
    for name, matrix in matrices:
        group_to_matrix_map[groups_map.get(name)].append(matrix)
    zmax_zmin_args_by_group: dict[Optional[int], dict[str, float]] = {}
    for i, ms in group_to_matrix_map.items():
        zmax_zmin_args_by_group[i] = {
            "zmax": max(m[~m.isnan()].max().item() for m in ms),
            "zmin": min(m[~m.isnan()].min().item() for m in ms),
            **groups_extra_args.get(i, {}),
        }
        if "zmid" in zmax_zmin_args_by_group[i]:
            zhalfrange = np.max(
                (
                    np.abs(
                        zmax_zmin_args_by_group[i]["zmax"]
                        - zmax_zmin_args_by_group[i]["zmid"]
                    ),
                    np.abs(
                        zmax_zmin_args_by_group[i]["zmin"]
                        - zmax_zmin_args_by_group[i]["zmid"]
                    ),
                )
            )
            zmax_zmin_args_by_group[i]["zmax"] = (
                zmax_zmin_args_by_group[i]["zmid"] + zhalfrange
            )
            zmax_zmin_args_by_group[i]["zmin"] = (
                zmax_zmin_args_by_group[i]["zmid"] - zhalfrange
            )
    zmax_zmin_args = {}
    for name, _ in matrices:
        zmax_zmin_args[name] = zmax_zmin_args_by_group[groups_map.get(name)]
    return zmax_zmin_args


def plot_tensors(
    matrices: Iterable[Tuple[str, Tensor]],
    *,
    plot_1D_kind: Literal["line", "scatter"] = "line",
    title="Subplots of Matrices",
    groups: Optional[Collection[Collection[str]]] = None,
    **kwargs,
) -> go.Figure:
    # Calculate grid size based on the number of matrices
    matrices = list(matrices)
    zmax_zmin_args = calculate_zmax_zmin_args(matrices, groups=groups)
    num_matrices = len(matrices)
    grid_size = int(np.ceil(np.sqrt(num_matrices)))
    subplot_titles = [name for name, _ in matrices] if len(matrices) else None

    # Create a subplot figure with calculated grid size
    fig = make_subplots(
        rows=grid_size,
        cols=grid_size,
        subplot_titles=subplot_titles,
    )

    # Initialize subplot position trackers
    row = 1
    col = 1

    # Iterate through each matrix in the dictionary
    for name, matrix in matrices:
        matrix = matrix.squeeze().cpu()  # Ensure matrix is 2D or 1D

        # Determine plot type based on matrix dimensions and add to subplot
        if len(matrix.shape) == 1:
            # 1D data - line plot
            fig.add_trace(
                go.Scatter(
                    y=matrix,
                    mode="lines" if plot_1D_kind == "line" else "markers",
                    name=name,
                ),
                row=row,
                col=col,
            )
        elif len(matrix.shape) == 2:
            # 2D data - heatmap
            fig.add_trace(
                go.Heatmap(z=matrix, name=name, **zmax_zmin_args.get(name, {})),
                row=row,
                col=col,
            )
            fig.update_yaxes(autorange="reversed", row=row, col=col)
        else:
            raise ValueError(f"Cannot plot tensor of shape {matrix.shape} ({name})")

        # Update subplot position for next plot
        col += 1
        if col > grid_size:
            col = 1
            row += 1

    # Update layout to adjust aspect ratio if necessary
    fig.update_layout(title_text=title, **kwargs)

    return fig


@torch.no_grad()
def log_tensor(
    logger: Run,
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
    logger.log({encode_4_byte_unicode(name): fig}, commit=False, **kwargs)
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
    add_mean: Union[
        Optional[Literal["pos_to_tok", "tok_to_pos"]],
        dict[int, Optional[Literal["pos_to_tok", "tok_to_pos"]]],
    ] = "pos_to_tok"
    plot_1D_kind: Literal["line", "scatter"] = "line"
    use_subplots: bool = True
    superplot_title: str = "model matrices"
    _superplot_title_extra: str = ""
    group_colorbars: bool = True
    shortformer: bool = False
    nanify_causal_attn: bool = True
    include_short_title: bool = True

    @staticmethod
    def all(**kwargs) -> ModelMatrixLoggingOptions:
        return ModelMatrixLoggingOptions(
            **(
                dict(
                    EQKE=True,
                    EQKP=True,
                    EU=True,
                    PU=True,
                    EVOU=True,
                    PVOU=True,
                    PQKP=True,
                    PQKE=True,
                )
                | kwargs
            ),  # type: ignore
        )

    @staticmethod
    def none(**kwargs) -> ModelMatrixLoggingOptions:
        return ModelMatrixLoggingOptions(
            **kwargs,
        )

    def __post_init__(self):
        if self.shortformer:
            self.PVOU = False
            self.PU = False

    def assert_model_supported(self, model: HookedTransformer, unsafe: bool = False):
        def error_unless(test: bool, message: str, warn_only: bool = False):
            if unsafe or warn_only:
                if not test:
                    logging.warning(message)
            else:
                assert test, message

        self._superplot_title_extra = ""

        error_unless(
            (model.cfg.normalization_type is None),
            f"Automatic logging for normalization type {model.cfg.normalization_type} is not yet implemented",
        )
        if not (
            model.cfg.attn_only
            or (not self.EVOU and not self.PVOU and model.cfg.n_layers == 1)
        ):
            logging.warning(
                "Automatic logging is only complete for attention-only models, or for 1L models with EVOU and PVOU logging turned off"
            )
            self._superplot_title_extra += " (missing MLPs)"

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
        sx_short: str,
        l: int,  # layer that we're computing input paths to
        reverse_strs: bool = False,
    ) -> Iterable[Tuple[str, str, str, Float[Tensor, "... a d_model"]]]:  # noqa: F722
        """Returns an iterable of ("VO"*, "lₙ{l}hₙ{h}"*, value) tuples of what x transforms to under repeated applications of apply_VO to layers up to and inlcuding l"""
        if l < 0:
            return
        if l == 0:
            for h in range(n_heads):
                cur_vo = f"{sx}VO" if not reverse_strs else f"OᵀVᵀ{sx}"
                cur_vo_short = f"{sx_short}VO" if not reverse_strs else f"OV{sx_short}"
                yield cur_vo, cur_vo_short, f"h{subscript(str(l))}{h}", apply_VO(
                    x, l, h
                )
            return
        vo2 = "VO" if not reverse_strs else "OᵀVᵀ"
        vo2_short = "VO" if not reverse_strs else "OV"
        for vo, vo_short, lh, value in ModelMatrixLoggingOptions._compute_paths(
            apply_VO, n_heads, x, sx, sx_short, l - 1, reverse_strs=reverse_strs
        ):
            for h in range(n_heads):
                lh2 = f"h{subscript(str(l-1))}{h}"
                cur_vo, cur_vo_short, cur_lh = (
                    (f"{vo}{vo2}", f"{vo_short}{vo2_short}", f"{lh}{lh2}")
                    if not reverse_strs
                    else (f"{vo2}{vo}", f"{vo2_short}{vo_short}", f"{lh2}{lh}")
                )
                yield cur_vo, cur_vo_short, cur_lh, apply_VO(value, l, h)

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
        sx_short: str,
        sx_direct: str,
        sx_direct_short: str,
        l: int,  # layer that we're computing input paths to
        reverse_strs: bool = False,
        *,
        skip_composition: bool = False,
    ) -> Iterable[
        Tuple[str, str, str, Float[Tensor, "... a d_model"], bool]  # noqa: F722
    ]:
        """Returns an iterable of ("VO"*, "VO"*, "lₙ{l}hₙ{h}"*, value, is_direct) tuples of what x transforms to under repeated applications of apply_VO to layers strictly before l"""
        yield sx_direct, sx_direct_short, "", x_direct, True
        if not skip_composition:
            for svo, svo_short, lh, val in ModelMatrixLoggingOptions._compute_paths(
                apply_VO, n_heads, x, sx, sx_short, l - 1, reverse_strs=reverse_strs
            ):
                yield svo, svo_short, lh, val, False

    @torch.no_grad()
    def matrices_to_log(
        self,
        model: HookedTransformer,
        *,
        unsafe: bool = False,
    ) -> Iterable[Tuple[str, Tensor]]:
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
            sEq: dict[int, str] = {}
            sEk: dict[int, str] = {}
            sPq: dict[int, str] = {}
            sPk: dict[int, str] = {}
            sEv: dict[int, str] = {}
            sPv: dict[int, str] = {}
            W_E_q: Union[
                dict[int, Float[Tensor, "d_model"]],  # noqa: F821
                dict[int, Float[Tensor, "d_vocab d_model"]],  # noqa: F722
            ] = {}
            W_E_k: Union[
                dict[int, Float[Tensor, "d_vocab-1 d_model"]],  # noqa: F722
                dict[int, Float[Tensor, "d_vocab d_model"]],  # noqa: F722
            ] = {}
            W_pos_q: dict[
                int,
                Union[
                    Float[Tensor, "d_model"],  # noqa: F821
                    Float[Tensor, "n_ctx d_model"],  # noqa: F722
                ],
            ] = {}
            W_pos_k: dict[
                int,
                Union[
                    Float[Tensor, "n_ctx-1 d_model"],  # noqa: F722
                    Float[Tensor, "n_ctx d_model"],  # noqa: F722
                ],
            ] = {}
            W_E_v: dict[int, Float[Tensor, "d_vocab d_model"]] = {}  # noqa: F722
            W_pos_v: dict[int, Float[Tensor, "n_ctx d_model"]] = {}  # noqa: F722

            bias: dict[int, Literal["tok", "pos"]] = {}

            for l in range(-1, W_Q.shape[0]):
                add_mean: Optional[Literal["pos_to_tok", "tok_to_pos"]] = (
                    self.add_mean
                    if not isinstance(self.add_mean, dict)
                    else self.add_mean.get(l, self.add_mean[0])
                )
                match add_mean:
                    case None:
                        if self.qtok is not None:
                            sEq[l] = f"E[{self.qtok}]"
                            W_E_q[l] = W_E[self.qtok]
                            if self.qtok % d_vocab == -1 % d_vocab:
                                sEk[l] = f"(E[:-1]-E[-1])"
                                W_E_k[l] = W_E[: self.qtok] - W_E_q[l]
                            elif self.qtok == 0:
                                sEk[l] = f"(E[1:]-E[0])"
                                W_E_k[l] = W_E[self.qtok + 1 :] - W_E_q[l]
                            else:
                                sEk[l] = (
                                    f"(E[:{self.qtok}]+E[{self.qtok+1}:]-E[{self.qtok}])"
                                )
                                W_E_k[l] = (
                                    torch.cat(
                                        [W_E[: self.qtok], W_E[self.qtok + 1 :]], dim=0
                                    )
                                    - W_E_q[l]
                                )
                        else:
                            sEq[l] = f"E"
                            W_E_q[l] = W_E
                            sEk[l] = f"E"
                            W_E_k[l] = W_E
                        if self.qpos is not None:
                            sPq[l] = f"P[{self.qpos}]"
                            W_pos_q[l] = W_pos[self.qpos]
                            if self.qpos % n_ctx == -1 % n_ctx:
                                sPk[l] = f"(P[:-1]-P[-1])"
                                W_pos_k[l] = W_pos[: self.qpos] - W_pos_q[l]
                            elif self.qpos == 0:
                                sPk[l] = f"(P[1:]-P[0])"
                                W_pos_k[l] = W_pos[self.qpos + 1 :] - W_pos_q[l]
                            else:
                                sPk[l] = (
                                    f"(P[:{self.qpos}]+P[{self.qpos+1}:]-P[{self.qpos}])"
                                )
                                W_pos_k[l] = (
                                    torch.cat(
                                        [W_pos[: self.qpos], W_pos[self.qpos + 1 :]],
                                        dim=0,
                                    )
                                    - W_pos_q[l]
                                )
                        else:
                            W_pos_q[l] = W_pos
                            sPq[l] = f"P"
                            W_pos_k[l] = W_pos
                            sPk[l] = f"P"
                    case "pos_to_tok":
                        if self.qtok is not None:
                            sEq[l] = f"E[{self.qtok}]"
                            W_E_q[l] = W_E[self.qtok]
                            if self.qtok % d_vocab == -1 % d_vocab:
                                sEk[l] = f"(E[:-1]-E[-1])"
                                W_E_k[l] = W_E[: self.qtok] - W_E_q[l]
                            elif self.qtok == 0:
                                sEk[l] = f"(E[1:]-E[0])"
                                W_E_k[l] = W_E[self.qtok + 1 :] - W_E_q[l]
                            else:
                                sEk[l] = (
                                    f"(E[:{self.qtok}]+E[{self.qtok+1}:]-E[{self.qtok}])"
                                )
                                W_E_k[l] = (
                                    torch.cat(
                                        [W_E[: self.qtok], W_E[self.qtok + 1 :]], dim=0
                                    )
                                    - W_E_q[l]
                                )
                        else:
                            sEq[l] = f"E"
                            W_E_q[l] = W_E
                            sEk[l] = f"E"
                            W_E_k[l] = W_E
                        if self.qpos is not None:
                            sPq[l] = f"P[{self.qpos}]"
                            W_pos_q[l] = W_pos[self.qpos]
                            if self.qpos % n_ctx == -1 % n_ctx:
                                sEk[l] = f"({sEk[l]}+𝔼(P[:-1]-P[-1]))"
                                sPk[l] = f"(P[:-1]-𝔼P[:-1])"
                                W_pos_k[l] = W_pos[: self.qpos] - W_pos_q[l]
                            elif self.qpos == 0:
                                sEk[l] = f"({sEk[l]}+𝔼(P[1:]-P[0]))"
                                sPk[l] = f"(P[1:]-𝔼P[1:])"
                                W_pos_k[l] = W_pos[self.qpos + 1 :] - W_pos_q[l]
                            else:
                                sEk[l] = (
                                    f"({sEk[l]}+𝔼(P[:{self.qpos}]+P[{self.qpos+1}:]-P[{self.qpos}]))"
                                )
                                sPk[l] = (
                                    f"(P[:{self.qpos}]+P[{self.qpos+1}:]-𝔼(P[:{self.qpos}]+P[{self.qpos+1}:]))"
                                )
                                W_pos_k[l] = (
                                    torch.cat(
                                        [W_pos[: self.qpos], W_pos[self.qpos + 1 :]],
                                        dim=0,
                                    )
                                    - W_pos_q[l]
                                )
                            W_E_q[l] = W_E_q[l] + W_pos_q[l]
                            W_pos_q[l] = W_pos_q[l] - W_pos_q[l]
                            W_pos_k_avg = W_pos_k[l].mean(dim=0)
                            W_E_k[l] = W_E_k[l] + W_pos_k_avg
                            W_pos_k[l] = W_pos_k[l] - W_pos_k_avg
                            sEq[l] = f"({sEq[l]}+{sPq[l]})"
                            sPq[l] = f"0"
                        else:
                            W_pos_q[l] = W_pos
                            sPq[l] = f"P"
                            W_pos_k[l] = W_pos
                            sPk[l] = f"P"
                            W_pos_k_avg = W_pos_k[l].mean(dim=0)
                            W_pos_q_avg = W_pos_q[l].mean(dim=0)
                            W_E_q[l] = W_E_q[l] + W_pos_q_avg
                            W_pos_q[l] = W_pos_q[l] - W_pos_q_avg
                            W_E_k[l] = W_E_k[l] + W_pos_k_avg
                            W_pos_k[l] = W_pos_k[l] - W_pos_k_avg
                            sEq[l] = f"({sEq[l]}+{str_mean(sPq[l])})"
                            sPq[l] = f"({sPq[l]}-{str_mean(sPq[l])})"
                            sEk[l] = f"({sEk[l]}+{str_mean(sPk[l])})"
                            sPk[l] = f"({sPk[l]}-{str_mean(sPk[l])})"
                    case "tok_to_pos":
                        if self.qpos is not None:
                            sPq[l] = f"P[{self.qpos}]"
                            W_pos_q[l] = W_pos[self.qpos]
                            if self.qpos % n_ctx == -1 % n_ctx:
                                sPk[l] = f"(P[:-1]-P[-1])"
                                W_pos_k[l] = W_pos[: self.qpos] - W_pos_q[l]
                            elif self.qpos == 0:
                                sPk[l] = f"(P[1:]-P[0])"
                                W_pos_k[l] = W_pos[self.qpos + 1 :] - W_pos_q[l]
                            else:
                                sPk[l] = (
                                    f"(P[:{self.qpos}]+P[{self.qpos+1}:]-P[{self.qpos}])"
                                )
                                W_pos_k[l] = (
                                    torch.cat(
                                        [W_pos[: self.qpos], W_pos[self.qpos + 1 :]],
                                        dim=0,
                                    )
                                    - W_pos_q[l]
                                )
                        else:
                            sPq[l] = f"P"
                            W_pos_q[l] = W_pos
                            sPk[l] = f"P"
                            W_pos_k[l] = W_pos
                        if self.qtok is not None:
                            sEq[l] = f"E[{self.qtok}]"
                            W_E_q[l] = W_E[self.qtok]
                            if self.qtok % d_vocab == -1 % d_vocab:
                                sEk[l] = f"({sEk[l]}+𝔼(E[:-1]-E[-1]))"
                                sEk[l] = f"(E[:-1]-𝔼E[:-1])"
                                W_E_k[l] = W_E[: self.qtok] - W_E_q[l]
                            elif self.qtok == 0:
                                sEk[l] = f"({sEk[l]}+𝔼(E[1:]-E[0]))"
                                sEk[l] = f"(E[1:]-𝔼E[1:])"
                                W_E_k[l] = W_E[self.qtok + 1 :] - W_E_q[l]
                            else:
                                sEk[l] = (
                                    f"({sEk[l]}+𝔼(E[:{self.qtok}]+E[{self.qtok+1}:]-E[{self.qtok}]))"
                                )
                                sEk[l] = (
                                    f"(E[:{self.qtok}]+E[{self.qtok+1}:]-𝔼(E[:{self.qtok}]+E[{self.qtok+1}:]))"
                                )
                                W_E_k[l] = (
                                    torch.cat(
                                        [W_E[: self.qtok], W_E[self.qtok + 1 :]], dim=0
                                    )
                                    - W_E_q[l]
                                )
                            W_pos_q[l] = W_pos_q[l] + W_E_q[l]
                            W_E_q[l] = W_E_q[l] - W_E_q[l]
                            W_E_k_avg = W_E_k[l].mean(dim=0)
                            W_pos_k[l] = W_pos_k[l] + W_E_k_avg
                            W_E_k[l] = W_E_k[l] - W_E_k_avg
                            sPq[l] = f"({sPq[l]}+{sEq[l]})"
                            sEq[l] = f"0"
                        else:
                            W_E_q[l] = W_E
                            sEq[l] = f"E"
                            W_E_k[l] = W_E
                            sEk[l] = f"E"
                            W_E_k_avg = W_E_k[l].mean(dim=0)
                            W_E_q_avg = W_E_q[l].mean(dim=0)
                            W_pos_q[l] = W_pos_q[l] + W_E_q_avg
                            W_E_q[l] = W_E_q[l] - W_E_q_avg
                            W_pos_k[l] = W_pos_k[l] + W_E_k_avg
                            W_E_k[l] = W_E_k[l] - W_E_k_avg
                            sPq[l] = f"({sPq[l]}+{str_mean(sEq[l])})"
                            sEq[l] = f"({sEq[l]}-{str_mean(sEq[l])})"
                            sPk[l] = f"({sPk[l]}+{str_mean(sEk[l])})"
                            sEk[l] = f"({sEk[l]}-{str_mean(sEk[l])})"

                W_E_v[l] = W_E
                W_pos_v[l] = W_pos
                sEv[l] = f"E"
                sPv[l] = f"P"
                match add_mean:
                    case "pos_to_tok":
                        bias[l] = "tok"
                        W_E_v[l] = W_E_v[l] + W_pos_v[l].mean(dim=0)
                        W_pos_v[l] = W_pos_v[l] - W_pos_v[l].mean(dim=0)
                        sEv[l] = f"({sEv[l]}+{str_mean(sPv[l])})"
                        sPv[l] = f"({sPv[l]}-{str_mean(sPv[l])})"
                    case "tok_to_pos":
                        bias[l] = "pos"
                        W_pos_v[l] = W_pos_v[l] + W_E_v[l].mean(dim=0)
                        W_E_v[l] = W_E_v[l] - W_E_v[l].mean(dim=0)
                        sPv[l] = f"({sPv[l]}+{str_mean(sEv[l])})"
                        sEv[l] = f"({sEv[l]}-{str_mean(sEv[l])})"
                    case None:
                        bias[l] = "pos"
                        pass
                sPk[l] = f"{sPk[l]}ᵀ"
                sEk[l] = f"{sEk[l]}ᵀ"

            def apply_U(
                x: Float[Tensor, "... d_model"],  # noqa: F722
                bias: bool = True,
            ) -> Float[Tensor, "... d_vocab_out"]:  # noqa: F722
                return x @ W_U + (b_U if bias else 0)

            def apply_VO(
                x: Float[Tensor, "... a d_model"],  # noqa: F722
                l: int,
                h: int,
                bias: bool = True,
            ) -> Float[Tensor, "... a d_model"]:  # noqa: F722
                return (x @ W_V[l, h, :, :] + b_V[l, h, None, :]) @ W_O[l, h, :, :] + (
                    b_O[l, None, None, :] if bias else 0
                )

            def apply_Q(
                x: Float[Tensor, "... a d_model"],  # noqa: F722
                l: int,
                h: int,
                bias: bool = True,
            ) -> Float[Tensor, "... a d_head"]:  # noqa: F722
                return x @ W_Q[l, h, :, :] + (b_Q[l, h, None, :] if bias else 0)

            def apply_KT(
                x: Float[Tensor, "... a d_model"],  # noqa: F722
                l: int,
                h: int,
                bias: bool = True,
            ) -> Float[Tensor, "... d_head a"]:  # noqa: F722
                return (
                    x @ W_K[l, h, :, :] + (b_K[l, h, None, :] if bias else 0)
                ).transpose(-1, -2)

            if self.EU:
                yield f"{sEq[-1]}U", apply_U(W_E_q[-1], bias=False)
            if self.PU and (sPq[-1] != "0" or self.log_zeros):
                yield f"{sPq[-1]}U", apply_U(W_pos_q[-1], bias=False)

            for l in range(W_Q.shape[0]):
                for h in range(W_Q.shape[1]):
                    for (
                        (
                            qx,
                            qx_direct,
                            qsx,
                            qsx_short,
                            qsx_direct,
                            qsx_direct_short,
                            qskip_composition,
                            qbias,
                        ),
                        (
                            kx,
                            kx_direct,
                            ksx,
                            ksx_short,
                            ksx_direct,
                            ksx_direct_short,
                            kskip_composition,
                            kbias,
                        ),
                        test,
                        nanify_above_diagonal_if_query_direct,
                    ) in (
                        (
                            (
                                W_E_v[l],
                                W_E_q[l],
                                sEv[l],
                                "E",
                                sEq[l],
                                "E",
                                False,
                                "tok",
                            ),
                            (
                                W_E_v[l],
                                W_E_k[l],
                                sEv[l],
                                "E",
                                sEk[l],
                                "E",
                                False,
                                "tok",
                            ),
                            self.EQKE,
                            False,
                        ),
                        (
                            (
                                W_E_v[l],
                                W_E_q[l],
                                sEv[l],
                                "E",
                                sEq[l],
                                "E",
                                False,
                                "tok",
                            ),
                            (
                                W_pos_v[l],
                                W_pos_k[l],
                                sPv[l],
                                "P",
                                sPk[l],
                                "P",
                                self.shortformer,
                                "pos",
                            ),
                            self.EQKP,
                            False,
                        ),
                        (
                            (
                                W_pos_v[l],
                                W_pos_q[l],
                                sPv[l],
                                "P",
                                sPq[l],
                                "P",
                                self.shortformer,
                                "pos",
                            ),
                            (
                                W_E_v[l],
                                W_E_k[l],
                                sEv[l],
                                "E",
                                sEk[l],
                                "E",
                                False,
                                "tok",
                            ),
                            self.PQKE,
                            False,
                        ),
                        (
                            (
                                W_pos_v[l],
                                W_pos_q[l],
                                sPv[l],
                                "P",
                                sPq[l],
                                "P",
                                self.shortformer,
                                "pos",
                            ),
                            (
                                W_pos_v[l],
                                W_pos_k[l],
                                sPv[l],
                                "P",
                                sPk[l],
                                "P",
                                self.shortformer,
                                "pos",
                            ),
                            self.PQKP,
                            self.nanify_causal_attn
                            and model.cfg.attention_dir == "causal",
                        ),
                    ):
                        if test:
                            for (
                                sq,
                                sq_short,
                                lh_q,
                                v_q,
                                is_direct_q,
                            ) in ModelMatrixLoggingOptions.compute_paths(
                                (
                                    lambda x, l, h: apply_VO(
                                        x, l, h, bias=bias[l] == qbias
                                    )
                                ),
                                model.cfg.n_heads,
                                x=qx,
                                x_direct=qx_direct,
                                sx=qsx,
                                sx_short=qsx_short,
                                sx_direct=qsx_direct,
                                sx_direct_short=qsx_direct_short,
                                l=l,
                                reverse_strs=False,
                                skip_composition=qskip_composition,
                            ):
                                for (
                                    sk,
                                    sk_short,
                                    lh_k,
                                    v_k,
                                    is_direct_k,
                                ) in ModelMatrixLoggingOptions.compute_paths(
                                    (
                                        lambda x, l, h: apply_VO(
                                            x, l, h, bias=bias[l] == kbias
                                        )
                                    ),
                                    model.cfg.n_heads,
                                    x=kx,
                                    x_direct=kx_direct,
                                    sx=f"{ksx}ᵀ",
                                    sx_short=f"{ksx_short}",
                                    sx_direct=ksx_direct,
                                    sx_direct_short=ksx_direct_short,
                                    l=l,
                                    reverse_strs=True,
                                    skip_composition=kskip_composition,
                                ):
                                    if sq != "0" or self.log_zeros:
                                        matrix = apply_Q(
                                            v_q, l, h, bias=bias[l] == qbias
                                        ) @ apply_KT(v_k, l, h, bias=bias[l] == kbias)
                                        if (
                                            nanify_above_diagonal_if_query_direct
                                            and is_direct_q
                                            and len(matrix.shape) >= 2
                                        ):
                                            # set everything above the main diagonal to NaN
                                            rows, cols = torch.triu_indices(
                                                *matrix.shape[-2:], offset=1
                                            )
                                            matrix[..., rows, cols] = float("nan")
                                        yield (
                                            f"{f'{sq_short}QK{sk_short}<br>' if self.include_short_title else ''}{sq}QKᵀ{sk}<br>.{lh_q}l{l}h{h}{lh_k}",
                                            matrix,
                                        )
                    if self.EVOU:
                        for (
                            sv,
                            sv_short,
                            lh_v,
                            v,
                            _,
                        ) in ModelMatrixLoggingOptions.compute_paths(
                            (lambda x, l, h: apply_VO(x, l, h, bias=bias[l] == "tok")),
                            model.cfg.n_heads,
                            x=W_E_v[l],
                            x_direct=W_E_v[l],
                            sx=sEv[l],
                            sx_short="E",
                            sx_direct=sEv[l],
                            sx_direct_short="E",
                            sx_direct=sEv[l],
                            l=l,
                            reverse_strs=False,
                        ):
                            yield (
                                f"{f'{sv_short}VOU<br>' if self.include_short_title else ''}{sv}VOU<br>.{lh_v}l{l}h{h}",
                                apply_U(
                                    apply_VO(v, l, h, bias=bias[l] == "tok"),
                                    bias=bias[l] == "tok",
                                ),
                            )
                    if self.PVOU:
                        for (
                            sv,
                            sv_short,
                            lh_v,
                            v,
                            _,
                        ) in ModelMatrixLoggingOptions.compute_paths(
                            (lambda x, l, h: apply_VO(x, l, h, bias=bias[l] == "pos")),
                            model.cfg.n_heads,
                            x=W_pos_v[l],
                            x_direct=W_pos_v[l],
                            sx=sPv[l],
                            sx_short="P",
                            sx_direct=sPv[l],
                            sx_direct_short="P",
                            l=l,
                            reverse_strs=False,
                            skip_composition=self.shortformer,
                        ):
                            yield (
                                f"{f'{sv_short}VOU<br>' if self.include_short_title else ''}{sv}VOU<br>.{lh_v}l{l}h{h}",
                                apply_U(
                                    apply_VO(v, l, h, bias=bias[l] == "pos"),
                                    bias=bias[l] == "pos",
                                ),
                            )

    @torch.no_grad()
    def log_matrices(
        self,
        logger: Run,
        model: HookedTransformer,
        *,
        unsafe: bool = False,
        **kwargs,
    ):
        matrices = dict(self.matrices_to_log(model, unsafe=unsafe))
        if self.use_subplots:
            OVs = tuple(name for name, _ in matrices.items() if "U" in name)
            QKs = tuple(name for name, _ in matrices.items() if "U" not in name)
            lh_s = set(val for name in QKs for val in re.findall(r"l\d+h\d+", name))
            QKs_by_layer = tuple(
                tuple(name for name in QKs if lh in name) for lh in lh_s
            )
            figs = (
                {
                    self.superplot_title: plot_tensors(
                        matrices.items(),
                        title=self.superplot_title + self._superplot_title_extra,
                        plot_1D_kind=self.plot_1D_kind,
                        groups=(
                            (
                                {
                                    OVs: dict(colorscale="Picnic_r", zmid=0),
                                }
                                | {
                                    QKss: dict(colorscale="Plasma")
                                    for QKss in QKs_by_layer
                                }
                            )
                            if self.group_colorbars
                            else None
                        ),
                    )
                }
                if matrices
                else {}
            )
        else:
            figs = {
                name: plot_tensors(
                    [(name, matrix)],
                    plot_1D_kind=self.plot_1D_kind,
                    title=name + self._superplot_title_extra,
                )
                for name, matrix in matrices.items()
            }
        logger.log(
            {encode_4_byte_unicode(k): v for k, v in figs.items()},
            commit=False,
            **kwargs,
        )
