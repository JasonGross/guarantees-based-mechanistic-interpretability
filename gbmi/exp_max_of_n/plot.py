from typing import Optional, Tuple, Literal, Union
import re
from functools import partial, reduce
import numpy as np
import torch
from torch import Tensor
import math
from jaxtyping import Float, Integer
import scipy.stats as stats
from transformer_lens import HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from gbmi.utils import shuffle_tensor
from gbmi.exp_max_of_n.analysis import (
    find_second_singular_contributions,
    find_size_and_query_direction,
)
from gbmi.analysis_tools.plot import (
    Colorscale,
    weighted_histogram,
    colorscale_to_cmap,
    imshow,
    line,
    scatter,
)
from gbmi.utils.images import trim_plotly_figure
from gbmi.analysis_tools.utils import pm_round, pm_mean_std, data_summary
from gbmi.analysis_tools.plot import hist
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
import gbmi.exp_max_of_n.verification.quadratic as quadratic
from gbmi.exp_max_of_n.analysis import find_size_and_query_direction
from gbmi.verification_tools.l1h1 import all_EQKE, all_EVOU, all_PVOU


@torch.no_grad()
def compute_QK(
    model: HookedTransformer,
    includes_eos: Optional[bool] = None,
    with_attn_scale: bool = True,
) -> dict:
    W_E, W_pos, W_Q, W_K = (
        model.W_E.to("cpu"),
        model.W_pos.to("cpu"),
        model.W_Q.to("cpu"),
        model.W_K.to("cpu"),
    )
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out

    attn_scale = 1.0
    sattn_scale = {"html": "", "latex": ""}
    if with_attn_scale and model.cfg.use_attn_scale:
        attn_scale = model.blocks[0].attn.attn_scale
        assert attn_scale == np.sqrt(
            model.cfg.d_head
        ), f"attn_scale: {attn_scale}, d_head: {model.cfg.d_head}"
        sattn_scale = {
            "html": " / √d<sub>head</sub>",  # <sup>-1</sup>",
            "latex": r" / \sqrt{d_{\mathrm{head}}}",  # ^{-1}}",
        }

    strings = {
        "html": (
            "<br>",
            "",
            "W<sub>E</sub>",
            "W<sub>pos</sub>",
            "W<sub>Q</sub>",
            "W<sub>K</sub>",
            "<sup>T</sup>",
            "𝔼",
            "<sub>dim=0</sub>",
            "<sub>p</sub>",
            "QK",
        ),
        "latex": (
            "\n",
            "$",
            "W_E",
            r"W_{\mathrm{pos}}",
            "W_Q",
            "W_K",
            "^T",
            r"\mathbb{E}",
            r"_{\mathrm{dim}=0}",
            "_p",
            r"\mathrm{QK}",
        ),
    }

    if includes_eos:
        QK = (
            (W_E[-1] + W_pos[-1])
            @ W_Q[0, 0]
            @ W_K[0, 0].T
            @ (W_E[:-1] + W_pos[:-1].mean(dim=0, keepdim=True)).T
        )
        QK_last = (
            (W_E[-1] + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E[-1] + W_pos[-1])
        )
        return {
            "data": (QK - QK_last).numpy() / attn_scale,
            "title": {
                key: f"Attention Score{nl}QK[p] := {smath}({sWe}[-1] + {sWpos}[-1]){sWq}{sWk}{sT}({sWe} + {sWpos}[p]){sT}{sattn_scale[key]}{smath}{nl}{smath}{sE}{s_dim0}({sQK}[:-1,:-1]) - {sQK}[-1, -1]{smath}"
                for key, (
                    nl,
                    smath,
                    sWe,
                    sWpos,
                    sWq,
                    sWk,
                    sT,
                    sE,
                    s_dim0,
                    s_p,
                    sQK,
                ) in strings.items()
            },
            "xaxis": "input token",
            "yaxis": "attention score pre-softmax",
        }
    else:
        QK = (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)).T
        return {
            "data": QK.numpy() / attn_scale,
            "title": {
                key: f"Attention Score{nl}EQKE := {smath}({sWe} + {sWpos}[-1]){sWq}{sWk}{sT}({sWe} + {sE}{s_p}{sWpos}[p]){sT}{sattn_scale[key]}{smath}"
                for key, (
                    nl,
                    smath,
                    sWe,
                    sWpos,
                    sWq,
                    sWk,
                    sT,
                    sE,
                    s_dim0,
                    s_p,
                    sQK,
                ) in strings.items()
            },
            "xaxis": "key token",
            "yaxis": "query token",
        }


@torch.no_grad()
def compute_l2_norm(model: HookedTransformer) -> float:
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.norm(param, 2).item() ** 2
    return math.sqrt(l2_norm)


@torch.no_grad()
def compute_OV(
    model: HookedTransformer, centered: bool = True, includes_eos: Optional[bool] = None
) -> dict:
    W_E, W_pos, W_V, W_O, W_U = (
        model.W_E.to("cpu"),
        model.W_pos.to("cpu"),
        model.W_V.to("cpu"),
        model.W_O.to("cpu"),
        model.W_U.to("cpu"),
    )
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    if includes_eos:
        OV = (W_E[:-1] + W_pos[:-1].mean(dim=0)) @ W_V[0, 0] @ W_O[0, 0] @ W_U
        W_E_pos_suffix = "[:-1]"
    else:
        OV = (W_E + W_pos.mean(dim=0)) @ W_V[0, 0] @ W_O[0, 0] @ W_U
        W_E_pos_suffix = ""
    result: dict = {"xaxis": "output logit token", "yaxis": "input token"}
    strings = {
        "html": (
            "<br>",
            "",
            "W<sub>E</sub>",
            "W<sub>U</sub>",
            "W<sub>pos</sub>",
            "W<sub>O</sub>",
            "W<sub>V</sub>",
            "<sup>T</sup>",
            "𝔼",
            "<sub>dim=0</sub>",
            "<sub>p</sub>",
            "OV",
            ".diag()",
            "None",
        ),
        "latex": (
            "\n",
            "$",
            "W_E",
            "W_U",
            r"W_{\mathrm{pos}}",
            "W_O",
            "W_V",
            "^T",
            r"\mathbb{E}",
            r"_{\mathrm{dim}=0}",
            "_p",
            r"\mathrm{OV}",
            r"\mathrm{.diag}()",
            r"\mathrm{None}",
        ),
    }

    if not centered:
        result.update(
            {
                "data": OV.numpy(),
                "title": {
                    key: f"Attention Computation: {smath}({sWe}{W_E_pos_suffix} + {sE}{s_p}{sWpos}{W_E_pos_suffix}[p]){sWv}{sWo}{sWu}{smath}"
                    for key, (
                        nl,
                        smath,
                        sWe,
                        sWu,
                        sWpos,
                        sWo,
                        sWv,
                        sT,
                        sE,
                        s_dim0,
                        s_p,
                        sOV,
                        sdiag,
                        sNone,
                    ) in strings.items()
                },
            }
        )
        return result
    result.update(
        {
            "data": (OV - OV.diag()[:, None]).numpy(),
            "title": {
                key: f"Attention Computation (centered){nl}{smath}{sOV} := ({sWe}{W_E_pos_suffix} + {sE}{s_p}{sWpos}{W_E_pos_suffix}[p]){sWv}{sWo}{sWu}{smath}{nl}{smath}{sOV} - {sOV}{sdiag}[:, {sNone}]{smath}"
                for key, (
                    nl,
                    smath,
                    sWe,
                    sWu,
                    sWpos,
                    sWo,
                    sWv,
                    sT,
                    sE,
                    s_dim0,
                    s_p,
                    sOV,
                    sdiag,
                    sNone,
                ) in strings.items()
            },
        }
    )
    return result


@torch.no_grad()
def compute_QK_by_position(
    model: HookedTransformer,
    includes_eos: Optional[bool] = None,
    with_attn_scale: bool = True,
) -> dict:
    W_E, W_pos, W_Q, W_K = (
        model.W_E.to("cpu"),
        model.W_pos.to("cpu"),
        model.W_Q.to("cpu"),
        model.W_K.to("cpu"),
    )
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    attn_scale = 1.0
    sattn_scale = {"html": "", "latex": ""}
    if with_attn_scale and model.cfg.use_attn_scale:
        attn_scale = model.blocks[0].attn.attn_scale
        assert attn_scale == np.sqrt(
            model.cfg.d_head
        ), f"attn_scale: {attn_scale}, d_head: {model.cfg.d_head}"
        sattn_scale = {
            "html": " / √d<sub>head</sub>",  # <sup>-1</sup>",
            "latex": r" / \sqrt{d_{\mathrm{head}}}",  # ^{-1}}",
        }
    strings = {
        "html": (
            "<br>",
            "",
            "W<sub>E</sub>",
            "W<sub>pos</sub>",
            "W<sub>Q</sub>",
            "W<sub>K</sub>",
            "<sup>T</sup>",
            "𝔼",
            "<sub>dim=0</sub>",
            "<sub>p</sub>",
            "QK",
        ),
        "latex": (
            "\n",
            "$",
            "W_E",
            r"W_{\mathrm{pos}}",
            "W_Q",
            "W_K",
            "^T",
            r"\mathbb{E}",
            r"_{\mathrm{dim}=0}",
            "_p",
            r"\mathrm{QK}",
        ),
    }
    if includes_eos:
        QK = (
            (W_E[-1] + W_pos[-1])
            @ W_Q[0, 0]
            @ W_K[0, 0].T
            @ (W_pos[:-1] - W_pos[:-1].mean(dim=0)).T
        ) / attn_scale
        return {
            "data": {"QK": QK.numpy()},
            "title": {
                key: f"Positional Contribution to Attention Score{nl}{smath}({sWe}[-1] + {sWpos}[-1]){sWq}{sWk}{sT}({sWpos}[:-1] - {sE}{s_dim0}{sWpos}[:-1]){sT}{sattn_scale[key]}{smath}"
                for key, (
                    nl,
                    smath,
                    sWe,
                    sWpos,
                    sWq,
                    sWk,
                    sT,
                    sE,
                    s_dim0,
                    s_p,
                    sQK,
                ) in strings.items()
            },
            "xaxis": "position",
            "yaxis": "attention score pre-softmax",
        }
    else:
        QK = (
            (W_E + W_pos[-1])
            @ W_Q[0, 0]
            @ W_K[0, 0].T
            @ (W_pos - W_pos.mean(dim=0)).T
            / attn_scale
        )
        return {
            "data": {"QK": QK.numpy()},
            "title": {
                key: f"Positional Contribution to Attention Score{nl}{smath}({sWe} + {sWpos}[-1]){sWq}{sWk}{sT}({sWpos} - {sE}{s_p}{sWpos}[p]){sT}{sattn_scale[key]}{smath}"
                for key, (
                    nl,
                    smath,
                    sWe,
                    sWpos,
                    sWq,
                    sWk,
                    sT,
                    sE,
                    s_dim0,
                    s_p,
                    sQK,
                ) in strings.items()
            },
            "xaxis": "key position",
            "yaxis": "query token",
        }


@torch.no_grad()
def compute_irrelevant(
    model: HookedTransformer,
    include_equals_OV: bool = False,
    includes_eos: Optional[bool] = None,
    title_kind: Literal["html", "latex"] = "html",
    pvou_as_2d: bool = True,
) -> dict:
    W_E, W_pos, W_V, W_O, W_U = (
        model.W_E.to("cpu"),
        model.W_pos.to("cpu"),
        model.W_V.to("cpu"),
        model.W_O.to("cpu"),
        model.W_U.to("cpu"),
    )
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    W_E_q = W_E[-1] if includes_eos else W_E
    W_pos_k = W_pos[:-1] if includes_eos else W_pos
    W_E_k = W_E[:-1] if includes_eos else W_E
    W_E_q_index = "[-1]" if includes_eos else ""
    W_pos_k_index = "[:-1]" if includes_eos else ""
    smath = "" if title_kind == "html" else "$"
    sWE = "W<sub>E</sub>" if title_kind == "html" else r"W_E"
    sWpos = "W<sub>pos</sub>" if title_kind == "html" else r"W_{\mathrm{pos}}"
    sWV = "W<sub>V</sub>" if title_kind == "html" else r"W_V"
    sWO = "W<sub>O</sub>" if title_kind == "html" else r"W_O"
    sWU = "W<sub>U</sub>" if title_kind == "html" else r"W_U"
    sE = "𝔼" if title_kind == "html" else r"\mathbb{E}"
    s_dim0 = "<sub>dim=0</sub>" if title_kind == "html" else r"_{\mathrm{dim}=0}"
    s_p = "<sub>p</sub>" if title_kind == "html" else r"_p"
    EU_key = f"{smath}({sWE}{W_E_q_index}+{sWpos}[-1]){sWU}{smath}"
    data = {
        EU_key: (((W_E_q + W_pos[-1]) @ W_U).numpy()),
    }
    if include_equals_OV:
        data.update(
            {
                f"{smath}({sWE}{W_E_q_index}+{sWpos}[-1]){sWV}{sWO}{sWU}{smath}": (
                    (W_E_q + W_pos[-1]) @ W_V[0, 0] @ W_O[0, 0] @ W_U
                ),
            }
        )
    PVOU_key = (
        f"{smath}({sWpos} - {sE}{s_p}{sWpos}{W_pos_k_index}[p]){sWV}{sWO}{sWU}{smath}"
    )
    if pvou_as_2d:
        data.update(
            {
                PVOU_key: (W_pos_k - W_pos_k.mean(dim=0))
                @ W_V[0, 0, :, :]
                @ W_O[0, 0, :, :]
                @ W_U
            }
        )
    data.update(
        {
            f"{smath}({sWpos}[{i}] - {sE}{s_p}{sWpos}{W_pos_k_index}[p]){sWV}{sWO}{sWU}{smath}": (
                (
                    (W_pos_k[i] - W_pos_k.mean(dim=0))
                    @ W_V[0, 0, :, :]
                    @ W_O[0, 0, :, :]
                    @ W_U
                ).numpy()
            )
            for i in range(W_pos_k.shape[0])
        }
    )

    return {
        "data": data,
        "title": "Irrelevant Contributions to logits" if include_equals_OV else "PVOU",
        "xaxis": "output logit token",
        "yaxis": {
            EU_key: "input token",
            1: "output logit value",
            PVOU_key: "input position",
        },
    }


@torch.no_grad()
def display_basic_interpretation(
    model: HookedTransformer,
    *,
    include_uncentered: bool = False,
    legend_at_bottom: bool = False,
    include_equals_OV: bool = False,
    includes_eos: Optional[bool] = None,
    OV_colorscale: Colorscale = "Picnic_r",
    QK_colorscale: Colorscale = "Plasma",  # "Sunsetdark_r"
    QK_SVD_colorscale: Colorscale = "Picnic_r",
    tok_dtick: Optional[int | float] = None,
    pos_dtick: Optional[int | float] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
) -> dict[str, Union[go.Figure, matplotlib.figure.Figure]]:
    QK_cmap = colorscale_to_cmap(QK_colorscale)
    QK_SVD_cmap = colorscale_to_cmap(QK_SVD_colorscale)
    OV_cmap = colorscale_to_cmap(OV_colorscale)
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    result = {}
    for attn_scale, with_attn_scale in (("", False), ("WithAttnScale", True)):
        QK = compute_QK(
            model, includes_eos=includes_eos, with_attn_scale=with_attn_scale
        )
        title_kind = "html" if plot_with == "plotly" else "latex"
        if includes_eos:
            match plot_with:
                case "plotly":
                    fig_qk = px.line(
                        {"QK": QK["data"]},
                        title=QK["title"]["html"],
                        labels={
                            "index": QK["xaxis"],
                            "variable": "",
                            "value": QK["yaxis"],
                        },
                    )
                    if show:
                        fig_qk.show(renderer=renderer)
                case "matplotlib":
                    fig_qk, ax = plt.subplots()
                    plt.close()
                    ax.plot(QK["data"])
                    ax.set_title(QK["title"]["latex"])
                    ax.set_xlabel(QK["xaxis"])
                    ax.set_ylabel(QK["yaxis"])
                    if show:
                        plt.figure(fig_qk)
                        fig_qk.show()
        else:
            fig_qk = imshow(
                QK["data"],
                title=QK["title"][title_kind],
                xaxis=QK["xaxis"],
                yaxis=QK["yaxis"],
                colorscale=QK_colorscale,
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            _, figs = find_size_and_query_direction(
                model,
                plot_heatmaps=True,
                renderer=renderer,
                colorscale=QK_SVD_colorscale,
                plot_with=plot_with,
                dtick=tok_dtick,
                show=show,
            )
            assert figs is not None
            for k, fig in figs.items():
                result[f"EQKE{attn_scale} {k}"] = fig
        result[f"EQKE{attn_scale}"] = fig_qk

    if include_uncentered:
        OV = compute_OV(model, centered=False, includes_eos=includes_eos)
        fig_ov = imshow(
            OV["data"],
            title=OV["title"][title_kind],
            xaxis=OV["xaxis"],
            yaxis=OV["yaxis"],
            colorscale=OV_colorscale,
            dtick_x=tok_dtick,
            dtick_y=tok_dtick,
            plot_with=plot_with,
            renderer=renderer,
            show=show,
        )
        result["EVOU"] = fig_ov
    OV = compute_OV(model, centered=True, includes_eos=includes_eos)
    fig_ov = imshow(
        OV["data"],
        title=OV["title"][title_kind],
        xaxis=OV["xaxis"],
        yaxis=OV["yaxis"],
        colorscale=OV_colorscale,
        dtick_x=tok_dtick,
        dtick_y=tok_dtick,
        plot_with=plot_with,
        renderer=renderer,
        show=show,
    )
    result["EVOU-centered"] = fig_ov

    for attn_scale, with_attn_scale in (("", False), ("WithAttnScale", True)):
        pos_QK = compute_QK_by_position(
            model, includes_eos=includes_eos, with_attn_scale=with_attn_scale
        )
        if includes_eos:
            fig_qk = px.scatter(
                pos_QK["data"],
                title=pos_QK["title"][title_kind],
                labels={
                    "index": pos_QK["xaxis"],
                    "variable": "",
                    "value": pos_QK["yaxis"],
                },
            )
            if show:
                fig_qk.show(renderer=renderer)
        else:
            fig_qk = imshow(
                pos_QK["data"]["QK"],
                title=pos_QK["title"][title_kind],
                colorscale=QK_colorscale,
                plot_with=plot_with,
                xaxis=pos_QK["xaxis"],
                yaxis=pos_QK["yaxis"],
                dtick_x=pos_dtick,
                dtick_y=tok_dtick,
                renderer=renderer,
                show=show,
            )
        result[f"EQKP{attn_scale}"] = fig_qk

    irrelevant = compute_irrelevant(
        model,
        include_equals_OV=include_equals_OV,
        includes_eos=includes_eos,
        title_kind=title_kind,
    )
    for key, data in irrelevant["data"].items():
        if len(data.shape) == 2:
            fig = imshow(
                data,
                title=key,
                colorscale=OV_colorscale,
                xaxis=irrelevant["xaxis"],
                yaxis=irrelevant["yaxis"][key],
                dtick_x=tok_dtick,
                dtick_y=pos_dtick if data.shape[0] == model.cfg.n_ctx else tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            result[f"irrelevant_{key}"] = fig
    if include_equals_OV:
        fig = scatter(
            {k: v for k, v in irrelevant["data"].items() if len(v.shape) == 1},
            title=irrelevant["title"],
            xaxis=irrelevant["xaxis"],
            caxis="",
            yaxis=irrelevant["yaxis"][1],
            legend_at_bottom=legend_at_bottom,
            plot_with=plot_with,
            renderer=renderer,
            show=show,
        )
    else:
        pass
    result["irrelevant"] = fig
    match plot_with:
        case "matplotlib":
            assert isinstance(fig, matplotlib.figure.Figure)
            fig.suptitle("")
        case "plotly":
            assert isinstance(fig, go.Figure)
            fig.update_layout(title="")
    return result


@torch.no_grad()
def EVOU_max_minus_diag_logit_diff(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    num_bins: Optional[int] = None,
) -> Tuple[Float[Tensor, "batch"], Integer[Tensor, "batch"]]:  # noqa: F821
    """
    If duplicate_by_sequence_count is True, bins are weighted according to how many sequences have the given maximum.
    """
    EVOU = all_EVOU(model)
    EVOU = EVOU - EVOU.diag()[:, None]
    # set diagonal to -inf
    EVOU[torch.eye(EVOU.shape[0], dtype=torch.bool)] = float("-inf")
    max_logit_minus_diag = EVOU.max(dim=-1).values
    if duplicate_by_sequence_count:
        n_ctx = model.cfg.n_ctx
        indices = torch.arange(max_logit_minus_diag.size(0))
        duplication_factors = (indices + 1) ** n_ctx - indices**n_ctx
    else:
        duplication_factors = torch.ones_like(max_logit_minus_diag)
    return max_logit_minus_diag, duplication_factors


@torch.no_grad()
def hist_EVOU_max_minus_diag_logit_diff(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    renderer: Optional[str] = None,
    num_bins: Optional[int] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    show: bool = True,
) -> Tuple[
    Union[go.Figure, matplotlib.figure.Figure],
    Tuple[Float[Tensor, "batch"], Integer[Tensor, "batch"]],  # noqa: F821
]:
    """
    If duplicate_by_sequence_count is True, bins are weighted according to how many sequences have the given maximum.
    """
    max_logit_minus_diag, duplication_factors = EVOU_max_minus_diag_logit_diff(
        model,
        duplicate_by_sequence_count=duplicate_by_sequence_count,
        num_bins=num_bins,
    )
    mean = np.average(max_logit_minus_diag.numpy(), weights=duplication_factors.numpy())
    std = np.average(
        (max_logit_minus_diag - mean).numpy() ** 2, weights=duplication_factors.numpy()
    )
    min, max = max_logit_minus_diag.min().item(), max_logit_minus_diag.max().item()
    mid, spread = (min + max) / 2, (max - min) / 2
    title_kind = "html" if plot_with == "plotly" else "latex"
    sEVOU = "EVOU" if title_kind == "html" else r"\mathrm{EVOU}"
    smath = "" if title_kind == "html" else "$"
    sT = "<sup>T</sup>" if title_kind == "html" else "^T"
    nl = "<br>" if title_kind == "html" else "\n"
    sWE = "W<sub>E</sub>" if title_kind == "html" else r"W_E"
    sWV = "W<sub>V</sub>" if title_kind == "html" else r"W_V"
    sWO = "W<sub>O</sub>" if title_kind == "html" else r"W_O"
    sWU = "W<sub>U</sub>" if title_kind == "html" else r"W_U"
    sdiag = ".diag()" if title_kind == "html" else r"\mathrm{.diag}()"
    smax = "max" if title_kind == "html" else r"\max"
    s_i = "<sub>i</sub>" if title_kind == "html" else r"_i"
    xbar = "x̄" if title_kind == "html" else r"\bar{x}"
    sigma = "σ" if title_kind == "html" else r"\sigma"
    pm = "±" if title_kind == "html" else r"\pm"
    sNone = "None" if title_kind == "html" else r"\mathrm{None}"
    sdotmax = ".max" if title_kind == "html" else r"\mathrm{.max}"
    sdim = "dim" if title_kind == "html" else r"\mathrm{dim}"
    shash = "#" if title_kind == "html" else r"\#"
    title = (
        f"{smath}{sEVOU} := {sWE}{sWV}{sWO}{sWU}{smath}"
        f"{'' if not duplicate_by_sequence_count else ' (weighted by sequence count)'}"
        f"{nl}{smath}({sEVOU} - {sEVOU}{sdiag}[:,{sNone}]){sdotmax}({sdim}=-1){smath} (excluding diagonal)"
        f"{nl}{smath}{xbar}{pm}{sigma}{smath}: {smath}{pm_round(mean, std, sep=f' {pm} ')}{smath}; range: {smath}{pm_round(mid, spread,sep=f' {pm} ')}{smath}"
    )
    if not duplicate_by_sequence_count:
        fig = hist(
            max_logit_minus_diag,
            title=title,
            xaxis="logit - diag",
            column_names="",
            variable="",
            renderer=renderer,
            plot_with=plot_with,
            show=show,
        )
    else:
        fig = weighted_histogram(
            max_logit_minus_diag.numpy(),
            duplication_factors.numpy(),
            title=title,
            xaxis="logit - diag",
            yaxis=f"count × {shash} sequences with given max",
            renderer=renderer,
            plot_with=plot_with,
            show=show,
        )
    return fig, (max_logit_minus_diag, duplication_factors)


@torch.no_grad()
def compute_attention_difference_vs_gap(
    model: HookedTransformer,
) -> Tuple[
    Integer[Tensor, "batch"],  # noqa F821
    Integer[Tensor, "batch"],  # noqa F821
    Float[Tensor, "batch"],  # noqa F821
]:
    """
    Returns (i-j, (E+P)QKE[i] - (E+P)QKE[j]), flattened across query
    """
    EQKE = all_EQKE(model)
    n_ctx = model.cfg.n_ctx
    idxs = torch.cartesian_prod(
        torch.arange(EQKE.shape[-1]), torch.arange(EQKE.shape[-1])
    )
    diffs = EQKE[:, idxs[:, 0]] - EQKE[:, idxs[:, 1]]
    sequence_counts = (idxs[:, 0] + 1) ** n_ctx - idxs[:, 0] ** n_ctx
    sequence_counts[idxs[:, 0] <= idxs[:, 1]] = 0
    repeated_sequence_counts = sequence_counts.repeat(EQKE.shape[0], 1)
    # zero repeated_sequence_counts[q, m] where q > m
    # Create a mask where q > m
    mask = torch.tril(repeated_sequence_counts, diagonal=-1).bool()
    repeated_sequence_counts[mask] = 0
    flat_idxs = (idxs[:, 0] - idxs[:, 1]).repeat(EQKE.shape[0], 1).flatten()
    flat_sequence_counts = repeated_sequence_counts.flatten()
    flat_diffs = diffs.flatten()
    flat_diffs /= model.blocks[0].attn.attn_scale
    return flat_sequence_counts, flat_idxs, flat_diffs


def scatter_attention_difference_vs_gap(
    model: HookedTransformer,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
) -> Union[go.Figure, matplotlib.figure.Figure]:
    _, flat_idxs, flat_diffs = compute_attention_difference_vs_gap(model)
    title_kind = {"plotly": "html", "matplotlib": "latex"}[plot_with]
    smath = "" if title_kind == "html" else "$"
    sdhead = "d<sub>head</sub>" if title_kind == "html" else r"d_{\mathrm{head}}"
    spowmhalf = "<sup>-½</sup>" if title_kind == "html" else r"^{-\sfrac{1}{2}}"
    sqWE = "(E+P)" if title_kind == "html" else r"\qWE "
    sbarWE = "E" if title_kind == "html" else r"\barWE "
    sT = "<sup>T</sup>" if title_kind == "html" else r"^T"
    fig = scatter(
        x=flat_idxs,
        y=flat_diffs,
        xaxis=f"{smath}i - j{smath}",
        yaxis=f"{smath}{sdhead}{spowmhalf}({sqWE}QK{sT}{sbarWE}{sT}[i] - {sqWE}QK{sT}{sbarWE}{sT}[j]){smath}",
        plot_with=plot_with,
        renderer=renderer,
        show=show,
    )
    return fig


@torch.no_grad()
def attention_difference_over_gap(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    num_bins: Optional[int] = None,
) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:  # noqa: F821
    """
    If duplicate_by_sequence_count is True, bins are weighted according to how many sequences have the given maximum.
    """
    sequence_counts, flat_idxs, flat_diffs = compute_attention_difference_vs_gap(model)
    flat_diffs /= flat_idxs
    sequence_counts, flat_idxs, flat_diffs = (
        sequence_counts[flat_diffs.isfinite()],
        flat_idxs[flat_diffs.isfinite()],
        flat_diffs[flat_diffs.isfinite()],
    )
    duplication_factors = (
        sequence_counts if duplicate_by_sequence_count else torch.ones_like(flat_diffs)
    )
    return flat_diffs, duplication_factors


@torch.no_grad()
def hist_attention_difference_over_gap(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    num_bins: Optional[int] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
) -> Tuple[
    Union[go.Figure, matplotlib.figure.Figure],
    Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]],  # noqa: F821
]:
    """
    If duplicate_by_sequence_count is True, bins are weighted according to how many sequences have the given maximum.
    """
    flat_diffs, duplication_factors = attention_difference_over_gap(
        model,
        duplicate_by_sequence_count=duplicate_by_sequence_count,
        num_bins=num_bins,
    )
    mean = np.average(flat_diffs.numpy(), weights=duplication_factors.numpy())
    std = np.average(
        (flat_diffs - mean).numpy() ** 2, weights=duplication_factors.numpy()
    )
    title_kind = "html" if plot_with == "plotly" else "latex"
    rm = lambda s: s if title_kind == "html" else r"\mathrm{" + s + "}"
    smath = "" if title_kind == "html" else "$"
    sT = "<sup>T</sup>" if title_kind == "html" else "^T"
    nl = "<br>" if title_kind == "html" else "\n"
    sWE = "W<sub>E</sub>" if title_kind == "html" else r"W_E"
    sWQ = "W<sub>Q</sub>" if title_kind == "html" else r"W_Q"
    sWK = "W<sub>K</sub>" if title_kind == "html" else r"W_K"
    sWpos = "W<sub>pos</sub>" if title_kind == "html" else r"W_{\mathrm{pos}}"
    sdhead = "d<sub>head</sub>" if title_kind == "html" else r"d_{\mathrm{head}}"
    sdiag = ".diag()" if title_kind == "html" else r"\mathrm{.diag}()"
    smax = "max" if title_kind == "html" else r"\max"
    s_i = "<sub>i</sub>" if title_kind == "html" else r"_i"
    xbar = "x̄" if title_kind == "html" else r"\bar{x}"
    sigma = "σ" if title_kind == "html" else r"\sigma"
    pm = "±" if title_kind == "html" else r"\pm"
    sNone = "None" if title_kind == "html" else r"\mathrm{None}"
    sdotmax = ".max" if title_kind == "html" else r"\mathrm{.max}"
    sdim = "dim" if title_kind == "html" else r"\mathrm{dim}"
    spowmhalf = "<sup>-½</sup>" if title_kind == "html" else r"^{-\sfrac{1}{2}}"
    shash = "#" if title_kind == "html" else r"\#"
    title = (
        f"{smath}{rm('EQKE')} := ({sWE} + {sWpos}[-1]){sWQ}{sWK}{sT}{sWE}{sT}{smath}"
        f"{'' if not duplicate_by_sequence_count else ' (weighted by sequence count)'}"
        f"{nl}{smath}{sdhead}{spowmhalf}({rm('EQKE')}[i] - {rm('EQKE')}[j]) / (i - j){smath}"
        f"{nl}{smath}{xbar}{pm}{sigma}{smath}: {smath}{pm_round(mean, std, sep=f' {pm} ')}{smath}"
    )
    xlabel = f"{smath}{sdhead}{spowmhalf}({rm('EQKE')}[i]-{rm('EQKE')}[j])/(i-j){smath}"
    if not duplicate_by_sequence_count:
        fig = hist(
            flat_diffs,
            xaxis=xlabel,
            column_names="",
            variable="",
            yaxis="count",
            title=title,
            plot_with=plot_with,
            renderer=renderer,
            show=show,
        )
    else:
        fig = weighted_histogram(
            flat_diffs.numpy(),
            duplication_factors.numpy(),
            title=title,
            num_bins=num_bins,
            xaxis=xlabel,
            yaxis=f"count × {shash} sequences with given max",
            renderer=renderer,
            plot_with=plot_with,
            show=show,
        )
    return fig, (flat_diffs, duplication_factors)


@torch.no_grad()
def make_better_slides_plots_00(
    model: HookedTransformer,
    OV_colorscale: Colorscale = "Picnic_r",
    QK_colorscale: Colorscale = "Plasma",
    tok_dtick: Optional[int | float] = None,
    pos_dtick: Optional[int | float] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
    do_print: bool = True,
) -> dict[str, go.Figure]:
    W_E, W_pos, W_U, W_V, W_O, W_Q, W_K = (
        model.W_E.cpu(),
        model.W_pos.cpu(),
        model.W_U.cpu(),
        model.W_V[0, 0].cpu(),
        model.W_O[0, 0].cpu(),
        model.W_Q[0, 0].cpu(),
        model.W_K[0, 0].cpu(),
    )
    attn_scale = model.blocks[0].attn.attn_scale
    EPq = W_E + W_pos[-1]
    EPk = W_E + W_pos.mean(dim=0)
    Pk = W_pos - W_pos.mean(dim=0)
    EPU = EPq @ W_U
    EVOU = EPk @ W_V @ W_O @ W_U
    EVOU_centered = EVOU - EVOU.diag()[:, None]
    PVOU = Pk @ W_V @ W_O @ W_U
    EQKE = EPq @ W_Q @ W_K.T @ EPk.T / attn_scale
    EQKP = EPq @ W_Q @ W_K.T @ Pk.T / attn_scale
    OV_zmax = np.max(
        [EVOU.abs().max().item(), PVOU.abs().max().item(), EPU.abs().max().item()]
    )
    QK_zmax = np.max([EQKE.abs().max().item(), EQKP.abs().max().item()])
    results = {}
    for key, zmax, colorscale in (
        ("OV", OV_zmax, OV_colorscale),
        ("QK", QK_zmax, QK_colorscale),
    ):
        match plot_with:
            case "plotly":
                results[f"{key}-colorbar"] = fig = go.Figure(
                    data=go.Heatmap(
                        z=[[0]],
                        colorscale=colorscale,
                        showscale=True,
                        zmin=-zmax,
                        zmax=zmax,
                        zmid=0,
                        colorbar=dict(x=0),
                    )
                )
                fig.add_trace(
                    go.Heatmap(
                        z=[[0]],
                        colorscale="Picnic_r",
                        showscale=False,
                        zmin=-zmax,
                        zmax=zmax,
                        zmid=0,
                    )
                )
                fig.update_layout(
                    width=75,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    xaxis_zeroline=False,
                    yaxis_zeroline=False,
                    xaxis_visible=False,
                    yaxis_visible=False,
                    margin=dict(l=0, r=0, b=0, t=0),
                )
                if show:
                    fig.show(renderer)
            case "matplotlib":
                cmap = colorscale_to_cmap(colorscale)
                results[f"{key}-colorbar"] = fig = plt.figure(figsize=(0.5, 4))
                plt.close()
                norm = matplotlib.colors.Normalize(vmin=-zmax, vmax=zmax)
                cbar = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=fig.gca(),
                    orientation="vertical",
                )
                # cbar = matplotlib.colorbar.ColorbarBase(
                #     plt.gca(), cmap=cmap, norm=norm, orientation="vertical"
                # )
                if show:
                    plt.figure(fig)
                    plt.show()
    to_latex = (
        lambda s: re.sub(r"([a-zA-Z]*)<sub>([^>]*)</sub>", r"$\1_{\2}$", s)
        .replace("position j", "position $j$")
        .replace("key position k", "position $k$")
    )
    maybe_to_latex = to_latex if plot_with == "matplotlib" else (lambda x: x)
    for m, title, colorscale, zmax, labels, dtick_x, dtick_y in (
        (
            EPU,
            "EPU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "query token t<sub>i</sub>"},
            tok_dtick,
            tok_dtick,
        ),
        (
            EVOU,
            "EVOU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "key token t<sub>j</sub>"},
            tok_dtick,
            tok_dtick,
        ),
        (
            PVOU,
            "PVOU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "position j"},
            tok_dtick,
            pos_dtick,
        ),
        (
            EQKE,
            "EQKE",
            QK_colorscale,
            QK_zmax,
            {"x": "key token t<sub>k</sub>", "y": "query token t<sub>q</sub>"},
            tok_dtick,
            tok_dtick,
        ),
        (
            EQKP,
            "EQKP",
            QK_colorscale,
            QK_zmax,
            {"x": "key position k", "y": "query token t<sub>q</sub>"},
            pos_dtick,
            tok_dtick,
        ),
    ):
        key = title
        results[key] = fig = imshow(
            m,
            title=title,
            colorscale=colorscale,
            zmax=zmax,
            zmin=-zmax,
            xaxis=maybe_to_latex(labels["x"]),
            yaxis=maybe_to_latex(labels["y"]),
            renderer=renderer,
            plot_with=plot_with,
            dtick_x=dtick_x,
            dtick_y=dtick_y,
            show=False,
        )
        match plot_with:
            case "plotly":
                assert isinstance(fig, go.Figure), f"fig: {type(fig)}"
                # fig.show(renderer)
                # remove title
                fig.update_layout(title_text="")
                fig.update(layout_coloraxis_showscale=False)
                # crop whitespace
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                trim_plotly_figure(fig)
                if show:
                    fig.show(renderer)
            case "matplotlib":
                assert isinstance(fig, matplotlib.figure.Figure), f"fig: {type(fig)}"
                ax, cbar_ax = fig.axes
                fig.tight_layout()
                if show:
                    plt.figure(fig)
                    plt.show()
                ax.set_title("")
                assert hasattr(cbar_ax, "_colorbar"), cbar_ax
                cbar = cbar_ax._colorbar
                cbar_ax.remove()
                # fig.colorbar(cbar_ax.collections[0], ax=ax, use_gridspec=False).remove()
                for c in ax.get_children():
                    if getattr(c, "colorbar", None) is cbar:
                        if do_print:
                            print(f"!! Manually removing colorbar from {c}")
                        del c.colorbar
                fig.tight_layout()
                if show:
                    plt.figure(fig)
                    plt.show()

    return results


# random resampling of EQKE_err
@torch.no_grad()
def resample_EQKE_err(
    *ms: Tuple[torch.Tensor, Tuple[dict[Literal["html", "latex"], str], str]],
    # QK_colorscale: Colorscale = "Plasma",
    # QK_SVD_colorscale: Colorscale = "Picnic_r",
    seed: int = 1234,
    nsamples: int = 100,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
    include_figures: bool = True,
    do_print: bool = False,
) -> Tuple[dict[str, Union[go.Figure, matplotlib.figure.Figure]], dict[str, float]]:
    results: dict = {}
    results_float = {}
    if include_figures:
        EQKE_err_exact = reduce(torch.matmul, [m for m, s in ms])
        for m, (title, fig_key) in ms:
            m_numpy = m.flatten().numpy()
            edges = np.histogram_bin_edges(m_numpy, bins="auto")
            counts, _ = np.histogram(m_numpy, bins=edges)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            pdf_values = stats.norm.pdf(
                bin_centers, loc=m.mean().item(), scale=m.std().item()
            )
            pdf_scaled = pdf_values * m.numel() * np.diff(edges)
            line_name = r"$\mathcal{N}(%s)$" % pm_round(
                m.mean().item(), m.std().item(), sep=", "
            )
            match plot_with:
                case "plotly":
                    fig = px.histogram(
                        {"": m_numpy},
                        nbins=len(edges) - 1,
                        title=title["html"],
                        labels={"variable": "", "value": "matrix element value"},
                    )
                    # f"𝒩({pm_round(m.mean().item(), m.std().item(), sep=', ')})"
                    fig.add_scatter(
                        x=bin_centers,
                        y=pdf_scaled,
                        mode="lines",
                        name=line_name,
                    )
                    if show:
                        fig.show(renderer)
                case "matplotlib":
                    fig, ax = plt.subplots()
                    plt.close()
                    ax.hist(
                        m_numpy,
                        bins=edges,
                    )
                    ax.plot(
                        bin_centers,
                        pdf_scaled,
                        linestyle="-",
                        color="r",
                        label=line_name,
                    )
                    ax.set_title(title["latex"])
                    ax.set_xlabel("matrix element value")
                    ax.set_ylabel("count")
                    ax.legend()
                    if show:
                        plt.figure(fig)
                        plt.show()
            results[fig_key] = fig
    # what if we randomize the order of all matrices without replacement?
    torch.manual_seed(seed)
    results_float["ResampleEQKEErrSeed"] = seed
    results_float["ResampleEQKEErrNumSamples"] = nsamples
    row_diffs = []
    max_row_diffs = []
    for _ in range(nsamples):
        ms_no_replacement = [shuffle_tensor(m) for m, _ in ms]
        result = reduce(torch.matmul, ms_no_replacement)
        row_diffs.extend(result.max(dim=-1).values - result.min(dim=-1).values)
        max_row_diffs.append(
            (result.max(dim=-1).values - result.min(dim=-1).values).max().item()
        )
    row_diffs = torch.stack(row_diffs)
    results_float |= data_summary(max_row_diffs, prefix="ResampleEQKEErr")
    max_row_diffs = torch.tensor(max_row_diffs)
    if do_print:
        print(f"max row diff (n = {nsamples}): {pm_mean_std(max_row_diffs)}")
    # print(f"row diff: {pm_mean_std(row_diffs)}")
    # sampling from normal
    row_diffs = []
    max_row_diffs = []
    for _ in range(nsamples):
        ms_normal = [torch.randn_like(m) * m.std() + m.mean() for m, _ in ms]
        result = reduce(torch.matmul, ms_normal)
        row_diffs.extend(result.max(dim=-1).values - result.min(dim=-1).values)
        max_row_diffs.append(
            (result.max(dim=-1).values - result.min(dim=-1).values).max().item()
        )
    row_diffs = torch.stack(row_diffs)
    results_float |= data_summary(max_row_diffs, prefix="ResampleNormalEQKEErr")
    if do_print:
        max_row_diffs = torch.tensor(max_row_diffs)
        m_descr = ", ".join(
            f"𝒩({pm_round(m.mean().item(), m.std().item(), sep=', ')})" for m, s in ms
        )
        print(
            f"max row diff (n = {nsamples}, m ~ {m_descr}): {pm_mean_std(max_row_diffs)}"
        )
    # print(f"row diff: {pm_mean_std(row_diffs)}")
    return results, results_float


@torch.no_grad()
def display_EQKE_SVD_analysis(
    model: HookedTransformer,
    *,
    QK_colorscale: Colorscale = "Plasma",
    QK_SVD_colorscale: Colorscale = "Picnic_r",
    tok_dtick: Optional[int | float] = None,
    pos_dtick: Optional[int | float] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
    include_figures: bool = True,
    do_print: bool = False,
) -> Tuple[dict[str, Union[go.Figure, matplotlib.figure.Figure]], dict[str, float]]:
    title_kind = "html" if plot_with == "plotly" else "latex"
    results = {}
    results_float = {}
    (
        size_direction,
        query_direction,
        size_query_singular_value,
    ), _ = find_size_and_query_direction(model)
    (second_key_direction, second_key_singular_value), (
        second_query_direction,
        second_query_singular_value,
    ) = find_second_singular_contributions(model, size_direction, query_direction)
    (W_Q_U, W_Q_S, W_Q_Vh), (W_Q_contrib, W_Q_err) = split_svd_contributions(
        model.W_Q[0, 0]
    )
    (W_K_U, W_K_S, W_K_Vh), (W_K_contrib, W_K_err) = split_svd_contributions(
        model.W_K[0, 0]
    )
    (
        (EQKE_query_key, err_accumulator),
        EQKE_pos_err,
        (err_upper_bound, (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)),
    ) = quadratic.decompose_EQKE_error_quadratic(
        model,
        key_direction=size_direction,
        query_direction=query_direction,
        second_key_direction=second_key_direction,
        second_query_direction=second_query_direction,
        W_Q_U=W_Q_U,
        W_K_U=W_K_U,
        sanity_check=False,
    )

    EQKE_pos_err_with_attn_scale = EQKE_pos_err / model.blocks[0].attn.attn_scale

    for attn_scale, cur_EQKE_pos_err in (
        ("", EQKE_pos_err),
        ("WithAttnScale", EQKE_pos_err_with_attn_scale),
    ):
        results_float |= data_summary(
            cur_EQKE_pos_err.flatten(), prefix=f"EQKP{attn_scale}"
        )
        results_float |= data_summary(
            cur_EQKE_pos_err.abs().flatten(), prefix=f"EQKP{attn_scale}Abs"
        )
        results_float |= data_summary(
            cur_EQKE_pos_err.amax(dim=-1) - cur_EQKE_pos_err.amin(dim=-1),
            prefix=f"EQKP{attn_scale}MaxRowDiff",
        )

    EQKE_err = W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
    EQKE_err_simple = EQKE_err + err_accumulator
    EQKE_exact = EQKE_query_key + EQKE_err_simple

    EQKE_query_key_with_attn_scale = EQKE_query_key / model.blocks[0].attn.attn_scale
    err_accumulator_with_attn_scale = err_accumulator / model.blocks[0].attn.attn_scale
    EQKE_err_with_attn_scale = EQKE_err / model.blocks[0].attn.attn_scale
    EQKE_err_simple_with_attn_scale = EQKE_err_simple / model.blocks[0].attn.attn_scale
    EQKE_exact_with_attn_scale = EQKE_exact / model.blocks[0].attn.attn_scale

    if include_figures:
        for attn_scale, attn_scale_value in (
            ("", 1.0),
            ("WithAttnScale", model.blocks[0].attn.attn_scale),
        ):
            fig = imshow(
                EQKE_exact / attn_scale_value,
                colorscale=QK_colorscale,
                title=f"EQKE{attn_scale}",
                xaxis="key token",
                yaxis="query token",
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"EQKE{attn_scale}"] = fig
            fig = imshow(
                EQKE_query_key.numpy() / attn_scale_value,
                title=(
                    f"EQKE{attn_scale}<sub>1</sub>"
                    if title_kind == "html"
                    else f"EQKE{attn_scale}$_1$"
                ),
                colorscale=QK_colorscale,
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"EQKE{attn_scale}1"] = fig
            fig = imshow(
                err_accumulator.numpy() / attn_scale_value,
                title=(
                    f"err_accumulator{attn_scale}"
                    if title_kind == "html"
                    else rf"err\_accumulator{attn_scale}"
                ),
                colorscale=QK_colorscale,
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"err_accumulator{attn_scale}"] = fig
            fig = imshow(
                (EQKE_query_key + err_accumulator) / attn_scale_value,
                title=(
                    f"EQKE{attn_scale}<sub>2</sub>"
                    if title_kind == "html"
                    else "EQKE$_2$"
                ),
                colorscale=QK_colorscale,
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"EQKE{attn_scale}2"] = fig
            smath = "" if title_kind == "html" else "$"
            sWE = "W<sub>E</sub>" if title_kind == "html" else "W_E"
            sWpos = "W<sub>pos</sub>" if title_kind == "html" else r"W_{\mathrm{pos}}"
            sWQ = "W<sub>Q</sub>" if title_kind == "html" else r"W_Q"
            sWK = "W<sub>K</sub>" if title_kind == "html" else r"W_K"
            sT = "<sup>T</sup>" if title_kind == "html" else "^T"
            sE = "𝔼" if title_kind == "html" else r"\mathbb{E}"
            s_p = "<sub>p</sub>" if title_kind == "html" else "_p"
            fig = imshow(
                EQKE_pos_err.numpy() / attn_scale_value,
                title=f"{smath}({sWE} + {sWpos}[-1]){sWQ}{sWK}{sT}({sWpos} - {sE}{s_p}{sWpos}[p]){sT}{smath}{attn_scale}",
                colorscale=QK_colorscale,
                dtick_x=pos_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"EQKE_pos_err{attn_scale}"] = fig

            zmax = EQKE_err.abs().max().item() / attn_scale_value
            fig = imshow(
                EQKE_err.numpy() / attn_scale_value,
                title=(
                    f"EQKE_err{attn_scale}"
                    if title_kind == "html"
                    else rf"EQKE\_err{attn_scale}"
                ),
                xaxis="key token",
                yaxis="query token",
                colorscale=QK_colorscale,
                zmax=zmax,
                zmin=-zmax,
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"EQKE_err{attn_scale}"] = fig
            fig = imshow(
                EQKE_err.numpy() / attn_scale_value,
                title=(
                    f"EQKE_err{attn_scale}"
                    if title_kind == "html"
                    else rf"EQKE\_err{attn_scale}"
                ),
                xaxis="",
                yaxis="",
                colorscale=QK_colorscale,
                zmax=zmax,
                zmin=-zmax,
                showticklabels=False,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )
            results[f"EQKE_err_noticks{attn_scale}"] = fig
            results[f"EQKE_err_svd{attn_scale}"] = analyze_svd(
                EQKE_err / attn_scale_value,
                descr=(
                    f"EQKE_err{attn_scale}"
                    if title_kind == "html"
                    else rf"EQKE\_err{attn_scale}"
                ),
                colorscale=QK_SVD_colorscale,
                plot_with=plot_with,
                renderer=renderer,
                show=show,
            )

            for m, s, key in (
                (
                    W_E_query_err2,
                    {
                        "html": "E<sub>q,2</sub><sup>⟂</sup>",
                        "latex": r"$E_{q,2}^{\perp}$",
                    },
                    "WEqqPerp",
                ),
                (
                    W_Q_err,
                    {"html": "Q<sup>⟂</sup>", "latex": r"$Q^{\perp}$"},
                    "WQqPerp",
                ),
                (
                    W_K_errT,
                    {"html": "K<sup>⟂</sup>", "latex": r"$K^{\perp}$"},
                    "WKkPerp",
                ),
                (
                    W_E_key_err2T,
                    {
                        "html": "E<sub>k,2</sub><sup>⟂</sup>",
                        "latex": r"$E_{k,2}^{\perp}$",
                    },
                    "WEkkPerp",
                ),
            ):
                fig = imshow(
                    m.numpy() / attn_scale_value,
                    title=f"{s[title_kind]}{attn_scale}",
                    colorscale=QK_colorscale,
                    zmax=zmax,
                    zmin=-zmax,
                    showticklabels=False,
                    plot_with=plot_with,
                    renderer=renderer,
                    show=show,
                )
                results[f"{key}{attn_scale}"] = fig
                fig = analyze_svd(
                    m / attn_scale_value,
                    scale_by_singular_value=False,
                    descr=f"{s[title_kind]}{attn_scale}",
                    colorscale=QK_SVD_colorscale,
                    plot_with=plot_with,
                    renderer=renderer,
                    show=show,
                )
                results[f"{key}-svd{attn_scale}"] = fig

    U, S, Vh = torch.linalg.svd(EQKE_exact)
    S_with_attn_scale = S / model.blocks[0].attn.attn_scale
    mindim = np.min(model.W_Q[0, 0].shape)
    for attn_scale, cur_S in (("", S), ("WithAttnScale", S_with_attn_scale)):
        results_float[f"EQKE{attn_scale}FirstSingularFloat"] = cur_S[0].item()
        results_float[f"EQKE{attn_scale}FirstSingularSqrtTwoFloat"] = cur_S[
            0
        ].item() * np.sqrt(2)
        if do_print:
            print(
                f"σ₁(EQKE_err)√2 = {cur_S[0].item()}√2 = {cur_S[0].item()*np.sqrt(2)}"
            )
        results_float[f"EQKE{attn_scale}SecondSingularFloat"] = cur_S[1].item()
        results_float[f"EQKE{attn_scale}ThirdSingularFloat"] = cur_S[2].item()
        results_float[f"EQKE{attn_scale}RatioFirstTwoSingularFloat"] = (
            cur_S[0] / cur_S[1]
        ).item()
        results_float |= data_summary(S[:mindim], prefix=f"EQKE{attn_scale}Singular")
    size_direction_diffs = size_direction.squeeze()[1:] - size_direction.squeeze()[:-1]
    results_float |= data_summary(size_direction, prefix="EQKESizeDirection")
    results_float |= data_summary(size_direction_diffs, prefix="EQKESizeDirectionDiffs")
    results_float |= data_summary(query_direction, prefix="EQKEQueryDirection")

    for cur_EQKE_err, descr in (
        (EQKE_err_simple, "Simple"),
        (EQKE_err, ""),
        (EQKE_err_simple_with_attn_scale, "SimpleWithAttnScale"),
        (EQKE_err_with_attn_scale, "WithAttnScale"),
    ):
        results_float[f"EQKEErr{descr}MaxRowDiffFloat"] = (
            (cur_EQKE_err.max(dim=-1).values - cur_EQKE_err.min(dim=-1).values)
            .max()
            .item()
        )
        results_float[f"EQKEErr{descr}MaxAbsFloat"] = cur_EQKE_err.abs().max().item()
        results_float[f"EQKEErr{descr}MeanDimZeroNormFloat"] = (
            cur_EQKE_err.mean(dim=0).norm().item()
        )
        results_float |= data_summary(cur_EQKE_err.flatten(), f"EQKEErr{descr}")
        s1 = torch.linalg.matrix_norm(cur_EQKE_err, ord=2)
        results_float[f"EQKEErr{descr}FirstSingularFloat"] = s1.item()
        results_float[f"EQKEErr{descr}FirstSingularSqrtTwoFloat"] = (
            s1 * np.sqrt(2)
        ).item()
        sf1 = torch.linalg.matrix_norm(cur_EQKE_err, ord="fro")
        results_float[f"EQKEErr{descr}FroNormFloat"] = sf1.item()
        results_float[f"EQKEErr{descr}FroNormSqrtTwoFloat"] = (sf1 * np.sqrt(2)).item()
        if do_print:
            print(f"σf₁(EQKE_err)√2 = {sf1}√2 = {sf1*np.sqrt(2)}")

    if do_print:
        for k in (
            "EQKEErrMaxRowDiffFloat",
            "EQKEErrMaxAbsFloat",
            "EQKEErrMeanDimZeroNormFloat",
        ):
            print(f"{k}: {results_float[k]}")

    ss = [
        torch.linalg.matrix_norm(m, ord=2).item()
        for m in (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)
    ]
    (
        results_float["WEqqPerpFirstSingularFloat"],
        results_float["WQqPerpFirstSingularFloat"],
        results_float["WKkPerpFirstSingularFloat"],
        results_float["WEkkPerpFirstSingularFloat"],
    ) = ss
    if do_print:
        print(f"singular values: {ss}")
        print(f"√2∏σ₁ = {np.prod(ss)}√2 = {np.prod(ss)*np.sqrt(2)}")

    results_float["EQKEErrProdFirstSingularFloat"] = np.prod(ss)
    results_float["EQKEErrProdFirstSingularSqrtTwoFloat"] = np.prod(ss) * np.sqrt(2)
    sfs = [
        torch.linalg.matrix_norm(m, ord="fro").item()
        for m in (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)
    ]
    (
        results_float["WEqqPerpFroNormFloat"],
        results_float["WQqPerpFroNormFloat"],
        results_float["WKkPerpFroNormFloat"],
        results_float["WEkkPerpFroNormFloat"],
    ) = sfs
    if do_print:
        print(f"singular fro values: {sfs}")
        print(f"√2∏σf₁ = {np.prod(sfs)}√2 = {np.prod(sfs)*np.sqrt(2)}")
        print(f"err_upper_bound: {err_upper_bound}")
    results_float["EQKEErrProdFroNormFloat"] = np.prod(sfs)
    results_float["EQKEErrProdFroNormSqrtTwoFloat"] = np.prod(sfs) * np.sqrt(2)

    resample_EQKE_err_results, resample_EQKE_err_results_float = resample_EQKE_err(
        (
            W_E_query_err2,
            (
                {"html": "E<sub>q,2</sub><sup>⟂</sup>", "latex": r"$E_{q,2}^\perp$"},
                "WEqqPerp-hist",
            ),
        ),
        (W_Q_err, ({"html": "Q<sup>⟂</sup>", "latex": r"$Q^\perp$"}, "WQqPerp-hist")),
        (W_K_errT, ({"html": "K<sup>⟂</sup>", "latex": r"$K^\perp$"}, "WKkPerp-hist")),
        (
            W_E_key_err2T,
            (
                {"html": "E<sub>k,2</sub><sup>⟂</sup>", "latex": r"$E_{k,2}^\perp$"},
                "WEkkPerp-hist",
            ),
        ),
    )
    # W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)
    results |= resample_EQKE_err_results
    results_float |= resample_EQKE_err_results_float

    return results, results_float
