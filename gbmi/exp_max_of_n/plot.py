from typing import Optional, Tuple, Literal, Union
import numpy as np
import torch
from torch import Tensor
import math
from jaxtyping import Float, Integer
from transformer_lens import HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gbmi.analysis_tools.plot import (
    Colorscale,
    weighted_histogram,
    colorscale_to_cmap,
    imshow,
    line,
    scatter,
)
from gbmi.analysis_tools.utils import pm_round
from gbmi.analysis_tools.plot import hist

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
            "html": "√d<sub>head</sub>",  # <sup>-1</sup>",
            "latex": r"\sqrt{d_{\mathrm{head}}}",  # ^{-1}}",
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
                key: f"Attention Score{nl}QK[p] := {smath}({sWe}[-1] + {sWpos}[-1]){sWq}{sWk}{sT}({sWe} + {sWpos}[p]){sT} / {sattn_scale[key]}{smath}{nl}{smath}{sE}{s_dim0}({sQK}[:-1,:-1]) - {sQK}[-1, -1]{smath}"
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
                key: f"Attention Score{nl}EQKE := {smath}({sWe} + {sWpos}[-1]){sWq}{sWk}{sT}({sWe} + {sE}{s_p}{sWpos}[p]){sT} / {sattn_scale[key]}{smath}"
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
    model: HookedTransformer, includes_eos: Optional[bool] = None
) -> dict:
    W_E, W_pos, W_Q, W_K = (
        model.W_E.to("cpu"),
        model.W_pos.to("cpu"),
        model.W_Q.to("cpu"),
        model.W_K.to("cpu"),
    )
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
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
        )
        return {
            "data": {"QK": QK.numpy()},
            "title": {
                key: f"Positional Contribution to Attention Score{nl}{smath}({sWe}[-1] + {sWpos}[-1]){sWq}{sWk}{sT}({sWpos}[:-1] - {sE}{s_dim0}{sWpos}[:-1]){sT}{smath}"
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
        QK = (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_pos - W_pos.mean(dim=0)).T
        return {
            "data": {"QK": QK.numpy()},
            "title": {
                key: f"Positional Contribution to Attention Score{nl}{smath}({sWe} + {sWpos}[-1]){sWq}{sWk}{sT}({sWpos} - {sE}{s_dim0}{sWpos}){sT}{smath}"
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
    data = {
        f"{smath}({sWE}{W_E_q_index}+{sWpos}[-1]){sWU}{smath}": (
            ((W_E_q + W_pos[-1]) @ W_U).numpy()
        ),
    }
    if include_equals_OV:
        data.update(
            {
                f"{smath}({sWE}{W_E_q_index}+{sWpos}[-1]){sWV}{sWO}{sWU}{smath}": (
                    (W_E_q + W_pos[-1]) @ W_V[0, 0] @ W_O[0, 0] @ W_U
                ),
            }
        )
    data.update(
        {
            f"{smath}({sWpos}[{i}] - {sE}{s_dim0}{sWpos}{W_pos_k_index}){sWV}{sWO}{sWU}{smath}": (
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
        "title": "Irrelevant Contributions to logits",
        "xaxis": "output logit token",
        "yaxis": "input token",
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
) -> dict[str, Union[go.Figure, matplotlib.figure.Figure]]:
    QK_cmap = colorscale_to_cmap(QK_colorscale)
    QK_SVD_cmap = colorscale_to_cmap(QK_SVD_colorscale)
    OV_cmap = colorscale_to_cmap(OV_colorscale)
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    QK = compute_QK(model, includes_eos=includes_eos)
    title_kind = "html" if plot_with == "plotly" else "latex"
    result = {}
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
                fig_qk.show(renderer=renderer)
            case "matplotlib":
                fig_qk, ax = plt.subplots()
                ax.plot(QK["data"])
                ax.set_title(QK["title"]["latex"])
                ax.set_xlabel(QK["xaxis"])
                ax.set_ylabel(QK["yaxis"])
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
        )
        _, figs = find_size_and_query_direction(
            model,
            plot_heatmaps=True,
            renderer=renderer,
            colorscale=QK_SVD_colorscale,
            plot_with=plot_with,
            dtick=tok_dtick,
        )
        assert figs is not None
        for k, fig in figs.items():
            result[f"EQKE {k}"] = fig
    result["EQKE"] = fig_qk

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
    )
    result["EVOU-centered"] = fig_ov

    pos_QK = compute_QK_by_position(model, includes_eos=includes_eos)
    if includes_eos:
        fig_qk = px.scatter(
            pos_QK["data"],
            title=pos_QK["title"][title_kind],
            labels={"index": pos_QK["xaxis"], "variable": "", "value": pos_QK["yaxis"]},
        )
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
        )
    result["EQKP"] = fig_qk

    irrelevant = compute_irrelevant(
        model,
        include_equals_OV=include_equals_OV,
        includes_eos=includes_eos,
        title_kind=title_kind,
    )
    irrelevant_plotly = compute_irrelevant(
        model,
        include_equals_OV=include_equals_OV,
        includes_eos=includes_eos,
        title_kind="html",
    )
    for key, data in irrelevant["data"].items():
        if len(data.shape) == 2:
            fig = imshow(
                data,
                title=key,
                colorscale=OV_colorscale,
                xaxis=irrelevant["xaxis"],
                yaxis=irrelevant["yaxis"],
                dtick_x=tok_dtick,
                dtick_y=tok_dtick,
                plot_with=plot_with,
                renderer=renderer,
            )
            result[f"irrelevant_{key}"] = fig
    fig = px.scatter(
        {k: v for k, v in irrelevant_plotly["data"].items() if len(v.shape) == 1},
        title=irrelevant_plotly["title"],
        labels={
            "index": irrelevant_plotly["xaxis"],
            "variable": "",
            "value": irrelevant_plotly["yaxis"],
        },
    )
    if legend_at_bottom:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.5,
                xanchor="center",
                x=0.5,
            )
        )
    result["irrelevant"] = fig
    fig.show(renderer=renderer)
    return result


@torch.no_grad()
def hist_EVOU_max_minus_diag_logit_diff(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    renderer: Optional[str] = None,
    num_bins: Optional[int] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
) -> Tuple[
    Union[go.Figure, matplotlib.figure.Figure],
    Tuple[Float[Tensor, "batch"], Integer[Tensor, "batch"]],  # noqa: F821
]:
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
        )
    else:
        fig = weighted_histogram(
            max_logit_minus_diag.numpy(),
            duplication_factors.numpy(),
            title=title,
            xaxis="logit - diag",
            yaxis=f"count * {shash} sequences with given max",
            renderer=renderer,
            plot_with=plot_with,
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
) -> Union[go.Figure, matplotlib.figure.Figure]:
    _, flat_idxs, flat_diffs = compute_attention_difference_vs_gap(model)
    title_kind = {"plotly": "html", "matplotlib": "latex"}[plot_with]
    smath = "" if title_kind == "html" else "$"
    sdhead = "d<sub>head</sub>" if title_kind == "html" else r"d_{\mathrm{head}}"
    spowmhalf = "<sup>-½</sup>" if title_kind == "html" else r"^{-\sfrac{1}{2}}"
    fig = scatter(
        x=flat_idxs,
        y=flat_diffs,
        xaxis=f"{smath}i - j{smath}",
        yaxis=f"{smath}{sdhead}{spowmhalf}((E+P)QKE[i] - (E+P)QKE[j]){smath}",
        plot_with=plot_with,
        renderer=renderer,
    )
    return fig


@torch.no_grad()
def hist_attention_difference_over_gap(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    num_bins: Optional[int] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
) -> Tuple[
    Union[go.Figure, matplotlib.figure.Figure],
    Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]],  # noqa: F821
]:
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
        )
    else:
        fig = weighted_histogram(
            flat_diffs.numpy(),
            duplication_factors.numpy(),
            title=title,
            num_bins=num_bins,
            xaxis=xlabel,
            yaxis=f"count * {shash} sequences with given max",
            renderer=renderer,
            plot_with=plot_with,
        )
    return fig, (flat_diffs, duplication_factors)
