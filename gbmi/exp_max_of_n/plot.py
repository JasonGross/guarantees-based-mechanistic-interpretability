from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import math
from jaxtyping import Float, Integer
from transformer_lens import HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
from gbmi.analysis_tools.plot import weighted_histogram
from gbmi.analysis_tools.utils import pm_round

from gbmi.exp_max_of_n.analysis import find_size_and_query_direction
from gbmi.verification_tools.l1h1 import all_EQKE, all_EVOU


@torch.no_grad()
def compute_QK(model: HookedTransformer, includes_eos: Optional[bool] = None) -> dict:
    W_E, W_pos, W_Q, W_K = (
        model.W_E.to("cpu"),
        model.W_pos.to("cpu"),
        model.W_Q.to("cpu"),
        model.W_K.to("cpu"),
    )
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out

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
            "data": (QK - QK_last).numpy(),
            "title": "Attention Score<br>QK[p] := (W<sub>E</sub>[-1] + W<sub>pos</sub>[-1])W<sub>Q</sub>W<sub>K</sub><sup>T</sup>(W<sub>E</sub> + W<sub>pos</sub>[p])<sup>T</sup><br>𝔼<sub>dim=0</sub>(QK[:-1,:-1]) - QK[-1, -1]",
            "xaxis": "input token",
            "yaxis": "attention score pre-softmax",
        }
    else:
        QK = (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)).T
        return {
            "data": QK.numpy(),
            "title": "Attention Score<br>EQKE := (W<sub>E</sub> + W<sub>pos</sub>[-1])W<sub>Q</sub>W<sub>K</sub><sup>T</sup>(W<sub>E</sub> + 𝔼<sub>p</sub>W<sub>pos</sub>[p])<sup>T</sup>",
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
    if not centered:
        result.update(
            {
                "data": OV.numpy(),
                "title": f"Attention Computation: (W<sub>E</sub>{W_E_pos_suffix} + 𝔼<sub>p</sub>W<sub>pos</sub>{W_E_pos_suffix}[p])W<sub>V</sub>W<sub>O</sub>W<sub>U</sub>",
            }
        )
        return result
    result.update(
        {
            "data": (OV - OV.diag()[:, None]).numpy(),
            "title": f"Attention Computation (centered)<br>OV := (W<sub>E</sub>{W_E_pos_suffix} + 𝔼<sub>dim=0</sub>W<sub>pos</sub>{W_E_pos_suffix})W<sub>V</sub>W<sub>O</sub>W<sub>U</sub><br>OV - OV.diag()[:, None]",
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
    if includes_eos:
        QK = (
            (W_E[-1] + W_pos[-1])
            @ W_Q[0, 0]
            @ W_K[0, 0].T
            @ (W_pos[:-1] - W_pos[:-1].mean(dim=0)).T
        )
        return {
            "data": {"QK": QK.numpy()},
            "title": "Positional Contribution to Attention Score<br>(W<sub>E</sub>[-1] + W<sub>pos</sub>[-1])W<sub>Q</sub>W<sub>K</sub><sup>T</sup>(W<sub>pos</sub>[:-1] - 𝔼<sub>dim=0</sub>W<sub>pos</sub>[:-1])<sup>T</sup>",
            "xaxis": "position",
            "yaxis": "attention score pre-softmax",
        }
    else:
        QK = (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_pos - W_pos.mean(dim=0)).T
        return {
            "data": {"QK": QK.numpy()},
            "title": "Positional Contribution to Attention Score<br>(W<sub>E</sub> + W<sub>pos</sub>[-1])W<sub>Q</sub>W<sub>K</sub><sup>T</sup>(W<sub>pos</sub> - 𝔼<sub>dim=0</sub>W<sub>pos</sub>)<sup>T</sup>",
            "xaxis": "key position",
            "yaxis": "query token",
        }


@torch.no_grad()
def compute_irrelevant(
    model: HookedTransformer,
    include_equals_OV: bool = False,
    includes_eos: Optional[bool] = None,
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
    data = {
        f"(W<sub>E</sub>{W_E_q_index}+W<sub>pos</sub>[-1])W<sub>U</sub>": (
            ((W_E_q + W_pos[-1]) @ W_U).numpy()
        ),
    }
    if include_equals_OV:
        data.update(
            {
                f"(W<sub>E</sub>{W_E_q_index}+W<sub>pos</sub>[-1])W<sub>V</sub>W<sub>O</sub>W<sub>U</sub>": (
                    (W_E_q + W_pos[-1]) @ W_V[0, 0] @ W_O[0, 0] @ W_U
                ),
            }
        )
    data.update(
        {
            f"(W<sub>pos</sub>[{i}] - 𝔼<sub>dim=0</sub>W<sub>pos</sub>{W_pos_k_index})W<sub>V</sub>W<sub>O</sub>W<sub>U</sub>": (
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
    include_uncentered: bool = False,
    legend_at_bottom: bool = False,
    include_equals_OV: bool = False,
    includes_eos: Optional[bool] = None,
    OV_colorscale: str = "Picnic_r",
    QK_colorscale: str = "Plasma",  # "Sunsetdark_r"
    QK_SVD_colorscale: str = "Picnic_r",
    renderer: Optional[str] = None,
) -> dict[str, go.Figure]:
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    QK = compute_QK(model, includes_eos=includes_eos)
    result = {}
    if includes_eos:
        fig_qk = px.line(
            {"QK": QK["data"]},
            title=QK["title"],
            labels={
                "index": QK["xaxis"],
                "variable": "",
                "value": QK["yaxis"],
            },
        )
        fig_qk.show(renderer=renderer)
    else:
        fig_qk = px.imshow(
            QK["data"],
            title=QK["title"],
            color_continuous_scale=QK_colorscale,
            labels={"x": QK["xaxis"], "y": QK["yaxis"]},
        )
        fig_qk.show(renderer=renderer)
        _, figs = find_size_and_query_direction(
            model, plot_heatmaps=True, renderer=renderer, colorscale=QK_SVD_colorscale
        )
        for k, fig in figs.items():
            result[f"EQKE {k}"] = fig
    result["EQKE"] = fig_qk

    if include_uncentered:
        OV = compute_OV(model, centered=False, includes_eos=includes_eos)
        fig_ov = px.imshow(
            OV["data"],
            title=OV["title"],
            color_continuous_scale=OV_colorscale,
            color_continuous_midpoint=0,
            labels={"x": OV["xaxis"], "y": OV["yaxis"]},
        )
        result["EVOU"] = fig_ov
        fig_ov.show(renderer=renderer)
    OV = compute_OV(model, centered=True, includes_eos=includes_eos)
    fig_ov = px.imshow(
        OV["data"],
        title=OV["title"],
        color_continuous_scale=OV_colorscale,
        labels={"x": OV["xaxis"], "y": OV["yaxis"]},
    )
    result["EVOU-centered"] = fig_ov
    fig_ov.show(renderer=renderer)

    pos_QK = compute_QK_by_position(model, includes_eos=includes_eos)
    if includes_eos:
        fig_qk = px.scatter(
            pos_QK["data"],
            title=pos_QK["title"],
            labels={"index": pos_QK["xaxis"], "variable": "", "value": pos_QK["yaxis"]},
        )
        fig_qk.show(renderer=renderer)
    else:
        fig_qk = px.imshow(
            pos_QK["data"]["QK"],
            title=pos_QK["title"],
            color_continuous_scale=QK_colorscale,
            labels={"x": pos_QK["xaxis"], "y": pos_QK["yaxis"]},
        )
        fig_qk.show(renderer=renderer)
    result["EQKP"] = fig_qk

    irrelevant = compute_irrelevant(
        model, include_equals_OV=include_equals_OV, includes_eos=includes_eos
    )
    for key, data in irrelevant["data"].items():
        if len(data.shape) == 2:
            fig = px.imshow(
                data,
                title=key,
                color_continuous_scale=OV_colorscale,
                labels={"x": irrelevant["xaxis"], "y": irrelevant["yaxis"]},
            )
            result[f"irrelevant_{key}"] = fig
            fig.show(renderer=renderer)
    fig = px.scatter(
        {k: v for k, v in irrelevant["data"].items() if len(v.shape) == 1},
        title=irrelevant["title"],
        labels={
            "index": irrelevant["xaxis"],
            "variable": "",
            "value": irrelevant["yaxis"],
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
):
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
    title = (
        f"EVOU := W_E @ W_V @ W_O @ W_U"
        f"{'' if not duplicate_by_sequence_count else ' (weighted by sequence count)'}"
        f"<br>(EVOU - EVOU.diag()[:,None]).max(dim=-1) (excluding diagonal)"
        f"<br>x̄±σ: {pm_round(mean, std)}; range: {pm_round(mid, spread)}"
    )
    if not duplicate_by_sequence_count:
        fig = px.histogram(
            {"": max_logit_minus_diag},
            title=title,
            labels={"value": "logit - diag", "variable": ""},
        )
    else:
        fig = weighted_histogram(
            max_logit_minus_diag.numpy(), duplication_factors.numpy()
        )
    fig.show(renderer=renderer)


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
    model: HookedTransformer, renderer: Optional[str] = None
):
    _, flat_idxs, flat_diffs = compute_attention_difference_vs_gap(model)
    px.scatter(
        x=flat_idxs,
        y=flat_diffs,
        labels={"x": "i - j", "y": "d_head<sup>-½</sup>((E+P)QKE[i] - (E+P)QKE[j])"},
    ).show(renderer)


@torch.no_grad()
def hist_attention_difference_over_gap(
    model: HookedTransformer,
    *,
    duplicate_by_sequence_count: bool = True,
    renderer: Optional[str] = None,
    num_bins: Optional[int] = None,
):
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
    title = (
        f"EQKE := (W<sub>E</sub> + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ W<sub>E</sub><sup>T</sup>"
        f"{'' if not duplicate_by_sequence_count else ' (weighted by sequence count)'}"
        f"<br>d_head<sup>-½</sup>(EQKE[i] - EQKE[j]) / (i - j)"
        f"<br>x̄±σ: {pm_round(mean, std)}"
    )
    xlabel = "d_head<sup>-½</sup>(EQKE[i]-EQKE[j])/(i-j)"
    if not duplicate_by_sequence_count:
        fig = px.histogram(
            {"": flat_diffs},
            labels={"value": "", "y": "count", "x": xlabel},
            title=title,
        )
    else:
        fig = weighted_histogram(
            flat_diffs.numpy(),
            duplication_factors.numpy(),
            title=title,
            num_bins=num_bins,
            labels={
                "x": xlabel,
                "y": "count * # sequences with given max",
            },
        )
    fig.show(renderer)
