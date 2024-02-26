from typing import Optional
import torch
from torch import Tensor
import math
from jaxtyping import Float
from transformer_lens import HookedTransformer
import plotly.express as px

from gbmi.exp_max_of_n.analysis import find_size_and_query_direction_no_figure


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
            "title": "Attention Score<br>QK[p] := (W<sub>E</sub>[-1] + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ (W<sub>E</sub> + W<sub>pos</sub>[p])<sup>T</sup><br>QK[:-1,:-1].mean(dim=0) - QK[-1, -1]",
            "xaxis": "input token",
            "yaxis": "attention score pre-softmax",
        }
    else:
        QK = (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)).T
        return {
            "data": QK.numpy(),
            "title": "Attention Score<br>QK := (W<sub>E</sub> + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ (W<sub>E</sub> + W<sub>pos</sub>.mean(dim=0))<sup>T</sup>",
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
    else:
        OV = (W_E + W_pos.mean(dim=0)) @ W_V[0, 0] @ W_O[0, 0] @ W_U
    result: dict = {"xaxis": "output logit token", "yaxis": "input token"}
    if not centered:
        result.update(
            {
                "data": OV.numpy(),
                "title": "Attention Computation: (W<sub>E</sub>[:-1] + W<sub>pos</sub>[:-1].mean(dim=0)) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub>",
            }
        )
        return result
    result.update(
        {
            "data": (OV - OV.diag()[:, None]).numpy(),
            "title": "Attention Computation (centered)<br>OV := (W<sub>E</sub>[:-1] + W<sub>pos</sub>[:-1].mean(dim=0)) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub><br>OV - OV.diag()[:, None]",
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
            "title": "Positional Contribution to Attention Score<br>(W<sub>E</sub>[-1] + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ (W<sub>pos</sub>[:-1] - W<sub>pos</sub>[:-1].mean(dim=0))<sup>T</sup>",
            "xaxis": "position",
            "yaxis": "attention score pre-softmax",
        }
    else:
        QK = (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_pos - W_pos.mean(dim=0)).T
        return {
            "data": {"QK": QK.numpy()},
            "title": "Positional Contribution to Attention Score<br>(W<sub>E</sub> + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ (W<sub>pos</sub> - W<sub>pos</sub>.mean(dim=0))<sup>T</sup>",
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
        f"(W<sub>E</sub>{W_E_q_index}+W<sub>pos</sub>[-1]) @ W<sub>U</sub>": (
            ((W_E_q + W_pos[-1]) @ W_U).numpy()
        ),
    }
    if include_equals_OV:
        data.update(
            {
                f"(W<sub>E</sub>{W_E_q_index}+W<sub>pos</sub>[-1]) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub>": (
                    (W_E_q + W_pos[-1]) @ W_V[0, 0] @ W_O[0, 0] @ W_U
                ),
            }
        )
    data.update(
        {
            f"(W<sub>pos</sub>[{i}] - W<sub>pos</sub>{W_pos_k_index}.mean(dim=0)) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub>": (
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
        "yaxis": "logit value",
    }


@torch.no_grad()
def display_basic_interpretation(
    model: HookedTransformer,
    include_uncentered: bool = False,
    legend_at_bottom: bool = False,
    include_equals_OV: bool = False,
    includes_eos: Optional[bool] = None,
    renderer: Optional[str] = None,
):
    if includes_eos is None:
        includes_eos = model.cfg.d_vocab != model.cfg.d_vocab_out
    QK = compute_QK(model, includes_eos=includes_eos)
    if includes_eos:
        px.line(
            {"QK": QK["data"]},
            title=QK["title"],
            labels={
                "index": QK["xaxis"],
                "variable": "",
                "value": QK["yaxis"],
            },
        ).show(renderer=renderer)
    else:
        px.imshow(
            QK["data"],
            title=QK["title"],
            color_continuous_scale="Sunsetdark",
            labels={"x": QK["xaxis"], "y": QK["yaxis"]},
        ).show(renderer=renderer)
        find_size_and_query_direction_no_figure(
            model, plot_heatmaps=True, renderer=renderer, colorscale="Picnic_r"
        )

    if include_uncentered:
        OV = compute_OV(model, centered=False, includes_eos=includes_eos)
        px.imshow(
            OV["data"],
            title=OV["title"],
            color_continuous_scale="Picnic_r",
            color_continuous_midpoint=0,
            labels={"x": OV["xaxis"], "y": OV["yaxis"]},
        ).show(renderer=renderer)
    OV = compute_OV(model, centered=True, includes_eos=includes_eos)
    px.imshow(
        OV["data"],
        title=OV["title"],
        color_continuous_scale="Picnic_r",
        labels={"x": OV["xaxis"], "y": OV["yaxis"]},
    ).show(renderer=renderer)

    pos_QK = compute_QK_by_position(model, includes_eos=includes_eos)
    if includes_eos:
        px.scatter(
            pos_QK["data"],
            title=pos_QK["title"],
            labels={"index": pos_QK["xaxis"], "variable": "", "value": pos_QK["yaxis"]},
        ).show(renderer=renderer)
    else:
        px.imshow(
            pos_QK["data"]["QK"],
            title=pos_QK["title"],
            color_continuous_scale="Sunsetdark",
            labels={"x": pos_QK["xaxis"], "y": pos_QK["yaxis"]},
        ).show(renderer=renderer)

    irrelevant = compute_irrelevant(
        model, include_equals_OV=include_equals_OV, includes_eos=includes_eos
    )
    for key, data in irrelevant["data"].items():
        if len(data.shape) == 2:
            px.imshow(
                data,
                title=key,
                color_continuous_scale="Picnic_r",
                labels={"x": irrelevant["xaxis"], "y": irrelevant["yaxis"]},
            ).show(renderer=renderer)
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
    fig.show(renderer=renderer)
