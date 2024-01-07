import torch
from plotly import express as px
from transformer_lens import HookedTransformer, utils as utils

from gbmi.analysis_tools.decomp import analyze_svd
from gbmi.analysis_tools.plot import imshow, line


def analyze_PVOU(model: HookedTransformer, colorscale="RdBu", renderer=None):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_pos @ W_V @ W_O @ W_U).detach()[0, 0, :, :]
    pos_indices = torch.arange(n_ctx)
    fig = px.imshow(
        utils.to_numpy(res),
        title="W_pos @ W_V @ W_O @ W_U",
        labels={"x": "logit affected", "y": "position"},
        color_continuous_midpoint=0.0,
        color_continuous_scale=colorscale,
    )
    fig.update_yaxes(tickvals=pos_indices, ticktext=pos_indices)
    fig.show(renderer)


def analyze_PU(model: HookedTransformer, colorscale="RdBu", renderer=None):
    W_U, W_pos = model.W_U, model.W_pos
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    res = (W_pos[-1, :] @ W_U).detach()
    line(
        res,
        title="W_pos[-1] @ W_U",
        xaxis="output token",
        showlegend=False,
        hovertemplate="Logit for %{x}: %{y}",
        renderer=renderer,
    )


def analyze_EU(model: HookedTransformer, colorscale="RdBu", renderer=None):
    W_U, W_E = model.W_U, model.W_E
    d_model, d_vocab = model.cfg.d_model, model.cfg.d_vocab
    assert W_U.shape == (d_model, d_vocab)
    assert W_E.shape == (d_vocab, d_model)
    res = (W_E @ W_U).detach()
    imshow(
        res,
        title="W_E @ W_U",
        renderer=renderer,
        xaxis="logit affected",
        yaxis="input token",
        colorscale=colorscale,
    )


def analyze_EVOU(
    model: HookedTransformer,
    colorscale="RdBu",
    renderer=None,
    scale_by_singular_value=True,
):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_E @ W_V @ W_O @ W_U).detach().cpu()[0, 0, :, :]
    imshow(
        res,
        title="W_E @ W_V @ W_O @ W_U",
        renderer=renderer,
        xaxis="logit affected",
        yaxis="input token",
        colorscale=colorscale,
    )
    analyze_svd(
        res,
        descr="W_E @ W_V @ W_O @ W_U",
        colorscale=colorscale,
        scale_by_singular_value=scale_by_singular_value,
        renderer=renderer,
    )
    line(
        res.diag(),
        title="(W_E @ W_V @ W_O @ W_U).diag()",
        xaxis="input token",
        showlegend=False,
        hovertemplate="Input Token: %{x}<br>Logit on %{x}: %{y}",
        renderer=renderer,
    )
