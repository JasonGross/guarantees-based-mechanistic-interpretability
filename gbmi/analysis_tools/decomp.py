import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import utils as utils

import gbmi.utils


def analyze_svd(
    M,
    descr="",
    scale_by_singular_value=True,
    colorscale="Picnic_r",
    singular_color="blue",
    renderer=None,
):
    U, S, Vh = torch.linalg.svd(M)
    V = gbmi.utils.T
    if scale_by_singular_value:
        U = U * S[None, :].sqrt()
        V = V * S[:, None].sqrt()
    if descr:
        descr = f" for {descr}"

    fig = make_subplots(rows=1, cols=3, subplot_titles=["U", "Singular Values", "V"])
    uzmax, vzmax = U.abs().max().item(), V.abs().max().item()
    fig.add_trace(
        go.Heatmap(
            z=utils.to_numpy(U),
            zmin=-uzmax,
            zmax=uzmax,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="U: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=utils.to_numpy(V),
            colorscale=colorscale,
            zmin=-vzmax,
            zmax=vzmax,
            showscale=False,
            hovertemplate="V: %{x}<br>Singular Index: %{y}<br>Value: %{z}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(S.shape[0]),
            y=utils.to_numpy(S),
            mode="lines+markers",
            marker=dict(color=singular_color),
            line=dict(color=singular_color),
            hovertemplate="Singular Value: %{y}<br>Singular Index: %{x}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title=f"SVD{descr}")  # , margin=dict(l=150, r=150))

    fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_layout(yaxis_scaleanchor="x")
    fig.update_yaxes(scaleanchor="x", autorange="reversed", row=1, col=1)
    fig.update_yaxes(scaleanchor="x", autorange="reversed", row=1, col=3)

    # fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[0, U.shape[0]], row=1, col=1)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[0, U.shape[1]], row=1, col=1)

    # fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[0, None], row=1, col=2)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[0, S.shape[0]], row=1, col=2)

    # fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[0, Vh.shape[0]], row=1, col=3)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[0, Vh.shape[1]], row=1, col=3)

    # fig.update_xaxes(range=[0, None], row=1, col=1)
    # fig.update_xaxes(range=[0, None], row=1, col=2)
    # fig.update_xaxes(range=[0, None], row=1, col=3)

    # fig.update_yaxes(range=[0, None], row=1, col=1)
    # fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_yaxes(range=[0, None], row=1, col=3)

    # fig.update_yaxes(title_text="Query Token", row=1, col=1)
    fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_yaxes(title_text="Key Token", row=1, col=3)

    fig.show(renderer)

    # line(S, title=f"Singular Values{descr}")
    # imshow(U, title=f"Principal Components on U{descr}")
    # imshow(Vh, title=f"Principal Components on Vh{descr}")
