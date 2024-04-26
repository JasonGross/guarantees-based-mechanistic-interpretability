from typing import (
    Literal,
    Tuple,
    Union,
    overload,
    Sequence,
    Optional,
    Any,
    List,
    Callable,
)
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import Tensor
from jaxtyping import Float, Shaped
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import utils as utils
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import matplotlib.pyplot as plt  # To use the colormap
import matplotlib.figure
import gbmi.utils
from gbmi.analysis_tools.plot import Colorscale, colorscale_to_cmap


def pca_2d(
    matrix: Shaped[Tensor, "n_vectors n_dims"],  # noqa: F722
    labels: Optional[List[str] | List[List[str]]] = None,
    markers: int | List[int] = 0,
    line: bool = False,
):
    def fn(matrix):
        pca = PCA(n_components=2)  # Reduce to two dimensions for visualization
        return pca.fit_transform(matrix)

    _analyse_2d(
        matrix,
        title="PCA Analysis",
        component_fn=fn,
        labels=labels,
        markers=markers,
        line=line,
    )


def tsne_2d(
    matrix: Shaped[Tensor, "n_vectors n_dims"],  # noqa: F722
    labels: Optional[List[str] | List[List[str]]] = None,
    markers: int | List[int] = 0,
    perplexity: float = 30,
    line: bool = False,
):
    # sensible range of perplexity: 5-50
    # BEWARE: see https://distill.pub/2016/misread-tsne/
    def fn(matrix):
        tsne = TSNE(
            n_components=2, perplexity=perplexity
        )  # Reduce to two dimensions for visualization
        return tsne.fit_transform(matrix)

    _analyse_2d(
        matrix,
        title="TSNE Analysis",
        component_fn=fn,
        labels=labels,
        markers=markers,
        line=line,
    )


def _analyse_2d(
    matrix: Shaped[Tensor, "n_vectors n_dims"],  # noqa: F722
    title: str,
    component_fn: Callable[
        [Float[np.ndarray, "n_vectors n_dims"]],  # noqa: F722
        Float[np.ndarray, "n_vectors n_components"],  # noqa: F722
    ],
    labels: Optional[List[str] | List[List[str]]] = None,
    markers: int | List[int] = 0,
    line: bool = False,
):
    # Labels [[a, b], [c, d]] will restart colouring at c.
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)

    # Prepare labels and colour maps
    if labels is None:
        true_labels = [[f"{i}" for i in range(len(matrix))]]
    elif isinstance(labels[0], Sequence):
        true_labels = labels
    else:
        true_labels = [labels]

    if isinstance(markers, int):
        markers = [markers for _ in labels]

    # Perform analysis
    principal_components = component_fn(matrix)

    # Create a scatter plot of the principal components

    ptr = 0
    traces = []
    for labels, marker in zip(true_labels, markers):
        slice = len(labels)
        colors = [
            "rgba({},{},{},{})".format(r * 255, g * 255, b * 255, a)
            for r, g, b, a in plt.cm.viridis(np.linspace(0, 1, slice))
        ]
        traces.append(
            go.Scatter(
                x=principal_components[ptr : ptr + slice, 0],
                y=principal_components[ptr : ptr + slice, 1],
                mode="line+markers+text" if line else "markers+text",
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(width=2, color="rgb(0, 0, 0)"),
                    symbol=marker,
                ),
                text=labels,
                textposition="top center",
            )
        )
        ptr += slice

    layout = go.Layout(
        title=title,
        xaxis=dict(title="Component 1"),
        yaxis=dict(title="Component 2"),
        showlegend=False,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    figure = go.Figure(data=traces, layout=layout)

    # Show the plot
    figure.show()


def _analyse(
    matrix: Shaped[Tensor, "n_vectors n_dims"],  # noqa: F722
    title: str,
    n_dims: Literal[2, 3],
    component_fn: Callable[
        [Float[np.ndarray, "n_vectors n_dims"]],  # noqa: F722
        Float[np.ndarray, "n_vectors n_components"],  # noqa: F722
    ],
    labels: Optional[List[str] | List[List[str]]] = None,
    markers: int | List[int] = 0,
    line: bool = False,
):
    # Labels [[a, b], [c, d]] will restart colouring at c.
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)

    # Prepare labels and colour maps
    if labels is None:
        true_labels = [[f"{i}" for i in range(len(matrix))]]
    elif isinstance(labels[0], Sequence):
        true_labels = labels
    else:
        true_labels = [labels]

    if isinstance(markers, int):
        markers = [markers for _ in labels]

    # Perform analysis
    principal_components = component_fn(matrix)

    # Create a scatter plot of the principal components

    ptr = 0
    traces = []
    for labels, marker in zip(true_labels, markers):
        slice = len(labels)
        colors = [
            "rgba({},{},{},{})".format(r * 255, g * 255, b * 255, a)
            for r, g, b, a in plt.cm.viridis(np.linspace(0, 1, slice))
        ]
        traces.append(
            (go.Scatter if n_dims == 2 else go.Scatter3d)(
                **{
                    "x": principal_components[ptr : ptr + slice, 0],
                    "y": principal_components[ptr : ptr + slice, 1],
                },
                **(
                    {}
                    if n_dims == 2
                    else {"z": principal_components[ptr : ptr + slice, 2]}
                ),
                **{
                    "mode": "line+markers+text" if line else "markers+text",
                    "marker": dict(
                        size=10,
                        color=colors,
                        line=dict(width=2, color="rgb(0, 0, 0)"),
                        symbol=marker,
                    ),
                    "text": labels,
                    "textposition": "top center",
                },
            )
        )

        ptr += slice

    layout = go.Layout(
        **dict(
            title=title,
            showlegend=False,
            margin=dict(l=40, r=40, b=40, t=40),
        ),
        **(
            dict(
                xaxis=dict(title="Component 1"),
                yaxis=dict(title="Component 2"),
            )
            if n_dims == 2
            else dict(
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3",
                )
            )
        ),
    )
    figure = go.Figure(data=traces, layout=layout)

    # Show the plot
    figure.show()


# Example usage:
# matrix_data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]
# plot_principal_components(matrix_data)


def analyze_svd(
    M,
    descr="",
    scale_by_singular_value=True,
    colorscale: Colorscale = "Picnic_r",
    singular_color="blue",
    figsize: Optional[Tuple[int, int]] = (15, 5),
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
):
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.T
    U, V = U[:, : S.shape[0]], V[:, : S.shape[0]]
    if scale_by_singular_value:
        U = U * S[None, : U.shape[1]].sqrt()
        V = V * S[None, : V.shape[1]].sqrt()
    if descr:
        descr = f" for {descr}"

    uzmax, vzmax = U.abs().max().item(), V.abs().max().item()
    cmap = colorscale_to_cmap(colorscale)
    match plot_with:
        case "plotly":
            fig = make_subplots(
                rows=1, cols=3, subplot_titles=["U", "Singular Values", "V"]
            )
            fig.add_trace(
                go.Heatmap(
                    z=utils.to_numpy(U),
                    zmin=-uzmax,
                    zmax=uzmax,
                    colorscale=colorscale,
                    zmid=0,
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
                    zmid=0,
                    showscale=False,
                    hovertemplate="V: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>",
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
        case "matplotlib":
            fig, axs = plt.subplots(1, 3, figsize=figsize)
            cax_u = axs[0].imshow(utils.to_numpy(U), cmap=cmap, vmin=-uzmax, vmax=uzmax)
            axs[0].set_title("U")
            fig.colorbar(cax_u, ax=axs[0], orientation="vertical").remove()

            axs[1].plot(
                np.arange(S.shape[0]),
                utils.to_numpy(S),
                marker="o",
                linestyle="-",
                color=singular_color,
            )
            axs[1].set_title("Singular Values")
            axs[1].set_ylim(bottom=0)

            cax_v = axs[2].matshow(
                utils.to_numpy(V), cmap=cmap, vmin=-vzmax, vmax=vzmax
            )
            axs[2].set_title("V")
            fig.colorbar(
                cax_v, ax=axs[2], orientation="vertical"
            ).remove()  # Remove colorbar if not needed

            for ax, ax_m in zip(axs, (U, None, V)):
                ax.tick_params(
                    axis="both",  # Changes apply to the x and y-axis
                    which="both",  # Both major and minor ticks are affected
                    bottom=True,
                    labelbottom=True,
                    left=True,
                    labelleft=True,
                    top=False,
                    labeltop=False,
                    right=False,
                    labelright=False,
                )
                if ax_m is not None:
                    ax.set_ylim(top=0, bottom=ax_m.shape[0])
                    # ax.invert_yaxis()
                    ax.set_facecolor("white")
                    # Get current ticks
                    yticks = ax.get_yticks()
                    yticks = yticks[(yticks >= 0) & (yticks <= ax_m.shape[0])]
                    ax.set_yticks(yticks)
                    # ax.set_yticklabels([f"{int(tick)}" for tick in yticks])

            fig.suptitle(f"SVD{descr}")
            plt.tight_layout(pad=3.0)
            plt.show()
            fig.suptitle("")
            axs[1].set_title(f"Singular Values{descr}")
            plt.figure(fig)
            plt.show()
    return fig


@torch.no_grad()
def split_svd_contributions(M: Float[Tensor, "r c"], n: int = 1) -> Tuple[  # noqa: F722
    Tuple[
        Float[Tensor, "r n"],  # noqa: F722
        Float[Tensor, "n"],  # noqa: F821
        Float[Tensor, "n c"],  # noqa: F722
    ],
    Tuple[Float[Tensor, "r c"], Float[Tensor, "r c"]],  # noqa: F722
]:
    """
    For U, S, Vh = torch.linalg.svd(M), return the first n components of (U, S, Vh) and (contrib, residual) where contrib = U @ S[:, None] @ Vh
    """
    U, S, Vh = torch.linalg.svd(M)
    U = U[:, :n]
    Vh = Vh[:n, :]
    S = S[:n]
    contrib = U @ S[:, None] @ Vh
    return (U, S, Vh), (contrib, M - contrib)
