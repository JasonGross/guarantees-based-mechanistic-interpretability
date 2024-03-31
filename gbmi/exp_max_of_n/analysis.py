from typing import Literal, Tuple, Optional, Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from transformer_lens import HookedTransformer, utils as utils

import gbmi.analysis_tools
from gbmi.analysis_tools.decomp import analyze_svd
import gbmi.utils
from gbmi.analysis_tools import plot
from gbmi.analysis_tools.fit import (
    cubic_func,
    quintic_func,
    quadratic_func,
    quartic_func,
    show_fits,
    sigmoid_func,
    inv_sigmoid_func,
)
from gbmi.analysis_tools.plot import Colorscale, colorscale_to_cmap, imshow, line
from gbmi.verification_tools.decomp import factor_contribution


@torch.no_grad()
def find_size_and_query_direction(
    model: HookedTransformer, plot_heatmaps=False, renderer=None, **kwargs
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[dict[str, go.Figure]]
]:
    """
    Approximates the size direction of the model.
    """
    W_pos, W_Q, W_K, W_E = model.W_pos, model.W_Q, model.W_K, model.W_E
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_pos.shape == (
        n_ctx,
        d_model,
    ), f"W_pos.shape = {W_pos.shape} != {(n_ctx, d_model)} = (n_ctx, d_model)"
    assert W_Q.shape == (
        1,
        1,
        d_model,
        d_model,
    ), f"W_Q.shape = {W_Q.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
    assert W_K.shape == (
        1,
        1,
        d_model,
        d_model,
    ), f"W_K.shape = {W_K.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
    assert W_E.shape == (
        d_vocab,
        d_model,
    ), f"W_E.shape = {W_E.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"

    QK = (
        (W_E + W_pos[-1])
        @ W_Q[0, 0, :, :]
        @ W_K[0, 0, :, :].T
        @ (W_E + W_pos.mean(dim=0)).T
    )
    assert QK.shape == (
        d_vocab,
        d_vocab,
    ), f"QK.shape = {QK.shape} != {(d_vocab, d_vocab)} = (d_vocab, d_vocab)"

    # take SVD:
    U, S, Vh = torch.linalg.svd(QK)
    # adjust the free parameter of sign
    sign = torch.sign(U[:, 0].mean())
    U, Vh = U * sign, Vh * sign

    # the size direction is the first column of Vh, normalized
    # query direction is the first column of U, normalized
    size_direction, query_direction = Vh[0, :], U[:, 0]
    size_query_singular_value = S[0] * size_direction.norm() * query_direction.norm()
    size_direction, query_direction = (
        size_direction / size_direction.norm(),
        query_direction / query_direction.norm(),
    )

    fig = None

    if plot_heatmaps:
        size_direction_resid, query_direction_resid = size_direction @ W_E + W_pos[
            -1
        ], query_direction @ W_E + W_pos.mean(dim=0)
        size_direction_QK, query_direction_QK = (
            size_direction_resid @ W_Q[0, 0, :, :],
            query_direction_resid @ W_K[0, 0, :, :],
        )

        fig = display_size_direction_stats(
            size_direction,
            query_direction,
            QK,
            U,
            Vh,
            S,
            # size_direction_resid=size_direction_resid, size_direction_QK=size_direction_QK,
            # query_direction_resid=query_direction_resid, query_direction_QK=query_direction_QK,
            renderer=renderer,
            **kwargs,
        )

    return (size_direction, query_direction, size_query_singular_value.item()), fig


@torch.no_grad()
def find_size_and_query_direction_no_figure(
    *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximates the size direction of the model.
    """
    return find_size_and_query_direction(*args, **kwargs)[0]


@torch.no_grad()
def find_size_direction(model: HookedTransformer, **kwargs):
    """
    Approximates the size direction of the model.
    """
    return find_size_and_query_direction_no_figure(model, **kwargs)[0]


@torch.no_grad()
def find_query_direction(model: HookedTransformer, **kwargs):
    """
    Approximates the query direction of the model.
    """
    return find_size_and_query_direction_no_figure(model, **kwargs)[1]


@torch.no_grad()
def find_second_singular_contributions(
    model: HookedTransformer,
    size_direction: torch.Tensor,
    query_direction: torch.Tensor,
    sanity_check: bool = True,
    plot_heatmaps: bool = False,
    renderer: Optional[str] = None,
) -> Tuple[Tuple[torch.Tensor, float], Tuple[torch.Tensor, float]]:
    """
    Finds the most significant singualar contribution to W_E + W_pos.mean(dim=0), and to W_E + W_pos[-1], after size_direction and query_direction have been removed.

    Returns ((query, singular value), (key, singular value))
    """
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )

    W_E_pos_k = W_E + W_pos.mean(dim=0)[None, :]
    W_E_pos_q = W_E + W_pos[-1][None, :]
    _W_E_size, W_E_size_err = factor_contribution(
        W_E_pos_k, size_direction, sanity_check=sanity_check
    )
    _W_E_query, W_E_query_err = factor_contribution(
        W_E_pos_q, query_direction, sanity_check=sanity_check
    )
    if plot_heatmaps:
        analyze_svd(
            W_E_size_err,
            descr="W_E_size_err",
            renderer=renderer,
            scale_by_singular_value=False,
        )
        analyze_svd(
            W_E_query_err,
            descr="W_E_query_err",
            renderer=renderer,
            scale_by_singular_value=False,
        )
    W_E_size_err_U, S_size, _ = torch.linalg.svd(W_E_size_err)
    W_E_query_err_U, S_query, _ = torch.linalg.svd(W_E_query_err)
    return (W_E_size_err_U[:, 0], S_size[0].item()), (
        W_E_query_err_U[:, 0],
        S_query[0].item(),
    )


@torch.no_grad()
def display_size_direction_stats(
    size_direction: torch.Tensor,
    query_direction: torch.Tensor,
    QK: torch.Tensor,
    U: torch.Tensor,
    Vh: torch.Tensor,
    S: torch.Tensor,
    size_direction_resid: Optional[torch.Tensor] = None,
    size_direction_QK: Optional[torch.Tensor] = None,
    query_direction_resid: Optional[torch.Tensor] = None,
    query_direction_QK: Optional[torch.Tensor] = None,
    *,
    do_exclusions: bool = True,
    include_contribution: bool = True,
    scale_by_singular_value: bool = True,
    renderer: Optional[str] = None,
    fit_funcs: Iterable = (cubic_func, quintic_func),
    delta_fit_funcs: Iterable = (quadratic_func, quartic_func),
    colorscale: Colorscale = "Plasma_r",
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    **kwargs,
):
    cmap = colorscale_to_cmap(colorscale)
    if scale_by_singular_value:
        U = U * S[None, :].sqrt()
        Vh = Vh * S[:, None].sqrt()
    results = {}

    match plot_with:
        case "plotly":
            fig = imshow(
                QK,
                title="Attention<br>(W<sub>E</sub> + W<sub>pos</sub>[-1])W<sub>Q</sub>W<sub>K</sub><sup>T</sup>(W<sub>E</sub> + 𝔼<sub>p</sub>W<sub>pos</sub>[p])<sup>T</sup>",
                xaxis="Key Token",
                yaxis="Query Token",
                renderer=renderer,
                colorscale=colorscale,
                **kwargs,
            )
        case "matplotlib":
            fig, ax = plt.subplots()
            sns.heatmap(QK, cmap=cmap, ax=ax)
            ax.set_title(
                "Attention\n$(W_E + W_{\\mathrm{pos}}[-1])W_Q W_K^T (W_E + \\mathbb{E}_p W_{\\mathrm{pos}}[p])^T$"
            )
            ax.set_xlabel("Key Token")
            ax.set_ylabel("Query Token")
            plt.show()
    results["Attention"] = fig

    uzmax, vzmax = U.abs().max().item(), Vh.abs().max().item()
    match plot_with:
        case "plotly":
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=["Query-Side SVD", "Singular Values", "Key-Side SVD"],
            )
            fig.add_trace(
                go.Heatmap(
                    z=utils.to_numpy(U),
                    colorscale=colorscale,
                    zmin=-uzmax,
                    zmax=uzmax,
                    showscale=False,
                    #  colorbar=dict(x=-0.15, # https://community.plotly.com/t/colorbar-ticks-left-aligned/60473/4
                    #             ticklabelposition='inside',
                    #             ticksuffix='     ',
                    #             ticklabeloverflow='allow',
                    #             tickfont_color='darkslategrey',),
                    hovertemplate="Query: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(S.shape[0]),
                    y=utils.to_numpy(S),
                    mode="lines+markers",
                    marker=dict(color="blue"),
                    line=dict(color="blue"),
                    hovertemplate="Singular Value: %{y}<br>Singular Index: %{x}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Heatmap(
                    z=utils.to_numpy(Vh.T),
                    colorscale=colorscale,
                    zmin=-vzmax,
                    zmax=vzmax,
                    showscale=False,
                    #  colorbar=dict(x=1.15),
                    hovertemplate="Key: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>",
                ),
                row=1,
                col=3,
            )
            fig.update_layout(title="Attention SVD")  # , margin=dict(l=150, r=150))
            fig.update_yaxes(title_text="Query Token", row=1, col=1)
            fig.update_yaxes(range=[0, None], row=1, col=2)
            fig.update_yaxes(title_text="Key Token", row=1, col=3)
            fig.show(renderer)
        case "matplotlib":
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # Plot 1: Query-Side SVD Heatmap
            cax1 = axs[0].matshow(
                U, cmap="viridis", vmin=-uzmax, vmax=uzmax
            )  # Adjust cmap to match your colorscale
            fig.colorbar(cax1, ax=axs[0])  # Optional: Add a colorbar
            axs[0].set_title("Query-Side SVD")
            axs[0].set_xlabel("Singular Index")
            axs[0].set_ylabel("Query Token")

            # Plot 2: Singular Values Line Plot
            axs[1].plot(np.arange(S.shape[0]), S, marker="o", color="blue")
            axs[1].set_title("Singular Values")
            axs[1].set_xlabel("Singular Index")
            axs[1].set_ylabel("Value")
            axs[1].set_ylim(bottom=0)  # Ensure y-axis starts at 0

            # Plot 3: Key-Side SVD Heatmap
            cax3 = axs[2].matshow(
                Vh.T, cmap="viridis", vmin=-vzmax, vmax=vzmax
            )  # Adjust cmap to match your colorscale
            fig.colorbar(cax3, ax=axs[2])  # Optional: Add a colorbar
            axs[2].set_title("Key-Side SVD")
            axs[2].set_xlabel("Singular Index")
            axs[2].set_ylabel("Key Token")

            # Adjust layout
            plt.tight_layout()
            plt.suptitle("Attention SVD")
            plt.show()
    results["Attention SVD"] = fig

    contribution_diff = None
    if include_contribution:
        contribution_diff, _ = compute_singular_contribution(
            QK,
            description="Attention",
            colorscale=colorscale,
            renderer=renderer,
            singular_value_count=1,
            xaxis="Key Token",
            yaxis="Query Token",
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Value: %{z}<extra></extra>",
            **kwargs,
        )

    # imshow(U, title="Query-Side SVD", yaxis="Query Token", renderer=renderer, **kwargs)
    # imshow(Vh.T, title="Key-Side SVD", yaxis="Key Token", renderer=renderer, **kwargs)
    # px.line({'singular values': training.to_numpy(S)}, title="Singular Values of QK Attention").show(renderer)

    match plot_with:
        case "plotly":
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Size", "Query"])
            fig.add_trace(
                go.Scatter(
                    x=np.arange(size_direction.shape[0]),
                    y=utils.to_numpy(size_direction),
                    mode="lines+markers",
                    marker=dict(color="blue"),
                    line=dict(color="blue"),
                    hovertemplate="Token: %{x}<br>Size: %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(query_direction.shape[0]),
                    y=utils.to_numpy(query_direction),
                    mode="lines+markers",
                    marker=dict(color="blue"),
                    line=dict(color="blue"),
                    hovertemplate="Token: %{x}<br>Query Value: %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            fig.update_layout(title="Directions in Token Space", showlegend=False)
            fig.show(renderer)
        case "matplotlib":
            fig, axs = plt.subplots(
                1, 2, figsize=(12, 6)
            )  # Adjust the figure size as needed

            # Plot 1: Size
            axs[0].plot(
                np.arange(size_direction.shape[0]),
                size_direction,
                marker="o",
                linestyle="-",
                color="blue",
            )
            axs[0].set_title("Size")
            axs[0].set_xlabel("Token")
            axs[0].set_ylabel("Size")

            # Plot 2: Query
            axs[1].plot(
                np.arange(query_direction.shape[0]),
                query_direction,
                marker="o",
                linestyle="-",
                color="blue",
            )
            axs[1].set_title("Query")
            axs[1].set_xlabel("Token")
            axs[1].set_ylabel("Query Value")

            # Set the main title for all subplots
            fig.suptitle("Directions in Token Space", fontsize=16)
            plt.tight_layout(
                rect=[0, 0, 1, 0.95]
            )  # Adjust layout to make room for the main title
            # Show plot
            plt.show()
    results["Directions in Token Space"] = fig

    # px.line({'size direction': training.to_numpy(size_direction)}, title="size direction in token space").show(renderer)
    # px.line({'query direction': training.to_numpy(query_direction)}, title="query direction in token space").show(renderer)
    if size_direction_resid is not None:
        fig = line(
            size_direction_resid,
            title="size direction in residual space",
            renderer=renderer,
        )
        results["Size Direction in Residual Space"] = fig
    if query_direction_resid is not None:
        fig = line(
            query_direction_resid,
            title="query direction in residual space",
            renderer=renderer,
        )
        results["Query Direction in Residual Space"] = fig
    if size_direction_QK is not None:
        fig = line(
            size_direction_QK, title="size direction in QK space", renderer=renderer
        )
        results["Size Direction in QK Space"] = fig
    if query_direction_QK is not None:
        fig = line(
            query_direction_QK, title="query direction in QK space", renderer=renderer
        )
        results["Query Direction in QK Space"] = fig

    reference_lines = []
    if contribution_diff is not None:
        # we make some reference lines for the plots of size[i+1] - size[i]
        # since we'll eventually multiply these by the singular value and the query direction entry, we want to divide by this product when comparing to values from the non-size-direction contributions
        # we compute the mean and worst-case behavior, and a more fine-grained worst-case adjacent difference
        singular_scale = S[0].item()
        scale_per_query = query_direction * singular_scale
        resid_diffs = contribution_diff[:, :-1] - contribution_diff[:, 1:]
        resid_max_diff = contribution_diff.max().item() - contribution_diff.min().item()
        resid_max_diff_per_query = (
            contribution_diff.max(dim=1).values - contribution_diff.min(dim=1).values
        )
        scale_mean, scale_min = (
            scale_per_query.mean(dim=0).item(),
            scale_per_query.min().item(),
        )
        resid_mean_diff = (
            (contribution_diff[:, :, None, None] - contribution_diff[None, None, :, :])
            .abs()
            .mean()
            .item()
        )
        resid_mean_diff_per_query = (
            (contribution_diff[:, :, None] - contribution_diff[:, None, :])
            .abs()
            .mean(dim=(-2, -1))
        )
        reference_lines = [
            (
                "resid.max - resid.min (worst-case independent query)",
                resid_max_diff / scale_min,
            ),
            (
                "resid.max - resid.min (average-case independent query)",
                resid_max_diff / scale_mean,
            ),
            (
                "resid.max - resid.min (worst-case query)",
                (resid_max_diff_per_query / scale_per_query).max().item(),
            ),
            (
                "(resid[i] - resid[i+1]).max (worst-case independent query)",
                (resid_diffs / scale_min).max().item(),
            ),
            (
                "(resid[i] - resid[i+1]).max (worst-case query)",
                (resid_diffs / scale_per_query[:, None]).max().item(),
            ),
            (
                "(resid[i] - resid[i+1]).abs.mean (average-case independent query)",
                (resid_diffs / scale_mean).abs().mean().item(),
            ),
            (
                "(resid[i] - resid[j]).abs.mean (average-case independent query)",
                resid_mean_diff / scale_mean,
            ),
            (
                "(resid[i] - resid[j]).abs.mean (average-case query)",
                (resid_mean_diff_per_query / scale_per_query).abs().mean().item(),
            ),
        ]

    size_direction_differences = size_direction[1:] - size_direction[:-1]
    figs = show_fits(
        size_direction,
        name="Size Direction",
        fit_funcs=(fit_func for fit_func in fit_funcs if fit_func is not sigmoid_func),
        do_exclusions=do_exclusions,
        renderer=renderer,
    )
    if figs is not None:
        results.update(figs)
    figs = show_fits(
        size_direction_differences,
        name="Size Direction Δ",
        reference_lines=reference_lines,
        fit_funcs=(
            fit_func for fit_func in delta_fit_funcs if fit_func is not sigmoid_func
        ),
        do_exclusions=do_exclusions,
        renderer=renderer,
    )
    if figs is not None:
        results.update(figs)

    y_data = size_direction.detach().cpu().numpy()
    x_data = np.linspace(1, len(y_data), len(y_data))

    for fit_func in fit_funcs:
        fit_func_name = fit_func.__name__
        if fit_func_name.endswith("_func"):
            fit_func_name = fit_func_name[: -len("_func")]

        if fit_func is sigmoid_func:
            # fit to sigmoid
            y_transposed = np.linspace(1, len(x_data), len(x_data))
            initial_params_transposed = [
                max(y_transposed),
                1 / np.mean(y_data),
                np.median(y_data),
            ]

            # Fit the curve with initial parameters

            params_transposed, covariance_transposed = curve_fit(
                sigmoid_func,
                y_data,
                y_transposed,
                p0=initial_params_transposed,
                maxfev=10000,
            )

            # Generate predicted y values with parameters
            y_pred_transposed = sigmoid_func(y_data, *params_transposed)
            # Calculating residuals
            residuals = y_transposed - y_pred_transposed

            # Creating subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            fig.suptitle(
                "Fitting a Sigmoid to the Size Vector Components and Residuals Analysis",
                fontsize=16,
            )

            # Plotting the original data and fitted curve
            plot.scatter(y_data, y_transposed, label="Data", color="blue")
            axs[0].plot(
                y_data,
                y_pred_transposed,
                color="red",
                label=rf"{inv_sigmoid_func.equation(params_transposed)}",
            )
            axs[0].set_xlabel("Component in Normalized Size Vector")
            axs[0].set_ylabel("Input Token")
            axs[0].legend()
            axs[0].grid(True)

            # Plotting residuals
            plot.scatter(y_data, residuals, color="green", label="Residuals")
            axs[1].axhline(y=0, color="r", linestyle="--", label="y=0")
            axs[1].set_xlabel("Component in Normalized Size Vector")
            axs[1].set_ylabel("Residual")
            axs[1].legend()
            axs[1].grid(True)

            # Displaying the plots
            plt.tight_layout(
                rect=(0, 0.03, 1, 0.95)
            )  # To prevent overlap between suptitle and subplots
            plt.show()
    return results


@torch.no_grad()
def compute_singular_contribution(
    M: torch.Tensor,
    plot_heatmaps=True,
    yaxis=None,
    xaxis=None,
    title=None,
    renderer=None,
    description=None,
    singular_value_count=1,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(M)
    (
        U[:, singular_value_count:],
        S[singular_value_count:],
        Vh[singular_value_count:, :],
    ) = (0, 0, 0)
    contribution = U @ torch.diag(S) @ Vh
    if plot_heatmaps:
        singular_value_str = (
            f"first {singular_value_count} singular values"
            if singular_value_count != 1
            else "first singular value"
        )
        to_description = f" to {description}" if description is not None else ""
        description = f"{description} " if description is not None else ""
        diff_zmax = (M - contribution).abs().max().item()
        zmax = np.max([contribution.abs().max().item(), diff_zmax])
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Contribution", "Residual", "Residual (rescaled)"],
        )
        fig.add_trace(
            go.Heatmap(
                z=utils.to_numpy(contribution),
                zmin=-zmax,
                zmax=zmax,
                showscale=True,
                colorbar=dict(x=-0.15, y=0.5),
                **kwargs,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=utils.to_numpy(M - contribution),
                zmin=-zmax,
                zmax=zmax,
                showscale=False,
                **kwargs,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=utils.to_numpy(M - contribution),
                zmin=-diff_zmax,
                zmax=diff_zmax,
                showscale=True,
                **kwargs,
            ),
            row=1,
            col=3,
        )
        if title is None:
            title = f"Contribution of the {singular_value_str}{to_description}"
        fig.update_layout(title=title, margin=dict(l=100))
        for col in range(3):
            if yaxis is not None:
                fig.update_yaxes(title_text=yaxis, row=1, col=col + 1)
            if xaxis is not None:
                fig.update_xaxes(title_text=xaxis, row=1, col=col + 1)
    fig.show(renderer)
    return M - contribution, contribution
