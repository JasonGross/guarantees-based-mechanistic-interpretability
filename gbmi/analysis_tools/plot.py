# %%
from pathlib import Path
from typing import Callable, Collection, Literal, Optional, Tuple, Union
import imageio
from matplotlib.colors import Colormap, to_hex, is_color_like
import torch
import numpy as np
from torch import Tensor
from jaxtyping import Float
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from scipy.optimize import curve_fit
from transformer_lens import HookedTransformer, utils as utils

import gbmi.analysis_tools
from gbmi.analysis_tools.fit import linear_func
from gbmi.analysis_tools.utils import pm_range, pm_mean_std, pm_round
from gbmi.verification_tools.l1h1 import all_EVOU, all_PVOU

Colorscale = Union[str, Collection[Collection[Union[float, str]]]]


def normalize_rgba(rgba):
    """Normalize RGBA values to the 0-1 range."""
    return tuple(v / 255 if i < 3 else v for i, v in enumerate(rgba))


def color_to_hex(color: str) -> str:
    """Convert an RGB(A) or hex color string to a hex color string."""
    if color.startswith("rgb"):  # Handle 'rgb' or 'rgba'
        # Extract numerical values and convert to RGBA tuple
        rgba = tuple(
            int(x)
            for x in color.replace("rgb", "").replace("a", "").strip("()").split(",")
        )
        return to_hex(normalize_rgba(rgba))  # type: ignore
    elif color.startswith("#") or is_color_like(color):
        return to_hex(color)
    else:
        raise ValueError(f"Unrecognized color format: {color}")


def colorscale_to_cmap(colorscale: Colorscale, *, name: str = "custom") -> Colormap:
    if isinstance(colorscale, str):
        try:
            return plt.get_cmap(colorscale)
        except ValueError:
            pass
        return colorscale_to_cmap(px.colors.get_colorscale(colorscale), name=colorscale)
    colorscale_hex = [(pos, color_to_hex(color)) for pos, color in colorscale]
    return LinearSegmentedColormap.from_list(name, colorscale_hex)


def imshow_plotly(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "",
    yaxis: str = "",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    showxticklabels: bool = True,
    showyticklabels: bool = True,
    showticklabels: bool = True,
    colorscale: Colorscale = "RdBu",
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
    **kwargs,
):
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale=colorscale,
        labels={"x": xaxis, "y": yaxis},
        zmin=zmin,
        zmax=zmax,
        **kwargs,
    )
    fig.update_xaxes(showticklabels=showticklabels and showxticklabels)
    fig.update_yaxes(showticklabels=showticklabels and showyticklabels)
    if show:
        fig.show(renderer)
    return fig


def imshow_matplotlib(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "",
    yaxis: str = "",
    colorscale: Colorscale = "RdBu",
    title: Optional[str] = None,
    showxticklabels: bool = True,
    showyticklabels: bool = True,
    showticklabels: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    show: bool = True,
    **kwargs,
):
    cmap = colorscale_to_cmap(colorscale)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        utils.to_numpy(tensor),
        ax=ax,
        center=0.0,
        cmap=cmap,
        vmin=zmin,
        vmax=zmax,
        # cbar_kws={"label": "Scale"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), visible=showticklabels and showxticklabels)
    ax.set_yticklabels(ax.get_yticklabels(), visible=showticklabels and showyticklabels)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()
    return fig


def imshow(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "",
    yaxis: str = "",
    colorscale: Colorscale = "RdBu",
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    show: bool = True,
    **kwargs,
):
    match plot_with:
        case "plotly":
            return imshow_plotly(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                colorscale=colorscale,
                show=show,
                **kwargs,
            )
        case "matplotlib":
            return imshow_matplotlib(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                colorscale=colorscale,
                show=show,
                **kwargs,
            )


def line_plotly(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "",
    yaxis: str = "",
    line_labels=None,
    showlegend=None,
    hovertemplate=None,
    show: bool = True,
    **kwargs,
):
    fig = px.line(
        utils.to_numpy(tensor),
        labels={"index": xaxis, "value": yaxis},
        y=line_labels,
        **kwargs,
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    if hovertemplate is not None:
        fig.update_traces(hovertemplate=hovertemplate)
    if show:
        fig.show(renderer)
    return fig


def line_matplotlib(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "",
    yaxis: str = "",
    line_labels=None,
    showlegend=None,
    hovertemplate=None,
    show: bool = True,
    **kwargs,
):
    raise NotImplementedError("Matplotlib line plots are not yet supported.")
    fig = px.line(
        utils.to_numpy(tensor),
        labels={"index": xaxis, "value": yaxis},
        y=line_labels,
        **kwargs,
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    if hovertemplate is not None:
        fig.update_traces(hovertemplate=hovertemplate)
    fig.show(renderer)
    return fig


def line(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "",
    yaxis: str = "",
    line_labels=None,
    showlegend=None,
    hovertemplate=None,
    show: bool = True,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    **kwargs,
):
    match plot_with:
        case "plotly":
            return line_plotly(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                line_labels=line_labels,
                showlegend=showlegend,
                hovertemplate=hovertemplate,
                show=show,
                **kwargs,
            )
        case "matplotlib":
            return line_matplotlib(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                line_labels=line_labels,
                showlegend=showlegend,
                hovertemplate=hovertemplate,
                show=show,
                **kwargs,
            )


def scatter_plotly(
    x,
    y,
    xaxis: str = "",
    yaxis: str = "",
    caxis: str = "",
    show: bool = True,
    renderer: Optional[str] = None,
    **kwargs,
):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    )
    if show:
        fig.show(renderer)
    return fig


def scatter_matplotlib(
    x,
    y,
    xaxis: str = "",
    yaxis: str = "",
    caxis: str = "",
    show: bool = True,
    renderer: Optional[str] = None,
    **kwargs,
):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if show:
        plt.show()
    return fig


def scatter(
    x,
    y,
    xaxis: str = "",
    yaxis: str = "",
    caxis: str = "",
    show: bool = True,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    **kwargs,
):
    match plot_with:
        case "plotly":
            return scatter_plotly(
                x,
                y,
                xaxis=xaxis,
                yaxis=yaxis,
                caxis=caxis,
                renderer=renderer,
                show=show,
                **kwargs,
            )
        case "matplotlib":
            return scatter_matplotlib(
                x,
                y,
                xaxis=xaxis,
                yaxis=yaxis,
                caxis=caxis,
                renderer=renderer,
                show=show,
                **kwargs,
            )


def hist_plotly(
    tensor,
    renderer: Optional[str] = None,
    xaxis: str = "value",
    yaxis: str = "count",
    variable: str = "variable",
    column_names: Optional[str | Collection[str]] = None,
    show: bool = True,
    **kwargs,
):
    data = utils.to_numpy(tensor)
    if isinstance(column_names, str):
        data = {column_names: data}
    elif column_names is not None:
        data = {name: data[i] for i, name in enumerate(column_names)}
    fig = px.histogram(
        data, labels={"value": xaxis, "y": yaxis, "variable": variable}, **kwargs
    )
    if show:
        fig.show(renderer)
    return fig


def hist_matplotlib(
    tensor,
    renderer: Optional[str] = None,
    xaxis: str = "value",
    yaxis: str = "count",
    variable: str = "variable",
    column_names: Optional[str | Collection[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show: bool = True,
    **kwargs,
):
    data = utils.to_numpy(tensor)
    if isinstance(column_names, str):
        data = {column_names: data}
    elif column_names is not None:
        data = {name: data[i] for i, name in enumerate(column_names)}
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=data, ax=ax, **kwargs)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()
    return fig


def hist(
    tensor,
    *,
    renderer: Optional[str] = None,
    xaxis: str = "value",
    yaxis: str = "count",
    variable: str = "variable",
    column_names: Optional[str | Collection[str]] = None,
    show: bool = True,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    **kwargs,
):
    match plot_with:
        case "plotly":
            return hist_plotly(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                variable=variable,
                column_names=column_names,
                show=show,
                **kwargs,
            )
        case "matplotlib":
            return hist_matplotlib(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                variable=variable,
                column_names=column_names,
                show=show,
                **kwargs,
            )


def summarize(
    values,
    name=None,
    histogram=False,
    renderer: Optional[str] = None,
    hist_args={},
    imshow_args=None,
    include_value=False,
    linear_fit=False,
    fit_function=None,
    fit_equation=None,
    fit_name=None,
    min=True,
    max=True,
    mean=True,
    median=True,
    range=True,
    range_size=True,
    firstn=None,
    abs_max=True,
):
    """[some somewhat-specific summarisation thing.]"""
    if histogram:
        hist_args_list = hist_args if isinstance(hist_args, list) else [hist_args]
        for hist_args in hist_args_list:
            hist_args = dict(hist_args)
            if "title" not in hist_args and name is not None:
                hist_args["title"] = f"Histogram of {name}"
            if "renderer" not in hist_args and renderer is not None:
                hist_args["renderer"] = renderer
            if "xaxis" not in hist_args:
                hist_args["xaxis"] = name if name is not None else "Value"
            if "yaxis" not in hist_args:
                hist_args["yaxis"] = "Count"
            hist(values, **hist_args)

    if imshow_args is not None:
        imshow_args = dict(imshow_args)
        if "title" not in imshow_args and name is not None:
            imshow_args["title"] = name
        if "renderer" not in imshow_args and renderer is not None:
            imshow_args["renderer"] = renderer
        if "xaxis" not in imshow_args and name is not None:
            imshow_args["xaxis"] = f"({name}).shape[1]"
        if "yaxis" not in imshow_args and name is not None:
            imshow_args["yaxis"] = f"({name}).shape[0]"
        if len(values.shape) == 1:
            line(values, **imshow_args)
        else:
            imshow(values, **imshow_args)

    if fit_function is None and linear_fit:
        fit_function = linear_func
    if fit_equation is None and fit_function is not None:
        fit_equation = fit_function.equation
    if fit_function is not None:
        assert fit_equation is not None
        assert len(values.shape) in (1, 2)
        if len(values.shape) == 1:
            x_vals = np.arange(values.shape[0])
            y_vals = utils.to_numpy(values)
            aggregated = ""
        else:
            x_vals = np.tile(np.arange(values.shape[1]), values.shape[0])
            y_vals = utils.to_numpy(values.flatten())
            aggregated = "Aggregated "
        name_space = "" if name is None else f"{name} "
        if fit_name is None:
            fit_name = fit_function.__name__
            if fit_name is not None and fit_name.endswith("_func"):
                fit_name = fit_name[: -len("_func")]
        fit_name_space = "" if not fit_name else f"{fit_name} "
        fit_title = f"{aggregated}{name_space}Data and {fit_name_space}Fit"
        resid_title = f"{aggregated}{name_space}Residual Errors"

        # Fit linear regression to the aggregated data
        popt, _ = curve_fit(fit_function, x_vals, y_vals)

        # Create a subplot with 1 row and 2 columns
        fig, axs = plt.subplots(
            1, 2, figsize=(12, 6)
        )  # Adjust the figure size to your liking

        # Scatter plot the data & best fit line on the first subplot
        scatter(x_vals, y_vals, label="Data", alpha=0.5, s=1)
        axs[0].plot(
            x_vals,
            fit_function(x_vals, *popt),
            "r-",
            label=f"Fit: {fit_equation(popt)}",
        )
        axs[0].set_title(fit_title)
        axs[0].legend()

        # Plot residual errors on the second subplot
        residuals = y_vals - fit_function(x_vals, *popt)
        order_indices = np.argsort(x_vals)
        scatter(x_vals[order_indices], residuals[order_indices], c="b", alpha=0.5)
        axs[1].set_title(resid_title)

        # Adjust the layout
        plt.tight_layout()
        plt.show()

    res = {}
    if include_value:
        res["value"] = values.detach().clone().cpu()
    if min:
        res["min"] = values.min().item()
    if max:
        res["max"] = values.max().item()
    if mean:
        res["mean"] = pm_mean_std(values.float())
    if median:
        res["median"] = values.median().item()
    if range:
        res["range"] = pm_range(values)
    if range_size:
        res["range_size"] = values.max().item() - values.min().item()
    if firstn is not None:
        res[f"first {firstn}"] = values[:firstn]
    if abs_max:
        res["abs(max)"] = values.abs().max().item()
    if fit_function is not None:
        res["fit_equation"] = f"y = {popt[0]}*x + {popt[1]}"
    if fit_function is not None:
        res["range_residuals"] = pm_range(residuals)
    if fit_function is not None:
        res["residuals"] = residuals[order_indices]
    if fit_function is not None:
        res["fit_params"] = popt

    return res


def hist_EVOU_max_logit_diff(
    model: HookedTransformer,
    mean_PVOU: bool = False,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
) -> Tuple[
    Union[go.Figure, matplotlib.figure.Figure], Float[Tensor, "d_vocab"]  # noqa: F821
]:
    EVOU = all_EVOU(model)
    title_kind = "html" if plot_with == "plotly" else "latex"
    WE_str = "W<sub>E</sub>" if title_kind == "html" else "W_E"
    smath = "" if title_kind == "html" else "$"
    sE = "𝔼" if title_kind == "html" else r"\mathbb{E}"
    s_p = "<sub>p</sub>" if title_kind == "html" else "_p"
    sWpos = "W<sub>pos</sub>" if title_kind == "html" else r"W_{\mathrm{pos}}"
    sEVOU = "EVOU" if title_kind == "html" else r"\mathrm{EVOU}"
    sWv = "W<sub>V</sub>" if title_kind == "html" else r"W_V"
    sWo = "W<sub>O</sub>" if title_kind == "html" else r"W_O"
    sWu = "W<sub>U</sub>" if title_kind == "html" else r"W_U"
    smax = "max" if title_kind == "html" else r"\max"
    smin = "min" if title_kind == "html" else r"\min"
    s_i = "<sub>i</sub>" if title_kind == "html" else "_i"
    s_j = "<sub>j</sub>" if title_kind == "html" else "_j"
    xbar = "x̄" if title_kind == "html" else r"\bar{x}"
    sigma = "σ" if title_kind == "html" else r"\sigma"
    pm = "±" if title_kind == "html" else r"\pm"
    nl = "<br>" if title_kind == "html" else "\n"
    if mean_PVOU:
        EVOU += all_PVOU(model).mean(dim=0)
        WE_str = f"({WE_str} + {sE}{s_p}{sWpos}[p])"
    max_logit_diff = EVOU.max(dim=-1).values - EVOU.min(dim=-1).values
    mean, std = max_logit_diff.mean().item(), max_logit_diff.std().item()
    min, max = max_logit_diff.min().item(), max_logit_diff.max().item()
    mid, spread = (min + max) / 2, (max - min) / 2
    title = f"{smath}{sEVOU} := {WE_str}{sWv}{sWo}{sWu}{smath}{nl}{smath}{smax}{s_i}{sEVOU}[:,i] - {smin}{s_j}EVOU[:,j]{smath}{nl}{smath}{xbar}{pm}{sigma}{smath}: {smath}{pm_round(mean, std, sep=f' {pm} ')}{smath}; range: {smath}{pm_round(mid, spread, sep=f' {pm} ')}{smath}"
    fig = hist(
        max_logit_diff,
        xaxis="logit diff",
        variable="",
        title=title,
        column_names="",
        renderer=renderer,
    )
    return fig, max_logit_diff


@torch.no_grad()
def weighted_histogram(
    data,
    weights,
    num_bins: Optional[int] = None,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    xaxis: str = "value",
    yaxis: str = "count",
    **kwargs,
):
    if num_bins is None:
        _, bin_edges = np.histogram(data, bins="auto")
        num_bins = len(bin_edges) - 1
    bins = np.linspace(data.min(), data.max(), num=num_bins)

    # Calculate counts for each bin
    hist_counts = np.zeros(len(bins) - 1, dtype=int)
    for i, value in enumerate(data):
        factor = weights[i]
        index = np.digitize(value, bins) - 1
        if 0 <= index < len(hist_counts):
            hist_counts[index] += factor

    # Plot using px.bar, manually setting the x (bins) and y (counts)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    match plot_with:
        case "plotly":
            fig = px.bar(
                x=bin_centers, y=hist_counts, labels={"x": xaxis, "y": yaxis}, **kwargs
            )
            fig.update_layout(bargap=0)
            # Iterate over each trace (bar chart) in the figure and adjust the marker properties
            for trace in fig.data:
                if trace.type == "bar":  # Check if the trace is a bar chart
                    # Update marker properties to remove the border line or adjust its appearance
                    trace.marker.line.width = (
                        0  # Set line width to 0 to remove the border
                    )
                    # Optionally, you can also set the line color to match the bar color exactly
                    trace.marker.line.color = trace.marker.color = (
                        "rgba(0, 0, 255, 1.0)"
                    )
            fig.show(renderer)
        case "matplotlib":
            fig, ax = plt.subplots()
            sns.histplot(
                data, weights=weights, bins=bins, ax=ax, x=xaxis, y=yaxis, **kwargs
            )
            # ax.bar(bin_centers, hist_counts, width=np.diff(bins), align="center")
            plt.show()
    return fig


def plotly_save_gif(
    fig: go.Figure,
    output_path: Union[str, Path],
    frames_dir: Optional[Union[str, Path]] = None,
    duration=0.5,
    tqdm_wrapper: Optional[Callable] = tqdm,
    hide: Collection[str] = ("sliders", "updatemenus"),
    cleanup_frames: bool = False,
):
    tqdm_wrapper = tqdm_wrapper or (lambda x, **kwargs: x)

    # Backup original layout components
    originals = {}
    # Hide sliders and buttons by setting them to empty
    for k in hide:
        originals[k] = getattr(fig.layout, k)
        setattr(fig.layout, k, [])

    frames_dir = frames_dir or Path(output_path).with_suffix("") / "frames"
    Path(frames_dir).mkdir(exist_ok=True, parents=True)
    filenames = []
    for i, frame in enumerate(tqdm_wrapper(fig.frames, desc="Exporting frames")):
        # Apply frame data to the figure's traces
        for trace, frame_data in zip(fig.data, frame.data):
            trace.update(frame_data)  # type: ignore

        # If the frame has a layout, update the figure's layout accordingly
        if frame.layout:
            fig.update_layout(frame.layout)

        # Save as image
        filename = f"{frames_dir}/frame_{i:04d}.png"
        fig.write_image(filename, height=fig.layout.height, width=fig.layout.width)
        filenames.append(filename)

    # Restore original layout components
    for k, original in originals.items():
        setattr(fig.layout, k, original)

    with imageio.get_writer(output_path, mode="I", duration=duration, loop=0) as writer:
        for filename in tqdm_wrapper(filenames, desc="Compiling GIF"):
            image = imageio.imread(filename)
            writer.append_data(image)  # type: ignore

    if cleanup_frames:
        for filename in filenames:
            filename.unlink()
