# %%
from pathlib import Path
from typing import Callable, Collection, Literal, Optional, Tuple, Union
import imageio
from matplotlib.colors import Colormap, to_hex, is_color_like
import numpy as np
from torch import Tensor
from jaxtyping import Float
import matplotlib.pyplot as plt
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
    renderer=None,
    xaxis="",
    yaxis="",
    colorscale: Colorscale = "RdBu",
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale=colorscale,
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    )
    fig.show(renderer)
    return fig


def imshow_matplotlib(
    tensor,
    *,
    renderer=None,
    xaxis="",
    yaxis="",
    colorscale: Colorscale = "RdBu",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    cmap = colorscale_to_cmap(colorscale)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        utils.to_numpy(tensor),
        ax=ax,
        center=0.0,
        cmap=cmap,
        # cbar_kws={"label": "Scale"},
    )
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if title is not None:
        ax.set_title(title)
    plt.show()
    return fig


def imshow(
    tensor,
    *,
    renderer=None,
    xaxis="",
    yaxis="",
    colorscale: Colorscale = "RdBu",
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
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
                **kwargs,
            )
        case "matplotlib":
            return imshow_matplotlib(
                tensor,
                renderer=renderer,
                xaxis=xaxis,
                yaxis=yaxis,
                colorscale=colorscale,
                **kwargs,
            )


def line_plotly(
    tensor,
    *,
    renderer=None,
    xaxis="",
    yaxis="",
    line_labels=None,
    showlegend=None,
    hovertemplate=None,
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
    fig.show(renderer)
    return fig


def line_matplotlib(
    tensor,
    *,
    renderer=None,
    xaxis="",
    yaxis="",
    line_labels=None,
    showlegend=None,
    hovertemplate=None,
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
    renderer=None,
    xaxis="",
    yaxis="",
    line_labels=None,
    showlegend=None,
    hovertemplate=None,
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
                **kwargs,
            )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    )
    fig.show(renderer)
    return fig


def hist(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    fig = px.histogram(
        utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs
    )
    fig.show(renderer)
    return fig


def summarize(
    values,
    name=None,
    histogram=False,
    renderer=None,
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
    model: HookedTransformer, mean_PVOU: bool = False, renderer: Optional[str] = None
) -> Tuple[go.Figure, Float[Tensor, "d_vocab"]]:  # noqa: F821
    EVOU = all_EVOU(model)
    WE_str = "W<sub>E</sub>"
    if mean_PVOU:
        EVOU += all_PVOU(model).mean(dim=0)
        WE_str = "(W<sub>E</sub> + 𝔼<sub>p</sub>W<sub>pos</sub>[p])"
    max_logit_diff = EVOU.max(dim=-1).values - EVOU.min(dim=-1).values
    mean, std = max_logit_diff.mean().item(), max_logit_diff.std().item()
    min, max = max_logit_diff.min().item(), max_logit_diff.max().item()
    mid, spread = (min + max) / 2, (max - min) / 2
    fig = px.histogram(
        {"": max_logit_diff},
        title=f"EVOU := {WE_str}W<sub>V</sub>W<sub>O</sub>W<sub>U</sub><br>max<sub>i</sub>EVOU[:,i] - min<sub>j</sub>EVOU[:,j]<br>x̄±σ: {pm_round(mean, std)}; range: {pm_round(mid, spread)}",
        labels={"value": "logit diff", "variable": ""},
    )
    fig.show(renderer)
    return fig, max_logit_diff


def weighted_histogram(data, weights, num_bins: Optional[int] = None, **kwargs):
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
    fig = px.bar(x=bin_centers, y=hist_counts, **kwargs)
    fig.update_layout(bargap=0)
    # Iterate over each trace (bar chart) in the figure and adjust the marker properties
    for trace in fig.data:
        if trace.type == "bar":  # Check if the trace is a bar chart
            # Update marker properties to remove the border line or adjust its appearance
            trace.marker.line.width = 0  # Set line width to 0 to remove the border
            # Optionally, you can also set the line color to match the bar color exactly
            trace.marker.line.color = trace.marker.color = "rgba(0, 0, 255, 1.0)"
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
