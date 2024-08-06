# %%
import math
from functools import partial
from pathlib import Path
from typing import (
    Callable,
    Collection,
    Literal,
    Optional,
    Tuple,
    Union,
    Sequence,
    Any,
    TypeVar,
)
from collections import defaultdict
import imageio
from matplotlib.colors import Colormap, hsv_to_rgb, rgb_to_hsv, to_hex, is_color_like
import torch
import numpy as np
from torch import Tensor
from jaxtyping import Float
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib as mpl
import matplotlib.axes
import matplotlib.axes._axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from cycler import cycler
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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
T = TypeVar("T")


def cmap_to_list(
    cmap: Union[str, ListedColormap], num_samples: int = 256
) -> list[Tuple[float, str]]:
    if isinstance(cmap, str):
        return cmap_to_list(plt.cm.get_cmap(cmap))

    if isinstance(cmap, mcolors.ListedColormap):
        colors = cmap.colors
        num_colors = len(colors)

        color_list = []

        for i, color in enumerate(colors):
            # Position as a fraction (0 to 1)
            position = i / (num_colors - 1)
            # Convert RGB tuple to hex string
            hex_color = mcolors.rgb2hex(color)
            color_list.append((position, hex_color))

        return color_list
    elif isinstance(cmap, mcolors.LinearSegmentedColormap):
        color_list = []

        for i in range(num_samples):
            # Position as a fraction (0 to 1)
            position = i / (num_samples - 1)
            # Get the color at this position
            rgba = cmap(position)
            # Convert RGBA tuple to hex string
            hex_color = mcolors.rgb2hex(rgba)
            color_list.append((position, hex_color))

        return color_list
    else:
        raise ValueError(
            f"{cmap} ({type(cmap)}) is not a ListedColormap nor LinearSegmentedColormap"
        )


def shift_cyclical_colorscale(
    colors: list[Tuple[float, T]], shift: int = 0
) -> list[Tuple[float, T]]:
    pos = [c[0] for c in colors]
    colors = [c[1] for c in colors]
    mid = len(colors) // 2
    return list(zip(pos, colors[mid + shift :] + colors[: mid + shift]))


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


def hex_to_rgb_float(hex_color: str) -> Tuple[float, float, float]:
    """Convert a hex color string to an RGB float tuple."""
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


GET_GRADIENT_N_STEPS_DEFAULT: int = 10


def get_gradient(
    hex_start: str, hex_end: str, n_steps: int = GET_GRADIENT_N_STEPS_DEFAULT
) -> np.ndarray:
    rgb_start = hex_to_rgb_float(hex_start)
    rgb_end = hex_to_rgb_float(hex_end)

    hsv_start = rgb_to_hsv(np.array([rgb_start]))[0]
    hsv_end = rgb_to_hsv(np.array([rgb_end]))[0]

    # Interpolate HSV values
    hues = np.linspace(hsv_start[0], hsv_end[0], n_steps)
    saturations = np.linspace(hsv_start[1], hsv_end[1], n_steps)
    values = np.linspace(hsv_start[2], hsv_end[2], n_steps)

    # Combine interpolated HSV values
    hsv_gradient = np.column_stack((hues, saturations, values))

    # Convert HSV to RGB
    rgb_gradient = hsv_to_rgb(hsv_gradient)

    return rgb_gradient


def interpolate_gradient(
    seq: Sequence[str],
    n_steps: Optional[int] = None,
    n_total_steps: Optional[int] = None,
) -> list[str]:
    if n_steps is None and n_total_steps is None:
        n_steps = GET_GRADIENT_N_STEPS_DEFAULT
    elif n_steps is None:
        n_steps = n_total_steps // (len(seq) - 1)
    gradient = []
    for i in range(len(seq) - 1):
        gradient.extend(get_gradient(seq[i], seq[i + 1], n_steps=n_steps))
    return [to_hex(color) for color in gradient]


def combine_interpolate_color_mapping(
    colors_1: Sequence[str],
    colors_2: Sequence[str],
    mid_color: Optional[str] = "#ffffff",
    mid: float = 0.485,
    n_steps: Optional[int] = None,
    n_total_steps: Optional[int] = None,
) -> list[Tuple[float, str]]:
    hex_1 = interpolate_gradient(colors_1, n_steps=n_steps, n_total_steps=n_total_steps)
    hex_2 = interpolate_gradient(colors_2, n_steps=n_steps, n_total_steps=n_total_steps)

    all_hex = []
    all_hex += list(zip(np.linspace(0, mid, len(hex_1)), hex_1))
    if mid_color is not None:
        all_hex += [(0.5, mid_color)]
    all_hex += list(zip(np.linspace(1 - mid, 1, len(hex_2)), hex_2))

    return all_hex


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
    dtick_x: Optional[float | int] = None,
    dtick_y: Optional[float | int] = None,
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
    xtickargs: dict = dict(showticklabels=showticklabels and showxticklabels)
    ytickargs: dict = dict(showticklabels=showticklabels and showyticklabels)
    if dtick_x is not None and xtickargs["showticklabels"]:
        xtickargs["dtick"] = dtick_x
    if dtick_y is not None and ytickargs["showticklabels"]:
        ytickargs["dtick"] = dtick_y
    fig.update_xaxes(**xtickargs)
    fig.update_yaxes(**ytickargs)
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
    dtick_x: Optional[float | int] = None,
    dtick_y: Optional[float | int] = None,
    show: bool = True,
    **kwargs,
):
    cmap = colorscale_to_cmap(colorscale)
    fig, ax = plt.subplots(figsize=figsize)
    plt.close()
    sns.heatmap(
        utils.to_numpy(tensor),
        ax=ax,
        center=0.0,
        cmap=cmap,
        vmin=zmin,
        vmax=zmax,
        # cbar_kws={"label": "Scale"},
    )
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    showxticklabels &= showticklabels
    showyticklabels &= showticklabels
    if showxticklabels and dtick_x is not None:
        xticks = np.arange(0, tensor.shape[1], dtick_x, dtype=type(dtick_x))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, visible=showxticklabels)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), visible=showxticklabels)
    if showyticklabels and dtick_y is not None:
        yticks = np.arange(0, tensor.shape[0], dtick_y, dtype=type(dtick_y))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, visible=showyticklabels)
    else:
        ax.set_yticklabels(ax.get_yticklabels(), visible=showyticklabels)
    ax.tick_params(
        bottom=showxticklabels,
        left=showyticklabels,
        top=False,
        labeltop=False,
        right=False,
        labelright=False,
    )
    if title is not None:
        ax.set_title(title)
    if show:
        plt.figure(fig)
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
    *data,
    # index: str = "index",
    # variable: str = "variable",
    # value: str = "value",
    xaxis: str = "",
    yaxis: str = "",
    caxis: str = "",
    show: bool = True,
    renderer: Optional[str] = None,
    legend_at_bottom: bool = False,
    reverse_xaxis: bool = False,
    reverse_yaxis: bool = False,
    color_order: Optional = None,
    yrange: Optional[Tuple[float, float]] = None,
    discontinuous_x: Sequence[float] = (),
    discontinuous_y: Sequence[float] = (),
    # prop_cycle: dict[str, Any] = {},
    **kwargs,
):
    # x = utils.to_numpy(x)
    # y = utils.to_numpy(y)
    # labels = {"x": xaxis, "y": yaxis, "color": caxis}
    labels = {"index": xaxis, "x": xaxis, "variable": caxis, "value": yaxis, "y": yaxis}
    fig = px.scatter(*data, labels=labels, **kwargs)
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
    if reverse_yaxis:
        fig.update_layout(yaxis=dict(autorange="reversed"))
    if reverse_xaxis:
        fig.update_layout(xaxis=dict(autorange="reversed"))
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    if show:
        fig.show(renderer)
    return fig


def better_unique_markers(n: int) -> list:
    # like sns._base.unique_markers, but more frindly to TeX export
    markers = sns._base.unique_markers(n * 10)
    str_markers = [m for m in markers if isinstance(m, str)]
    num_markers = [m for m in markers if not isinstance(m, str)]
    return (str_markers + num_markers)[:n]


def scatter_matplotlib(
    *data,
    xaxis: str = "",
    yaxis: str = "",
    caxis: str = "",
    show: bool = True,
    title: Optional[str] = None,
    legend_at_bottom: bool = False,
    renderer: Optional[str] = None,
    log_x: Union[bool, int] = False,
    log_y: Union[bool, int] = False,
    reverse_xaxis: bool = False,
    reverse_yaxis: bool = False,
    legend: Optional[bool] = None,
    yrange: Optional[Tuple[float, float]] = None,
    discontinuous_x: Sequence[float] = (),
    discontinuous_y: Sequence[float] = (),
    # prop_cycle: dict[str, Any] = {},
    **kwargs,
):
    # x = utils.to_numpy(x)
    # y = utils.to_numpy(y)
    fig, axes = plt.subplots(
        nrows=len(discontinuous_y) + 1,
        ncols=len(discontinuous_x) + 1,
        sharey="row",
        sharex="col",
        squeeze=False,
    )
    if not show:
        plt.close()
    # for ax in axes:
    #     ax.set_prop_cycle(cycler(**prop_cycle))
    data_minx: dict[int, float] = defaultdict(lambda: np.inf)
    data_maxx: dict[int, float] = defaultdict(lambda: -np.inf)
    data_miny: dict[int, float] = defaultdict(lambda: np.inf)
    data_maxy: dict[int, float] = defaultdict(lambda: -np.inf)
    missing: dict[Tuple[int, int], bool] = defaultdict(lambda: False)

    def axes_scatter(all_x, all_y, **kwargs):
        remaining_points = list(zip(all_x, all_y))
        y_ubounds = [-np.inf] + list(discontinuous_y) + [np.inf]
        x_ubounds = [-np.inf] + list(discontinuous_x) + [np.inf]
        for r, row in enumerate(axes if reverse_yaxis else reversed(axes)):
            for c, ax in enumerate(row if not reverse_xaxis else reversed(row)):
                y_ubound, x_ubound = y_ubounds[r + 1], x_ubounds[c + 1]
                y_lbound, x_lbound = y_ubounds[r], x_ubounds[c]
                cur_x, cur_y = [], []
                next_points = []
                for x, y in remaining_points:
                    # convert to float so we don't overflow int64 when looking at min, max below
                    x = float(x) if isinstance(x, int) else x
                    y = float(y) if isinstance(y, int) else y
                    if x_lbound < x <= x_ubound and y_lbound < y <= y_ubound:
                        cur_x.append(x)
                        cur_y.append(y)
                    else:
                        next_points.append((x, y))
                remaining_points = next_points
                # print(r, c, cur_x, cur_y, next_points)
                if cur_x:
                    ax.scatter(cur_x, cur_y, **kwargs)
                    data_minx[c] = np.min([data_minx[c], *cur_x])
                    data_maxx[c] = np.max([data_maxx[c], *cur_x])
                    data_miny[r] = np.min([data_miny[r], *cur_y])
                    data_maxy[r] = np.max([data_maxy[r], *cur_y])
                    # print(r, c, dict(data_minx), dict(data_maxx), dict(data_miny), dict(data_maxy))
                else:
                    ax.scatter(all_x[:1], all_y[:1], **kwargs)
                    missing[(r, c)] = True
        assert len(remaining_points) == 0, (remaining_points, x_ubounds, y_ubounds)

    def on_all_axes(f: Callable[[matplotlib.axes._axes.Axes], Any], axes=axes):
        if isinstance(axes, np.ndarray):
            return [on_all_axes(f, ax) for ax in axes]
        return f(axes)

    right_ax = axes[0, -1]
    # sns_kwargs = {}
    # if len(data) == 1 and isinstance(data[0], dict):
    #     data = list(data)
    #     print(data)
    #     data[0] = pd.DataFrame(data[0])
    #     data[0] = pd.melt(data[0], var_name="variable", value_name="value")
    #     kwargs |= dict(x="variable", y="value")
    #     print(data)
    # manual scatter plotting so that we get better LaTeX export
    if len(data) == 1 and isinstance(data[0], dict):
        for (k, v), marker in zip(data[0].items(), better_unique_markers(len(data[0]))):
            x = range(len(v))
            axes_scatter(x, v, label=k, marker=marker)
        if not legend_at_bottom:
            box = right_ax.get_position()
            right_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            right_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    elif (
        len(data) == 1
        and isinstance(data[0], pd.DataFrame)
        and "x" in kwargs
        and "y" in kwargs
    ):
        data = data[0]
        if "color" in kwargs:
            groups = kwargs.get("color_order", data[kwargs["color"]].unique())
            if legend is None:
                legend = True
            for group, marker in zip(groups, better_unique_markers(len(groups))):
                subset = data[data[kwargs["color"]] == group]
                axes_scatter(
                    subset[kwargs["x"]],
                    subset[kwargs["y"]],
                    label=group,
                    # color=
                )
        else:
            axes_scatter(
                data[kwargs["x"]],
                data[kwargs["y"]],
                # label=group,
                # color=
            )
    else:
        sns_remap = {"color": "hue", "color_order": "hue_order"}
        for k in list(kwargs.keys()):
            if k in sns_remap and sns_remap[k] not in kwargs:
                kwargs[sns_remap[k]] = kwargs.pop(k)
        on_all_axes(lambda ax: sns.scatterplot(*data, ax=ax, **kwargs))
    if legend_at_bottom:
        right_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), shadow=True)
    elif legend:
        box = right_ax.get_position()
        right_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        right_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # ax.legend(loc='upper right', bbox_to_anchor=(0, 0, 1, 1))
    rmid = axes.shape[0] // 2
    cmid = axes.shape[1] // 2
    axes[-1, cmid].set_xlabel(xaxis)
    axes[rmid, 0].set_ylabel(yaxis)
    if log_x:
        if isinstance(log_x, bool):
            on_all_axes(lambda ax: ax.set_xscale("log"))
        else:
            on_all_axes(lambda ax: ax.set_xscale("log", base=log_x))
    if log_y:
        if isinstance(log_y, bool):
            on_all_axes(lambda ax: ax.set_yscale("log"))
        else:
            on_all_axes(lambda ax: ax.set_yscale("log", base=log_y))
    if reverse_xaxis:
        on_all_axes(lambda ax: ax.invert_xaxis())
    if reverse_yaxis:
        on_all_axes(lambda ax: ax.invert_yaxis())
    if yrange is not None:
        on_all_axes(lambda ax: ax.set_ylim(*yrange))
    if discontinuous_y:
        fig.subplots_adjust(hspace=0.15)
    if discontinuous_x:
        fig.subplots_adjust(wspace=0.15)
    for row in axes[:-1]:
        for ax in row:
            ax.xaxis.tick_top()
    for row in axes[-1:]:
        for ax in row:
            ax.xaxis.tick_bottom()
    for row in axes:
        for ax in row[:1]:
            ax.yaxis.tick_left()
        for ax in row[1:]:
            ax.yaxis.tick_right()
    for row in axes:
        # hide the spines between axes
        for ax in row[:-1]:
            ax.spines["right"].set_visible(False)
        # for ax in row[:1]:
        #     ax.yaxis.tick_left()
        for ax in row[1:]:
            # ax.yaxis.tick_right()
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft="off", left=False)
            for tick in ax.get_yticklabels():
                tick.set_visible(False)
            # ax.set_yticks([])
    for row in axes[:-1]:
        for ax in row:
            # ax.xaxis.tick_top()
            ax.spines.bottom.set_visible(False)
            ax.tick_params(labelbottom="off", bottom=False)
            for tick in ax.get_xticklabels():
                tick.set_visible(False)
            # ax.set_xticks([])
    for row in axes[1:]:
        for ax in row:
            ax.spines.top.set_visible(False)
    on_all_axes(lambda ax: ax.tick_params(labeltop="off", labelright="off"))

    def adjust_range(lo, hi, log_base):
        if log_base is True:
            log_base = 10
        do_log = (lambda x: x) if not log_base else (lambda x: math.log(x, log_base))
        do_exp = (lambda x: x) if not log_base else partial(math.pow, log_base)
        lo = do_log(lo)
        hi = do_log(hi)
        r = 1.1 * (hi - lo) or 2
        return do_exp(lo - r), do_exp(hi + r)

    if discontinuous_y:
        for r, row in enumerate(axes):
            for c, ax in enumerate(row):
                if (
                    np.isfinite(data_maxy[r])
                    and np.isfinite(data_miny[r])
                    and missing[(r, c)]
                ):
                    # print(f"axes[{r}, {c}].set_ylim({data_miny[r] - extra}, {data_maxy[r] + extra})")
                    ax.set_ylim(*adjust_range(data_miny[r], data_maxy[r], log_y))
    if discontinuous_x:
        for r, row in enumerate(axes):
            for c, ax in enumerate(row):
                # print(data_minx, data_maxx)
                # print(r, c, np.isfinite(data_maxx[c]), np.isfinite(data_minx[c]), missing[(r, c)])
                if (
                    np.isfinite(data_maxx[c])
                    and np.isfinite(data_minx[c])
                    and missing[(r, c)]
                ):
                    # print(f"axes[{r}, {c}].set_xlim({data_minx[c] - extra}, {data_maxx[c] + extra})")
                    ax.set_xlim(*adjust_range(data_minx[c], data_maxx[c], log_x))
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.figure(fig)
        plt.show()
    return fig


def scatter(
    *data,
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
                *data,
                xaxis=xaxis,
                yaxis=yaxis,
                caxis=caxis,
                renderer=renderer,
                show=show,
                **kwargs,
            )
        case "matplotlib":
            return scatter_matplotlib(
                *data,
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
    plt.close()
    sns.histplot(data=data, ax=ax, **kwargs)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if not column_names and ax.get_legend() is not None:
        ax.get_legend().remove()
    if title is not None:
        ax.set_title(title)
    if show:
        plt.figure(fig)
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


def colorbar_plotly(
    zmin: float,
    zmax: float,
    colorscale: Colorscale = "RdBu",
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    show: bool = True,
    renderer: Optional[str] = None,
    **kwargs,
):
    fig = go.Figure(
        data=go.Heatmap(
            z=[[0]],
            colorscale=colorscale,
            showscale=True,
            zmin=zmin,
            zmax=zmax,
            # zmid=0,
            colorbar=dict(x=0),
        )
    )
    fig.add_trace(
        go.Heatmap(
            z=[[0]],
            colorscale="Picnic_r",
            showscale=False,
            zmin=zmin,
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
    return fig


def colorbar_matplotlib(
    zmin: float,
    zmax: float,
    colorscale: Colorscale = "RdBu",
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    show: bool = True,
    renderer: Optional[str] = None,
    **kwargs,
):
    cmap = colorscale_to_cmap(colorscale)
    fig = plt.figure(figsize=(0.5, 4) if orientation == "vertical" else (4, 0.5))
    plt.close()
    norm = matplotlib.colors.Normalize(vmin=zmin, vmax=zmax)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=fig.gca(),
        orientation=orientation,
    )
    # cbar = matplotlib.colorbar.ColorbarBase(
    #     plt.gca(), cmap=cmap, norm=norm, orientation="vertical"
    # )
    if show:
        plt.figure(fig)
        plt.show()
    return fig


def colorbar(
    zmin: float,
    zmax: float,
    colorscale: Colorscale = "RdBu",
    *,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    show: bool = True,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    **kwargs,
):
    match plot_with:
        case "plotly":
            return colorbar_plotly(
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                show=show,
                orientation=orientation,
                **kwargs,
            )
        case "matplotlib":
            return colorbar_matplotlib(
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                show=show,
                orientation=orientation,
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


def EVOU_max_logit_diff(
    model: HookedTransformer,
    mean_PVOU: bool = False,
) -> Float[Tensor, "d_vocab"]:  # noqa: F821
    EVOU = all_EVOU(model)
    if mean_PVOU:
        EVOU += all_PVOU(model).mean(dim=0)
    max_logit_diff = EVOU.max(dim=-1).values - EVOU.min(dim=-1).values
    return max_logit_diff


def hist_EVOU_max_logit_diff(
    model: HookedTransformer,
    mean_PVOU: bool = False,
    plot_with: Literal["plotly", "matplotlib"] = "plotly",
    renderer: Optional[str] = None,
    show: bool = True,
) -> Tuple[
    Union[go.Figure, matplotlib.figure.Figure], Float[Tensor, "d_vocab"]  # noqa: F821
]:
    max_logit_diff = EVOU_max_logit_diff(model, mean_PVOU=mean_PVOU)
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
        WE_str = f"({WE_str} + {sE}{s_p}{sWpos}[p])"
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
        plot_with=plot_with,
        renderer=renderer,
        show=show,
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
    column_name: str = "",
    title: Optional[str] = None,
    show: bool = True,
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
                x=bin_centers,
                y=hist_counts,
                labels={"x": xaxis, "y": yaxis},
                title=title,
                **{k: v for k, v in kwargs.items() if k != "edgecolor"},
            )
            fig.update_layout(bargap=0)
            # Iterate over each trace (bar chart) in the figure and adjust the marker properties
            if "edgecolor" in kwargs:
                for trace in fig.data:
                    if trace.type == "bar":  # Check if the trace is a bar chart
                        if kwargs["edgecolor"] == "none":
                            # Update marker properties to remove the border line or adjust its appearance
                            trace.marker.line.width = (
                                0  # Set line width to 0 to remove the border
                            )
                            # Optionally, you can also set the line color to match the bar color exactly
                            trace.marker.line.color = trace.marker.color = (
                                "rgba(0, 0, 255, 1.0)"
                            )
                        else:
                            trace.marker.line.color = kwargs["edgecolor"]
            if show:
                fig.show(renderer)
        case "matplotlib":
            fig, ax = plt.subplots()
            plt.close()
            df = pd.DataFrame({"data": data, "weights": weights})
            sns.histplot(
                df,
                weights="weights",
                bins=num_bins,
                ax=ax,
                x="data",
                **kwargs,
            )
            if not column_name and ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.set(xlabel=xaxis, ylabel=yaxis)
            if title is not None:
                ax.set_title(title)
            # ax.bar(bin_centers, hist_counts, width=np.diff(bins), align="center")
            if show:
                plt.figure(fig)
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


# %%
def discriminate_figure(
    fig: Union[go.Figure, matplotlib.figure.Figure],
    *,
    plotly_attrs=("update_layout",),
    matplotlib_attrs=("axes",),
) -> Literal["plotly", "matplotlib"]:
    has_plotly_attrs = [hasattr(fig, attr) for attr in plotly_attrs]
    has_matplotlib_attrs = [hasattr(fig, attr) for attr in matplotlib_attrs]
    is_plotly = [isinstance(fig, go.Figure), all(has_plotly_attrs)] + has_plotly_attrs
    is_matplotlib = [
        isinstance(fig, matplotlib.figure.Figure),
        all(has_matplotlib_attrs),
    ] + has_matplotlib_attrs
    while len(is_plotly) > 0 or len(is_matplotlib) > 0:
        if len(is_plotly) > 0 and is_plotly[0]:
            return "plotly"
        elif len(is_matplotlib) > 0 and is_matplotlib[0]:
            return "matplotlib"
        else:
            if len(is_plotly) > 0:
                is_plotly.pop(0)
            if len(is_matplotlib) > 0:
                is_matplotlib.pop(0)
    assert len(is_plotly) == 0 and len(is_matplotlib) == 0, (is_plotly, is_matplotlib)
    raise ValueError(f"Could not determine the type of figure {fig} ({type(fig)})")


def remove_titles(
    fig: Union[go.Figure, matplotlib.figure.Figure],
):
    match discriminate_figure(fig):
        case "plotly":
            for trace in fig.data:
                trace.update(name="")
            fig.update_layout(title_text="")
        case "matplotlib":
            for ax in fig.axes:
                ax.set_title("")
            fig.suptitle("")
    return fig


def remove_axis_labels(
    fig: Union[go.Figure, matplotlib.figure.Figure],
):
    match discriminate_figure(fig):
        case "matplotlib":
            for ax in fig.axes:
                ax.set_xlabel("")
                ax.set_ylabel("")
        case "plotly":
            fig.update_layout(xaxis_title="", yaxis_title="")
    return fig


def remove_colorbars(
    fig: Union[go.Figure, matplotlib.figure.Figure],
):
    match discriminate_figure(fig):
        case "matplotlib":
            for ax in fig.axes:
                for cax in ax.get_children():
                    if getattr(cax, "colorbar", None) is not None:
                        cax.colorbar.remove()
        case "plotly":
            fig.update(layout_coloraxis_showscale=False)
            # for trace in fig.data:
            #     if "colorbar" in trace:
            #         trace.colorbar = None
    return fig


def remove_axis_ticklabels(
    fig: Union[go.Figure, matplotlib.figure.Figure],
    *,
    remove_tickmarks: bool = False,
):
    match discriminate_figure(fig):
        case "matplotlib":
            for ax in fig.axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if remove_tickmarks:
                    ax.tick_params(axis="both", which="both", length=0)
        case "plotly":
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            if remove_tickmarks:
                fig.update_xaxes(ticks="")
                fig.update_yaxes(ticks="")
    return fig
