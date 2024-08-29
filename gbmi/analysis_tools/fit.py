import itertools
from inspect import signature
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import plotly.colors
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from transformer_lens import utils as utils


def linear_func(x, a, b):
    """Linear function: f(x) = a * x + b"""
    return a * x + b


def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c


def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def quartic_func(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def quintic_func(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


def absolute_shift_func(x, a, b, c):
    return a * np.abs(x - b) + c


def linear_sinusoid_func(x, a, b, c, d):
    return (a * x + b) * np.sin(c * x + d)


def quadratic_sinusoid_func(x, a, b, c, d, e):
    return (a * x**2 + b * x + c) * np.sin(d * x + e)


def absolute_shift_sinusoid_func(x, a, b, c, d, e):
    return (a * np.abs(x - b) + c) * np.sin(d * x + e)


def linear_abs_sinusoid_func(x, a, b, c, d):
    return (a * x + b) * np.abs(np.sin(c * x + d))


def quadratic_abs_sinusoid_func(x, a, b, c, d, e):
    return (a * x**2 + b * x + c) * np.abs(np.sin(d * x + e))


def absolute_shift_abs_sinusoid_func(x, a, b, c, d, e):
    return (a * np.abs(x - b) + c) * np.abs(np.sin(d * x + e))


def sigmoid_func(x, K, B, M):
    return K / (1 + np.exp(-B * (x - M)))


def inv_sigmoid_func(y, K, B, M):
    return M - np.log(K / y - 1) / B


linear_func.equation = lambda popt: f"y = {popt[0]:.3f}*x + {popt[1]:.3f}"

quadratic_func.equation = (
    lambda popt: f"y = {popt[0]:.3e}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}"
)

cubic_func.equation = (
    lambda popt: f"y = {popt[0]:.3e}*x^3 + {popt[1]:.3e}*x^2 + {popt[2]:.3f}*x + {popt[3]:.3f}"
)

quartic_func.equation = (
    lambda popt: f"y = {popt[0]:.3e}*x^4 + {popt[1]:.3e}*x^3 + {popt[2]:.3e}*x^2 + {popt[3]:.3f}*x + {popt[4]:.3f}"
)

quintic_func.equation = (
    lambda popt: f"y = {popt[0]:.3e}*x^5 + {popt[1]:.3e}*x^4 + {popt[2]:.3e}*x^3 + {popt[3]:.3e}*x^2 + {popt[4]:.3f}*x + {popt[5]:.3f}"
)

absolute_shift_func.equation = (
    lambda popt: f"y = {popt[0]:.3f}*|x - {popt[1]:.3f}| + {popt[2]:.3f}"
)

linear_sinusoid_func.equation = (
    lambda popt: f"y = ({popt[0]:.3f}*x + {popt[1]:.3f}) * sin({popt[2]:.3f}*x + {popt[3]:.3f})"
)

quadratic_sinusoid_func.equation = (
    lambda popt: f"y = ({popt[0]:.3f}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}) * sin({popt[3]:.3f}*x + {popt[4]:.3f})"
)

absolute_shift_sinusoid_func.equation = (
    lambda popt: f"y = ({popt[0]:.3f}*|x - {popt[1]:.3f}| + {popt[2]:.3f}) * sin({popt[3]:.3f}*x + {popt[4]:.3f})"
)

linear_abs_sinusoid_func.equation = (
    lambda popt: f"y = ({popt[0]:.3f}*x + {popt[1]:.3f}) * |sin({popt[2]:.3f}*x + {popt[3]:.3f})|"
)

quadratic_abs_sinusoid_func.equation = (
    lambda popt: f"y = ({popt[0]:.3f}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}) * |sin({popt[3]:.3f}*x + {popt[4]:.3f})|"
)

absolute_shift_abs_sinusoid_func.equation = (
    lambda popt: f"y = ({popt[0]:.3f}*|x - {popt[1]:.3f}| + {popt[2]:.3f}) * |sin({popt[3]:.3f}*x + {popt[4]:.3f})|"
)

sigmoid_func.equation = (
    lambda popt: f"y = {popt[0]:.3f} / (1 + exp(-{popt[1]:.3f} * (x - {popt[2]:.3f})))"
)

inv_sigmoid_func.equation = (
    lambda popt: f"x = {popt[2]:.3f} - ln({popt[0]:.3f} / y - 1) / {popt[1]:.3f}"
)


def fit_name_of_func(fit_function):
    fit_name = fit_function.__name__
    if fit_name is not None and fit_name.endswith("_func"):
        fit_name = fit_name[: -len("_func")]
    return fit_name


@torch.no_grad()
def make_fit(values: torch.Tensor, fit_function, exclude_count=None):
    assert len(values.shape) in (1, 2)
    if len(values.shape) == 1:
        x_vals = np.arange(values.shape[0])
        y_vals = utils.to_numpy(values)
    else:
        x_vals = np.tile(np.arange(values.shape[1]), values.shape[0])
        y_vals = utils.to_numpy(values.flatten())

    x_vals_fit, y_vals_fit = x_vals, y_vals
    if exclude_count is not None:
        x_vals_fit, y_vals_fit = (
            x_vals[exclude_count:-exclude_count],
            y_vals[exclude_count:-exclude_count],
        )
    popt, _ = curve_fit(fit_function, x_vals_fit, y_vals_fit)

    residuals = y_vals - fit_function(x_vals, *popt)
    order_indices = np.argsort(x_vals)

    return (
        popt,
        (x_vals, y_vals),
        (x_vals, fit_function(x_vals, *popt)),
        (x_vals[order_indices], residuals[order_indices]),
    )


def make_fit_traces(
    values: torch.Tensor,
    fit_function,
    exclude_count=None,
    fit_equation: Optional[Callable] = None,
    reference_lines: Optional[List[Tuple[str, float]]] = None,
    reference_colors=plotly.colors.qualitative.Dark24,
):
    popt, points, fit, resid = make_fit(
        values, fit_function, exclude_count=exclude_count
    )
    if fit_equation is None:
        fit_equation = fit_function.equation
    assert fit_equation is not None
    if reference_lines is None:
        reference_lines = []
    reference_line_traces = [
        go.Scatter(
            x=np.arange(points[0].shape[0]),
            y=np.full(points[0].shape, val),
            name=name,
            mode="lines",
            line=dict(color=color, dash="dash"),
            hovertemplate=f"{val}<extra>{name}</extra>",
            showlegend=False,
            legendgroup=fit_function.__name__,
        )
        for (name, val), color in zip(
            reference_lines, itertools.cycle(reference_colors)
        )
    ]
    # , size=1
    return (
        popt,
        [
            go.Scatter(
                x=points[0],
                y=points[1],
                name="Data",
                mode="markers",
                marker=dict(color="red", opacity=0.5),
                showlegend=True,
                legendgroup=fit_function.__name__,
            ),
            go.Scatter(
                x=fit[0],
                y=fit[1],
                name=f"Fit: {fit_equation(popt)}",
                mode="lines",
                line=dict(color="blue"),
                showlegend=True,
                legendgroup=fit_function.__name__,
            ),
            go.Scatter(
                x=resid[0],
                y=resid[1],
                name="Residuals",
                mode="markers",
                marker=dict(color="red", opacity=0.5),
                showlegend=False,
            ),
        ],
        reference_line_traces,
    )


def show_fits(
    values: torch.Tensor,
    name: str,
    fit_funcs: Iterable[Callable],
    do_exclusions=True,
    renderer=None,
    show: bool = True,
    **kwargs,
):
    assert len(values.shape) == 1
    fit_funcs = list(fit_funcs)
    fig = make_subplots(
        rows=len(fit_funcs),
        cols=2,
        subplot_titles=[
            title
            for fit_func in fit_funcs
            for title in (f"{fit_name_of_func(fit_func)} Fit", "Residuals")
        ],
    )
    for i, fit_func in enumerate(fit_funcs):
        popt, (points, fit, resid), reference_line_traces = make_fit_traces(
            values, fit_func, exclude_count=None, **kwargs
        )
        fig.add_trace(points, row=i + 1, col=1)
        fig.add_trace(fit, row=i + 1, col=1)
        fig.add_trace(resid, row=i + 1, col=2)
        for trace in reference_line_traces:
            fig.add_trace(trace, row=i + 1, col=1)
    fig.update_layout(
        title=f"{name} Data & Fit",
        legend=dict(
            bgcolor="rgba(255,255,255,0.5)",
            yanchor="middle",
            y=0.5,  # Y=1 anchors the legend to the top of the plot area
            xanchor="left",
            x=0,
        ),
        height=300 * len(fit_funcs) + 100,
    )

    if do_exclusions:
        max_param_count = max(
            [len(signature(fit_func).parameters) for fit_func in fit_funcs]
        )
        frames = [
            go.Frame(
                data=[
                    trace
                    for fit_func in fit_funcs
                    for trace_list in make_fit_traces(
                        values, fit_func, exclude_count=exclude_count, **kwargs
                    )[1:]
                    for trace in trace_list
                ],
                name=(str(exclude_count) if exclude_count is not None else "0"),
            )
            for exclude_count in [None]
            + list(range(1, (values.shape[0] - max_param_count) // 2))
        ]

        fig.frames = frames

        sliders = [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=20),
                    prefix="# End Points to Exclude:",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [frame.name],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        method="animate",
                        label=frame.name,
                    )
                    for frame in fig.frames
                ],
            )
        ]

        fig.update_layout(sliders=sliders)

    if show:
        fig.show(renderer)
