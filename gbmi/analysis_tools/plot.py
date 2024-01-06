import numpy as np
from matplotlib import pyplot as plt
from plotly import express as px
from scipy.optimize import curve_fit
from transformer_lens import utils as utils

import gbmi.analysis_tools
from gbmi.analysis_tools.fit import linear_func
from gbmi.analysis_tools.utils import pm_range, pm_mean_std


def imshow(tensor, renderer=None, xaxis="", yaxis="", colorscale="RdBu", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale=colorscale,
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(
    tensor,
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


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


def hist(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.histogram(
        utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs
    ).show(renderer)


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
        gbmi.analysis.plot.scatter(x_vals, y_vals, label="Data", alpha=0.5, s=1)
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
        gbmi.analysis.plot.scatter(
            x_vals[order_indices], residuals[order_indices], c="b", alpha=0.5
        )
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
