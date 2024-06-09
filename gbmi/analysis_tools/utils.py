from typing import Optional, Literal, Union, Sequence, Collection
import numpy as np
import scipy.stats as stats
from torch import Tensor
from jaxtyping import Integer, Float
import torch


def _item(obj):
    try:
        return obj.item()
    except AttributeError:
        return obj


def pm_round(
    mean: float,
    std: float,
    extra_digits: int = 1,
    total_digits: Optional[int] = None,
    format_specifier: Literal["f", "e"] = "f",
    sep: str = " ± ",
) -> str:
    if total_digits is None:
        if np.isnan(std):
            return f"{mean}{sep}{std}"
        else:
            total_digits = int(1 + extra_digits - np.log10(std))
    if total_digits < 0:
        mean, std = round(mean, total_digits), round(std, total_digits)
        total_digits = 0
    return f"{mean:.{total_digits}{format_specifier}}{sep}{std:.{total_digits}{format_specifier}}"


def pm_range(values, round: bool = True):
    maxv, minv = _item(values.max()), _item(values.min())
    mid, half_range = (maxv + minv) / 2.0, (maxv - minv) / 2.0
    return pm_round(mid, half_range) if round else f"{mid} ± {half_range}"


def pm_mean_std(values, round: bool = True):
    mean, std = _item(values.mean()), _item(values.std())
    return pm_round(mean, std) if round else f"{mean} ± {std}"


def center_by_mid_range(
    tensor: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    maxv, minv = (
        tensor.max(dim=dim, keepdim=True).values,
        tensor.min(dim=dim, keepdim=True).values,
    )
    return tensor - (maxv + minv) / 2.0


def make_local_tqdm(tqdm):
    if tqdm is None:
        return lambda arg, **kwargs: arg
    else:
        return tqdm


def replace_nans_with_row_max(tensor):
    # Step 1: Identify the nan values
    nan_mask = torch.isnan(tensor)

    # Step 2: Compute the maximum value for each row, ignoring nans
    non_nan_tensor = torch.where(
        nan_mask, torch.tensor(float("-inf")).to(tensor.device), tensor
    )
    row_max, _ = torch.max(non_nan_tensor, dim=1, keepdim=True)

    # Replace nan with the max value of the respective row
    tensor[nan_mask] = row_max.expand_as(tensor)[nan_mask]

    return tensor


@torch.no_grad()
def layernorm_noscale(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(axis=-1, keepdim=True)


@torch.no_grad()
def layernorm_scales(
    x: torch.Tensor, eps: float = 1e-5, recip: bool = True
) -> torch.Tensor:
    x = layernorm_noscale(x)
    scale = (x.pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    if recip:
        scale = 1 / scale
    return scale


# from https://stackoverflow.com/a/29677616/377022
@torch.no_grad()
def weighted_quantile(
    values: Float[Union[Tensor, np.ndarray], "..."],
    quantiles: Float[Union[Tensor, np.ndarray], "..."],
    sample_weight: Optional[Float[Union[Tensor, np.ndarray], "..."]] = None,
    values_sorted: bool = False,
    old_style: bool = False,
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    if isinstance(values, Tensor):
        values = values.numpy()
    if isinstance(quantiles, Tensor):
        quantiles = quantiles.numpy()
    if isinstance(sample_weight, Tensor):
        sample_weight = sample_weight.numpy()
    values = np.array(values)  # type: ignore
    quantiles = np.array(quantiles)  # type: ignore
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.array(sample_weight)
    assert (quantiles >= 0).all() and (
        quantiles <= 1
    ).all(), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = values.argsort()
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= sample_weight.sum()
    return np.interp(quantiles, weighted_quantiles, values)


def data_summary_percentiles():
    s = twenty_five_percent_in_std_dev = stats.norm.ppf(0.75) * 2
    percentiles = stats.norm.cdf([-3 * s, -2 * s, -s, 0, s, 2 * s, 3 * s])
    percentile_names = [
        "LowerWhiskerBottomEnd",
        "LowerWhiskerCrosshatch",
        "QuartileOne",
        "Median",
        "QuartileThree",
        "UpperWhiskerCrosshatch",
        "UpperWhiskerTopEnd",
    ]
    return percentile_names, percentiles


def data_summary(
    data, prefix: str = "", float_postfix: str = "Float", int_postfix: str = ""
):
    if isinstance(data, dict):
        keys = list(data.keys())
        values = [data[k] for k in keys]
    else:
        keys = None
        values = data
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif not isinstance(values, np.ndarray):
        values = np.array(values)  # turn to float

    wf = lambda k: f"{prefix}{k}{float_postfix}"

    result = {
        f"{prefix}Len{int_postfix}": len(values.flatten()),
        wf("Min"): values.min(),
        wf("Max"): values.max(),
    }
    values = values + 0.0  # floatify
    result |= {
        wf("Mean"): values.mean(),
        wf("StdDev"): values.std(),
        wf("SqrMean"): (values**2).mean(),
    }

    percentile_names, percentiles = data_summary_percentiles()
    percentile_values = np.percentile(values, percentiles)

    result.update({wf(pct): v for pct, v in zip(percentile_names, percentile_values)})

    if keys is not None:
        closest_keys = {}

        def find_closest_key(value):
            return keys[np.argmin(np.abs(values - value))]

        closest_keys.update(
            {
                f"{prefix}MeanKey": find_closest_key(values.mean()),
                f"{prefix}MinKey": find_closest_key(values.min()),
                f"{prefix}MaxKey": find_closest_key(values.max()),
            }
        )

        for pct, value in zip(percentile_names, percentile_values):
            closest_keys[f"{prefix}{pct}Key"] = find_closest_key(value)

        result.update(closest_keys)

    return result
