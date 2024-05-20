from typing import Optional, Literal, Union, Sequence, Collection
import numpy as np
from torch import Tensor
from jaxtyping import Integer, Float
import torch


def pm_round(
    mean: float,
    std: float,
    extra_digits: int = 1,
    total_digits: Optional[int] = None,
    format_specifier: Literal["f", "e"] = "f",
    sep: str = " ± ",
) -> str:
    if total_digits is None:
        total_digits = int(1 + extra_digits - np.log10(std))
    return f"{mean:.{total_digits}{format_specifier}}{sep}{std:.{total_digits}{format_specifier}}"


def pm_range(values, round: bool = True):
    mid, half_range = (values.max().item() + values.min().item()) / 2.0, (
        values.max().item() - values.min().item()
    ) / 2.0
    return pm_round(mid, half_range) if round else f"{mid} ± {half_range}"


def pm_mean_std(values, round: bool = True):
    mean, std = values.mean().item(), values.std().item()
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
