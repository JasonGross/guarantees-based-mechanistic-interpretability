from typing import Optional

import torch


def pm_range(values):
    return f"{(values.max().item() + values.min().item()) / 2.0} ± {(values.max().item() - values.min().item()) / 2.0}"


def pm_mean_std(values):
    return f"{values.mean().item()} ± {values.std().item()}"


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
