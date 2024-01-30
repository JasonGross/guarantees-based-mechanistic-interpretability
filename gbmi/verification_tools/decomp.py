from typing import Literal, Tuple, Union, overload
from torch import Tensor
from jaxtyping import Float
import torch
from gbmi.utils.lowrank import LowRankTensor


@torch.no_grad()
def factor_right_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "c"],  # noqa: F821
    sanity_check: bool = True,
    show: bool = True,
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    v = v / v.norm(dim=-1, keepdim=True)
    assert (
        m.shape[-1] == v.shape[-1]
    ), f"m.shape[-1] must match the shape of v ({m.shape[-1]} != {v.shape[-1]}, m.shape: {m.shape}, v.shape: {v.shape})"
    v_alt = m @ v
    contrib = LowRankTensor(
        v_alt[..., None], v[..., None, :], check=sanity_check, show=show
    )
    if sanity_check:
        assert contrib.check(torch.stack([v * (row @ v) for row in m], dim=0))
    return contrib, m - contrib


@torch.no_grad()
def factor_left_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "r"],  # noqa: F821
    sanity_check: bool = True,
    show: bool = True,
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    contrib, resid = factor_right_contribution(
        m.T, v, sanity_check=sanity_check, show=show
    )
    return contrib.T, resid.T


@overload
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "r"],  # noqa: F821
    *,
    sanity_check: bool = True,
    show: bool = True,
    side: Literal["left"] = "left",
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    ...


@overload
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "c"],  # noqa: F821
    *,
    sanity_check: bool = True,
    show: bool = True,
    side: Literal["right"],
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    ...


@torch.no_grad()
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Union[Float[Tensor, "r"], Float[Tensor, "c"]],  # noqa: F821
    *,
    sanity_check: bool = True,
    show: bool = True,
    side: Literal["left", "right"] = "left",
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    if side == "left":
        return factor_left_contribution(m, v, sanity_check=sanity_check)
    elif side == "right":
        return factor_right_contribution(m, v, sanity_check=sanity_check)
    else:
        raise ValueError(f"side must be left or right, not {side}")
