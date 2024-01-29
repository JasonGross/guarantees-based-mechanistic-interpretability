from typing import Literal, Tuple, Union, overload
from torch import Tensor
from jaxtyping import Float
import torch


@torch.no_grad()
def factor_right_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "c"],  # noqa: F821
    sanity_check: bool = True,
) -> Tuple[
    Float[Tensor, "r"],  # noqa: F821
    Tuple[Float[Tensor, "r c"], Float[Tensor, "r c"]],  # noqa: F722
]:
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    v = v / v.norm(dim=-1, keepdim=True)
    assert (
        m.shape[-1] == v.shape[-1]
    ), f"m.shape[-1] must match the shape of v ({m.shape[-1]} != {v.shape[-1]}, m.shape: {m.shape}, v.shape: {v.shape})"
    v_alt = m @ v
    contrib = v_alt[..., None] @ v[..., None, :]
    if sanity_check:
        contrib_alt = torch.stack([v * (row @ v) for row in m], dim=0)
        assert torch.allclose(contrib, contrib_alt)
    return v_alt, (contrib, m - contrib)


@torch.no_grad()
def factor_left_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "r"],  # noqa: F821
    sanity_check: bool = True,
) -> Tuple[
    Float[Tensor, "c"],  # noqa: F821
    Tuple[Float[Tensor, "r c"], Float[Tensor, "r c"]],  # noqa: F722
]:
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    v_alt, (contrib, resid) = factor_right_contribution(
        m.T, v, sanity_check=sanity_check
    )
    return v_alt, (contrib.T, resid.T)


@overload
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Float[Tensor, "r"],  # noqa: F821
    *,
    sanity_check: bool = True,
    side: Literal["left"] = "left",
) -> Tuple[
    Float[Tensor, "c"],  # noqa: F821
    Tuple[Float[Tensor, "r c"], Float[Tensor, "r c"]],  # noqa: F722
]:
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
    side: Literal["right"],
) -> Tuple[
    Float[Tensor, "r"],  # noqa: F821
    Tuple[Float[Tensor, "r c"], Float[Tensor, "r c"]],  # noqa: F722
]:
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
    side: Literal["left", "right"] = "left",
) -> Tuple[
    Union[Float[Tensor, "c"], Float[Tensor, "r"]],  # noqa: F821
    Tuple[Float[Tensor, "r c"], Float[Tensor, "r c"]],  # noqa: F722
]:
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    if side == "left":
        return factor_left_contribution(m, v, sanity_check=sanity_check)
    elif side == "right":
        return factor_right_contribution(m, v, sanity_check=sanity_check)
    else:
        raise ValueError(f"side must be left or right, not {side}")
