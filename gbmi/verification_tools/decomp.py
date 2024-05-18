from typing import Literal, Optional, Tuple, Union, overload
from functools import reduce
from torch import Tensor
from jaxtyping import Float
import torch
import numpy as np

# from transformer_lens import FactoredMatrix
from gbmi.utils.FactoredMatrix import FactoredMatrix
from gbmi.utils.lowrank import LowRankTensor


@torch.no_grad()
def factor_right_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Union[Float[Tensor, "c"], Float[Tensor, "c n"]],  # noqa: F821, F722
    sanity_check: bool = True,
    show: bool = True,
    checkparams: Optional[dict] = None,
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c n)
    """
    v = v / v.norm(dim=0, keepdim=True)
    if len(v.shape) == 1:
        v = v[:, None]
    assert (
        m.shape[-1] == v.shape[0]
    ), f"m.shape[-1] must match the shape of v ({m.shape[-1]} != {v.shape[0]}, m.shape: {m.shape}, v.shape: {v.shape})"
    v_alt = m @ v
    contrib = LowRankTensor(
        v_alt,
        v.transpose(-2, -1),
        check=sanity_check,
        show=show,
        checkparams=checkparams,
    )
    if sanity_check:
        assert contrib.check(
            torch.stack([(row @ v) @ v.transpose(-2, -1) for row in m], dim=0),
            do_assert=True,
        )
    global gcontrib
    global gm
    gcontrib = contrib
    gm = m
    return contrib, m - contrib


@torch.no_grad()
def factor_left_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Union[Float[Tensor, "r"], Float[Tensor, "n r"]],  # noqa: F821, F722
    sanity_check: bool = True,
    show: bool = True,
    checkparams: Optional[dict] = None,
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c n)
    """
    if len(v.shape) == 1:
        v = v[None, :]
    contrib, resid = factor_right_contribution(
        m.T, v.T, sanity_check=sanity_check, show=show, checkparams=checkparams
    )
    return contrib.T, resid.T


@overload
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Union[Float[Tensor, "r"], Float[Tensor, "n r"]],  # noqa: F821, F722
    *,
    sanity_check: bool = True,
    show: bool = True,
    checkparams: Optional[dict] = None,
    side: Literal["left"] = "left",
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c n)
    """
    ...


@overload
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Union[Float[Tensor, "c"], Float[Tensor, "c n"]],  # noqa: F821, F722
    *,
    sanity_check: bool = True,
    show: bool = True,
    checkparams: Optional[dict] = None,
    side: Literal["right"],
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c n)
    """
    ...


@torch.no_grad()
def factor_contribution(
    m: Float[Tensor, "r c"],  # noqa: F722
    v: Union[
        Float[Tensor, "r"],  # noqa: F821
        Float[Tensor, "n r"],  # noqa: F722
        Float[Tensor, "c"],  # noqa: F821
        Float[Tensor, "c n"],  # noqa: F722
    ],
    *,
    sanity_check: bool = True,
    show: bool = True,
    checkparams: Optional[dict] = None,
    side: Literal["left", "right"] = "left",
) -> Tuple[Float[LowRankTensor, "r c"], Float[Tensor, "r c"]]:  # noqa: F722
    """Returns the contribution of v to m, and the residual
    Complexity: O(r c)
    """
    if side == "left":
        return factor_left_contribution(
            m, v, sanity_check=sanity_check, show=show, checkparams=checkparams
        )
    elif side == "right":
        return factor_right_contribution(
            m, v, sanity_check=sanity_check, show=show, checkparams=checkparams
        )
    else:
        raise ValueError(f"side must be left or right, not {side}")


# %%
@torch.no_grad()
def max_row_diffs_per_dim_2(
    A: Float[Tensor, "... a b"],  # noqa: F722
    B: Float[Tensor, "... b c"],  # noqa: F722
    use_mean_row: bool = False,
) -> Float[Tensor, "... a"]:  # noqa: F722
    r"""Computes the maximum difference between elements in the same row of the product of A and B

    Complexity: O(ab + bc)

    $$\begin{align*}
    &\max_{r,i,j} (AB)_{r,i} - (AB)_{r,j} \\
    &= \max_{r,i,j} \sum_k \left(A_{r,k} B_{k,i} - A_{r,k} B_{k,j}\right) \\
    &= \max_{r,i,j} \sum_k A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
    &\le \max_r \sum_k \max_{i,j} A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
    &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ \min_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
    &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ -\max_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
    &= \max_r \sum_k \left|A_{r,k}\max_{i,j}  \left(B_{k,i} - B_{k,j}\right)\right| \\
    &= \max_r \sum_k \left|A_{r,k}\right|\left|\max_{i,j}  \left(B_{k,i} - B_{k,j}\right)\right| \\
    \end{align*}$$

    If use_mean_row is True, we combine this with the mean+diff trick over r.

    Postconditions:
        \forall r, i, j:
            -return_r <= (AB)_{r,i} - (AB)_{r,j} <= return_r
    """
    max_B_diffs = B.max(dim=-1).values - B.min(dim=-1).values
    if use_mean_row:
        EA = A.mean(dim=-2, keepdim=True)
        EAB = EA @ B
        EAB_diffs = EAB.max(dim=-1).values - EAB.min(dim=-1).values
        A = A - EA
        return EAB_diffs + A.abs() @ max_B_diffs
    else:
        return A.abs() @ max_B_diffs


# %%
@torch.no_grad()
def max_row_diffs_per_dim(*m: Tensor, use_mean_row: bool = False) -> Tensor:
    r"""Computes the maximum difference between elements in the same row of the product of the passed matrices by considering all points to break the product at

    Complexity: O(  \sum_{0 ≤ i < j < len(m) - 1} m[0].shape[-2] * m[i].shape[-1] * m[j].shape[-1]
    Complexity:   + \sum_{0 < i < j ≤ len(m) - 1} m[i].shape[-2] * m[j].shape[-2] * m[-1].shape[-1]
    Complexity:   + \sum_{0 ≤ i < len(m) - 1} m[0].shape[-2] * m[i].shape[-1] + m[i+1].shape[-2] * m[-1].shape[-1])

    Preconditions:
        \forall i: m[i].shape[-1] == m[i + 1].shape[-2]
    Postconditions:
        Define
            M := \prod_i m[i]
        \forall r, i, j:
            -return_r <= M_{r,i} - M_{r,j} <= return_r
    """
    partial_products_l = [m[0]]
    partial_products_r = [m[-1]]
    for ml, mr in zip(m[1:-1], reversed(m[1:-1])):
        partial_products_l.append(partial_products_l[-1] @ ml)
        partial_products_r.append(mr @ partial_products_r[-1])
    max_row_diffs = [
        max_row_diffs_per_dim_2(l, r, use_mean_row=use_mean_row)
        for l, r in zip(partial_products_l, reversed(partial_products_r))
    ]
    # all estimates in max_row_diffs are valid, so we can reduce over them
    max_row_diffs_stacked = torch.stack(max_row_diffs, dim=-1)
    return max_row_diffs_stacked.min(dim=-1).values


@torch.no_grad()
def max_row_diffs_per_dim_no_multipy(
    *m: Tensor, use_mean_row: bool = False, use_mean_row_recursively: bool = False
) -> Tensor:
    r"""Computes the maximum difference between elements in the same row of the product of the passed matrices by using the max row diff trick recursively.

    Complexity: O(\sum_i m[i].shape[-2] * m[i].shape[-1]))

    Preconditions:
        \forall i: m[i].shape[-1] == m[i + 1].shape[-2]
    Postconditions:
        Define
            M := \prod_i m[i]
        \forall r, i, j:
            -return_r <= M_{r,i} - M_{r,j} <= return_r
    """
    if len(m) == 1:
        return m[0].max(dim=-1).values - m[0].min(dim=-1).values
    A, m = m[0], m[1:]
    if use_mean_row:
        EA = A.mean(dim=-2, keepdim=True)
        EAm = reduce(torch.matmul, m, EA)
        EAm_diffs = EAm.max(dim=-1).values - EAm.min(dim=-1).values
        A = A - EA
    else:
        EAm_diffs = 0

    return EAm_diffs + A.abs() @ max_row_diffs_per_dim_no_multipy(
        *m,
        use_mean_row=use_mean_row_recursively,
        use_mean_row_recursively=use_mean_row_recursively,
    )


# %%
@torch.no_grad()
def bound_max_row_diff_by_SVD(
    *matrices: Tensor,
) -> Tuple[Float[Tensor, ""], Tuple[Tensor, ...]]:  # noqa: F722
    r"""
    Let M denote the product of the elements of `matrices` (under matrix multiplication)

    Complexity: max_{a, b s.t. \exists m\in matrices, m.shape = (a, b)} O(a b min(a, b))

    We compute an upper bound on the difference between elements in the same row of the product of the matrices:
    Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt{2} \sigma_1(M)$.
    This is the value we return, computing an upper bound on the first singular value by multiplying the first singular values of each matrix.

    Preconditions:
        the matrices in `matrices` can be multiplied
    Postconditions:
        forall r.
          max_{i,j} M_{r, i} - M_{r, j} <= return[0]
        return[1] == matrices
    """
    # take the product of the first singular values in each matrix to get a bound on the singular value of the product
    prod_max_singular = torch.tensor(
        [torch.linalg.matrix_norm(m, ord=2) for m in matrices]
    ).prod()
    # since \sigma_1(M) = \sup_x \| M x \| / \|x\|, considering vectorswith one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is sqrt(2) * \sigma_1(M)
    return (
        prod_max_singular * np.sqrt(2),
        matrices,
    )


def split_SVD(
    *matrices: Tensor,
    n_principle_components: int = 1,
    sanity_check: bool = False,
    checkparams: Optional[dict] = None,
) -> Tuple[LowRankTensor, LowRankTensor]:
    """Splits the product of matrices matrices into a low-rank factored matrix M consisting of the first n_principle_components singular values and vectors, and a residual matrix R"""
    assert len(matrices) > 0
    if len(matrices) == 1:
        A = matrices[0]
        return LowRankTensor(
            A, torch.eye(A.shape[-1]).to(A), check=sanity_check, checkparams=checkparams
        ), LowRankTensor(
            torch.zeros(*A.shape[:-1], 1).to(A),
            torch.zeros(*A.shape[:-2], 1, A.shape[-1]).to(A),
            check=sanity_check,
            checkparams=checkparams,
        )
    A, B, ms = matrices[0], matrices[1], matrices[2:]
    AB = FactoredMatrix(A, B)
    m = reduce(FactoredMatrix.__matmul__, ms, AB)  # type: ignore
    U, S, Vh = m.svd()
    A, B = m.A, m.B
    U = U[..., :n_principle_components]
    S = S[..., :n_principle_components]
    Vh = Vh[..., :n_principle_components]
    A0, Ar = factor_left_contribution(
        A, U.transpose(-2, -1), sanity_check=sanity_check, checkparams=checkparams
    )
    B0, Br = factor_right_contribution(
        B, Vh, sanity_check=sanity_check, checkparams=checkparams
    )
    return A0 @ B0, LowRankTensor(Ar, Br, check=sanity_check, checkparams=checkparams)
