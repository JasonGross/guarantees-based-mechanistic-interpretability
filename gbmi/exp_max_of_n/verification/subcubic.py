from typing import Union, Optional, Tuple
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from transformer_lens import HookedTransformer
from gbmi.utils.lowrank import LowRankTensor
from gbmi.verification_tools.decomp import split_SVD
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig


@torch.no_grad()
def decompose_EQKE_error(
    model: HookedTransformer,
    *,
    sanity_check: bool = True,
    atol: float = 1e-4,
    approximation_rank: int = 1,
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
) -> Tuple[
    Float[LowRankTensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    Tuple[
        Union[Float[Tensor, ""], Float[Tensor, "d_vocab_q"]],  # noqa: F722, F821
        Tuple[
            Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
            Float[Tensor, "d_model d_vocab_k"],  # noqa: F722
        ],
    ],
]:
    r"""
    Returns:
        (EQKE_query_key, EQKE_pos_err, (remaining_error_upper_bound, two matrices whose product is the exact remaining error))
    where
        EQKE_query_key is the rank approximation_rank approximation of the query-key contribution to the EQKE matrix
        EQKE_pos_err is the contribution of the position embeddings to the error
        remaining_error_upper_bound is a bound on the maximum difference between two elements in the same row of the remaining error of EQKE, and may be either a float or a tensor indexed by query token, depending on the configuration of tricks

    Note that EQKE is actually computed as (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0, keepdim=True)).T

    Complexity: O(d_vocab * (d_vocab + d_model * n_ctx) + d_vocab * d_model^2)

    The d_model^2 term comes from having to do low-rank SVD

    Preconditions:
        (none)
    Postconditions:
        Define err := EQKE - EQKE_query_key
        Then we guarantee:
        . max_{i,j} err_{r, i} - err_{r, j} <= remaining_error_upper_bound
        . EQKE_pos_err[p] := (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_pos[p] - W_pos.mean(dim=0, keepdim=True)).T

    We compute as follows:
    $$
    \begin{align*}
    \widetilde{E_q} & := W_E + W_\text{pos}[-1] \\
    \widetilde{E_k} & := W_E + \mathbb{E}_p W_\text{pos}[p] \\
    \text{EQKE}_p
    & := \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \widetilde{E_q}W_QW_K^T(W_{\text{pos}}[p] - \mathbb{E}_{p'} W_\text{pos}[p'])^T \\
    & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \text{EQKE\_pos\_err}
    \end{align*}
    $$

    TODO add description of low-rank SVD

    In the default case, we use the svd method for the final component:
    We compute an upper bound on what the final component can contribute to differences in elements in the same row:
    Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt{2} \sigma_1(M)$.
    This is the value we return, computing an upper bound on the first singular value by multiplying the first singular values of each matrix.

    If tricks.attention_error_handling is "max_diff", then we instead compute the maximum difference in a more clever way.
    $$\begin{align*}
    &\max_{r,i,j} (AB)_{r,i} - (AB)_{r,j} \\
    &= \max_{r,i,j} \sum_k \left(A_{r,k} B_{k,i} - A_{r,k} B_{k,j}\right) \\
    &= \max_{r,i,j} \sum_k A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
    &\le \max_r \sum_k \max_{i,j} A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
    &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ \min_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
    &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ -\max_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
    &= \max_r \sum_k \left|A_{r,k}\max_{i,j}  \left(B_{k,i} - B_{k,j}\right)\right| \\
    &= \max_r \sum_k \left|A_{r,k}\right|\max_{i,j}  \left(B_{k,i} - B_{k,j}\right) \\
    \end{align*}$$
    """
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )

    W_E_pos_k = W_E + W_pos.mean(dim=0)[None, :]
    W_pos_err = W_pos - W_pos.mean(dim=0)[None, :]
    W_E_pos_q = W_E + W_pos[-1][None, :]
    EQKE_pos_err = W_E_pos_q @ (W_Q[0, 0] @ (W_K[0, 0].T @ W_pos_err.T))
    EQKE_query_key, EQKE_err = split_SVD(
        W_E_pos_q,
        W_Q[0, 0],
        W_K[0, 0].T,
        W_E_pos_k.T,
        n_principle_components=approximation_rank,
        sanity_check=sanity_check,
        checkparams=dict(atol=atol),
    )
    return (
        EQKE_query_key,
        EQKE_pos_err,
        (
            tricks.bound_attention_error(EQKE_err.A, EQKE_err.B),
            (EQKE_err.A, EQKE_err.B),
        ),
    )
