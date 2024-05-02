from typing import Callable, Union, Optional, Tuple
from functools import reduce
import time
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from transformer_lens import HookedTransformer
from gbmi.utils.lowrank import LowRankTensor
from gbmi.verification_tools.decomp import split_SVD
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
import gbmi.exp_max_of_n.verification.quadratic as quadratic
from gbmi.verification_tools.utils import complexity_of
from gbmi.verification_tools.l1h1 import (
    all_EVOU,
    all_PVOU,
    all_EVOU_nocache,
    all_PVOU_nocache,
)


@torch.no_grad()
def decompose_EQKE_error(
    model: HookedTransformer,
    *,
    key_direction: Optional[Tensor] = None,
    query_direction: Optional[Tensor] = None,
    second_key_direction: Optional[Tensor] = None,
    second_query_direction: Optional[Tensor] = None,
    W_Q_U: Optional[Tensor] = None,
    W_K_U: Optional[Tensor] = None,
    layer: int = 0,
    head: int = 0,
    sanity_check: bool = True,
    atol: float = 1e-4,
    approximation_rank: int = 1,
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
) -> Tuple[
    Union[
        Float[LowRankTensor, "d_vocab_q d_vocab_k"],  # noqa: F722
        Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    ],
    Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    Tuple[
        Union[Float[Tensor, ""], Float[Tensor, "d_vocab_q"]],  # noqa: F722, F821
        Union[
            Tuple[
                Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
                Float[Tensor, "d_model d_vocab_k"],  # noqa: F722
            ],
            Tuple[
                Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
                Float[Tensor, "d_model d_model"],  # noqa: F722
                Float[Tensor, "d_model d_model"],  # noqa: F722
                Float[Tensor, "d_model d_vocab_k"],  # noqa: F722
            ],
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

    If tricks.attention_error_handling is "max_diff_subproduct" or "mean+max_diff_subproduct", then we use quadratic.decompose_EQKE_error_quadratic to compute the low-rank factorization.
    """
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q[layer, head],
        model.W_K[layer, head],
    )

    W_E_pos_k = W_E + W_pos.mean(dim=0)[None, :]
    W_pos_err = W_pos - W_pos.mean(dim=0)[None, :]
    W_E_pos_q = W_E + W_pos[-1][None, :]
    EQKE_pos_err = W_E_pos_q @ (W_Q @ (W_K.T @ W_pos_err.T))

    if "subproduct" in tricks.attention_error_handling:
        assert (
            key_direction is not None
        ), f"key_direction must be provided if using {tricks.attention_error_handling}"
        assert (
            query_direction is not None
        ), f"query_direction must be provided if using {tricks.attention_error_handling}"
        assert (
            second_key_direction is not None
        ), f"second_key_direction must be provided if using {tricks.attention_error_handling}"
        assert (
            second_query_direction is not None
        ), f"second_query_direction must be provided if using {tricks.attention_error_handling}"
        assert (
            W_Q_U is not None
        ), f"W_Q_U must be provided if using {tricks.attention_error_handling}"
        assert (
            W_K_U is not None
        ), f"W_K_U must be provided if using {tricks.attention_error_handling}"
        (
            (EQKE_query_key, err_accumulator),
            EQKE_pos_err,
            (_err_upper_bound, err_matrices),
        ) = quadratic.decompose_EQKE_error_quadratic(
            model,
            key_direction=key_direction,
            query_direction=query_direction,
            second_key_direction=second_key_direction,
            second_query_direction=second_query_direction,
            W_Q_U=W_Q_U,
            W_K_U=W_K_U,
            sanity_check=True,
            atol=atol,
            layer=layer,
            head=head,
        )
        EQKE_query_key = EQKE_query_key + err_accumulator
    else:
        EQKE_query_key, EQKE_err = split_SVD(
            W_E_pos_q,
            W_Q,
            W_K.T,
            W_E_pos_k.T,
            n_principle_components=approximation_rank,
            sanity_check=sanity_check,
            checkparams=dict(atol=atol),
        )
        err_matrices = (EQKE_err.A, EQKE_err.B)

    return (
        EQKE_query_key,
        EQKE_pos_err,
        (
            tricks.bound_attention_error(*err_matrices),
            err_matrices,
        ),
    )


def verify_proof(
    model: HookedTransformer,
    *,
    use_exact_EQKE: bool,
    W_EP_direction: Optional[Float[Tensor, "d_model"]],  # noqa: F821
    key_direction: Tensor,
    query_direction: Tensor,
    second_key_direction: Tensor,
    second_query_direction: Tensor,
    W_Q_U: Tensor,
    W_K_U: Tensor,
    min_gaps: Integer[
        Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"  # noqa: F722
    ],
    layer: int = 0,
    head: int = 0,
    atol: float = 1e-4,
    approximation_rank: int = 1,
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
    print_complexity: Union[bool, Callable[[str], None]] = True,
    print_results: Union[bool, Callable[[str], None]] = True,
    sanity_check: bool = True,
):
    if isinstance(print_complexity, bool):
        print_complexity = print if print_complexity else lambda x: None
    if isinstance(print_results, bool):
        print_results = print if print_results else lambda x: None

    prooftimes = []

    def add_time(f, *args, **kwargs):
        starttime = time.time()
        result = f(*args, **kwargs)
        prooftimes.append(time.time() - starttime)
        return result

    (
        EQKE_query_key,
        EQKE_pos_err,
        (err_upper_bound, err_matrices),
    ) = add_time(
        decompose_EQKE_error,
        model,
        key_direction=key_direction,
        query_direction=query_direction,
        second_key_direction=second_key_direction,
        second_query_direction=second_query_direction,
        W_Q_U=W_Q_U,
        W_K_U=W_K_U,
        layer=layer,
        head=head,
        sanity_check=sanity_check,
        atol=atol,
        approximation_rank=approximation_rank,
        tricks=tricks,
    )
    print_complexity(
        f"Complexity of decompose_EQKE_error: {complexity_of(decompose_EQKE_error)}"
    )

    if use_exact_EQKE:
        print_complexity(f"Complexity of using exact EQKE: O(d_vocab^2 d_model)")
        err_exact = add_time(reduce, torch.matmul, err_matrices)
        cur_EQKE = add_time(lambda: EQKE_query_key + err_exact)
        EQKE_err_upper_bound = torch.tensor(0)
    else:
        print_complexity(f"Complexity of using approximate EQKE: O(d_vocab^2)")
        cur_EQKE = EQKE_query_key + 0.0
        EQKE_err_upper_bound = err_upper_bound

    extreme_right_attention = add_time(
        quadratic.compute_extreme_right_attention_quadratic,
        cur_EQKE,
        min_gap=min_gaps,
    )

    print_complexity(
        f"Complexity of compute_extreme_right_attention_quadratic: {complexity_of(quadratic.compute_extreme_right_attention_quadratic)}"
    )  # O(d_vocab^2)
    if not isinstance(EQKE_err_upper_bound, Tensor):
        EQKE_err_upper_bound = torch.tensor(EQKE_err_upper_bound)
    if EQKE_err_upper_bound.ndim < 1:
        EQKE_err_upper_bound = EQKE_err_upper_bound[None]
    min_right_attention = extreme_right_attention[0]
    print_results(
        str(
            (
                (min_right_attention > EQKE_err_upper_bound[:, None, None])[
                    ~min_right_attention.isnan()
                ]
            )
            .sum()
            .item()
        )
    )

    def adjust_extreme_right_attention():
        extreme_right_attention_adjusted = extreme_right_attention.clone()
        extreme_right_attention_adjusted[0] -= EQKE_err_upper_bound[:, None, None]
        extreme_right_attention_adjusted[1] += EQKE_err_upper_bound[:, None, None]
        return extreme_right_attention_adjusted

    extreme_right_attention_adjusted = add_time(adjust_extreme_right_attention)
    extreme_right_attention_softmaxed = add_time(
        quadratic.compute_extreme_softmaxed_right_attention_quadratic,
        extreme_right_attention_adjusted,
        EQKE_pos_err,
        min_gap=min_gaps,
        attn_scale=model.blocks[0].attn.attn_scale,
    )
    print_complexity(
        f"Complexity of compute_extreme_softmaxed_right_attention: {complexity_of(quadratic.compute_extreme_softmaxed_right_attention_quadratic)}"
    )  # O(d_vocab^2 * n_ctx^2)
    EVOU: Float[Tensor, "d_vocab d_vocab_out"] = add_time(  # noqa: F722
        all_EVOU_nocache, model
    )
    print_complexity(
        f"Complexity of EVOU: {complexity_of(all_EVOU)}"
    )  # O(d_vocab^2 * d_model)
    PVOU: Float[Tensor, "n_ctx d_vocab_out"] = add_time(  # noqa: F722
        all_PVOU_nocache, model
    )
    print_complexity(
        f"Complexity of PVOU: {complexity_of(all_PVOU)}"
    )  # O(n_ctx * d_vocab * d_model)
    W_EP: Float[Tensor, "d_vocab_q d_model"] = add_time(  # noqa: F722
        lambda: model.W_E + model.W_pos.mean(dim=0, keepdim=True)
    )
    print_complexity(f"Complexity of W_EP: O((d_vocab + n_ctx) * d_model)")

    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ] = add_time(
        quadratic.compute_largest_wrong_logit_quadratic,
        extreme_right_attention_softmaxed,
        W_EP=W_EP,
        W_U=model.W_U,
        EVOU=EVOU,
        PVOU=PVOU,
        min_gap=min_gaps,
        W_EP_direction=W_EP_direction,
        tricks=tricks,
    )
    print_complexity(
        f"Complexity of compute_largest_wrong_logit_quadratic: {complexity_of(quadratic.compute_largest_wrong_logit_quadratic)}"
    )  # O(d_vocab^2 * n_ctx^2)
    accuracy_bound, (
        correct_count,
        total_sequences,
    ) = add_time(
        quadratic.compute_accuracy_lower_bound_from,
        largest_wrong_logit,
        min_gap=min_gaps,
    )
    print_results(
        f"Accuracy lower bound: {accuracy_bound} ({correct_count} correct sequences of {total_sequences})"
    )

    prooftime = sum(prooftimes)
    print_results(f"Subcubic Proof time: {prooftime}s")

    left_behind = quadratic.count_unaccounted_for_by_gap(min_gaps, collapse_n_ctx=False)
    print_results(
        f"We leave on the floor {left_behind} sequences ({left_behind / total_sequences:.2%})"
    )

    return {
        "err_upper_bound": err_upper_bound,
        "largest_wrong_logit": largest_wrong_logit,
        "accuracy_lower_bound": accuracy_bound,
        "correct_count_lower_bound": correct_count,
        "total_sequences": total_sequences,
        "prooftime": prooftime,
        "left_behind": left_behind,
    }
