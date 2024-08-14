from typing import Union, Optional
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from gbmi.exp_max_of_n.analysis import (
    find_second_singular_contributions,
    find_size_and_query_direction,
)
from gbmi.analysis_tools.decomp import split_svd_contributions
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.exp_max_of_n.verification.quadratic import (
    compute_extreme_right_attention_quadratic,
    compute_extreme_softmaxed_right_attention_quadratic,
    compute_largest_wrong_logit_quadratic,
    decompose_EQKE_error_quadratic,
)
from gbmi.utils import compress_int_tensor


@torch.no_grad()
def find_min_gaps(
    *,
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    EQKE_err_upper_bound: Union[
        float, Float[Tensor, ""], Float[Tensor, "d_vocab_q"]  # noqa: F722, F821
    ],
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    attn_scale: Union[Float[Tensor, ""], float],  # noqa: F722
    position: Optional[int] = None,
    leave: Optional[bool] = None,
    desc: Optional[str] = None,
    pbar: Optional[tqdm] = None,
    **compute_largest_wrong_logit_quadratic_kwargs,
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    """
    Run the argument across all possible min_gaps, and return the min_gap that works for each query token and max token.

    Since here we are finding the argument/proof rather than verifying it, the complexity does not matter.
    """
    d_vocab_q, d_vocab_k = EQKE.shape
    _d_vocab_q, n_ctx = EQKE_pos_err.shape
    min_gaps = torch.ones((d_vocab_q, d_vocab_k, n_ctx), dtype=torch.long)
    if not isinstance(EQKE_err_upper_bound, Tensor):
        EQKE_err_upper_bound = torch.tensor(EQKE_err_upper_bound)
    if EQKE_err_upper_bound.ndim < 1:
        EQKE_err_upper_bound = EQKE_err_upper_bound[None]
    min_gap_list = list(reversed(range(1, d_vocab_k)))
    if pbar is None:
        min_gap_list = tqdm(
            min_gap_list,
            position=position,
            leave=leave,
            desc=desc,
        )
    for min_gap in min_gap_list:
        extreme_right_attention: Float[
            Tensor, "minmax=2 d_vocab_q d_vocab_max n_ctx_copies_nonmax"  # noqa: F722
        ] = compute_extreme_right_attention_quadratic(EQKE, min_gap=min_gap)
        extreme_right_attention_adjusted = extreme_right_attention.clone()
        extreme_right_attention_adjusted[0] -= EQKE_err_upper_bound[:, None, None]
        extreme_right_attention_adjusted[1] += EQKE_err_upper_bound[:, None, None]
        extreme_right_attention_softmaxed = (
            compute_extreme_softmaxed_right_attention_quadratic(
                extreme_right_attention_adjusted,
                EQKE_pos_err,
                min_gap=min_gap,
                attn_scale=attn_scale,
            )
        )
        largest_wrong_logit = compute_largest_wrong_logit_quadratic(
            extreme_right_attention_softmaxed,
            min_gap=min_gap,
            **compute_largest_wrong_logit_quadratic_kwargs,
        )
        # if the largest wrong logit is negative, then this gap works
        min_gaps[largest_wrong_logit < 0] = min_gap

        if pbar is not None:
            pbar.update(1)

    # we compact the tensor to take up less space on disk, since it's pretty big
    return compress_int_tensor(min_gaps, allow_negative=False)


@torch.no_grad()
def compress_min_gaps_over_query(
    min_gaps: Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]  # noqa: F722
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    """We don't have the compute budget to treat min gaps separately for each (query, max, copies) triplet.  So we compress over the query dimension.  We do this by computing how many sequences would succeed for each choice of gap (all the queries with gap <= the given gap) and how many sequences we pick up (max - gap) ** (num copies nonmax) * (nctx - 1 choose num copies nonmax - 1).
    The (nctx - 1 choose num copies nonmax - 1) factor is the same and doesn't impact sorting order, so we can ignore it.
    """
    sorted_min_gaps = min_gaps.sort(dim=0).values
    _, d_vocab, n_ctx = sorted_min_gaps.shape
    num_choices = (
        torch.arange(d_vocab, device=min_gaps.device)[None, :, None] - sorted_min_gaps
    ).clamp(min=0)
    num_sequences = num_choices ** (
        torch.arange(n_ctx, device=min_gaps.device)[None, None, :] - 1
    ).clamp(min=0)
    num_queries = (1 + torch.arange(d_vocab, device=min_gaps.device))[:, None, None]
    num_sequences_including_queries = num_sequences * num_queries
    return sorted_min_gaps.gather(
        0, num_sequences_including_queries.argmax(dim=0).unsqueeze(0)
    ).squeeze(0)


@torch.no_grad()
def W_EP_direction_for_tricks_kwargs(model: HookedTransformer):
    W_EP: Float[Tensor, "d_vocab d_model"] = model.W_E + model.W_pos.mean(  # noqa: F722
        dim=0, keepdim=True
    )
    return {"W_EP": W_EP, "W_U": model.W_U}


@torch.no_grad()
def W_EP_direction_for_tricks(
    *,
    W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
    W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
    tricks: Optional[LargestWrongLogitQuadraticConfig] = None,
) -> Optional[Float[Tensor, "d_model"]]:  # noqa F722
    if (
        tricks is None or tricks.EUPU_handling == "svd_query+max_diff"
    ):  # the only one that makes use of the direction
        U, _, Vh = torch.linalg.svd(W_EP @ W_U)
        W_EP_svd_query = U[:, 0] @ W_EP
        W_EP_mean_query = W_EP.mean(dim=0)
        if ((W_EP - W_EP_svd_query) @ W_U).norm(dim=-1).mean() > (
            (W_EP + W_EP_svd_query) @ W_U
        ).norm(dim=-1).mean():
            # svd got the sign wrong :-/
            W_EP_svd_query = -W_EP_svd_query
        return W_EP_svd_query
    return None


def find_EKQE_error_directions(
    model: HookedTransformer, *, layer: int = 0, head: int = 0
):
    (
        size_direction,
        query_direction,
        size_query_singular_value,
    ), _ = find_size_and_query_direction(model)
    (second_key_direction, second_key_singular_value), (
        second_query_direction,
        second_query_singular_value,
    ) = find_second_singular_contributions(model, size_direction, query_direction)
    (W_Q_U, W_Q_S, W_Q_Vh), (W_Q_contrib, W_Q_err) = split_svd_contributions(
        model.W_Q[layer, head]
    )
    (W_K_U, W_K_S, W_K_Vh), (W_K_contrib, W_K_err) = split_svd_contributions(
        model.W_K[layer, head]
    )
    return {
        "key_direction": size_direction,
        "query_direction": query_direction,
        "second_key_direction": second_key_direction,
        "second_query_direction": second_query_direction,
        "W_Q_U": W_Q_U,
        "W_K_U": W_K_U,
    }


# %%
@torch.no_grad()
def find_min_gaps_with_EQKE(
    model: HookedTransformer,
    *,
    key_direction: Tensor,
    query_direction: Tensor,
    second_key_direction: Tensor,
    second_query_direction: Tensor,
    W_Q_U: Tensor,
    W_K_U: Tensor,
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
    W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
    sanity_check: bool = True,
    atol: float = 1e-4,
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
    use_exact_EQKE: bool = False,
    # svd_EUPU: bool = False,
    attn_scale: Union[Float[Tensor, ""], float],  # noqa: F722
    position: Optional[int] = None,
    leave: Optional[bool] = None,
    desc: Optional[str] = None,
    pbar: Optional[tqdm] = None,
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    (
        (EQKE_query_key, err_accumulator),
        EQKE_pos_err,
        (err_upper_bound, (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)),
    ) = decompose_EQKE_error_quadratic(
        model,
        key_direction=key_direction,
        query_direction=query_direction,
        second_key_direction=second_key_direction,
        second_query_direction=second_query_direction,
        W_Q_U=W_Q_U,
        W_K_U=W_K_U,
        sanity_check=sanity_check,
        atol=atol,
    )

    err_exact = W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
    cur_EQKE = EQKE_query_key + err_accumulator + (err_exact if use_exact_EQKE else 0)
    EQKE_err_upper_bound = torch.tensor(0) if use_exact_EQKE else err_upper_bound

    W_EP_direction = W_EP_direction_for_tricks(W_EP=W_EP, W_U=W_U, tricks=tricks)
    # cur_EUPU_low_rank = EUPU_lowrank if svd_EUPU else None
    # cur_EUPU_high_rank = torch.zeros_like(EUPU) if svd_EUPU else EUPU
    # cur_EUPU_max_err = torch.tensor(0) if not svd_EUPU else EUPU_err_upper_bound

    return find_min_gaps(
        EQKE=cur_EQKE,
        EQKE_err_upper_bound=EQKE_err_upper_bound,
        EQKE_pos_err=EQKE_pos_err,
        EVOU=EVOU,
        PVOU=PVOU,
        tricks=tricks,
        attn_scale=attn_scale,
        position=position,
        leave=leave,
        desc=desc,
        pbar=pbar,
        W_EP=W_EP,
        W_U=W_U,
        W_EP_direction=W_EP_direction,
    )
