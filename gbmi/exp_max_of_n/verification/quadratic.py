from typing import Tuple, Union, Optional
import math
import numpy as np
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from transformer_lens import HookedTransformer
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.utils.lowrank import LowRankTensor
from gbmi.utils.sequences import count_sequences
from gbmi.verification_tools.decomp import factor_contribution


@torch.no_grad()
def compute_extreme_softmaxed_right_attention_quadratic(
    extreme_right_attention: Float[
        Tensor, "minmax=2 d_vocab_q d_vocab_max n_ctx_copies_nonmax"  # noqa: F722
    ],
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"]  # noqa: F722
    ] = 1,
    *,
    attn_scale: Union[Float[Tensor, ""], float],  # noqa: F722
) -> Float[Tensor, "minmax=2 d_vocab_q d_vocab_max n_ctx_copies_nonmax"]:  # noqa: F722
    r"""
    Computes the extreme post-softmax attention paid to the maximum token by each query token, for each number of copies of a non-max token.

    min_gap is used only to determine when the result should be nan

    Complexity: O(d_vocab^2 * n_ctx^2)

    Preconditions:
        . attn_scale is correct for the model
        Define:
        EQKE[q, p, k] := (W_E[q] + W_pos[-1]) @ W_Q[layer, head] @ W_K[layer, head].T @ (W_E[k] + W_pos[p]).T
        Then we demand:
        . \forall q, m, p1, p2, k, n:
          if ((q == m) or (q <= m - min_gap[q, m, n])) and (k <= m - min_gap[q, m, n]):
            extreme_right_attention[0, q, m, n] + EQKE_pos_error[q, p1] - EKQE_pos_error[q, p2]
            <= EQKE[q, p1, m] - EQKE[q, p2, k]
            <= extreme_right_attention[1, q, m, n] + EQKE_pos_error[q, p1] - EKQE_pos_error[q, p2]
    Postconditions:
        \forall q, m, n_copies_nonmax:
          if q > m: return[:, q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[:, q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax] and n_copies_nonmax != 0: return[:, q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[:, q, m, 0] = nan
          else: return[0, q, m, n_copies_nonmax] <= (post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m (including possibly the query token), and all other tokens <= m - min_gap[q, m, n_copies_nonmax])
                            <= return[1, q, m, n_copies_nonmax]
    """
    minmax, d_vocab_q, d_vocab_max, n_ctx_copies_nonmax = extreme_right_attention.shape
    n_ctx = EQKE_pos_err.shape[-1]
    extreme_right_attention = extreme_right_attention.expand(
        *extreme_right_attention.shape[:-1], n_ctx
    )
    result = torch.zeros_like(extreme_right_attention)
    tmp = torch.zeros(
        (
            minmax,
            n_ctx,
        )
    )
    EQKE_pos_err = EQKE_pos_err - EQKE_pos_err[:, -1].unsqueeze(
        -1
    )  # softmax is invariant to adding a constant to all inputs, so we offset by the attention paid to the query position; this lets us uniformly fill in 0 for the attention paid to the query position further down, without it interfering with sorting
    EQKE_pos_err = EQKE_pos_err[:, :-1].sort(dim=-1).values
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            if max_tok < q_tok:
                result[:, q_tok, max_tok] = float("nan")
                continue
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
                if n_copies_nonmax == 0 and max_tok != q_tok:
                    result[:, q_tok, max_tok, n_copies_nonmax] = float("nan")
                    continue
                if max_tok < cur_min_gap and n_copies_nonmax != 0:
                    result[:, q_tok, max_tok, n_copies_nonmax] = float("nan")
                    continue
                if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                    result[:, q_tok, max_tok, n_copies_nonmax] = float("nan")
                    continue
                tmp[0, :-1] = EQKE_pos_err[q_tok]
                tmp[1, :-1] = EQKE_pos_err[q_tok].flip(dims=[0])
                tmp[:, -1] = 0
                n_copies_max = n_ctx - n_copies_nonmax
                # we handle max tok in the query position specially
                # put the max tokens in the least favored slots, where attention is lowest
                max_indices = (
                    list(range(n_copies_max))
                    if max_tok != q_tok
                    else (list(range(n_copies_max - 1)) + [-1])
                )
                tmp[:, max_indices] += extreme_right_attention[
                    :, q_tok, max_tok, n_copies_nonmax
                ].unsqueeze(-1)
                tmp = (tmp / attn_scale).softmax(dim=-1)
                result[:, q_tok, max_tok, n_copies_nonmax] = tmp[:, max_indices].sum(
                    dim=-1
                )
    return result


@torch.no_grad()
def compute_largest_wrong_logit_quadratic(
    extreme_softmaxed_right_attention: Float[
        Tensor, "minmax=2 d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    *,
    W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
    W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]  # noqa: F722
    ] = 1,
    W_EP_direction: Optional[Float[Tensor, "d_model"]] = None,  # noqa F722
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    r"""
    Computes the largest gap between the wrong logit and the right logit for each query token, max token, and number of copies of a non-max token.

    Complexity: O(d_vocab^2 * n_ctx^2) (+ d_vocab * d_model^2 if W_EP_direction is None and tricks.EUPU_handling == "svd_query+max_diff")

    Preconditions:
        W_EP := W_E + W_pos[-1]
        EVOU = W_E @ W_V @ W_O @ W_U
        PVOU = W_pos @ W_V @ W_O @ W_U
        \forall q, m, n_copies_nonmax:
          if q > m: extreme_softmaxed_right_attention[:, q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: extreme_softmaxed_right_attention[:, q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax] and n_copies_nonmax != 0: extreme_softmaxed_right_attention[:, q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: extreme_softmaxed_right_attention[:, q, m, 0] = nan
          else: extreme_softmaxed_right_attention[0, q, m, n_copies_nonmax] <= (post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m, and all other tokens <= m - min_gap[q, m, n_copies_nonmax])
                        <= extreme_softmaxed_right_attention[1, q, m, n_copies_nonmax]
    Postconditions:
        \forall q, m, n_copies_nonmax, x:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[q, m, 0] = nan
          else: for all sequences with query q, max token m, n_copies_nonmax tokens not equal to m (including the query when the query is not equal to m), and all tokens either equal to m or less than or equal to m - min_gap[q, m, n_copies_nonmax], we have:
            return[q, m, n_copies_nonmax] <= model(sequence)[-1, x] - model(sequence)[-1, m]
    """
    results = torch.zeros_like(extreme_softmaxed_right_attention[0]) + float("nan")
    minmax, d_vocab_q, d_vocab_max, n_ctx = extreme_softmaxed_right_attention.shape
    EVOU_max_logit_diff: Float[Tensor, "d_vocab_k"] = (  # noqa: F821
        EVOU.max(dim=-1).values - EVOU.min(dim=-1).values
    )  # for when we're paying attention to the wrong token

    # EUPU is too expensive to center with respect to the max token
    # so we split it
    # this one we can center with respect to the max token
    EUPU_mean_query: Float[Tensor, "d_vocab_out"]  # noqa: F821
    # this one we pessimize over the wrong token
    EUPU_per_query_max_gap: Float[Tensor, "d_vocab_q"]  # noqa: F821
    EUPU_mean_query, EUPU_per_query_max_gap = tricks.split_EPU(
        W_EP=W_EP, W_U=W_U, W_EP_mean_query=W_EP_direction
    )

    # center EVOU with respect to the diagonal, so it's convenient for the max token
    EVOU = EVOU - EVOU.diag()[:, None]
    for max_tok in range(d_vocab_max):
        # center PVOU according to max token, O(d_vocab * n_ctx)
        PVOU = PVOU - PVOU[:, max_tok].unsqueeze(-1)
        # center EUPU_mean_query according to max token, O(d_vocab)
        EUPU_mean_query = EUPU_mean_query - EUPU_mean_query[max_tok].item()

        # Pessimization over position:
        # relax to PVOU attention being indepenent of EVOU attention, and also relax to it being possible to pay 100% attention to one PVOU position (this is reasonable, the gap in pre-softmax attention between adjacent tokens is like 20, 1e-20 is essentially 0 in float32.  EDIT: except I forgot to divide by attn_scale when doing this, attn_scale is 5.7, 20/5.7 is 3.5, exp(3.5) is 0.03, so it's not quite 100% attention.  Probably still pretty small)
        cur_PVOU: Float[Tensor, "d_vocab_out"] = PVOU.max(dim=0).values  # noqa: F821

        # handle the case with only the max token
        # here we can use EUPU exactly
        logits_only_max: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
            W_EP[max_tok, :] @ W_U + EVOU[max_tok, :] + cur_PVOU
        )
        # O(d_model * d_vocab)
        logits_only_max -= logits_only_max[max_tok].item()
        logits_only_max[max_tok] = float(
            "-inf"
        )  # so we can max the logits across the non-max tokens
        results[max_tok, max_tok, 0] = logits_only_max.max().item()

        # now handle the cases with at least one non-max token
        # query-independent logits from the skip connection / PVOU independence
        logits: Float[Tensor, "d_vocab_out"] = EUPU_mean_query + cur_PVOU  # noqa: F821
        assert logits[max_tok] == 0  # sanity check from centering above
        # exact logits from paying attention to the right thing
        right_attention_logits: Float[Tensor, "d_vocab_out"] = EVOU[  # noqa: F821
            max_tok
        ]
        assert right_attention_logits[max_tok] == 0  # sanity check from centering above
        right_attention_logits_tmp = right_attention_logits.detach().clone()
        right_attention_logits_tmp[max_tok] = float("-inf")
        # maximum added to the wrong logit from paying attention to the right thing
        right_attention_logits_max: Float[Tensor, ""] = (  # noqa: F722
            right_attention_logits_tmp.max()
        )
        for n_copies_nonmax in range(1, n_ctx):
            cur_min_gap = (
                min_gap
                if isinstance(min_gap, int)
                else int(min_gap[: max_tok + 1, max_tok, n_copies_nonmax].min().item())
            )
            if max_tok < cur_min_gap:
                # we must have only the max token
                continue
            # pessimize over the thing we're not supposed to be paying attention to (w.r.t. the token that is non-max that we're paying attention)
            # maximum added to the wrong logit from paying attention to the wrong thing
            wrong_attention_logits: Float[Tensor, ""] = (  # noqa: F722
                EVOU_max_logit_diff[: max_tok - cur_min_gap + 1].max()
            )

            # if the maximum non-max logit is negative (more generally: smaller for the right thing than the wrong thing),
            # we do worst by attending as little as possible to the max token,
            # but if it's positive (more generally: larger for the right thing than the wrong thing),
            # we do poorly if we attend as much as possible to the max token
            min_max_index = (
                0
                if right_attention_logits_max.item() < wrong_attention_logits.item()
                else 1
            )

            # First we combine query-independent logit information, then we reduce over output tokens and loop over queries
            # drop the nan values where the query token is invalid given the number of copies and the max token
            average_right_attention: Float[Tensor, ""]  # noqa: F722
            right_attention_adjustment: Float[Tensor, "d_vocab_q"]
            (
                average_right_attention,
                right_attention_adjustment,
            ) = tricks.split_extreme_softmaxed_right_attention(
                extreme_softmaxed_right_attention[
                    min_max_index, :, max_tok, n_copies_nonmax
                ],
                max_tok=max_tok,
            )
            cur_copies_logits = (
                logits + average_right_attention * right_attention_logits
            )
            assert (
                cur_copies_logits[max_tok] == 0
            ), f"cur_copies_logits[{max_tok}] == {cur_copies_logits[max_tok]} != 0"  # sanity check from centering above
            cur_copies_logits[max_tok] = float("-inf")
            # find maximum wrong logit
            average_wrong_logit = (
                cur_copies_logits.max()
                + (1 - average_right_attention) * wrong_attention_logits
            )
            for q_tok in range(max_tok + 1):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
                if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                    continue
                cur_extra_right_attention = (
                    extreme_softmaxed_right_attention[
                        min_max_index, q_tok, max_tok, n_copies_nonmax
                    ]
                    - average_right_attention
                )
                results[q_tok, max_tok, n_copies_nonmax] = (
                    average_wrong_logit + EUPU_per_query_max_gap[q_tok]
                )
                # add attention correction factor on right attention
                results[q_tok, max_tok, n_copies_nonmax] += (
                    cur_extra_right_attention * right_attention_logits_max
                )
                # add attention correction factor on right attention, using += - instead of -= to make the negation more obvious
                results[q_tok, max_tok, n_copies_nonmax] += (
                    -cur_extra_right_attention * wrong_attention_logits
                )
    return results


@torch.no_grad()
def compute_extreme_right_attention_quadratic(
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"]  # noqa: F722
    ] = 1,
) -> Float[Tensor, "minmax=2 d_vocab_q d_vocab_max n_ctx_copies_nonmax"]:  # noqa: F722
    r"""
    Computes a tensor of extreme right attention (more attention paid to the max than to a single instance of a non-max token at least min_gap less than the max token) for each query token and each max token
    When the query token is larger than the max token, the matrix holds nan.

    Complexity: O(d_vocab^2 n_ctx)

    Preconditions:
        (none)
    Postconditions:
        \forall q, m, n_copies_nonmax, k <= m - min_gap[q, m, n_copies_nonmax]:
          if q > m: return[q, m, :] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[:, q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[:, q, m, n_copies_nonmax] = 0
          else: return[0, q, m, n_copies_nonmax] <= EQKE[q, k] - EQKE[q, m] <= return[1, q, m, n_copies_nonmax]
    """
    n_ctx = min_gap.shape[-1] if not isinstance(min_gap, int) else 1
    result = torch.zeros((2, EQKE.shape[0], EQKE.shape[1], n_ctx)).to(EQKE.device)
    for q_tok in range(EQKE.shape[0]):
        # running_extrema[:, k] is inclusive of attention paid to token value k
        running_extrema = torch.zeros((2, EQKE[q_tok].shape[0])).to(EQKE.device)
        for max_tok in range(EQKE.shape[1]):
            running_extrema[:, max_tok] = EQKE[q_tok, max_tok].item()
            if max_tok > 0:
                # running_extrema.shape[0] is flipped from result.shape[0] because we subtract running extrema;
                # 0 is max here, 1 is min
                assert (
                    running_extrema[:, max_tok - 1 : max_tok + 1].shape[1] == 2
                )  # sanity check for slicing by (max_tok-1, max_tok)
                running_extrema[0, max_tok] = running_extrema[
                    0, max_tok - 1 : max_tok + 1
                ].max()
                running_extrema[1, max_tok] = running_extrema[
                    1, max_tok - 1 : max_tok + 1
                ].min()
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
                if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                    result[:, q_tok, max_tok, n_copies_nonmax] = float("nan")
                elif max_tok < cur_min_gap:
                    result[:, q_tok, max_tok, n_copies_nonmax] = 0
                else:
                    result[:, q_tok, max_tok, n_copies_nonmax] = (
                        EQKE[q_tok, max_tok] - running_extrema[:, max_tok - cur_min_gap]
                    )
    return result


@torch.no_grad()
def decompose_EQKE_error_quadratic(
    model: HookedTransformer,
    *,
    key_direction: Tensor,
    query_direction: Tensor,
    second_key_direction: Tensor,
    second_query_direction: Tensor,
    W_Q_U: Tensor,
    W_K_U: Tensor,
    layer: int = 0,
    head: int = 0,
    sanity_check: bool = True,
    atol: float = 1e-4,
) -> Tuple[
    Tuple[
        Float[LowRankTensor, "d_vocab_q d_vocab_k"],  # noqa: F722
        Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    ],
    Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    Tuple[
        Union[Float[Tensor, ""], Float[Tensor, "d_vocab_q"]],  # noqa: F722, F821
        Tuple[
            Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
            Float[Tensor, "d_model d_model"],  # noqa: F722
            Float[Tensor, "d_model d_model"],  # noqa: F722
            Float[Tensor, "d_model d_vocab_k"],  # noqa: F722
        ],
    ],
]:
    r"""
    Returns:
        ((EQKE_query_key, err_accumulator), EQKE_pos_err, (remaining_error_upper_bound, four matrices whose product is the exact remaining error))
    where
        EQKE_query_key is the rank 1 approximation of the query-key contribution to the EQKE matrix
        err_accumulator is the sum of the efficiently-computable (O(d_vocab^2)) error terms
        EQKE_pos_err is the contribution of the position embeddings to the error
        remaining_error_upper_bound is a bound on the maximum difference between two elements in the same row of the remaining error of EQKE, and may be either a float or a tensor indexed by query token, depending on the configuration of tricks

    Note that EQKE is actually computed as (W_E + W_pos[-1]) @ W_Q[layer, head] @ W_K[layer, head].T @ (W_E + W_pos.mean(dim=0, keepdim=True)).T

    Complexity: O(d_vocab * (d_vocab + d_model * n_ctx))

    Preconditions:
        (none)
    Postconditions:
        Define err := EQKE - (EQKE_query_key + err_accumulator)
        Then we guarantee:
        . max_{i,j} err_{r, i} - err_{r, j} <= remaining_error_upper_bound
        . EQKE_pos_err[p] := (W_E + W_pos[-1]) @ W_Q[layer, head] @ W_K[layer, head].T @ (W_pos[p] - W_pos.mean(dim=0, keepdim=True)).T

    EQKE_query_key uses key_direction and query_direction for the rank 1 approximation

    We compute as follows:
    $$
    \begin{align*}
    \overline{W_\text{pos}} & := W_\text{pos}\text{.mean}(\text{dim}=0) \\
    \widetilde{E_q} & := W_E + W_\text{pos}[-1] \\
    \widetilde{E_k} & := W_E + \overline{W_\text{pos}} \\
    \text{EQKE}_p
    & := \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \widetilde{E_q}W_QW_K^T(W_{\text{pos}}[p] - \overline{W_\text{pos}})^T \\
    & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \text{EQKE\_pos\_err}
    \end{align*}
    $$
    We can decompose $\widetilde{E_k}$ as a sum of a rank 1 matrix in the given key direction and a matrix orthogonal to the key direction, say $E_k = E_{k,\text{key}} + E_{k,\text{key}}^\perp$.
    We can decompose $\widetilde{E_q}$ as a sum of a rank 1 matrix in the given query direction and a matrix orthogonal to the query direction, say $E_q = E_{q,\text{query}} + E_{q,\text{query}}^\perp$.
    We can decompose $E_{k,\text{key}}^\perp$, $E_{q,\text{query}}^\perp$, $W_Q$, and $W_K$ as sums of rank 1 matrices in the second key direction, second query direction, W\_Q\_U, and W\_K\_U, respectively.
    $$
    \begin{align*}
    E_{k,\text{key}}^\perp & = E_{k,\text{key},\text{second}} + E_{k,\text{key},\text{second}}^\perp \\
    E_{q,\text{query}}^\perp & = E_{q,\text{query},\text{second}} + E_{q,\text{query},\text{second}}^\perp \\
    W_Q & = W_{Q,\text{U}} + W_{Q,\text{U}}^\perp \\
    W_K & = W_{K,\text{U}} + W_{K,\text{U}}^\perp
    \end{align*}
    $$
    Then we can write
    $$
    \begin{align*}
    \text{EQKE}_p - \text{EQKE\_pos\_err}
    & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T \\
    & = E_{k,\text{key}}W_QW_K^T E_{q,\text{query}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key}}W_QW_K^T {E_{q,\text{query}}^\perp}^T + E_{k,\text{key}}^\perp W_QW_K^T E_{q,\text{query}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}W_QW_K^T E_{q,\text{query},\text{second}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}W_QW_K^T {E_{q,\text{query},\text{second}}^\perp}^T + E_{k,\text{key},\text{second}}^\perp W_QW_K^T E_{q,\text{query},\text{second}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}W_{K,\text{U}}^T {E_{q,\text{query},\text{second}}^\perp}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}^\perp W_{K,\text{U}}^T {E_{q,\text{query},\text{second}}^\perp}^T + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}} {W_{K,\text{U}}^\perp}^T {E_{q,\text{query},\text{second}}^\perp}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}^\perp {W_{K,\text{U}}^\perp}^T {E_{q,\text{query},\text{second}}^\perp}^T
    \end{align*}
    $$
    Note that the first component is returned as EQKE_query_key, the middle components are accumulated in err_accumulator.

    Except for the last line, all of these components are rank 1 matrices, and we can compute them efficiently.

    The final value we compute by attempting to bound the largest singular value of the remaining error term.
    We compute an upper bound on what the final component can contribute to differences in elements in the same row:
    Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt{2} \sigma_1(M)$.
    This is the value we return, computing an upper bound on the first singular value by multiplying the first singular values of each matrix.

    However, we don't have the compute budget for computing $\sigma_1$ exactly, so we approximate it as the product of the frobenius norms.
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
    W_E_key, W_E_key_err = factor_contribution(
        W_E_pos_k,
        key_direction.squeeze(),
        sanity_check=sanity_check,
        checkparams=dict(atol=atol),
    )  # O(d_vocab * d_model)
    W_E_key.setcheckparams(atol=atol)
    W_E_query, W_E_query_err = factor_contribution(
        W_E_pos_q,
        query_direction.squeeze(),
        sanity_check=sanity_check,
        checkparams=dict(atol=atol),
    )  # O(d_vocab * d_model)
    W_E_query.setcheckparams(atol=atol)
    EQKE_query_key = (W_E_query @ W_Q[layer, head]) @ (
        W_K[layer, head].T @ W_E_key.T
    )  # O(d_vocab * d_vocab)
    err_accumulator = torch.zeros_like(EQKE_query_key.AB)  # O(d_vocab^2)
    EQKE_query_cross_err = (
        (W_E_query @ W_Q[layer, head]) @ W_K[layer, head].T
    ) @ W_E_key_err.T  # O(d_vocab * d_model)
    err_accumulator += EQKE_query_cross_err
    EQKE_err_cross_key = W_E_query_err @ (
        W_Q[layer, head] @ (W_K[layer, head].T @ W_E_key.T)
    )  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_cross_key

    # This is a differently-shaped error term, and will be treated separately
    EQKE_pos_err = W_E_pos_q @ (
        W_Q[layer, head] @ (W_K[layer, head].T @ W_pos_err.T)
    )  # O(d_vocab * d_model * n_ctx)

    # We'd like a faster way to estimate the quantity (EQKE_err_err_check.max(dim=-1) - EQKE_err_err_check.min(dim=-1)).max()
    # The naive computation is O(d_vocab^2 * d_model), and we can only get this down to O(d_vocab * d_model^2) by using SVD
    # To improve our error bounds a bit, first we again peel off the leading singular values
    W_E_second_key, W_E_key_err2 = factor_contribution(
        W_E_key_err, second_key_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_second_key.setcheckparams(atol=1e-4)
    (
        W_E_second_query,
        W_E_query_err2,
    ) = factor_contribution(
        W_E_query_err, second_query_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_second_query.setcheckparams(atol=1e-4)
    EQKE_err_second_query_key = (W_E_second_query @ W_Q[layer, head]) @ (
        W_K[layer, head].T @ W_E_second_key.T
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_second_query_key
    EQKE_err_second_query_cross_err = (
        (W_E_second_query @ W_Q[layer, head]) @ W_K[layer, head].T
    ) @ W_E_key_err2.T  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_second_query_cross_err
    EQKE_err_err_cross_second_key = W_E_query_err2 @ (
        W_Q[layer, head] @ (W_K[layer, head].T @ W_E_second_key.T)
    )  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_err_cross_second_key

    # Now we peel off the first singular vectors of W_Q and W_K
    W_Q_rank1, W_Q_err = factor_contribution(
        W_Q[layer, head], W_Q_U.squeeze(), sanity_check=sanity_check
    )  # O(d_model * d_model)
    W_Q_rank1.setcheckparams(atol=1e-4)
    W_K_rank1, W_K_err = factor_contribution(
        W_K[layer, head], W_K_U.squeeze(), sanity_check=sanity_check
    )  # O(d_model * d_model)
    W_K_rank1.setcheckparams(atol=1e-4)

    EQKE_err_err_err__first_singular = (W_E_query_err2 @ W_Q_rank1) @ (
        W_K_rank1.T @ W_E_key_err2.T
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_err_err__first_singular

    EQKE_err_err_err__Q_cross_err = (
        (W_E_query_err2 @ W_Q_rank1) @ W_K_err.T
    ) @ W_E_key_err2.T  # O(d_vocab * d_voacb)
    err_accumulator += EQKE_err_err_err__Q_cross_err
    EQKE_err_err_err__err_cross_K = W_E_query_err2 @ (
        W_Q_err @ (W_K_rank1.T @ W_E_key_err2.T)
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_err_err__err_cross_K

    # We would like a faster way to compute EQKE_err_err_err__err_cross_err
    # unfortunately, we can only get this down to O(d_vocab * d_model^2) by using SVD

    return (
        (EQKE_query_key, err_accumulator),
        EQKE_pos_err,
        (
            np.sqrt(2)
            * np.prod(
                [
                    torch.linalg.matrix_norm(m, ord="fro")
                    for m in (W_E_query_err2, W_Q_err, W_K_err.T, W_E_key_err2.T)
                ]
            ),
            (W_E_query_err2, W_Q_err, W_K_err.T, W_E_key_err2.T),
        ),
    )


@torch.no_grad()
def count_unaccounted_for_by_gap(
    min_gap: Integer[Tensor, "d_vocab_q d_vocab_max n_ctx"],  # noqa: F722
    collapse_n_ctx: bool = False,
) -> int:
    """Computes the number of sequences that we are leaving on the table by using gaps"""
    d_vocab_q, d_vocab_max, n_ctx = min_gap.shape
    unaccounted_for: int = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            if q_tok > max_tok:
                continue
            if collapse_n_ctx:
                gaps = min_gap[q_tok, max_tok]
                gap = gaps[~gaps.isnan()].max().long().item()
                if q_tok == max_tok:
                    if max_tok < gap:
                        unaccounted_for += (max_tok + 1) ** (n_ctx - 1)
                    else:
                        unaccounted_for += (1 + max_tok) ** (n_ctx - 1) - (
                            1 + (max_tok - gap + 1)
                        ) ** (n_ctx - 1)
                else:
                    if max_tok < gap:
                        unaccounted_for += (max_tok + 1) ** (n_ctx - 1) - max_tok ** (
                            n_ctx - 1
                        )
                    else:
                        unaccounted_for += (
                            (max_tok + 1) ** (n_ctx - 1) - max_tok ** (n_ctx - 1)
                        ) - (
                            (1 + (max_tok - gap + 1)) ** (n_ctx - 1)
                            - (max_tok - gap + 1) ** (n_ctx - 1)
                        )
            else:
                for n_copies_nonmax in range(n_ctx):
                    if n_copies_nonmax == n_ctx - 1 and max_tok != q_tok:
                        continue
                    gap = min_gap[q_tok, max_tok, n_copies_nonmax].long().item()
                    unaccounted_for += (
                        max_tok**n_copies_nonmax
                        - (1 + max_tok - gap) ** n_copies_nonmax
                    ) * math.comb(n_ctx - 1, n_copies_nonmax)
    return unaccounted_for


@torch.no_grad()
def count_correct_sequences(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]  # noqa: F722
    ] = 1,
) -> int:
    d_vocab_q, d_vocab_max, n_ctx = largest_wrong_logit.shape
    correct_count = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
                # use not to also catch nans
                if (
                    (not largest_wrong_logit[q_tok, max_tok, n_copies_nonmax] < 0)
                    or q_tok > max_tok
                    or (q_tok != max_tok and n_copies_nonmax == 0)
                    or (q_tok != max_tok and max_tok - q_tok < cur_min_gap)
                    or (max_tok == 0 and n_copies_nonmax > 0)
                ):
                    continue
                # N.B. Here, n_copies_nonmax DOES include the query token when it's not equal to max_tok
                nonmax_pre_query_count = (
                    n_copies_nonmax - 1 if q_tok != max_tok else n_copies_nonmax
                )
                if nonmax_pre_query_count == 0:
                    correct_count += 1
                else:
                    # count the number of sequences of length n_ctx - 1 with nonmax_pre_query_count tokens less than or equal to max_tok - cur_min_gap and the remaining tokens equal to max_tok, where order matters
                    correct_count += count_sequences(
                        n_ctx - 1, nonmax_pre_query_count, max_tok - cur_min_gap
                    )
    return correct_count


def compute_accuracy_lower_bound_from(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]  # noqa: F722
    ] = 1,
) -> Tuple[float, Tuple[int, int]]:
    """
    returns correct_count / total_sequences, (correct_count, total_sequences)
    """
    d_vocab_q, d_vocab_max, n_ctx = largest_wrong_logit.shape
    correct_count = count_correct_sequences(largest_wrong_logit, min_gap=min_gap)
    total_sequences = d_vocab_max**n_ctx
    return correct_count / total_sequences, (correct_count, total_sequences)
