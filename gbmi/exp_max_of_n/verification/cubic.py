import math
import time
from typing import Callable, Optional, Tuple, Union
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from gbmi.utils.sequences import count_sequences
from gbmi.verification_tools.general import EU_PU
from gbmi.verification_tools.l1h1 import all_EQKE, all_EQKP, all_EVOU, all_PVOU
from gbmi.verification_tools.utils import complexity_of


@torch.no_grad()
def compute_extreme_softmaxed_right_attention_cubic_simple(
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    EQKP: Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    attn_scale: Union[Float[Tensor, ""], float],  # noqa F722
    position: Optional[int] = None,
    *,
    pbar: Optional[tqdm] = None,
) -> Float[
    Tensor,
    "minmax=2 attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
]:
    r"""
    Computes the extreme (min-attn-to-max is minmax=0, max-attn-to-max is minmax=1) post-softmax attention (extremized over sequence orderings) paid to the maximum token (attn=0) and
    to the non-maximum token (attn=1) and to the query token (attn=2):
        - by each possible value of the query token,
        - for each possible value of the max token,
        - for each possible value of the nonmax token,
        - for each number of copies of the non max token

    Basically, this attempts to lower bound the attention that is paid to the max token and the query token, by
    pessimising over the order of the non-query tokens.

    Note that we could do a bit of a better job than this by taking in min_gap and a pre-computed extreme_attention
    matrix, if we wanted to.

    Time Complexity: O(d_vocab^3 * n_ctx^2)

    Preconditions:
        . attn_scale is correct for the model (\sqrt(d_head) by default)
        . EQKE[q, k] is the attention paid from query q to key token k
        . EQKP[q, p] is the attention paid from query q to key position p
    Postconditions:
        \forall w \in {0,1,2}, q, m, k, n_copies_nonmax: ("w" for which)
          if q > m or k > m: return[:, w, q, m, k, n_copies_nonmax] = nan
                (That is, the answer is undefined if the query token is greater than the max token, or if the non-max
                token is greater than the max token)
          elif m = k and n_copies_nonmax != 0: return[:, w, q, m, k, n_copies_nonmax] = nan
                (That is, the answer is undefined if the non-max token is equal to the max token and there are non-zero
                copies of non-max tokens)
          elif q != m and n_copies_nonmax >= n_ctx - 1: return[:, w, q, m, k, n_copies_nonmax] = nan
                (That is, the answer is undefined if the query token is not equal to the max token and there are n_ctx
                - 1 or more copies of the non-max token, because then the max token would be missing)
          else: amongst all permutations of [the non-query-tokens in] the sequence with query q, n_copies_nonmax copies
                of k, and all other tokens equal to m:
                return[0, 0, q, m, k, n_copies_nonmax] <= post-softmax attention paid to max token m     <= return[1, 0, q, m, k, n_copies_nonmax]
                return[0, 1, q, m, k, n_copies_nonmax] <= post-softmax attention paid to non-max token k <= return[1, 1, q, m, k, n_copies_nonmax]
                return[0, 2, q, m, k, n_copies_nonmax] <= post-softmax attention paid to query token q   <= return[1, 2, q, m, k, n_copies_nonmax]

    """
    d_vocab, n_ctx = EQKE.shape[-1], EQKP.shape[-1]
    result = torch.zeros((2, 3, d_vocab, d_vocab, d_vocab, n_ctx)).to(EQKE) + float(
        "nan"
    )
    tmp = torch.zeros(
        (
            2,
            n_ctx,
        )
    ).to(EQKE)
    # constants for indices so we don't have 0 and 1 floating around
    w_max = 0
    w_nmx = 1
    w_qry = 2
    # we sort EQKP so that higher-attention positions are at the back, so we can put the max token at the front.
    EQKP, EQKPm1 = EQKP[:, :-1].sort(dim=-1).values, EQKP[:, -1]

    max_tok_range = range(d_vocab)
    if pbar is None:
        max_tok_range = tqdm(max_tok_range, desc="max_tok", position=position)
    for max_tok in max_tok_range:
        for q_tok in range(max_tok + 1):
            tmp[:, -1] = EQKE[q_tok, q_tok] + EQKPm1[q_tok]
            for k_tok in range(max_tok + 1):
                if pbar is not None:
                    pbar.update(1)
                if k_tok == max_tok:
                    if q_tok == max_tok:
                        # only max tok, so we pay 100% attention to it
                        result[:, w_max, q_tok, max_tok, k_tok, 0] = 1
                        result[:, w_nmx, q_tok, max_tok, k_tok, 0] = 0
                        result[:, w_qry, q_tok, max_tok, k_tok, 0] = 0
                        continue
                    tmp[:, :-1] = EQKP[q_tok] + EQKE[q_tok, k_tok]
                    tmp_sm = (tmp / attn_scale).softmax(dim=-1)
                    result[:, w_max, q_tok, max_tok, k_tok, 0] = tmp_sm[:, :-1].sum(
                        dim=-1
                    )
                    result[:, w_nmx, q_tok, max_tok, k_tok, 0] = 0
                    result[:, w_qry, q_tok, max_tok, k_tok, 0] = tmp_sm[:, -1]
                    continue
                for n_copies_nonmax in range(n_ctx):
                    n_copies_max_nonquery = n_ctx - n_copies_nonmax - 1
                    if q_tok != max_tok and n_copies_nonmax >= n_ctx - 1:
                        continue
                    tmp[0, :-1] = EQKP[q_tok]
                    tmp[1, :-1] = EQKP[q_tok].flip(dims=[0])

                    tmp[:, :n_copies_max_nonquery] += EQKE[q_tok, max_tok]
                    # attention paid to non-max tokens other than in the query position
                    tmp[:, n_copies_max_nonquery:-1] += EQKE[q_tok, k_tok]
                    tmp_sm = (tmp / attn_scale).softmax(dim=-1)
                    result[:, w_max, q_tok, max_tok, k_tok, n_copies_nonmax] = tmp_sm[
                        :, :n_copies_max_nonquery
                    ].sum(dim=-1) + (tmp_sm[:, -1] if q_tok == max_tok else 0)
                    result[:, w_nmx, q_tok, max_tok, k_tok, n_copies_nonmax] = result[
                        :, w_qry, q_tok, max_tok, k_tok, n_copies_nonmax
                    ] = (tmp_sm[:, -1] if q_tok != max_tok else 0)
    return result


@torch.no_grad()
def compute_largest_wrong_logit_cubic(
    extreme_softmaxed_right_attention: Float[
        Tensor,
        "minmax=2 attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
    ],
    *,
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    # permitted_nonmax_tokens: Optional[Bool[
    #     Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    # ]] = None,
) -> Float[
    Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
]:
    r"""
        Computes the largest difference between the wrong logit and the right logit for each query token, max token, nonmax
        token, and number of copies of the non-max token.

        Complexity: O(d_vocab^3 * n_ctx^2)

        Preconditions:
            extreme_softmaxed_right_attention satisfies the postcondition of compute_extreme_softmaxed_right_attention_cubic_simple
                (a lower bound on the post-softmax attention paid to the max token and the query token,
                by each possible value of the query token,
                for each possible value of the max token,
                for each possible value of the non-max token,
                for each number of copies of the non max token)
            EUPU = (W_E + W_pos[-1]) @ W_U
            EVOU = W_E @ W_V @ W_O @ W_U
            PVOU = W_pos @ W_V @ W_O @ W_U
        Postconditions:
            \forall q, m, k, n_copies_nonmax, x:
              if q > m or k > m: return[q, m, k, n_copies_nonmax] = nan
              elif m = k and n_copies_nonmax != 0: return[q, m, m, n_copies_nonmax] = nan
              elif m != k and n_copies_nonmax == 0: return[q, m, k, 0] = nan
              elif q != m and n_copies_nonmax >= n_ctx - 1: return[q, m, k, n_copies_nonmax] = nan
                (That is, in these cases, the answer is undefined because the query token is greater than the max token, or
                there isn't enough room for the non-max token or the max token in the sequence.)
              else: for all sequences with query q, max token m, n_copies_nonmax [non-query] copies of k (and the rest of
                the non-query tokens equal to m), we have:
                return[q, m, k, n_copies_nonmax] <= model(sequence)[-1, x] - model(sequence)[-1, m]
                That is, we return a lower bound on the difference between the wrong logit and the right logit for this
                combination of query token, max token, non-max token, and number of copies of the non-max token.

    The main idea here is that by pessimizing over the positional attention independently of the embedding attention, we can
        later use a convexity argument for the embedding attention.
    """
    # if permitted_nonmax_tokens is None:
    #     permitted_nonmax_tokens = torch.ones_like(min_softmaxed_right_attention[0], dtype=torch.bool)
    results = torch.zeros_like(
        extreme_softmaxed_right_attention[0, 0, :, :, :, :]
    ) + float("nan")
    _, _, d_vocab, _, _, n_ctx = extreme_softmaxed_right_attention.shape
    w_max = 0
    w_nmx = 1
    w_qry = 2
    for max_tok in range(d_vocab):
        # center PVOU according to max token, O(d_vocab * n_ctx)
        PVOU = PVOU - PVOU[:, max_tok].unsqueeze(-1)
        # center EUPU according to max token, O(d_vocab^2)
        EUPU = EUPU - EUPU[:, max_tok].unsqueeze(-1)
        # center EVOU according to max token, O(d_vocab^2)
        EVOU = EVOU - EVOU[:, max_tok].unsqueeze(-1)
        # to make convexity go through, we need to consider a larger region of phase space bounded by points where
        # positional attention is independent of token attention.
        # Here we pessimize over positional attention, assuming that we pay 100% of attention to the worst position
        PVOU_pessimized: Float[Tensor, "d_vocab_out"] = PVOU.max(  # noqa: F821
            dim=0
        ).values

        # handle the case with only the max token
        logits_only_max: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
            EUPU[max_tok, :] + EVOU[max_tok, :] + PVOU_pessimized
        )
        logits_only_max -= logits_only_max[max_tok].item()
        logits_only_max[max_tok] = float(
            "-inf"
        )  # so we can max the logits across the non-max tokens
        results[max_tok, max_tok, max_tok, 0] = logits_only_max.max().item()

        # now handle the cases with only the query token and n_ctx - 1 copies of the max token
        for q_tok in range(max_tok):
            cur_extreme_right_attention = extreme_softmaxed_right_attention[
                :, :, q_tok, max_tok, max_tok, 0
            ]
            # N.B. because EVOU[q_tok, max_tok] == 0 by centering above, we just take the maximum attention paid to the query
            logits_only_q_and_max: Float[
                Tensor, "minmax=2 d_vocab_out"  # noqa: F722
            ] = (
                EUPU[q_tok, :]
                + PVOU_pessimized
                + EVOU[max_tok, :] * cur_extreme_right_attention[:, w_max].unsqueeze(-1)
                + EVOU[q_tok, :] * cur_extreme_right_attention[:, w_qry].unsqueeze(-1)
            )
            logits_only_q_and_max -= logits_only_q_and_max[:, max_tok].unsqueeze(-1)
            logits_only_q_and_max[:, max_tok] = float("-inf")
            results[q_tok, max_tok, max_tok, 0] = logits_only_q_and_max.max().item()

        # precompose pessimization for EUPU over output logit, so we have enough compute budget
        EUPU_tmp: Float[Tensor, "d_vocab_q d_vocab_out"] = (  # noqa: F722
            EUPU.detach().clone()
        )
        EUPU_tmp[:, max_tok] = float("-inf")
        EUPU_per_query_pessimized: Float[Tensor, "d_vocab_q"] = (  # noqa: F821
            EUPU_tmp.max(dim=-1).values
        )

        # Ditto for EVOU
        # distribute PVOU over EVOU, to avoid premature pessimization
        # TODO: mean+diff
        EPVOU_tmp: Float[Tensor, "d_vocab_k d_vocab_out"] = (  # noqa: F722
            EVOU + PVOU_pessimized
        )
        EPVOU_tmp[:, max_tok] = float("-inf")
        EPVOU_per_key_pessimized: Float[Tensor, "d_vocab_k"] = (  # noqa: F821
            EPVOU_tmp.max(dim=-1).values
        )

        # now handle the cases with at least one non-max non-query token
        for nonmax_tok in range(max_tok):
            for n_copies_nonmax in range(1, n_ctx):
                # distribute PVOU over EVOU, to avoid premature pessimization

                # pessimize over the thing we're not supposed to be paying attention to (w.r.t. the token that is non-max that we're paying attention)
                # maximum added to the wrong logit from paying attention to the wrong thing
                wrong_attention_logits: Float[Tensor, ""] = (  # noqa: F72
                    EPVOU_per_key_pessimized[nonmax_tok]
                )

                # pessimize also over the thing we are paying attention to
                right_attention_wrong_logits: Float[Tensor, ""] = (  # noqa: F722
                    EPVOU_per_key_pessimized[max_tok]
                )

                for q_tok in range(max_tok + 1):
                    if q_tok != max_tok and n_copies_nonmax >= n_ctx - 1:
                        continue
                    query_wrong_logits: Float[Tensor, ""] = (  # noqa: F722
                        EPVOU_per_key_pessimized[q_tok]
                    )
                    right_attn = extreme_softmaxed_right_attention[
                        :, w_max, q_tok, max_tok, nonmax_tok, n_copies_nonmax
                    ]
                    q_attn = extreme_softmaxed_right_attention[
                        :, w_qry, q_tok, max_tok, nonmax_tok, n_copies_nonmax
                    ]
                    wrong_attn = extreme_softmaxed_right_attention[
                        :, w_nmx, q_tok, max_tok, nonmax_tok, n_copies_nonmax
                    ]
                    results[q_tok, max_tok, nonmax_tok, n_copies_nonmax] = (
                        EUPU_per_query_pessimized[q_tok]
                        + (
                            right_attn * right_attention_wrong_logits
                            + q_attn * query_wrong_logits
                            + wrong_attn * wrong_attention_logits
                        ).max()
                    ).item()
    return results


@torch.no_grad()
def count_correct_sequences_cubic(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    ],
) -> int:
    d_vocab_q, d_vocab_max, d_vocab_nonmax, n_ctx = largest_wrong_logit.shape
    correct_count: int = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            for n_copies_nonmax in range(n_ctx):
                if q_tok > max_tok or (
                    q_tok != max_tok and n_copies_nonmax >= n_ctx - 1
                ):
                    continue
                if n_copies_nonmax == 0:
                    cur_largest_wrong_logit = largest_wrong_logit[
                        q_tok, max_tok, max_tok, 0
                    ]
                    correct_count += int((cur_largest_wrong_logit < 0).sum().item())
                else:
                    cur_largest_wrong_logit = largest_wrong_logit[
                        q_tok, max_tok, :max_tok, n_copies_nonmax
                    ]  # consider wrong logits only when non-max token is less than max token
                    num_nonmax_tok_choices = cur_largest_wrong_logit[
                        ~cur_largest_wrong_logit.isnan() & (cur_largest_wrong_logit < 0)
                    ].size(0)
                    cur_count = count_sequences(
                        n_ctx - 1, n_copies_nonmax, num_nonmax_tok_choices
                    )
                    if n_copies_nonmax > 0 and max_tok == 0:
                        assert (
                            cur_count == 0
                        ), f"count: {cur_count} == count_sequences({n_ctx - 1}, {n_copies_nonmax}, {num_nonmax_tok_choices})"
                    else:
                        max_possible_count = max_tok**n_copies_nonmax * math.comb(
                            n_ctx - 1, n_copies_nonmax
                        )
                        assert (
                            cur_count <= max_possible_count
                        ), f"count: {cur_count} == count_sequences({n_ctx - 1}, {n_copies_nonmax}, {num_nonmax_tok_choices}) > {max_possible_count}"
                    # cur_largest_wrong_logit < 0 -> the model gets it right (and the non-nan just ensures its valid)
                    # N.B. Here, n_copies_nonmax does NOT include the query token
                    correct_count += cur_count
                    # consider the space where there's one dimension for each input token that controls "do we use this or not"
                    # we consider a subspace where "did the model get it right" is convex in this space
                    # i.e. if you get it for non-max token = x and non-max token = y, you get it for all sequences
                    # that contain any combination of x and y (note that there can only be 1, 2, or 3 non-max tokens in
                    # the sequence (where 3 occurs only when the query token is the max token))
                    # You can extend this to 3 non-max token choices by induction

    return correct_count


def compute_accuracy_lower_bound_from_cubic(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    ],
) -> Tuple[float, Tuple[int, int]]:
    """
    returns correct_count / total_sequences, (correct_count, total_sequences)
    """
    d_vocab_q, d_vocab_max, _, n_ctx = largest_wrong_logit.shape
    correct_count = count_correct_sequences_cubic(largest_wrong_logit)
    total_sequences = d_vocab_max**n_ctx
    return correct_count / total_sequences, (correct_count, total_sequences)


def find_proof(model: HookedTransformer):
    return {}


def verify_proof(
    model: HookedTransformer,
    proof_args: dict,
    *,
    print_complexity: Union[bool, Callable[[str], None]] = True,
    print_results: Union[bool, Callable[[str], None]] = True,
    sanity_check: bool = True,
    pbar: Optional[tqdm] = None,
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

    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = add_time(EU_PU, model)  # noqa: F722
    print_complexity(
        f"Complexity of EU_PU: {complexity_of(EU_PU)}"
    )  # O(d_vocab^2 * d_model)
    EVOU: Float[Tensor, "d_vocab d_vocab_out"] = add_time(all_EVOU, model)  # noqa: F722
    print_complexity(
        f"Complexity of EVOU: {complexity_of(all_EVOU)}"
    )  # O(d_vocab^2 * d_model)
    PVOU: Float[Tensor, "n_ctx d_vocab_out"] = add_time(all_PVOU, model)  # noqa: F722
    print_complexity(
        f"Complexity of PVOU: {complexity_of(all_PVOU)}"
    )  # O(n_ctx * d_vocab * d_model)
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"] = add_time(all_EQKE, model)  # noqa: F722
    print_complexity(
        f"Complexity of EQKE: {complexity_of(all_EQKE)}"
    )  # O(d_vocab^2 * d_model)
    EQKP: Float[Tensor, "d_vocab_q n_ctx_k"] = add_time(all_EQKP, model)  # noqa: F722
    print_complexity(
        f"Complexity of EQKP: {complexity_of(all_EQKP)}"
    )  # O(d_vocab * d_model * n_ctx)

    extreme_right_attention_softmaxed_cubic: Float[
        Tensor,
        "minmax=2 attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
    ] = add_time(
        compute_extreme_softmaxed_right_attention_cubic_simple,
        EQKE=EQKE,
        EQKP=EQKP,
        attn_scale=model.blocks[0].attn.attn_scale,
        pbar=pbar,
    )
    print_complexity(
        f"Complexity of compute_extreme_softmaxed_right_attention_cubic_simple: {complexity_of(compute_extreme_softmaxed_right_attention_cubic_simple)}"
    )  # O(d_vocab^3 * n_ctx^2)
    if sanity_check:
        sanity_check_diff = (
            extreme_right_attention_softmaxed_cubic[1, 0]
            - extreme_right_attention_softmaxed_cubic[0, 0]
        )
        sanity_check_diff_failed_indices = [
            tuple(i.item() for i in j)
            for j in zip(*torch.where(sanity_check_diff < -1e-6))
        ]
        assert (
            len(sanity_check_diff_failed_indices) == 0
        ), f"Sanity check failed: {sanity_check_diff_failed_indices}, {extreme_right_attention_softmaxed_cubic[(slice(None), slice(None)) + sanity_check_diff_failed_indices[0]]}"

    largest_wrong_logit_cubic: Float[
        Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    ] = add_time(
        compute_largest_wrong_logit_cubic,
        extreme_right_attention_softmaxed_cubic,
        EUPU=EUPU,
        EVOU=EVOU,
        PVOU=PVOU,
    )
    print_complexity(
        f"Complexity of compute_largest_wrong_logit_cubic: {complexity_of(compute_largest_wrong_logit_cubic)}"
    )  # O(d_vocab^3 * n_ctx^2)

    accuracy_bound_cubic, (
        correct_count_cubic,
        total_sequences,
    ) = add_time(compute_accuracy_lower_bound_from_cubic, largest_wrong_logit_cubic)
    print_results(
        f"Cubic Accuracy lower bound: {accuracy_bound_cubic} ({correct_count_cubic} correct sequences of {total_sequences})"
    )
    prooftime = sum(prooftimes)
    print_results(f"Cubic Proof time: {prooftime}s")
    return {
        "largest_wrong_logit": largest_wrong_logit_cubic,
        "accuracy_lower_bound": accuracy_bound_cubic,
        "correct_count_lower_bound": correct_count_cubic,
        "total_sequences": total_sequences,
        "prooftime": prooftime,
    }
