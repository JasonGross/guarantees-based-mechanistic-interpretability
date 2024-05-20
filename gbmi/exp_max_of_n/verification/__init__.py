# N.B. DO NOT import annotations from __future__ or else enumerate_dataclass_values will break on LargestWrongLogitQuadraticConfig
import dataclasses
from typing import ClassVar, Literal, Tuple, Union, Dict, Any, Optional
from enum import Enum

import numpy as np
import torch
from jaxtyping import Float, Integer
from torch import Tensor
from functools import reduce, cache
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)  # , FactoredMatrix
from gbmi.utils.FactoredMatrix import FactoredMatrix

from gbmi.utils import dropnan
from gbmi.analysis_tools.plot import summarize
from gbmi.analysis_tools.utils import make_local_tqdm
from gbmi.utils.sequences import generate_all_sequences_for_model
from gbmi.utils.sequences import generate_all_sequences
from gbmi.verification_tools.l1h1 import all_EVOU, all_PVOU, all_attention_scores
from gbmi.verification_tools.decomp import (
    max_row_diffs_per_dim,
    max_row_diffs_per_dim_no_multipy,
    bound_max_row_diff_by_SVD,
)
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils import bits_of_type


@torch.no_grad()
def logit_delta_of_results(
    all_tokens: Integer[Tensor, "batch n_ctx"],  # noqa: F722
    predicted_logits: Float[Tensor, "d_vocab_out"],  # noqa: F821
    renderer=None,
    histogram_all_incorrect_logit_differences: bool = False,
    return_summary: bool = False,
    hist_args={},
) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max.
    """
    (batch, n_ctx), (_batch, d_vocab_out) = all_tokens.shape, predicted_logits.shape
    assert predicted_logits.shape == (
        batch,
        d_vocab_out,
    ), f"predicted_logits.shape = {predicted_logits.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"

    # Extract statistics for each row
    # Use values in all_tokens as indices to gather correct logits
    indices_of_max = all_tokens.max(dim=-1, keepdim=True).values
    assert indices_of_max.shape == (
        batch,
        1,
    ), f"indices_of_max.shape = {indices_of_max.shape} != {(batch, 1)} = (batch, 1)"
    correct_logits = torch.gather(predicted_logits, -1, indices_of_max)
    assert correct_logits.shape == (
        batch,
        1,
    ), f"correct_logits.shape = {correct_logits.shape} != {(batch, 1)} = (batch, 1)"
    logits_above_correct = correct_logits - predicted_logits
    assert logits_above_correct.shape == (
        batch,
        d_vocab_out,
    ), f"logits_above_correct.shape = {logits_above_correct.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"
    # replace correct logit indices with large number so that they don't get picked up by the min
    logits_above_correct[
        torch.arange(logits_above_correct.shape[0]), indices_of_max.squeeze()
    ] = float("inf")
    min_incorrect_logit = logits_above_correct.min(dim=-1).values
    assert min_incorrect_logit.shape == (
        batch,
    ), f"min_incorrect_logit.shape = {min_incorrect_logit.shape} != {(batch,)} = (batch,)"

    if histogram_all_incorrect_logit_differences:
        all_incorrect_logits = logits_above_correct[
            logits_above_correct != float("inf")
        ]
        summarize(
            all_incorrect_logits,
            name="all incorrect logit differences",
            histogram=True,
            hist_args=hist_args,
            renderer=renderer,
        )

    if return_summary:
        return summarize(
            min_incorrect_logit,
            name="min(correct logit - incorrect logit)",
            renderer=renderer,
            histogram=True,
        )

    else:
        return min_incorrect_logit.min().item()


@torch.no_grad()
def logit_delta(
    model: HookedTransformer,
    renderer=None,
    histogram_all_incorrect_logit_differences: bool = False,
    return_summary: bool = False,
    hist_args={},
) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max.
    Complexity: O(d_vocab^n_ctx * fwd_pass)
    Complexity: fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    Complexity: n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = (
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_vocab_out,
        model.cfg.d_model,
    )

    all_tokens = generate_all_sequences_for_model(model=model)
    assert all_tokens.shape == (
        d_vocab**n_ctx,
        n_ctx,
    ), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits = model(all_tokens)[:, -1, :].detach().cpu()
    assert predicted_logits.shape == (
        d_vocab**n_ctx,
        d_vocab_out,
    ), f"predicted_logits.shape = {predicted_logits.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"

    return logit_delta_of_results(
        all_tokens=all_tokens,
        predicted_logits=predicted_logits,
        renderer=renderer,
        histogram_all_incorrect_logit_differences=histogram_all_incorrect_logit_differences,
        return_summary=return_summary,
        hist_args=hist_args,
    )


@torch.no_grad()
def compute_gap(
    all_tokens: Integer[Tensor, "batch n_ctx"]  # noqa: F722
) -> Integer[Tensor, "batch"]:  # noqa: F821
    """
    computes the gap between the max token and the second max token in each row of all_tokens
    """
    maxv = all_tokens.max(dim=-1, keepdim=True).values
    all_but_maxv = all_tokens.clone()
    all_but_maxv[all_but_maxv == maxv] = -all_tokens.max().item()
    second_maxv = all_but_maxv.max(dim=-1, keepdim=True).values
    second_maxv[second_maxv < 0] = maxv[second_maxv < 0]
    return (maxv - second_maxv)[:, 0]


@torch.no_grad()
def all_tokens_small_gap(
    model: HookedTransformer, max_min_gap: int = 1
) -> Integer[Tensor, "batch n_ctx"]:  # noqa: F722
    """
    All sequences of tokens with the constraint that some token z in the sequence satisfies true_max - max_min_gap <= z < true_max
    Complexity: O(d_vocab ^ (n_ctx - 1) * (max_min_gap * 2 + 1))
    """
    n_ctx, d_vocab = model.cfg.n_ctx, model.cfg.d_vocab

    all_tokens_after_start = generate_all_sequences(
        n_digits=d_vocab, sequence_length=n_ctx - 1
    )
    all_tokens_after_start_max = all_tokens_after_start.max(dim=-1, keepdim=True).values
    all_tokens_after_start_max_minf = all_tokens_after_start.clone()
    all_tokens_after_start_max_minf[
        all_tokens_after_start_max_minf == all_tokens_after_start_max
    ] = (-max_min_gap - 1)
    all_tokens_after_start_second_max = all_tokens_after_start_max_minf.max(
        dim=-1, keepdim=True
    ).values
    first_token_max = all_tokens_after_start_max + max_min_gap + 1
    gap_already_present = (
        all_tokens_after_start_second_max >= all_tokens_after_start_max - max_min_gap
    )
    first_token_upper_min = all_tokens_after_start_max + gap_already_present.long()
    first_token_min = torch.zeros_like(first_token_max)
    first_token_min[~gap_already_present] = (
        all_tokens_after_start_max[~gap_already_present] - max_min_gap
    )
    first_token_min[first_token_min < 0] = 0
    first_token_max[first_token_max >= d_vocab] = d_vocab
    first_token_upper_min[first_token_upper_min >= d_vocab] = d_vocab
    assert first_token_max.shape == (
        d_vocab ** (n_ctx - 1),
        1,
    ), f"first_token_max.shape = {first_token_max.shape} != {(d_vocab**(n_ctx - 1), 1)} = (d_vocab**(n_ctx - 1), 1)"
    assert first_token_upper_min.shape == (
        d_vocab ** (n_ctx - 1),
        1,
    ), f"first_token_upper_min.shape = {first_token_upper_min.shape} != {(n_ctx, 1)} = (d_vocab**(n_ctx - 1), 1)"
    assert all_tokens_after_start_max.shape == (
        d_vocab ** (n_ctx - 1),
        1,
    ), f"all_tokens_after_start_max.shape = {all_tokens_after_start_max.shape} != {(d_vocab**(n_ctx - 1), 1)} = (d_vocab**(n_ctx - 1), 1)"
    assert first_token_min.shape == (
        d_vocab ** (n_ctx - 1),
        1,
    ), f"first_token_min.shape = {first_token_min.shape} != {(d_vocab**(n_ctx - 1), 1)} = (d_vocab**(n_ctx - 1), 1)"
    (
        first_token_max,
        first_token_upper_min,
        all_tokens_after_start_max,
        first_token_min,
    ) = (
        first_token_max[:, 0],
        first_token_upper_min[:, 0],
        all_tokens_after_start_max[:, 0],
        first_token_min[:, 0],
    )
    first_token_ranges = [
        torch.cat([torch.arange(lower, mid), torch.arange(lower_big, upper)])
        for lower, mid, lower_big, upper in zip(
            first_token_min,
            all_tokens_after_start_max,
            first_token_upper_min,
            first_token_max,
        )
    ]
    all_tokens_with_small_gap = torch.cat(
        [
            torch.cartesian_prod(first_tokens, *rest_tokens[:, None])
            for first_tokens, rest_tokens in zip(
                first_token_ranges, all_tokens_after_start
            )
        ]
    )

    return all_tokens_with_small_gap


@torch.no_grad()
def logit_delta_small_gap_exhaustive(
    model: HookedTransformer,
    max_min_gap: int = 1,
    renderer=None,
    histogram_all_incorrect_logit_differences: bool = False,
    return_summary: bool = False,
    hist_args={},
) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max, with the constraint that some token z in the sequence satisfies true_max - max_min_gap <= z < true_max
    Complexity: O(d_vocab ^ (n_ctx - 1) * (max_min_gap * 2 + 1) * fwd_pass)
    Complexity: fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    Complexity: n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = (
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_vocab_out,
        model.cfg.d_model,
    )

    all_tokens = all_tokens_small_gap(model, max_min_gap=max_min_gap)
    assert (
        len(all_tokens.shape) == 2 and all_tokens.shape[1] == n_ctx
    ), f"all_tokens.shape = {all_tokens.shape} != (_, {n_ctx}) = (_, n_ctx)"
    predicted_logits = model(all_tokens)[:, -1, :].detach().cpu()
    assert (
        len(predicted_logits.shape) == 2 and predicted_logits.shape[1] == d_vocab_out
    ), f"predicted_logits.shape = {predicted_logits.shape} != (_, {d_vocab_out}) = (_, d_vocab_out)"

    return logit_delta_of_results(
        all_tokens=all_tokens,
        predicted_logits=predicted_logits,
        renderer=renderer,
        histogram_all_incorrect_logit_differences=histogram_all_incorrect_logit_differences,
        return_summary=return_summary,
        hist_args=hist_args,
    )


@torch.no_grad()
def logit_delta_by_gap(
    model: HookedTransformer,
    renderer=None,
    histogram_all_incorrect_logit_differences: bool = False,
    return_summary: bool = False,
    hist_args={},
) -> Dict[int, Union[float, Dict[str, Any]]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max, with the constraint that all non-max tokens in the sequence are strictly more than gap away from the true max, indexed by gap
    Complexity: O(d_vocab ^ n_ctx * fwd_pass)
    Complexity: fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    Complexity: n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = (
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_vocab_out,
        model.cfg.d_model,
    )

    all_tokens = generate_all_sequences_for_model(model=model)
    assert all_tokens.shape == (
        d_vocab**n_ctx,
        n_ctx,
    ), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits = model(all_tokens)[:, -1, :].detach().cpu()
    assert predicted_logits.shape == (
        all_tokens.shape[0],
        d_vocab_out,
    ), f"predicted_logits.shape = {predicted_logits.shape} != {(all_tokens.shape[0], d_vocab_out)} = (all_tokens.shape[0], d_vocab_out)"
    gaps = compute_gap(all_tokens)
    assert gaps.shape == (
        all_tokens.shape[0],
    ), f"gaps.shape = {gaps.shape} != {(all_tokens.shape[0],)} = (all_tokens.shape[0],)"
    return {
        gap: logit_delta_of_results(
            all_tokens=all_tokens[gaps == gap, :],
            predicted_logits=predicted_logits[gaps == gap, :],
            renderer=renderer,
            histogram_all_incorrect_logit_differences=histogram_all_incorrect_logit_differences,
            return_summary=return_summary,
            hist_args=hist_args,
        )
        for gap in range(d_vocab)
    }


@torch.no_grad()
def worst_PVOU_gap_for(
    model: HookedTransformer,
    query_tok: int,
    max_tok: int,
    min_gap: int = 0,
    PVOU: Optional[Float[Tensor, "n_ctx d_vocab_out"]] = None,  # noqa: F722
    attention_score_map: Optional[
        Float[Tensor, "n_ctx_k d_vocab_q d_vocab_k"]  # noqa: F722
    ] = None,
    optimize_max_query_comparison=True,
) -> Float[Tensor, "d_vocab_out"]:  # noqa: F821
    """
    Returns a map of non_max_output_tok to PVOU with the worst (largest) value of PVOU[non_max_output_tok] - PVOU[max_tok],
        across all possible attention scalings for the query token and for token values <= max_tok - min_gap.
    Complexity: O(PVOU + attention_score_map + d_vocab_out * n_ctx^2)
    Complexity: ~ O(n_ctx * d_vocab * d_model^2 (from PVOU) + d_vocab * d_head^2 * d_model * n_ctx (from attention_score_map) + (n_ctx * log(n_ctx) (sorting) + n_ctx^2) * d_vocab)
    Complexity: (for n_ctx=2) O(POVU + attention_score_map + n_ctx)
    N.B. Clever caching could reduce n_ctx^2 to n_ctx, leaving n_ctx log(n_ctx) from sorting as the dominant factor
    N.B. If optimize_max_query_comparison is set, and n_ctx is 2, then whenever query_tok != max_tok we know exactly what the sequence is and can just compute the attention
    """
    assert max_tok >= query_tok, f"max_tok = {max_tok} < {query_tok} = query_tok"
    assert (
        max_tok == query_tok or max_tok >= query_tok + min_gap
    ), f"max_tok = {max_tok} < {query_tok} + {min_gap} = query_tok + min_gap"
    n_ctx, d_vocab_out, d_vocab = (
        model.cfg.n_ctx,
        model.cfg.d_vocab_out,
        model.cfg.d_vocab,
    )
    if PVOU is None:
        PVOU = all_PVOU(model)
    assert PVOU.shape == (
        n_ctx,
        d_vocab_out,
    ), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    if attention_score_map is None:
        attention_score_map = all_attention_scores(model)
    assert attention_score_map.shape == (
        n_ctx,
        d_vocab,
        d_vocab,
    ), f"attention_scores.shape = {attention_score_map.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    worst_attention_score = torch.zeros((n_ctx,))
    worst_attention_score[-1] = attention_score_map[-1, query_tok, query_tok]
    if n_ctx == 2 and optimize_max_query_comparison and query_tok != max_tok:
        worst_attention_score[0] = attention_score_map[0, query_tok, max_tok]
        worst_PVOU = worst_attention_score.softmax(dim=-1) @ PVOU
        return worst_PVOU - worst_PVOU[max_tok]
    elif max_tok - min_gap < 0:
        # everything must be the max
        worst_PVOU = attention_score_map[:, query_tok, max_tok].softmax(dim=-1) @ PVOU
        return worst_PVOU - worst_PVOU[max_tok]
    else:
        # compute the min and max attention scores for each position and query token where the key token is either max_tok or <= max_tok - gap
        min_attention_scores_below_gap, max_attention_scores_below_gap = (
            attention_score_map[:-1, query_tok, : max_tok + 1 - min_gap]
            .min(dim=-1)
            .values,
            attention_score_map[:-1, query_tok, : max_tok + 1 - min_gap]
            .max(dim=-1)
            .values,
        )
        assert min_attention_scores_below_gap.shape == (
            n_ctx - 1,
        ), f"min_attention_scores.shape = {min_attention_scores_below_gap.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        assert max_attention_scores_below_gap.shape == (
            n_ctx - 1,
        ), f"max_attention_scores.shape = {max_attention_scores_below_gap.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        min_attention_scores = torch.minimum(
            attention_score_map[:-1, query_tok, max_tok], min_attention_scores_below_gap
        )
        max_attention_scores = torch.maximum(
            attention_score_map[:-1, query_tok, max_tok], max_attention_scores_below_gap
        )
        assert min_attention_scores.shape == (
            n_ctx - 1,
        ), f"min_attention_scores.shape = {min_attention_scores.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        assert max_attention_scores.shape == (
            n_ctx - 1,
        ), f"max_attention_scores.shape = {max_attention_scores.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        worst_attention_score[:-1] = min_attention_scores
        PVOU = PVOU.T
        assert PVOU.shape == (
            d_vocab_out,
            n_ctx,
        ), f"PVOU.T.shape = {PVOU.shape} != {(d_vocab_out, n_ctx)} = (d_vocab_out, n_ctx)"
        worst_PVOU = torch.zeros((d_vocab_out,))
        d_PVOU = PVOU[:, :] - PVOU[max_tok, :][None, :]
        assert d_PVOU.shape == (
            d_vocab_out,
            n_ctx,
        ), f"d_PVOU.shape = {d_PVOU.shape} != {(d_vocab_out, n_ctx)} = (d_vocab_out, n_ctx)"
        # sort d_PVOU in descending order
        _, d_PVOU_idxs = d_PVOU[:, :-1].sort(dim=-1, descending=True)
        for non_max_output_tok in range(d_vocab_out):
            worst_attention_score[:-1] = min_attention_scores
            for i in d_PVOU_idxs[non_max_output_tok, :]:
                # compare d_PVOU weighted by softmax of worst_attention_score for worst_attention_score[i] in (min_attention_scores[i], max_attention_scores[i])
                # set worst_attention_score[i] to whichever one is worse (more positive)
                # print(d_PVOU.shape, worst_attention_score.softmax(dim=-1).shape)
                min_d_PVOU = (
                    worst_attention_score.softmax(dim=-1)
                    @ d_PVOU[non_max_output_tok, :]
                )
                worst_attention_score[i] = max_attention_scores[i]
                max_d_PVOU = (
                    worst_attention_score.softmax(dim=-1)
                    @ d_PVOU[non_max_output_tok, :]
                )
                if min_d_PVOU > max_d_PVOU:
                    worst_attention_score[i] = min_attention_scores[i]
            worst_PVOU[non_max_output_tok] = (
                worst_attention_score.softmax(dim=-1) @ d_PVOU[non_max_output_tok, :]
            )
            # print(i, min_attention_scores[i], worst_attention_score[i], max_attention_scores[i], min_d_PVOU, max_d_PVOU, d_PVOU[i])
        # return the PVOU for the worst_attention_score
        return worst_PVOU


@torch.no_grad()
def all_worst_PVOU(
    model: HookedTransformer, min_gap: int = 0, tqdm=None, **kwargs
) -> Float[Tensor, "d_vocab_q d_vocab_max d_vocab_out"]:  # noqa: F722
    """
    Returns the mixture of PVOUs with the worst (largest) value of PVOU[non_max_output_tok] - PVOU[max_tok], across all possible attention scalings for the query token and for token values <= max_tok - min_gap.
    Complexity: O(PVOU + attention_score_map + n_ctx^2 * d_vocab^3)
    Complexity: ~ O(n_ctx * d_vocab * d_model^2 (from PVOU) + d_vocab * d_head^2 * d_model * n_ctx (from attention_score_map) + (n_ctx * log(n_ctx) (sorting) + n_ctx^2) * d_vocab^3)
    Complexity: (for n_ctx=2) O(PVOU + attention_score_map + n_ctx * d_vocab^2)
    N.B. Clever caching could reduce n_ctx^2 to n_ctx, leaving n_ctx log(n_ctx) * d_vocab^3 from sorting as the dominant factor.
    N.B. for max_of_{two,three}, this is maybe? worse than exhaustive enumeration (oops)
    """
    local_tqdm = make_local_tqdm(tqdm)
    n_ctx, d_vocab_out, d_vocab = (
        model.cfg.n_ctx,
        model.cfg.d_vocab_out,
        model.cfg.d_vocab,
    )
    PVOU = all_PVOU(model)
    assert PVOU.shape == (
        n_ctx,
        d_vocab_out,
    ), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    attention_score_map = all_attention_scores(model)
    assert attention_score_map.shape == (
        n_ctx,
        d_vocab,
        d_vocab,
    ), f"attention_scores.shape = {attention_score_map.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    result = torch.zeros((d_vocab, d_vocab, d_vocab_out)) + float("nan")
    for query_tok in local_tqdm(range(d_vocab), total=d_vocab):
        for max_tok in [query_tok] + list(
            range(query_tok + np.max([1, min_gap]), d_vocab)
        ):
            result[query_tok, max_tok, :] = worst_PVOU_gap_for(
                model,
                query_tok,
                max_tok,
                min_gap=min_gap,
                PVOU=PVOU,
                attention_score_map=attention_score_map,
                **kwargs,
            )

    return result


@torch.no_grad()
def worst_EVOU_gap_for(
    model: HookedTransformer,
    query_tok: int,
    max_tok: int,
    min_gap: int = 0,
    EVOU: Optional[Float[Tensor, "d_vocab d_vocab_out"]] = None,  # noqa: F722
    attention_score_map: Optional[
        Float[Tensor, "n_ctx_k d_vocab_q d_vocab_k"]  # noqa: F722
    ] = None,
    optimize_max_query_comparison=True,
) -> Float[Tensor, "d_vocab_out"]:  # noqa: F821
    """
    Returns the map of non_max_output_tok to worst (largest) value of EVOU[non_max_output_tok] - EVOU[max_tok], across all possible attention scalings for the query token
        and for token values <= max_tok - min_gap.
    To deal with the fact that attention and EVOU are not truly independent, we relax the "worst" calculation by saying that the attention paid to a given token in a given position
        is the min of (most attention paid to this token in this position) and (most attention paid to any token < max in this position).
    "<" is relaxed to "<=" when the token under consideration is the max token.

    Complexity: O(EVOU + attention_score_map + n_ctx * d_vocab + d_vocab^2)
    Complexity: (for n_ctx=2) O(EOVU + attention_score_map + d_vocab + n_ctx)
    #N.B. If optimize_max_query_comparison is set, and n_ctx is 2, then whenever query_tok != max_tok we know exactly what the sequence is and can just compute the attention
    """
    assert max_tok >= query_tok, f"max_tok = {max_tok} < {query_tok} = query_tok"
    assert (
        max_tok == query_tok or max_tok >= query_tok + min_gap
    ), f"max_tok = {max_tok} < {query_tok} + {min_gap} = query_tok + min_gap"
    n_ctx, d_vocab_out, d_vocab = (
        model.cfg.n_ctx,
        model.cfg.d_vocab_out,
        model.cfg.d_vocab,
    )
    if EVOU is None:
        EVOU = all_EVOU(model)
    assert EVOU.shape == (
        d_vocab,
        d_vocab_out,
    ), f"EVOU.shape = {EVOU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    if attention_score_map is None:
        attention_score_map = all_attention_scores(model)
    assert attention_score_map.shape == (
        n_ctx,
        d_vocab,
        d_vocab,
    ), f"attention_scores.shape = {attention_score_map.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    if n_ctx == 2 and optimize_max_query_comparison and query_tok != max_tok:
        worst_attention_score = torch.zeros((n_ctx,))
        worst_attention_score[-1] = attention_score_map[-1, query_tok, query_tok]
        worst_attention_score[0] = attention_score_map[0, query_tok, max_tok]
        worst_EVOU = (
            worst_attention_score.softmax(dim=-1)
            @ EVOU[torch.tensor([max_tok, query_tok]), :]
        )
        return worst_EVOU - worst_EVOU[max_tok]
    elif max_tok - min_gap < 0:
        # everything must be the max
        assert max_tok == query_tok, f"max_tok = {max_tok} != {query_tok} = query_tok"
        worst_EVOU = EVOU[max_tok, :]
        return worst_EVOU - worst_EVOU[max_tok]
    else:
        # for each non-query position, compute the min and max attention scores for that position and query token where the key token is < max_tok, and also when the key token is <= max_tok
        max_nonmax_tok = np.min([max_tok - 1, max_tok - min_gap])
        min_attention_scores_without_max, max_attention_scores_without_max = (
            attention_score_map[:-1, query_tok, : max_nonmax_tok + 1]
            .min(dim=-1)
            .values,
            attention_score_map[:-1, query_tok, : max_nonmax_tok + 1]
            .max(dim=-1)
            .values,
        )
        assert min_attention_scores_without_max.shape == (
            n_ctx - 1,
        ), f"min_attention_scores_without_max.shape = {min_attention_scores_without_max.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        assert max_attention_scores_without_max.shape == (
            n_ctx - 1,
        ), f"max_attention_scores_without_max.shape = {max_attention_scores_without_max.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        # for each key token below the max, compute the min and max attention scores for that token and query token where the key token is <= max_tok
        # if query token is max, we assume all other tokens are the same; otherwise, we pick the minimal attention slot for the max token and the other slots for the non-max, except when we consider all maxes but the query
        # we must subtract off the maximum to avoid overflow, as per https://github.com/pytorch/pytorch/blob/bc047ec906d8e1730e2ccd8192cef3c3467d75d1/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L115-L136
        attention_to_query = attention_score_map[-1, query_tok, query_tok]
        attentions_to_max = attention_score_map[:-1, query_tok, max_tok]
        attention_offset = torch.maximum(attentions_to_max.max(), attention_to_query)
        attention_to_max_exp = (attentions_to_max - attention_offset).exp().sum()
        attention_to_query_exp = (attention_to_query - attention_offset).exp()
        attention_sum = attention_to_max_exp + attention_to_query_exp
        EVOUs = torch.zeros((max_tok + 1, d_vocab_out))
        EVOUs[max_tok, :] = (
            EVOU[max_tok, :] * attention_to_max_exp / attention_sum
            + EVOU[query_tok, :] * attention_to_query_exp / attention_sum
        )
        assert EVOUs[max_tok, :].shape == (
            d_vocab_out,
        ), f"EVOU_all_maxes.shape = {EVOUs[max_tok, :].shape} != {(d_vocab_out,)} = (d_vocab_out,)"

        # consider all tokens < max, compute EVOU for each
        attention_to_max = attention_score_map[:-1, query_tok, max_tok].min()
        for non_max_tok in range(max_nonmax_tok + 1):
            # we need to relax attention to non-max, picking the attention to this slot from the min of largest attention to this token and largest attention to this slot
            max_attention_to_non_max = attention_score_map[
                :-1, query_tok, non_max_tok
            ].max()
            attention_to_non_max = torch.minimum(
                max_attention_to_non_max, max_attention_scores_without_max
            )
            if query_tok == max_tok:
                # we must subtract off the maximum to avoid overflow, as per https://github.com/pytorch/pytorch/blob/bc047ec906d8e1730e2ccd8192cef3c3467d75d1/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L115-L136
                attention_offset = torch.maximum(
                    attention_to_query, attention_to_non_max.max()
                )
                attention_to_max_exp = (attention_to_max - attention_offset).exp()
                attention_to_query_exp = (attention_to_query - attention_offset).exp()
                attention_to_non_max_exp = (
                    (attention_to_non_max - attention_offset).exp().sum()
                )
                attention_sum = attention_to_non_max_exp + attention_to_query_exp
                EVOUs[non_max_tok, :] = (
                    EVOU[non_max_tok, :] * attention_to_non_max_exp / attention_sum
                    + EVOU[query_tok, :] * attention_to_query_exp / attention_sum
                )
            else:
                # we must subtract off the maximum to avoid overflow, as per https://github.com/pytorch/pytorch/blob/bc047ec906d8e1730e2ccd8192cef3c3467d75d1/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L115-L136
                attention_offset = torch.maximum(
                    torch.maximum(attention_to_max, attention_to_query),
                    attention_to_non_max.max(),
                )
                attention_to_non_max_exp = (
                    attention_to_non_max - attention_offset
                ).exp()
                # drop the smallest value in attention_to_non_max
                attention_to_non_max_exp = (
                    attention_to_non_max_exp.sum() - attention_to_non_max_exp.min()
                )
                attention_to_max_exp = (attention_to_max - attention_offset).exp()
                attention_to_query_exp = (attention_to_query - attention_offset).exp()
                attention_sum = (
                    attention_to_non_max_exp
                    + attention_to_query_exp
                    + attention_to_max_exp
                )
                EVOUs[non_max_tok, :] = (
                    EVOU[non_max_tok, :] * attention_to_non_max_exp / attention_sum
                    + EVOU[query_tok, :] * attention_to_query_exp / attention_sum
                    + EVOU[max_tok, :] * attention_to_max_exp / attention_sum
                )
        # subtract off the max_tok EVOU
        # print(EVOUs)
        EVOUs = EVOUs - EVOUs[:, max_tok][:, None]
        # return the worst EVOU
        return EVOUs.max(dim=0).values


@torch.no_grad()
def all_worst_EVOU(
    model: HookedTransformer, min_gap: int = 0, tqdm=None, **kwargs
) -> Float[Tensor, "d_vocab_q d_vocab_max d_vocab_out"]:  # noqa: F722
    """
    Returns the mixture of EVOUs with the worst (largest) value of EVOU[non_max_output_tok] - EVOU[max_tok], across all possible attention scalings for the query token and for token values <= max_tok - min_gap.
    Complexity: O(EVOU + attention_score_map + (n_ctx + d_vocab) * d_vocab^3)
    Complexity: (for n_ctx=2) O(EVOU + attention_score_map + (n_ctx + d_vocab) * d_vocab^2)
    N.B. for max_of_{two,three}, this is maybe? worse than exhaustive enumeration (oops)
    """
    local_tqdm = make_local_tqdm(tqdm)
    n_ctx, d_vocab_out, d_vocab = (
        model.cfg.n_ctx,
        model.cfg.d_vocab_out,
        model.cfg.d_vocab,
    )
    EVOU = all_EVOU(model)
    assert EVOU.shape == (
        d_vocab,
        d_vocab_out,
    ), f"EVOU.shape = {EVOU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    attention_score_map = all_attention_scores(model)
    assert attention_score_map.shape == (
        n_ctx,
        d_vocab,
        d_vocab,
    ), f"attention_scores.shape = {attention_score_map.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    result = torch.zeros((d_vocab, d_vocab, d_vocab_out)) + float("nan")
    for query_tok in local_tqdm(range(d_vocab), total=d_vocab):
        for max_tok in [query_tok] + list(
            range(query_tok + np.max([1, min_gap]), d_vocab)
        ):
            result[query_tok, max_tok, :] = worst_EVOU_gap_for(
                model,
                query_tok,
                max_tok,
                min_gap=min_gap,
                EVOU=EVOU,
                attention_score_map=attention_score_map,
                **kwargs,
            )

    return result


@dataclasses.dataclass
class EffectiveDimensionApproximation:
    R_exponent: int = 0
    multiplicand: int = 1
    divisor: int = 1

    def __add__(
        self, other: "EffectiveDimensionApproximation"
    ) -> "EffectiveDimensionApproximation":
        return EffectiveDimensionApproximation(
            R_exponent=self.R_exponent + other.R_exponent,
            multiplicand=self.multiplicand * other.multiplicand,
            divisor=self.divisor * other.divisor,
        )

    def __int__(self) -> int:
        return self.R_exponent

    def __float__(self, float_type) -> float:
        nbits = bits_of_type(float_type)
        return int(self) + (np.log2(self.multiplicand) - np.log2(self.divisor)) / nbits


@dataclasses.dataclass
class LargestWrongLogitQuadraticConfig:
    EUPU_handling: Literal[
        "mean_query+max_diff",
        "svd_query+max_diff",
        "max_diff",
        "max_diff_exact",
        "global_max_diff_exact",
    ] = "mean_query+max_diff"
    attention_handling: Literal[
        "mean_query+diff", "drop_average_query_per_output_logit_reasoning"
    ] = "mean_query+diff"
    attention_error_handling: Literal[
        "svd",
        "max_diff",
        "max_diff_subproduct",
        "max_diff_subproduct_recursive",
        "mean+max_diff",
        "mean+max_diff_subproduct",
        "mean+max_diff_subproduct_recursive",
        "mean_recursive+max_diff_subproduct_recursive",
        "exact_EQKE+max_diff_exact",
        "max_diff_exact",
    ] = "max_diff"

    EUPU_OFF: ClassVar[Literal["global_max_diff_exact"]] = "global_max_diff_exact"
    attention_handling_OFF: ClassVar[
        Literal["drop_average_query_per_output_logit_reasoning"]
    ] = "drop_average_query_per_output_logit_reasoning"
    attention_error_handling_OFF: ClassVar[Literal["svd"]] = "svd"

    @property
    def is_subcubic(self) -> bool:
        return (
            self.EUPU_handling_subcubic
            and self.attention_handling_subcubic
            and self.attention_error_handling_subcubic
        )

    def effective_dimension_estimate(
        self, cfg: HookedTransformerConfig
    ) -> EffectiveDimensionApproximation:
        return (
            self.EUPU_handling_effective_dimension_estimate(cfg)
            + self.attention_handling_effective_dimension_estimate(cfg)
            + self.attention_error_handling_effective_dimension_estimate(cfg)
        )

    @classmethod
    def OFF(cls) -> "LargestWrongLogitQuadraticConfig":
        return cls(
            EUPU_handling=cls.EUPU_OFF,
            attention_handling=cls.attention_handling_OFF,
            attention_error_handling=cls.attention_error_handling_OFF,
        )

    @classmethod
    @cache
    def all_values(cls) -> Tuple["LargestWrongLogitQuadraticConfig", ...]:
        return tuple(enumerate_dataclass_values(cls))

    @classmethod
    @cache
    def _parsing_dict(
        cls, latex: bool = False
    ) -> dict[str, "LargestWrongLogitQuadraticConfig"]:
        all_values = [(v.short_description(latex=latex), v) for v in cls.all_values()]
        result = dict(all_values)
        NL = "\n"
        assert len(result) == len(
            all_values
        ), f"The following values have equal descriptions (with latex={latex}): {NL.join(repr((k, v, result[k]) for k, v in result.items()))}"
        return result

    @classmethod
    def parse(
        cls, short_description: str, latex: bool = False
    ) -> "LargestWrongLogitQuadraticConfig":
        return cls._parsing_dict(latex=latex)[short_description]

    def split_EPU(
        self,
        W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa F722
        W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa F722
        W_EP_mean_query: Optional[Float[Tensor, "d_model"]] = None,  # noqa F722
    ) -> Tuple[Float[Tensor, "d_vocab_out"], Float[Tensor, "d_vocab_q"]]:  # noqa F821
        """
        Returns (EUPU_mean_query, EUPU_per_query_max_logit_diff)

        Note that this function is correct regardless of what direction is passed for W_EP_mean_query, which merely determines how good the bound is in the svd_query+max_diff case only

        Complexity: O(d_vocab_q * d_model + d_model * d_vocab_out) (+ d_vocab_q * d_model^2 if W_EP_mean_query is None and self.EUPU_handling == "svd_query+max_diff")
        """
        if self.EUPU_handling in ("global_max_diff_exact", "max_diff_exact"):
            return self.split_EUPU(W_EP @ W_U)
        if self.EUPU_handling == "mean_query+max_diff":
            W_EP_mean_query = W_EP.mean(dim=0)
        elif self.EUPU_handling == "svd_query+max_diff":
            if W_EP_mean_query is None:
                U, _, Vh = torch.linalg.svd(W_EP)
                W_EP_mean_query = U[:, 0] @ W_EP
        else:
            W_EP_mean_query = torch.zeros_like(W_EP).mean(dim=0)
        # help the type checker
        assert W_EP_mean_query is not None
        EUPU_mean_query: Float[Tensor, "d_vocab_out"] = (  # noqa F821
            W_EP_mean_query @ W_U
        )
        W_EP_per_query: Float[Tensor, "d_vocab_q d_model"] = (  # noqa F722
            W_EP - W_EP_mean_query[None, :]
        )
        W_U_per_query_max_logit_diff: Float[Tensor, "d_model"] = (  # noqa F821
            W_U.max(dim=-1).values - W_U.min(dim=-1).values
        )
        EUPU_per_query_max_logit_diff: Float[Tensor, "d_vocab_q"] = (  # noqa F821
            W_EP_per_query.abs() @ W_U_per_query_max_logit_diff
        )
        return EUPU_mean_query, EUPU_per_query_max_logit_diff

    def split_EUPU(
        self,
        EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa F722
    ) -> Tuple[Float[Tensor, "d_vocab_out"], Float[Tensor, "d_vocab_q"]]:  # noqa F821
        """
        Returns (EUPU_mean_query, EUPU_per_query_max_logit_diff)
        """
        EUPU_mean_query: Float[Tensor, "d_vocab_out"] = (  # noqa F821
            EUPU.mean(dim=0)
            if self.EUPU_handling == "mean_query+max_diff"
            else torch.zeros_like(EUPU).mean(dim=0)
        )
        EUPU_per_query: Float[Tensor, "d_vocab_q d_vocab_out"] = (  # noqa F722
            EUPU - EUPU_mean_query[None, :]
        )
        EUPU_per_query_max_logit_diff: Float[Tensor, "d_vocab_q"] = (  # noqa F821
            EUPU_per_query.max(dim=-1).values - EUPU_per_query.min(dim=-1).values
        )
        if self.EUPU_handling == "global_max_diff_exact":
            EUPU_per_query_max_logit_diff[:] = (
                EUPU_per_query.max() - EUPU_per_query.min()
            )
        return EUPU_mean_query, EUPU_per_query_max_logit_diff

    @property
    def EUPU_handling_quadratic(self) -> bool:
        if self.EUPU_handling in ("global_max_diff_exact", "max_diff"):
            return False
        return True

    @property
    def EUPU_handling_subcubic(self) -> bool:
        if self.EUPU_handling_quadratic:
            return True
        return True

    def EUPU_handling_effective_dimension_estimate(
        self, cfg: HookedTransformerConfig
    ) -> EffectiveDimensionApproximation:
        match self.EUPU_handling:
            case "global_max_diff_exact":
                # we find just the max minus the min, so we have just dimension 1 (or 2)
                return EffectiveDimensionApproximation(2, divisor=2)
            case "max_diff_exact":
                return EffectiveDimensionApproximation(cfg.d_vocab * 2, divisor=2)
            case "mean_query+max_diff" | "svd_query+max_diff":
                return EffectiveDimensionApproximation(
                    cfg.d_vocab + cfg.d_vocab * cfg.d_model + 2, divisor=2 * 2
                )  # TODO fix divisor from .abs() and max-min ordering
            case "max_diff":
                return EffectiveDimensionApproximation(
                    cfg.d_vocab * cfg.d_model + 2, divisor=2 * 2
                )  # TODO fix divisor from .abs() and max-min ordering

    def bound_attention_error(
        self, *matrices: Tensor
    ) -> Union[Float[Tensor, ""], Float[Tensor, "d_vocab_q"]]:  # noqa F821
        match self.attention_error_handling:
            case "svd":
                A, B, ms = matrices[0], matrices[1], matrices[2:]
                AB = FactoredMatrix(A, B)
                m = reduce(FactoredMatrix.__matmul__, ms, AB)  # type: ignore
                U, S, Vh = m.svd()
                return S[0] * np.sqrt(2)
            case (
                "max_diff"
                | "max_diff_subproduct"
                | "mean+max_diff"
                | "mean+max_diff_subproduct"
            ):
                use_mean_row = self.attention_error_handling.startswith("mean")
                return max_row_diffs_per_dim(*matrices, use_mean_row=use_mean_row)
            case (
                "max_diff_subproduct_recursive"
                | "mean+max_diff_subproduct_recursive"
                | "mean_recursive+max_diff_subproduct_recursive"
            ):
                use_mean_row = self.attention_error_handling.startswith("mean")
                use_mean_row_recursively = self.attention_error_handling.startswith(
                    "mean_recursively"
                )
                return max_row_diffs_per_dim_no_multipy(
                    *matrices,
                    use_mean_row=use_mean_row,
                    use_mean_row_recursively=use_mean_row_recursively,
                )
            case "max_diff_exact":
                m = reduce(torch.matmul, matrices)
                return m.max(dim=-1).values - m.min(dim=-1).values
            case "exact_EQKE+max_diff_exact":
                for m in matrices:
                    assert torch.allclose(
                        m, torch.zeros_like(m)
                    ), f"matrices should be zero when passing {self.attention_error_handling}, not {m}"
                return torch.tensor(0).to(matrices[0])

    @property
    def attention_error_handling_quadratic(self) -> bool:
        match self.attention_error_handling:
            case (
                "max_diff_subproduct_recursive"
                | "mean+max_diff_subproduct_recursive"
                | "mean_recursive+max_diff_subproduct_recursive"
            ):
                return True
            case _:
                return False

    @property
    def attention_error_handling_subcubic(self) -> bool:
        if self.attention_error_handling_quadratic:
            return True
        match self.attention_error_handling:
            case "svd":
                # low rank svd is considered subcubic
                return True
            case (
                "max_diff"
                | "max_diff_subproduct"
                | "mean+max_diff"
                | "mean+max_diff_subproduct"
            ):
                return True
            case (
                "max_diff_subproduct_recursive"
                | "mean+max_diff_subproduct_recursive"
                | "mean_recursive+max_diff_subproduct_recursive"
            ):
                assert False  # handled by quadratic
            case "max_diff_exact" | "exact_EQKE+max_diff_exact":
                return False  # involves full matmul

    def attention_error_handling_handling_effective_dimension_estimate(
        self, cfg: HookedTransformerConfig
    ) -> EffectiveDimensionApproximation:
        match self.attention_error_handling:
            case "svd":
                # d_vocab * 2 for principle components (singular value derived from those)
                return EffectiveDimensionApproximation(cfg.d_vocab * 2)
            case (
                "max_diff"
                | "max_diff_subproduct"
                | "mean+max_diff"
                | "mean+max_diff_subproduct"
            ):
                # we still do svd, but now we also keep some info about the error
                mean_count = (
                    cfg.d_vocab
                    if self.attention_error_handling.startswith("mean")
                    else 0
                )
                return EffectiveDimensionApproximation(
                    cfg.d_vocab * 2 + mean_count + cfg.d_model * 2
                )
            case (
                "max_diff_subproduct_recursive"
                | "mean+max_diff_subproduct_recursive"
                | "mean_recursive+max_diff_subproduct_recursive"
            ):
                pass
            #     mean_count = cfg.d_vocab if self.attention_error_handling.startswith("mean") else 0
            #     use_mean_row_recursively = self.attention_error_handling.startswith(
            #         "mean_recursively"
            #     )
            #     return max_row_diffs_per_dim_no_multipy(
            #         *matrices,
            #         use_mean_row=use_mean_row,
            #         use_mean_row_recursively=use_mean_row_recursively,
            #     )
            case "max_diff_exact":
                pass
            #     m = reduce(torch.matmul, matrices)
            #     return m.max(dim=-1).values - m.min(dim=-1).values
            case "exact_EQKE+max_diff_exact":
                pass
            #     for m in matrices:
            #         assert torch.allclose(
            #             m, torch.zeros_like(m)
            #         ), f"matrices should be zero when passing {self.attention_error_handling}, not {m}"
            #     return torch.tensor(0).to(matrices[0])

    def split_extreme_softmaxed_right_attention(
        self,
        extreme_softmaxed_right_attention: Float[Tensor, "d_vocab_q"],  # noqa F821
        *,
        max_tok: int,
    ) -> Tuple[Float[Tensor, ""], Float[Tensor, "d_vocab_q"]]:  # noqa F821
        """
        Returns (average_right_attention, right_attention_adjustment)

        Postconditions:
            average_right_attention is not nan, inf, -inf
            average_right_attention + right_attention_adjustment = min_softmaxed_right_attention
        """
        average_right_attention = (
            dropnan(extreme_softmaxed_right_attention).mean()
            if self.attention_handling == "mean_query+diff"
            else torch.zeros([]).to(extreme_softmaxed_right_attention)
        )
        right_attention_adjustment = (
            extreme_softmaxed_right_attention - average_right_attention
        )
        return average_right_attention, right_attention_adjustment

    @property
    def attention_handling_quadratic(self) -> bool:
        return True

    @property
    def attention_handling_subcubic(self) -> bool:
        return True

    @staticmethod
    def transform_description(description: str, *, latex: bool = False) -> str:
        if latex:
            return "".join(
                v.capitalize() for v in description.replace("+", "_").split("_")
            )
        else:
            return description.replace("_", "-")

    def short_description(self, latex: bool = False) -> str:
        transform = lambda s: self.transform_description(s, latex=latex)
        if latex:
            return f"EUPU{transform(self.EUPU_handling)}Attn{transform(self.attention_handling)}AttnErr{transform(self.attention_error_handling)}"
        else:
            return f"EUPU-{transform(self.EUPU_handling)}--attn-{transform(self.attention_handling)}--attn-err-{transform(self.attention_error_handling)}"

    def __str__(self) -> str:
        return self.short_description()

    def __hash__(self):
        return hash(
            (self.EUPU_handling, self.attention_handling, self.attention_error_handling)
        )
