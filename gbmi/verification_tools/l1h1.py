import math
from typing import Union

import numpy as np
import torch
from torch import Tensor
from fancy_einsum import einsum
from transformer_lens import HookedTransformer
from jaxtyping import Float

from gbmi.verification_tools.general import EU_PU


@torch.no_grad()
def all_EVOU(
    model: HookedTransformer,
) -> Float[Tensor, "d_vocab d_vocab_out"]:  # noqa: F722
    """
    Returns all OV results, ignoring position, of shape (d_vocab, d_vocab_out)
    Complexity: O(d_vocab * (d_model * d_head + d_model * d_vocab_out)) ~ O(d_vocab^2 * d_model)
    """
    W_E, W_O, W_V, W_U = model.W_E, model.W_O, model.W_V, model.W_U
    d_model, d_vocab, d_head, d_vocab_out = (
        model.cfg.d_model,
        model.cfg.d_vocab,
        model.cfg.d_head,
        model.cfg.d_vocab_out,
    )
    assert W_E.shape == (d_vocab, d_model)
    assert W_O.shape == (1, 1, d_model, d_head)
    assert W_V.shape == (1, 1, d_head, d_model)
    assert W_U.shape == (d_model, d_vocab_out)

    EVOU = (
        (W_E @ W_V[0, 0, :, :]) @ W_O[0, 0, :, :]
    ) @ W_U  # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    assert EVOU.shape == (
        d_vocab,
        d_vocab_out,
    ), f"EVOU.shape = {EVOU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    return EVOU


@torch.no_grad()
def all_PVOU(
    model: HookedTransformer,
) -> Float[Tensor, "n_ctx d_vocab_out"]:  # noqa: F722
    """
    Returns all OV results, position only, of shape (n_ctx, d_vocab_out)
    Complexity: O(n_ctx * (d_model * d_head + d_model * d_vocab_out)) ~ O(n_ctx * d_vocab * d_model)
    """
    W_pos, W_O, W_V, W_U = model.W_pos, model.W_O, model.W_V, model.W_U
    d_model, n_ctx, d_head, d_vocab_out = (
        model.cfg.d_model,
        model.cfg.n_ctx,
        model.cfg.d_head,
        model.cfg.d_vocab_out,
    )
    assert W_pos.shape == (n_ctx, d_model)
    assert W_O.shape == (1, 1, d_model, d_head)
    assert W_V.shape == (1, 1, d_head, d_model)
    assert W_U.shape == (d_model, d_vocab_out)

    PVOU = (
        (W_pos @ W_V[0, 0, :, :]) @ W_O[0, 0, :, :]
    ) @ W_U  # (n_ctx, d_vocab_out). PVOU[i, j] is how copying at position i affects logit j.
    assert PVOU.shape == (
        n_ctx,
        d_vocab_out,
    ), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    return PVOU


@torch.no_grad()
def all_attention_scores(
    model: HookedTransformer,
) -> Float[Tensor, "n_ctx_k d_vocab_q d_vocab_k"]:  # noqa: F722
    """
    Returns pre-softmax attention of shape (n_ctx_k, d_vocab_q, d_vocab_k)
    Complexity: O(d_vocab^2 * d_model * n_ctx)
    """
    W_E, W_pos, W_Q, W_K = model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, n_ctx, d_vocab, d_head = (
        model.cfg.d_model,
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_head,
    )
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_Q.shape == (1, 1, d_model, d_head)
    assert W_K.shape == (1, 1, d_model, d_head)

    last_resid = (
        W_E + W_pos[-1]
    )  # (d_vocab, d_model). Rows = possible residual streams.
    assert last_resid.shape == (
        d_vocab,
        d_model,
    ), f"last_resid.shape = {last_resid.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"
    key_tok_resid = (
        W_E + W_pos[:, None, :]
    )  # (n_ctx, d_vocab, d_model). Dim 1 = possible residual streams.
    assert key_tok_resid.shape == (
        n_ctx,
        d_vocab,
        d_model,
    ), f"key_tok_resid.shape = {key_tok_resid.shape} != {(n_ctx, d_vocab, d_model)} = (n_ctx, d_vocab, d_model)"
    q = last_resid @ W_Q[0, 0, :, :]  # (d_vocab, d_head).
    assert q.shape == (
        d_vocab,
        d_head,
    ), f"q.shape = {q.shape} != {(d_vocab, d_head)} = (d_vocab, d_head)"
    k = einsum(
        "n_ctx d_vocab d_head, d_head d_model_k -> n_ctx d_model_k d_vocab",
        key_tok_resid,
        W_K[0, 0, :, :],
    )
    assert k.shape == (
        n_ctx,
        d_head,
        d_vocab,
    ), f"k.shape = {k.shape} != {(n_ctx, d_head, d_vocab)} = (n_ctx, d_head, d_vocab)"
    x_scores = einsum(
        "d_vocab_q d_head, n_ctx d_head d_vocab_k -> n_ctx d_vocab_q d_vocab_k", q, k
    )
    assert x_scores.shape == (
        n_ctx,
        d_vocab,
        d_vocab,
    ), f"x_scores.shape = {x_scores.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    # x_scores[pos, qt, kt] is the score from query token qt to key token kt at position pos

    return x_scores


@torch.no_grad()
def find_all_d_attention_scores(
    model: HookedTransformer, min_gap: int = 1
) -> Union[
    Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    Float[
        Tensor,
        "d_vocab_q n_ctx_max n_ctx_non_max d_vocab_k_max d_vocab_k_nonmax",  # noqa: F722
    ],
]:
    """
    If input tokens are x, y, with x - y > min_gap, the minimum values of
    score(x) - score(y).

    Complexity: O(d_vocab * d_model^2 * n_ctx + d_vocab^min(3,n_ctx) * n_ctx^min(2,n_ctx-1))
    Returns: d_attention_score indexed by
        if n_ctx <= 2:
            (d_vocab_q, d_vocab_k)
        if n_ctx > 2:
            (d_vocab_q, n_ctx_max, n_ctx_non_max, d_vocab_k_max, d_vocab_k_nonmax)
    """
    n_ctx, d_vocab = model.cfg.n_ctx, model.cfg.d_vocab
    x_scores = all_attention_scores(model)
    assert x_scores.shape == (
        n_ctx,
        d_vocab,
        d_vocab,
    ), f"x_scores.shape = {x_scores.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    # x_scores[pos, qt, kt] is the score from query token qt to key token kt at position pos

    if n_ctx <= 2:
        # when there are only two cases, it must be the case that either the max is in the query slot, or the non-max is in the query slot
        scores = torch.zeros((d_vocab, d_vocab)) + float("inf")
        for q_tok in range(d_vocab):
            for k_tok in range(d_vocab):
                if math.abs(k_tok - q_tok) >= min_gap:
                    # q_tok is always in the last position
                    scores[q_tok, k_tok] = (
                        x_scores[0, q_tok, k_tok].item()
                        - x_scores[-1, q_tok, q_tok].item()
                    ) * np.sign(k_tok - q_tok)
    else:
        # when there are more than two cases, we need to consider all cases
        scores = torch.zeros((d_vocab, n_ctx, n_ctx, d_vocab, d_vocab)) + float("inf")
        for q_tok in range(d_vocab):
            for pos_of_max in range(n_ctx):
                for k_tok_max in range(d_vocab):
                    if pos_of_max == n_ctx - 1 and k_tok_max != q_tok:
                        continue
                    for pos_of_non_max in range(n_ctx):
                        if pos_of_max == pos_of_non_max:
                            continue
                        for k_tok_non_max in range(k_tok_max - (min_gap - 1)):
                            if pos_of_non_max == n_ctx - 1 and k_tok_non_max != q_tok:
                                continue
                            scores[
                                q_tok,
                                pos_of_max,
                                pos_of_non_max,
                                k_tok_max,
                                k_tok_non_max,
                            ] = (
                                x_scores[pos_of_max, q_tok, k_tok_max].item()
                                - x_scores[pos_of_non_max, q_tok, k_tok_non_max].item()
                            )

    return scores


@torch.no_grad()
def find_min_d_attention_score(
    model: HookedTransformer, min_gap: int = 1, reduce_over_query=False
) -> Union[float, Float[Tensor, "d_vocab_q"]]:  # noqa: F821
    """
    If input tokens are x, y, with x - y > min_gap, the minimum value of
    score(x) - score(y).

    Complexity: O(d_vocab * d_model^2 * n_ctx + d_vocab^min(3,n_ctx) * n_ctx^min(2,n_ctx-1))
    Returns: float if reduce_over_query else torch.Tensor[d_vocab] (indexed by query token)
    """
    scores = find_all_d_attention_scores(model, min_gap=min_gap)
    while len(scores.shape) != 1:
        scores = scores.min(dim=-1).values
    if reduce_over_query:
        scores = scores.min(dim=0).values.item()
    return scores


@torch.no_grad()
def EU_PU_PVOU(
    model: HookedTransformer,
    attention_pattern: Float[Tensor, "batch n_ctx"],  # noqa F722
) -> Float[Tensor, "batch d_vocab_q d_vocab_out"]:  # noqa: F722
    """
    Calculates logits from EU, PU, and the positional part of the OV path for a given batch of attentions
    attention_pattern: (batch, n_ctx) # post softmax
    Returns: (batch, d_vocab_q, d_vocab_out)
    Complexity: O(d_vocab^2 * d_model + d_vocab^2 * d_model^2 + batch * n_ctx * d_vocab_out + batch * d_vocab^2)
    """
    n_ctx, d_vocab, d_vocab_out = (
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_vocab_out,
    )
    batch, _ = attention_pattern.shape
    assert attention_pattern.shape == (
        batch,
        n_ctx,
    ), f"attention_post_softmax.shape = {attention_pattern.shape} != {(batch, n_ctx)} = (batch, n_ctx)"
    EUPU = EU_PU(model)
    assert EUPU.shape == (
        d_vocab,
        d_vocab_out,
    ), f"EUPU.shape = {EUPU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    PVOU = all_PVOU(model)
    assert PVOU.shape == (
        n_ctx,
        d_vocab_out,
    ), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    PVOU_scaled = attention_pattern @ PVOU
    assert PVOU_scaled.shape == (
        batch,
        d_vocab_out,
    ), f"PVOU_scaled.shape = {PVOU_scaled.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"
    result = EUPU[None, :, :] + PVOU_scaled[:, None, :]
    assert result.shape == (
        batch,
        d_vocab,
        d_vocab_out,
    ), f"result.shape = {result.shape} != {(batch, d_vocab, d_vocab_out)} = (batch, d_vocab, d_vocab_out)"

    return result
