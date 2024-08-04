from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

import gbmi.utils.ein as ein
from gbmi.verification_tools.general import EU_PU
from gbmi.verification_tools.l1h1 import (
    all_attention_scores,
    all_EVOU_nocache,
    all_PVOU_nocache,
)


@torch.no_grad()
def run_model_cached(
    model: HookedTransformer,
    inputs: Float[Tensor, "batch n_ctx"],  # noqa: F722
    *,
    cache: Optional[dict[str, Tensor]] = None,
) -> Float[Tensor, "batch d_vocab"]:  # noqa: F722
    """
    Runs the model on the given inputs, caching matrices for amortized speedup
    Complexity: O(|inputs| * n_ctx * d_vocab_out + d_vocab^2 * d_model + n_ctx * d_model * d_vocab)
    """
    if cache is None:
        cache = {}
    n_ctx, d_vocab = model.cfg.n_ctx, model.cfg.d_vocab
    attn_scale = model.blocks[0].attn.attn_scale
    all_attn: Float[Tensor, "n_ctx_k d_vocab_q d_vocab_k"]  # noqa: F722
    all_attn_exp: Float[Tensor, "n_ctx_k d_vocab_max d_vocab_q d_vocab_k"]  # noqa: F722
    EVOU: Float[Tensor, "d_vocab d_vocab_out"]  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"]  # noqa: F722
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"]  # noqa: F722
    if "attn_exp" not in cache:
        all_attn = all_attention_scores(model, bias=True) / attn_scale
        all_attn_exp = ein.array(
            lambda pos, m, q, k: (all_attn[pos, q, k] - all_attn[pos, q, m]).exp(),
            sizes=[n_ctx, d_vocab, d_vocab, d_vocab],
            device=all_attn.device,
        )
        cache["attn_exp"] = all_attn_exp
    else:
        all_attn_exp = cache["attn_exp"]
    EVOU = cache["EVOU"] = (
        cache["EVOU"] if "EVOU" in cache else all_EVOU_nocache(model, bias=True)
    )
    PVOU = cache["PVOU"] = (
        cache["PVOU"] if "PVOU" in cache else all_PVOU_nocache(model, bias=False)
    )
    EUPU = cache["EUPU"] = (
        cache["EUPU"] if "EUPU" in cache else EU_PU(model, bias=False)
    )
    EPVOU_scaled: Float[Tensor, "n_ctx_k d_vocab_max d_vocab_q d_vocab_k"]  # noqa: F722
    EPVOU_scaled = cache["EPVOU_scaled"] = (
        cache["EPVOU_scaled"]
        if "EPVOU_scaled" in cache
        else ein.array(
            lambda pos, max_tok, query_tok, key_tok: (
                (EUPU[query_tok] + PVOU[pos] + EVOU[key_tok])
                * all_attn_exp[pos, max_tok, query_tok, key_tok]
            ),
            sizes=[n_ctx, d_vocab, d_vocab, d_vocab],
            device=all_attn.device,
        )
    )

    maxes = inputs.amax(dim=-1, keepdim=True)
    queries = inputs[..., -1].unsqueeze(-1)
    sequences = torch.arange(n_ctx, device=EPVOU_scaled.device)
    EPVOU_scaled_batch: Float[Tensor, "batch n_ctx_k d_vocab_out"]  # noqa: F722
    EPVOU_scaled_batch = EPVOU_scaled[sequences, maxes, queries, inputs]
    scale_factor: Float[Tensor, "batch 1"]  # noqa: F722
    scale_factor = all_attn_exp[sequences, maxes, queries, inputs].sum(
        dim=-1, keepdim=True
    )
    return EPVOU_scaled_batch.sum(dim=-2) / scale_factor
