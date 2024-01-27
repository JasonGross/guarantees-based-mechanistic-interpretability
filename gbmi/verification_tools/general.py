import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer


@torch.no_grad()
def EU_PU(
    model: HookedTransformer, renderer=None, pos: int = -1
) -> Float[Tensor, "d_vocab_q d_vocab_out"]:  # noqa: F722
    """
    Calculates logits from just the EU and PU paths in position pos.
    Complexity: O(d_vocab^2 * d_model)
    Return shape: (d_vocab, d_vocab_out) (indexed by query token)
    """
    W_E, W_pos, W_U = model.W_E, model.W_pos, model.W_U
    d_model, n_ctx, d_vocab, d_vocab_out = (
        model.cfg.d_model,
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_vocab_out,
    )
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_U.shape == (d_model, d_vocab_out)

    result = (W_E + W_pos[pos][None, :]) @ W_U
    assert result.shape == (d_vocab, d_vocab_out)

    return result
