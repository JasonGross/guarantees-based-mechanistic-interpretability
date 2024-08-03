from typing import Iterable, Optional, Tuple
import torch
from torch import Tensor
from jaxtyping import Float
import math
from transformer_lens import HookedTransformer


def count_sequences_for(
    n_ctx: int,
    max_tok: int,
    query_tok: int,
    num_copies_nonmax: int,
    largest_nonmax_tok: Optional[int],
) -> int:
    """
    Counts the number of sequences that can be generated according to the given parameters.
    """
    if max_tok == 0:
        assert query_tok == max_tok
        assert num_copies_nonmax == 0
        assert largest_nonmax_tok is None
        return 1

    if num_copies_nonmax == 0:
        assert largest_nonmax_tok is None
        return 1

    assert largest_nonmax_tok is not None

    if largest_nonmax_tok == 0:
        return math.comb(n_ctx - 1, num_copies_nonmax)

    assert largest_nonmax_tok < max_tok, (largest_nonmax_tok, max_tok)
    assert query_tok <= max_tok, (query_tok, max_tok)
    assert num_copies_nonmax < n_ctx, (num_copies_nonmax, n_ctx)
    if query_tok != max_tok:
        assert num_copies_nonmax < n_ctx - 1, (num_copies_nonmax, n_ctx)

    return math.comb(n_ctx - 1, num_copies_nonmax) * (
        (largest_nonmax_tok + 1) ** num_copies_nonmax
        - (largest_nonmax_tok**num_copies_nonmax)
    )


def probability_mass_for(
    n_ctx: int,
    d_vocab: int,
    max_tok: int,
    query_tok: int,
    num_copies_nonmax: int,
    largest_nonmax_tok: Optional[int],
) -> int:
    """
    Computes the probability mass for the given parameters."""
    return (
        count_sequences_for(
            n_ctx, max_tok, query_tok, num_copies_nonmax, largest_nonmax_tok
        )
        / d_vocab**n_ctx
    )


def sample(
    n_ctx: int,
    d_vocab: int,
    max_samples: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[str | torch.device] = None,
) -> Iterable[Tuple[Tensor, float]]:
    """
    Generates samples according to importance sampling, based on max token, largest nonmax token, query token, and number of copies of the nonmax token.  The pair of (sample, weight) is returned for each sample.
    """
    randint = lambda high: int(
        torch.randint(0, high, (1,), generator=generator, dtype=torch.long).item()
    )
    nsamples = 0
    nsequences = d_vocab**n_ctx
    while max_samples is None or nsamples < max_samples:
        max_tok = randint(d_vocab)
        if max_tok == 0:
            yield torch.zeros(n_ctx, device=device), 1.0 / nsequences
            nsamples += 1
            continue

        query_tok = randint(max_tok + 1)
        num_copies_nonmax = randint(n_ctx if max_tok == query_tok else n_ctx - 1)

        if num_copies_nonmax == 0:
            yield torch.tensor(
                [max_tok] * (n_ctx - 1) + [query_tok], dtype=torch.long, device=device
            ), 1.0 / nsequences
            continue

        max_nonmax_tok = randint(max_tok)

        max_tok_list = [max_tok] * (n_ctx - 1 - num_copies_nonmax)

        if max_nonmax_tok == 0:
            # yield random sequence with num_copies_nonmax 0s and n_ctx - 1 - num_copies_nonmax max_tok in some order, followed by query_tok; , math.comb(n_ctx-1, num_copies_nonmax) / nsequences
            seq = torch.tensor(
                [0] * num_copies_nonmax + max_tok_list,
                dtype=torch.long,
            )
            seq = seq[torch.randperm(n_ctx - 1, generator=generator)]
            seq = torch.tensor([*seq, query_tok], dtype=torch.long, device=device)
            yield seq, math.comb(n_ctx - 1, num_copies_nonmax) / nsequences
            nsamples += 1
            continue

        nsubset_sequences = math.comb(n_ctx - 1, num_copies_nonmax) * (
            (max_nonmax_tok + 1) ** num_copies_nonmax
            - (max_nonmax_tok**num_copies_nonmax)
        )

        other_numbers = torch.randint(
            0,
            max_nonmax_tok + 1,
            (num_copies_nonmax - 1,),
            generator=generator,
            dtype=torch.long,
        )
        # yield random shuffle of cat([other_numbers, [max_nonmax_tok] + [max_tok] * (n_ctx - 1 - num_copies_nonmax)] + [query_tok]); , nsubset_sequences / nsequences
        seq = torch.cat(
            [
                other_numbers,
                torch.tensor(
                    [max_nonmax_tok] + max_tok_list,
                    dtype=torch.long,
                ),
            ]
        )
        seq = seq[torch.randperm(n_ctx - 1, generator=generator)]
        seq = torch.tensor([*seq, query_tok], dtype=torch.long, device=device)
        yield seq, nsubset_sequences / nsequences
        nsamples += 1
