from itertools import combinations
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor

from gbmi.utils.sequences import count_sequences, generate_all_sequences


def probability_mass_for(
    n_ctx: int,
    d_vocab: int,
    num_copies_nonmax: int,
    largest_nonmax_tok: Optional[int],
) -> int:
    """
    Computes the probability mass for the given parameters."""
    if num_copies_nonmax == 0:
        return 1
    assert largest_nonmax_tok is not None
    return (
        count_sequences(
            n_ctx - 1, num_copies_nonmax, largest_nonmax_tok + 1, nonmax_strict=True
        )
        / d_vocab**n_ctx
    )


def sample(
    n_ctx: int,
    d_vocab: int,
    max_samples: Optional[int] = None,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[str | torch.device] = None,
) -> Iterable[Tuple[Tensor, float]]:
    """
    Generates samples according to importance sampling, based on max token, largest nonmax token, query token, and number of copies of the nonmax token.  The pair of (sample, weight) is returned for each sample.
    """
    randints = lambda high, size: torch.randint(
        0, high, size, generator=generator, dtype=torch.long
    )
    randint = lambda high: int(randints(high, (1,)).item())
    nsamples = 0
    nsequences = d_vocab**n_ctx
    while max_samples is None or nsamples < max_samples:
        max_tok = randint(d_vocab)
        query_tok = randint(max_tok + 1)
        num_copies_nonmax = randint(n_ctx if max_tok == query_tok else n_ctx - 1)
        seq = torch.full((n_ctx,), max_tok, dtype=torch.long)
        seq[-1] = query_tok
        if max_tok == 0 or num_copies_nonmax == 0:
            yield seq.to(device=device), 1.0 / nsequences
            nsamples += 1
            continue

        max_nonmax_tok = randint(max_tok)
        other_tokens = randints(max_nonmax_tok + 1, (num_copies_nonmax - 1,))
        seq[: num_copies_nonmax - 1] = other_tokens
        seq[num_copies_nonmax - 1] = max_nonmax_tok
        seq[:-1] = seq[torch.randperm(n_ctx - 1, generator=generator)]
        yield seq.to(device), probability_mass_for(
            n_ctx, d_vocab, num_copies_nonmax, max_nonmax_tok
        )
        nsamples += 1


def all_keys_cubic(
    n_ctx: int, d_vocab: int
) -> Iterable[Tuple[int, int, int, Optional[int]]]:
    """
    Generates all keys for the cubic model, in the order (max token, query token, number of copies of the nonmax token, largest nonmax token).  The largest nonmax token is None if the number of copies of the nonmax token is 0.
    """
    for max_tok in range(d_vocab):
        for query_tok in range(max_tok + 1):
            for num_copies_nonmax in range(
                1 if max_tok == 0 else n_ctx if max_tok == query_tok else n_ctx - 1
            ):
                if max_tok == 0 or num_copies_nonmax == 0:
                    yield max_tok, query_tok, 0, None
                    continue

                for largest_nonmax_tok in range(max_tok):
                    yield max_tok, query_tok, num_copies_nonmax, largest_nonmax_tok


def sample_include_all_keys(
    n_ctx: int,
    d_vocab: int,
    nsamples_per_key: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[str | torch.device] = None,
) -> Iterable[Tuple[Tensor, float]]:
    """
    Generates samples according to importance sampling, based on max token, largest nonmax token, query token, and number of copies of the nonmax token.  The pair of (sample, weight) is returned for each sample.
    """
    randints = lambda high, size: torch.randint(
        0, high, size, generator=generator, dtype=torch.long
    )
    nsequences = d_vocab**n_ctx
    for max_tok, query_tok, num_copies_nonmax, largest_nonmax_tok in all_keys_cubic(
        n_ctx, d_vocab
    ):
        if max_tok == 0 or num_copies_nonmax == 0:
            seq = torch.full((n_ctx,), max_tok, dtype=torch.long)
            seq[-1] = query_tok
            yield seq.to(device=device), 1.0 / nsequences
            continue

        assert largest_nonmax_tok is not None, (max_tok, query_tok, num_copies_nonmax)

        seq_count = count_sequences(
            n_ctx - 1,
            num_copies_nonmax,
            largest_nonmax_tok + 1,
            nonmax_strict=True,
        )

        if seq_count <= nsamples_per_key:
            for num_smaller_nonmax in range(num_copies_nonmax):
                for other_small_tokens in generate_all_sequences(
                    largest_nonmax_tok, num_smaller_nonmax
                ):
                    for smaller_nonmax_tok_pos in combinations(
                        range(num_copies_nonmax), num_smaller_nonmax
                    ):
                        other_tokens = torch.full(
                            (num_copies_nonmax, 1), largest_nonmax_tok, dtype=torch.long
                        )
                        other_tokens[smaller_nonmax_tok_pos] = other_small_tokens
                        for nonmax_tok_pos in combinations(
                            range(n_ctx - 1), num_copies_nonmax
                        ):
                            seq = torch.full((n_ctx,), max_tok, dtype=torch.long)
                            seq[-1] = query_tok
                            seq[nonmax_tok_pos] = other_tokens
                            yield seq.to(device), seq_count / nsequences
        else:
            for _ in range(nsamples_per_key):
                other_tokens = randints(
                    largest_nonmax_tok + 1, (num_copies_nonmax - 1,)
                )
                seq = torch.full((n_ctx,), max_tok, dtype=torch.long)
                seq[-1] = query_tok
                seq[: num_copies_nonmax - 1] = other_tokens
                seq[num_copies_nonmax - 1] = largest_nonmax_tok
                seq[:-1] = seq[torch.randperm(n_ctx - 1, generator=generator)]
                yield seq.to(device), seq_count / nsequences
