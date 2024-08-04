import time
from functools import partial
from itertools import combinations
from typing import Iterable, Literal, Optional, Tuple

import torch
from jaxtyping import Float, Integer
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

import gbmi.utils as utils
from gbmi.exp_max_of_n.verification.brute_force import run_model_cached
from gbmi.utils import batched
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
    normalize_weight: bool = True,
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
        seqcount = count_sequences(
            n_ctx - 1, num_copies_nonmax, max_nonmax_tok + 1, nonmax_strict=True
        )
        yield seq.to(device), (
            (seqcount / nsequences) if normalize_weight else seqcount
        )
        nsamples += 1


def all_keys_cubic(
    n_ctx: int,
    d_vocab: int,
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
    normalize_weight: bool = True,
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
            yield seq.to(device=device), ((1.0 / nsequences) if normalize_weight else 1)
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
                    for smaller_nonmax_tok_pos in map(
                        list,
                        combinations(range(num_copies_nonmax), num_smaller_nonmax),
                    ):
                        other_tokens = torch.full(
                            (num_copies_nonmax,), largest_nonmax_tok, dtype=torch.long
                        )
                        if num_smaller_nonmax > 0:
                            other_tokens[smaller_nonmax_tok_pos] = other_small_tokens
                        for nonmax_tok_pos in map(
                            list,
                            combinations(range(n_ctx - 1), num_copies_nonmax),
                        ):
                            seq = torch.full((n_ctx,), max_tok, dtype=torch.long)
                            seq[-1] = query_tok
                            seq[list(nonmax_tok_pos)] = other_tokens
                            yield seq.to(device), (
                                (1.0 / nsequences) if normalize_weight else 1
                            )
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
                yield seq.to(device), (
                    (seq_count / nsamples_per_key) / nsequences
                    if normalize_weight
                    else seq_count / nsamples_per_key
                )


@torch.no_grad()
def importance_sample_model_batch(
    model: HookedTransformer,
    samples: Iterable[Tuple[Float[Tensor, "n_ctx"], float]],  # noqa: F821
    *,
    cache: Optional[dict[str, Tensor]] = None,
    device: Optional[str | torch.device] = None,
) -> Tuple[
    float,
    float,
    float,
    int,
    int,
    Integer[Tensor, "batch n_ctx"],  # noqa: F722
    float,
]:
    """returns total_weight, unnormalized_loss, unnormalized_accuracy, num_correct, num_incorrect, incorrect_sequences, duration"""
    start = time.time()
    samples = list(samples)
    xs: Float[Tensor, "batch n_ctx"]  # noqa: F722
    labels: Integer[Tensor, "batch"]  # noqa: F821
    weights: Float[Tensor, "batch"]  # noqa: F821
    logits: Float[Tensor, "batch d_vocab"]  # noqa: F722
    xs = torch.stack([sample for sample, _ in samples], dim=0).to(device)
    labels = xs.amax(dim=-1)
    weights = torch.tensor([weight for _, weight in samples], device=device)
    if device is not None:
        model.to(device, print_details=False)
    logits = run_model_cached(model, xs, cache=cache)
    log_probs = utils.log_softmax(logits, dim=-1)
    correct_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze()
    full_accuracy = labels == logits.argmax(dim=-1)
    total_weight = weights.sum().item()
    unnormalized_loss = -correct_log_probs.dot(weights).item()
    unnormalized_accuracy = full_accuracy.float().dot(weights).item()
    incorrect_sequences = xs[~full_accuracy]
    num_correct = len(xs) - len(incorrect_sequences)
    num_incorrect = len(incorrect_sequences)
    duration = time.time() - start
    return (
        total_weight,
        unnormalized_loss,
        unnormalized_accuracy,
        num_correct,
        num_incorrect,
        incorrect_sequences,
        duration,
    )


@torch.no_grad()
def importance_sample_model(
    model: HookedTransformer,
    samples: Iterable[Tuple[Float[Tensor, "n_ctx"], float]],  # noqa: F821
    batch_size: Optional[int] = None,
    *,
    cache: Optional[dict[str, Tensor]] = None,
    device: Optional[str | torch.device] = None,
    pbar: Optional[tqdm] = None,
) -> dict[
    Literal[
        "loss",
        "accuracy",
        "num_correct",
        "num_incorrect",
        "incorrect_sequences",
        "duration",
    ],
    Tensor | int | float,
]:
    total_weight = 0.0
    unnormalized_loss = 0.0
    unnormalized_accuracy = 0.0
    num_correct = 0
    num_incorrect = 0
    incorrect_sequences_list = []
    start = time.time()
    sample_batches = (
        batched(samples, batch_size) if batch_size is not None else [samples]
    )
    if cache is None:
        cache = {}
    for sample_batch in sample_batches:
        if pbar is not None:
            pbar.update(1)
        (
            cur_total_weight,
            cur_unnormalized_loss,
            cur_unnormalized_accuracy,
            cur_num_correct,
            cur_num_incorrect,
            cur_incorrect_sequences,
            _cur_duration,
        ) = importance_sample_model_batch(
            model, sample_batch, cache=cache, device=device
        )
        total_weight += cur_total_weight
        unnormalized_loss += cur_unnormalized_loss
        unnormalized_accuracy += cur_unnormalized_accuracy
        num_correct += cur_num_correct
        num_incorrect += cur_num_incorrect
        incorrect_sequences_list.append(cur_incorrect_sequences)
        if pbar is not None:
            pbar.set_postfix(
                {
                    "loss": unnormalized_loss / total_weight,
                    "accuracy": unnormalized_accuracy / total_weight,
                    "weight": total_weight,
                }
            )
    incorrect_sequences = (
        torch.cat(incorrect_sequences_list, dim=0)
        if incorrect_sequences_list
        else torch.empty((0,), device=device)
    )
    duration = time.time() - start
    loss = unnormalized_loss / total_weight if total_weight != 0 else float("nan")
    accuracy = (
        unnormalized_accuracy / total_weight if total_weight != 0 else float("nan")
    )
    return {
        "loss": loss,
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "incorrect_sequences": incorrect_sequences,
        "duration": duration,
    }


@torch.no_grad()
def importance_sample(
    model: HookedTransformer,
    nsamples: int | Tuple[int, Literal["per_key"]],
    batch_size: Optional[int] = None,
    *,
    cache: Optional[dict[str, Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[str | torch.device] = None,
    normalize_weight: bool = True,
    pbar: Optional[tqdm] = None,
    seed: Optional[int] = None,
) -> dict[
    Literal[
        "loss",
        "accuracy",
        "num_correct",
        "num_incorrect",
        "incorrect_sequences",
        "duration",
    ],
    Tensor | int | float,
]:
    if seed is not None:
        torch.manual_seed(seed)
        if generator is not None:
            generator.manual_seed(seed)
    if isinstance(nsamples, int):
        samples = sample(
            model.cfg.n_ctx,
            model.cfg.d_vocab,
            nsamples,
            generator=generator,
            device=device,
            normalize_weight=normalize_weight,
        )
    else:
        nsamples_per_key, _per_key = nsamples
        assert _per_key == "per_key", f"{_per_key} is not 'per_key'"
        samples = sample_include_all_keys(
            model.cfg.n_ctx,
            model.cfg.d_vocab,
            nsamples_per_key,
            generator=generator,
            device=device,
            normalize_weight=normalize_weight,
        )
    return importance_sample_model(
        model, samples, batch_size=batch_size, cache=cache, device=device, pbar=pbar
    )
