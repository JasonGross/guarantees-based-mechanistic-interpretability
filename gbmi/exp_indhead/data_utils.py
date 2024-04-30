# %%
from typing import Collection, Iterable, Optional, Sequence, Tuple, Union
import itertools
from itertools import chain, cycle
import einops
from jaxtyping import Float, Integer
import torch
from torch import Tensor
from tqdm.auto import tqdm
import gbmi.utils as utils
from gbmi.utils.english_ngram import ngram_count_table, DEFAULT_CORPUS

# %%


class ExactBigramTask:

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        labels: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        *,
        use_bos: bool,
        only_eos: Optional[int] = None,
        only_strong_signal: bool = False,
        high_precision: bool = True,
        # _xs only for logging purposes
        _xs: Optional[Integer[Tensor, "batch pos"]] = None,  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        if use_bos:
            logits = logits[:, 1:, :]
            labels = labels[:, 1:, :]
        if only_eos is not None:
            logits = logits[:, -only_eos:, :]
            labels = labels[:, -only_eos:, :]
        # # Note that this rearrangement is not necessary because we do boolean indexing
        # logits = einops.rearrange(logits, "b p v -> (b p) v")
        # labels = einops.rearrange(labels, "b p v -> (b p) v")
        # remove nans from the labels, which are used to mark locations where we don't want to test
        mask = ~labels.isnan().any(dim=-1)
        logits = logits[mask, :]
        labels = labels[mask, :]
        if only_strong_signal:
            mask = (labels != 0).sum(dim=-1) == 1
            assert mask.any(
                dim=-1
            ).all(), f"All sequences must have at least one location with exactly one possibility, but got {mask.any(dim=-1)} on\nlogits={logits}\nlabels={labels}\nmask={mask}"
            # for _xsi, labelsi, logitsi, maski in zip(_xs, labels, logits, mask):
            #     input((_xsi, labelsi[maski].argmax(dim=-1), maski.nonzero()))
            logits = logits[mask, :]
            labels = labels[mask, :]
        assert len(logits.shape) == 2, logits.shape
        assert len(labels.shape) == 2, labels.shape
        if high_precision:
            logits = logits.to(torch.float64)
            loss = utils.cross_entropy(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    @staticmethod
    def generator(
        *, seed: int, num_tokens: int, seq_length: int, max_length: int
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        generator = torch.Generator(device=default_device)
        generator.manual_seed(seed)
        n_samples = 0
        while True:
            bigram_dist = torch.rand(num_tokens, num_tokens)
            bigram_dist = bigram_dist / bigram_dist.sum(dim=-1, keepdim=True)
            yield sample_ngrams(
                num=seq_length, ngram_counts_table=bigram_dist, generator=generator
            )
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return


class ExhaustiveTask:
    @staticmethod
    def dist_generator_helper(
        *, num_tokens: int, seq_length: int, generator: torch.Generator
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        for x in torch.randperm(num_tokens, generator=generator):
            x = torch.tensor([x.item()], dtype=torch.long)
            if seq_length == 1:
                yield x
            else:
                for xs in ExhaustiveTask.dist_generator_helper(
                    num_tokens=num_tokens,
                    seq_length=seq_length - 1,
                    generator=generator,
                ):
                    yield torch.cat([x, xs])

    @staticmethod
    def dist_generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        force_strong_signal: bool = True,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        generator = torch.Generator(device=default_device)
        generator.manual_seed(seed)
        return filter(
            (
                lambda x: not force_strong_signal
                or (calculate_batch_probabilities(x, num_tokens) == 1).any()
            ),
            ExhaustiveTask.dist_generator_helper(
                num_tokens=num_tokens, seq_length=seq_length, generator=generator
            ),
        )

    @staticmethod
    def generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        force_strong_signal: bool = True,
        max_length: Optional[int] = None,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        n_samples = 0
        for x in tqdm(
            ExhaustiveTask.dist_generator(
                seed=seed,
                num_tokens=num_tokens,
                seq_length=seq_length,
                force_strong_signal=force_strong_signal,
            ),
            desc="Exhaustive datagen",
            total=num_tokens**seq_length,
        ):
            yield x
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return


class EnglishExactNgramTask:
    @staticmethod
    def dist_generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        max_length: Optional[int],
        force_strong_signal: bool = True,
        ngram_counts_table: Float[Tensor, "num_tokens*"],  # noqa F722
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        assert all(
            i == num_tokens for i in ngram_counts_table.shape
        ), f"ngram_counts_table.shape={ngram_counts_table.shape} != num_tokens* = {num_tokens}*"
        generator = torch.Generator(device=default_device)
        generator.manual_seed(seed)
        n_samples = 0
        while True:
            yield (
                sample_ngrams_with_at_least_one_unique(
                    num=seq_length,
                    ngram_counts_table=ngram_counts_table,
                    generator=generator,
                )
                if force_strong_signal
                else sample_ngrams(
                    num=seq_length,
                    ngram_counts_table=ngram_counts_table,
                    generator=generator,
                )
            )
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return

    @staticmethod
    def generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        max_length: int,
        force_strong_signal: bool = True,
        ngram: int = 3,
        corpus: str = DEFAULT_CORPUS,
        allow_language_truncation: bool = True,
        alpha_mix_uniform: Optional[float] = None,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        ngram_counts_table = construct_ngram_counts_table(
            num_tokens=num_tokens,
            ngram=ngram,
            corpus=corpus,
            allow_language_truncation=allow_language_truncation,
            alpha_mix_uniform=alpha_mix_uniform,
        )
        yield from EnglishExactNgramTask.dist_generator(
            seed=seed,
            num_tokens=num_tokens,
            seq_length=seq_length,
            max_length=max_length,
            force_strong_signal=force_strong_signal,
            ngram_counts_table=ngram_counts_table,
        )


class ABCBCTask:

    # based on https://github.com/TomFrederik/mvp_induction/blob/main/datasets.py
    @staticmethod
    def generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        max_length: int,
        skip_end: bool = False,
        b_unique: bool = False,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        generator = torch.Generator(device=default_device)
        generator.manual_seed(seed)
        n_samples = 0
        n_cs = seq_length - 3
        while True:
            tokens = torch.randperm(num_tokens, generator=generator)
            (a, b), cs = tokens[:2], tokens[(2 if b_unique else 1) :]
            cs = cs[torch.randint(0, cs.size(0), (n_cs,), generator=generator)]
            split_index1, split_index2 = (
                torch.randint(1, cs.size(0) + 1, (2,), generator=generator)
                .sort()
                .values
            )
            cs1, cs2, cs3 = (
                cs[:split_index1],
                cs[split_index1:split_index2],
                cs[split_index2:],
            )
            if skip_end:
                cs2, cs3 = torch.cat([cs2, cs3], dim=0), []
            yield torch.tensor(
                [*cs1, a, b, *cs2, a, b, *cs3][:-1],
                dtype=torch.long,
                device=default_device,
            )
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return


class ABCBCEnglishTask:
    # based on https://github.com/TomFrederik/mvp_induction/blob/main/datasets.py
    @staticmethod
    def dist_generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        max_length: Optional[int],
        skip_end: bool = False,
        a_unique: bool = True,
        b_unique: bool = False,
        ngram_counts_table: Float[Tensor, "num_tokens*"],  # noqa F722
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        assert all(
            i == num_tokens for i in ngram_counts_table.shape
        ), f"ngram_counts_table.shape={ngram_counts_table.shape} != num_tokens* = {num_tokens}*"
        generator = torch.Generator(device=default_device)
        generator.manual_seed(seed)
        n_samples = 0
        n_cs = seq_length - 3
        while True:
            n_cs1 = int(torch.randint(0, n_cs + 1, (1,), generator=generator).item())
            n_cs2 = int(
                torch.randint(0, n_cs - n_cs1 + 1, (1,), generator=generator).item()
            )
            if torch.rand(1, generator=generator) < 0.5:
                n_cs1, n_cs2 = n_cs2, n_cs1
            n_cs3 = n_cs - n_cs1 - n_cs2
            assert (
                n_cs1 + n_cs2 + n_cs3 == n_cs
            ), f"{n_cs1} + {n_cs2} + {n_cs3} != {n_cs}"
            assert n_cs1 >= 0, n_cs1
            assert n_cs2 >= 0, n_cs2
            assert n_cs3 >= 0, n_cs3
            cs1 = sample_ngrams(
                num=int(n_cs1) + 2,
                ngram_counts_table=ngram_counts_table,
                generator=generator,
            )

            cs1, a, b = cs1[:-2], int(cs1[-2]), int(cs1[-1])
            if a == b and (a_unique or b_unique):
                continue
            if (a_unique and a in cs1) or (b_unique and b in cs1):
                continue
            avoid = []
            avoid += [a] if a_unique else []
            avoid += [b] if b_unique else []
            cs2 = sample_ngrams(
                a,
                b,
                num=int(n_cs2),
                ngram_counts_table=ngram_counts_table,
                generator=generator,
                avoid=avoid,
            )

            if skip_end:
                before_cs3_0, before_cs3_1 = torch.tensor(
                    [a, b, *cs2], dtype=torch.long
                )[-2:]
            else:
                before_cs3_0, before_cs3_1 = a, b
            cs3 = sample_ngrams(
                int(before_cs3_0),
                int(before_cs3_1),
                num=int(n_cs3),
                ngram_counts_table=ngram_counts_table,
                generator=generator,
                avoid=avoid,
            )
            if skip_end:
                cs2, cs3 = torch.cat([cs2, cs3], dim=0), []
            assert (
                len(cs1) + len(cs2) + len(cs3) + 4 == seq_length + 1
            ), f"{len(cs1)} + {len(cs2)} + {len(cs3)} + 4 != {seq_length + 1}"
            yield torch.tensor(
                [*cs1, a, b, *cs2, a, b, *cs3][:-1],
                dtype=torch.long,
                device=default_device,
            )
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return

    # based on https://github.com/TomFrederik/mvp_induction/blob/main/datasets.py
    @staticmethod
    def generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        max_length: int,
        skip_end: bool = False,
        a_unique: bool = True,
        b_unique: bool = False,
        ngram: int = 3,
        corpus: str = DEFAULT_CORPUS,
        allow_language_truncation: bool = True,
        alpha_mix_uniform: Optional[float] = None,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        ngram_counts_table = construct_ngram_counts_table(
            num_tokens=num_tokens,
            ngram=ngram,
            corpus=corpus,
            allow_language_truncation=allow_language_truncation,
            alpha_mix_uniform=alpha_mix_uniform,
        )
        yield from ABCBCEnglishTask.dist_generator(
            seed=seed,
            num_tokens=num_tokens,
            seq_length=seq_length,
            max_length=max_length,
            skip_end=skip_end,
            a_unique=a_unique,
            b_unique=b_unique,
            ngram_counts_table=ngram_counts_table,
        )


class ABCABCExhaustiveTask:
    # generates all ngrams N (final character is allowed to duplicate a previous one),
    # then for each ngram picks k sequences of random characters distinct from the ngram,
    # then consider all splits of the sequence into XYZ (allowed to be empty)
    # and consider XNYN[:-1]Z, read off the final character of the second N to predict N[-1]

    #     def
    # abcabx
    # xxxxabcab
    # abcbc
    # xabcb
    pass


def all_sequences_avoiding_iter(
    *, num_tokens: int, length: int, avoid: Collection[int] = tuple()
) -> Iterable[Integer[Tensor, "..."]]:
    avoid = set(avoid)
    valid_toks = torch.tensor(
        [tok for tok in range(num_tokens) if tok not in avoid], dtype=torch.long
    )
    for seq_idxs in itertools.product(range(num_tokens - len(avoid)), repeat=length):
        yield torch.tensor([valid_toks[idx] for idx in seq_idxs], dtype=torch.long)


def all_ngrams_iter(
    *,
    num_tokens: int,
    ngram: int,
    avoid: Iterable[int] = tuple(),
    avoid_nodup: Iterable[int] = tuple(),
    avoid_duplicates_initial_pattern: Iterable[bool] = tuple(),
    avoid_duplicates: bool = False,
) -> Iterable[Integer[Tensor, "ngram"]]:  # noqa F821
    if ngram == 0:
        yield torch.tensor([], dtype=torch.long)
        return
    avoid, avoid_nodup = tuple(avoid), tuple(avoid_nodup)
    avoid_duplicates_initial_pattern = list(avoid_duplicates_initial_pattern) + [
        avoid_duplicates
    ]
    avoid_duplicates_here, avoid_duplicates_initial_pattern = (
        avoid_duplicates_initial_pattern[0],
        avoid_duplicates_initial_pattern[1:],
    )
    for t in range(num_tokens):
        if t not in avoid and (not avoid_duplicates_here or t not in avoid_nodup):
            t = torch.tensor([t], dtype=torch.long)
            for ts in all_ngrams_iter(
                num_tokens=num_tokens,
                ngram=ngram - 1,
                avoid=avoid,
                avoid_nodup=[*avoid_nodup, int(t)],
                avoid_duplicates_initial_pattern=avoid_duplicates_initial_pattern,
                avoid_duplicates=avoid_duplicates,
            ):
                yield torch.cat([t, ts], dim=-1)


def all_ngrams(
    *,
    num_tokens: int,
    ngram: int,
    avoid: Iterable[int] = tuple(),
    avoid_duplicates_initial_pattern: Iterable[bool] = tuple(),
    avoid_duplicates: bool = False,
) -> Integer[Tensor, "batch ngram"]:  # noqa F722
    return torch.tensor(
        list(
            all_ngrams_iter(
                num_tokens=num_tokens,
                ngram=ngram,
                avoid=avoid,
                avoid_duplicates_initial_pattern=avoid_duplicates_initial_pattern,
                avoid_duplicates=avoid_duplicates,
            )
        ),
        dtype=torch.long,
    )


def construct_ngram_counts_table(
    *,
    num_tokens: int,
    ngram: int = 3,
    corpus: str = DEFAULT_CORPUS,
    allow_language_truncation: bool = True,
    alpha_mix_uniform: Optional[float] = None,
) -> Integer[Tensor, "..."]:
    ngram_counts_table = increment_zero_counts(
        torch.tensor(ngram_count_table(n=ngram, corpus=corpus))
    )
    if allow_language_truncation:
        ngram_counts_table = ngram_counts_table[
            tuple(slice(num_tokens) for _ in ngram_counts_table.shape)
        ]
    assert ngram_counts_table.shape == tuple(
        [num_tokens] * ngram
    ), f"ngram_table.shape={ngram_counts_table.shape} != ({', '.join(['num_tokens'] * ngram)}) = ({', '.join([str(num_tokens)] * ngram)})"
    if alpha_mix_uniform is not None:
        ngram_counts_table = ngram_counts_table / ngram_counts_table.sum(
            dim=-1, keepdim=True
        )
        uniform_counts_table = torch.ones_like(ngram_counts_table)
        uniform_counts_table /= uniform_counts_table.sum(dim=-1, keepdim=True)
        ngram_counts_table = (
            ngram_counts_table * (1 - alpha_mix_uniform)
            + uniform_counts_table * alpha_mix_uniform
        )

    return ngram_counts_table


def increment_zero_counts(table: Tensor) -> Tensor:
    if (table == 0).any():
        return table + 1 / (1 + table.max() * table.numel())
    else:
        return table


def sample_ngrams_iter(
    *start: int,
    num: int,
    ngram_counts_table: Tensor,
    avoid: Iterable[int] = tuple(),
    avoid_duplicates_initial_pattern: Iterable[bool] = tuple(),
    avoid_duplicates: bool = False,
    generator: torch.Generator,
) -> Iterable[int]:

    def truncate(prev: Iterable[int]) -> list[int]:
        if len(ngram_counts_table.shape) == 1:
            return []
        return list(prev)[-(len(ngram_counts_table.shape) - 1) :]

    ngram_counts_table = increment_zero_counts(ngram_counts_table.clone())
    prev = truncate(list(start))
    ngram_counts_table[..., list(avoid)] = 0
    ngram_counts_table_nodup = ngram_counts_table.clone()
    avoid_duplicates_pattern = chain(
        avoid_duplicates_initial_pattern, cycle([avoid_duplicates])
    )
    for _i, avoid_duplicates_here in zip(range(num), avoid_duplicates_pattern):
        cur_table = (
            ngram_counts_table_nodup if avoid_duplicates_here else ngram_counts_table
        )
        cur_table = cur_table[tuple(prev)]
        if len(cur_table.shape) > 1:
            cur_table = cur_table.sum(dim=tuple(range(1, len(cur_table.shape))))
        assert (
            cur_table.sum() != 0
        ), f"Cannot avoid duplicates ({avoid_duplicates_initial_pattern}, {avoid_duplicates}; at {_i}, {avoid_duplicates_here}; avoiding {avoid}) with {prev}"
        cur_table = cur_table / cur_table.sum()
        next_token = int(torch.multinomial(cur_table, 1, generator=generator).item())
        yield next_token
        ngram_counts_table_nodup[..., next_token] = 0
        prev.append(next_token)
        prev = truncate(prev)


def sample_ngrams(
    *start: int,
    num: int,
    ngram_counts_table: Tensor,
    avoid: Iterable[int] = tuple(),
    avoid_duplicates_initial_pattern: Iterable[bool] = tuple(),
    avoid_duplicates: bool = False,
    generator: torch.Generator,
) -> Integer[Tensor, "num"]:  # noqa: F821
    return torch.tensor(
        list(
            sample_ngrams_iter(
                *start,
                num=num,
                ngram_counts_table=ngram_counts_table,
                avoid=avoid,
                avoid_duplicates=avoid_duplicates,
                avoid_duplicates_initial_pattern=avoid_duplicates_initial_pattern,
                generator=generator,
            )
        ),
        dtype=torch.long,
    )


def sample_ngrams_with_at_least_one_unique(
    *start: int,
    num: int,
    ngram_counts_table: Tensor,
    avoid: Iterable[int] = tuple(),
    generator: torch.Generator,
) -> Integer[Tensor, "seq_length"]:  # noqa: F821
    avoid = set(avoid)
    while True:
        ngram = sample_ngrams(
            *start,
            num=num,
            ngram_counts_table=ngram_counts_table,
            avoid=avoid,
            generator=generator,
        )
        if num >= 2 and len(set(ngram.tolist())) == len(ngram):
            # resample the last character only to speed things up
            ngram = ngram[:-1]
            avoid |= set(ngram.tolist())
            return torch.cat(
                [
                    ngram,
                    sample_ngrams(
                        *ngram.tolist(),
                        num=1,
                        ngram_counts_table=ngram_counts_table,
                        avoid=avoid,
                        generator=generator,
                    ),
                ]
            )
        if (
            calculate_batch_probabilities(ngram, ngram_counts_table.shape[0]) == 1
        ).any():
            return ngram


def calculate_batch_probabilities(
    batch_input: Integer[Tensor, "... seq_length"], num_tokens: int  # noqa: F821, F722
) -> Float[Tensor, "... seq_length num_tokens"]:  # noqa: F821, F722
    # Convert batch input to a PyTorch tensor
    # Convert batch input to a PyTorch tensor
    batch_tensor = (
        torch.tensor(batch_input, dtype=torch.long)
        if not isinstance(batch_input, torch.Tensor)
        else batch_input.long()
    )

    # Get the shape of the batch tensor
    batch_dims, seq_length = batch_tensor.shape[:-1], batch_tensor.shape[-1]

    # Initialize a tensor to store the probability distributions
    # Starting with a uniform distribution for the first position
    probability_distributions = (
        torch.ones(batch_dims + (seq_length, num_tokens), dtype=torch.float)
        / num_tokens
    )

    # Create tensors to count occurrences and calculate cumulative probabilities
    for i in range(1, seq_length):
        # Count occurrences of each token in positions before the current one
        tokens = torch.zeros(batch_dims)
        tokens = batch_tensor[..., i]
        token_occurrences = torch.zeros(batch_dims + (num_tokens,))
        for next_token in range(num_tokens):
            token_occurrences[..., next_token] = (
                (
                    (
                        batch_tensor[..., :i]
                        == tokens[...].unsqueeze(-1).expand_as(batch_tensor[..., :i])
                    )
                    & (batch_tensor[..., 1 : i + 1] == next_token)
                )
                .float()
                .sum(dim=-1)
            )
        normalized_token_occurrences = token_occurrences / token_occurrences.sum(
            dim=-1, keepdim=True
        )
        normalized_token_occurrences[normalized_token_occurrences.isnan()] = (
            1 / num_tokens
        )
        probability_distributions[..., i, :] = normalized_token_occurrences

        # Normalize to get probabilities for positions from the second onwards

    return probability_distributions


def cat_bos_token(
    tokens: Integer[Tensor, "... seq_length"], *, bos: Optional[int]  # noqa: F722
) -> Integer[Tensor, "... num_tokens"]:  # noqa: F722
    if bos is None:
        return tokens
    return torch.cat(
        [
            torch.full(
                tokens.shape[:-1] + (1,),
                bos,
                dtype=torch.long,
                device=tokens.device,
            ),
            tokens,
        ],
        dim=-1,
    )


def cat_bos_uniform_labels(
    labels: Float[Tensor, "... seq_length num_tokens"],  # noqa: F722
    *,
    bos: Optional[int],
) -> Float[Tensor, "... seq_length num_tokens"]:  # noqa: F722
    if bos is None:
        return labels
    num_tokens = labels.shape[-1]
    return torch.cat(
        [
            torch.full(
                labels.shape[:-2] + (1, num_tokens),
                1 / num_tokens,
                dtype=labels.dtype,
                device=labels.device,
            ),
            labels,
        ],
        dim=-2,
    )
