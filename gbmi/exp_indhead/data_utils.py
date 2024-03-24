from typing import Collection, Iterable, Optional, Sequence, Tuple
import einops
from jaxtyping import Float, Integer
import torch
from torch import Tensor
from tqdm.auto import tqdm
from gbmi.utils.english_ngram import ngram_count_table, DEFAULT_CORPUS


class ExactBigramTask:

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        labels: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        *,
        use_bos: bool,
        only_eos: Optional[int] = None,
        only_strong_signal: bool = False,
        # _xs only for logging purposes
        _xs: Optional[Integer[Tensor, "batch pos"]] = None,  # noqa: F722
    ) -> Float[Tensor, ""]:  # noqa: F722
        if use_bos:
            logits = logits[:, 1:, :]
            labels = labels[:, 1:, :]
        if only_eos is not None:
            logits = logits[:, -only_eos:, :]
            labels = labels[:, -only_eos:, :]
        if only_strong_signal:
            mask = (labels != 0).sum(dim=-1) == 1
            assert mask.any(
                dim=-1
            ).all(), f"All sequences must have at least one location with exactly one possibility, but got {mask.any(dim=-1)} on\nlogits={logits}\nlabels={labels}\nmask={mask}"
            # for _xsi, labelsi, logitsi, maski in zip(_xs, labels, logits, mask):
            #     input((_xsi, labelsi[maski].argmax(dim=-1), maski.nonzero()))
            logits = logits[mask, :]
            labels = labels[mask, :]
        else:
            # Note that this rearrangement is already taken care of by boolean indexing above
            logits = einops.rearrange(logits, "b p v -> (b p) v")
            labels = einops.rearrange(labels, "b p v -> (b p) v")
        assert len(logits.shape) == 2, logits.shape
        assert len(labels.shape) == 2, labels.shape
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    @staticmethod
    def sample_bigram(
        num_tokens: int,
        seq_length: int,
        bigram_dist: Float[Tensor, "num_tokens num_tokens"],  # noqa: F722
        g: torch.Generator,
    ) -> Iterable[int]:
        token = int(torch.randint(num_tokens, (1,), generator=g).item())
        yield token
        for _ in range(seq_length - 1):
            token = int(torch.multinomial(bigram_dist[token], 1, generator=g).item())
            yield token

    @staticmethod
    def generator(
        *, seed: int, num_tokens: int, seq_length: int, max_length: int
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        g = torch.Generator(device=default_device)
        g.manual_seed(seed)
        n_samples = 0
        while True:
            bigram_dist = torch.rand(num_tokens, num_tokens)
            bigram_dist = bigram_dist / bigram_dist.sum(dim=-1, keepdim=True)
            yield torch.tensor(
                list(
                    ExactBigramTask.sample_bigram(
                        num_tokens, seq_length, bigram_dist, g
                    )
                ),
                dtype=torch.long,
                device=default_device,
            )
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return


class EnglishExactTrigramTask:
    @staticmethod
    def sample_trigram_iter(
        seq_length: int,
        monogram_dist: Float[Tensor, "num_tokens"],  # noqa: F821
        bigram_dist: Float[Tensor, "num_tokens num_tokens"],  # noqa: F722
        trigram_dist: Float[Tensor, "num_tokens num_tokens num_tokens"],  # noqa: F722
        g: torch.Generator,
    ) -> Iterable[int]:
        prevtoken = int(torch.multinomial(monogram_dist, 1, generator=g).item())
        yield prevtoken
        if seq_length == 1:
            return
        token = int(torch.multinomial(bigram_dist[prevtoken], 1, generator=g).item())
        yield token
        for _ in range(seq_length - 2):
            prevtoken, token = token, int(
                torch.multinomial(trigram_dist[prevtoken, token], 1, generator=g).item()
            )
            yield token

    @staticmethod
    def sample_trigram(
        seq_length: int,
        monogram_dist: Float[Tensor, "num_tokens"],  # noqa: F821
        bigram_dist: Float[Tensor, "num_tokens num_tokens"],  # noqa: F722
        trigram_dist: Float[Tensor, "num_tokens num_tokens num_tokens"],  # noqa: F722
        g: torch.Generator,
        device: torch.device,
    ) -> Integer[Tensor, "seq_length"]:  # noqa: F821
        return torch.tensor(
            list(
                EnglishExactTrigramTask.sample_trigram_iter(
                    seq_length, monogram_dist, bigram_dist, trigram_dist, g
                )
            ),
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def sample_trigram_with_at_least_one_unique(
        seq_length: int,
        monogram_dist: Float[Tensor, "num_tokens"],  # noqa: F821
        bigram_dist: Float[Tensor, "num_tokens num_tokens"],  # noqa: F722
        trigram_dist: Float[Tensor, "num_tokens num_tokens num_tokens"],  # noqa: F722
        g: torch.Generator,
        device: torch.device,
    ) -> Integer[Tensor, "seq_length"]:  # noqa: F821
        while True:
            trigram = EnglishExactTrigramTask.sample_trigram(
                seq_length, monogram_dist, bigram_dist, trigram_dist, g, device=device
            )
            if seq_length >= 2 and len(set(trigram.tolist())) == len(trigram):
                # resample the last character to speed things up
                final_token_dist_ref = trigram_dist[trigram[-2], trigram[-1]]
                final_token_dist = torch.zeros_like(final_token_dist_ref)
                idxs = torch.tensor(list(set(trigram[:-1])))
                final_token_dist[idxs] = final_token_dist_ref[idxs]
                if final_token_dist.sum() == 0:
                    final_token_dist[idxs] = trigram_dist[:, trigram[-1], idxs].sum(
                        dim=0
                    )
                if final_token_dist.sum() == 0:
                    final_token_dist[idxs] = (
                        trigram_dist[:, :, idxs].sum(dim=0).sum(dim=0)
                    )
                final_token_dist /= final_token_dist.sum()
                trigram[-1] = int(
                    torch.multinomial(final_token_dist, 1, generator=g).item()
                )
            if (
                calculate_batch_probabilities(trigram, monogram_dist.shape[0]) == 1
            ).any():
                return trigram

    @staticmethod
    def generator(
        *,
        seed: int,
        num_tokens: int,
        seq_length: int,
        max_length: int,
        force_strong_signal: bool = True,
        corpus: str = DEFAULT_CORPUS,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        monogram_table = torch.tensor(
            ngram_count_table(n=1, corpus=corpus), device=default_device
        )
        bigram_table = torch.tensor(
            ngram_count_table(n=2, corpus=corpus), device=default_device
        )
        trigram_table = torch.tensor(
            ngram_count_table(n=3, corpus=corpus), device=default_device
        )
        assert monogram_table.shape == (
            num_tokens,
        ), f"monogram_table.shape={monogram_table.shape} != (num_tokens,) = ({num_tokens},)"
        assert bigram_table.shape == (
            num_tokens,
            num_tokens,
        ), f"bigram_table.shape={bigram_table.shape} != (num_tokens, num_tokens) = ({num_tokens}, {num_tokens})"
        assert trigram_table.shape == (
            num_tokens,
            num_tokens,
            num_tokens,
        ), f"trigram_table.shape={trigram_table.shape} != (num_tokens, num_tokens, num_tokens) = ({num_tokens}, {num_tokens}, {num_tokens})"

        monogram_table /= monogram_table.sum(dim=-1, keepdim=True)
        bigram_table = torch.where(
            bigram_table.sum(dim=-1, keepdim=True) != 0,
            bigram_table,
            bigram_table.sum(dim=0),
        )
        bigram_table /= bigram_table.sum(dim=-1, keepdim=True)
        trigram_table = torch.where(
            trigram_table.sum(dim=-1, keepdim=True) != 0,
            trigram_table,
            trigram_table.sum(dim=0),
        )
        trigram_table = torch.where(
            trigram_table.sum(dim=-1, keepdim=True) != 0,
            trigram_table,
            trigram_table.sum(dim=0).sum(dim=0),
        )
        trigram_table /= trigram_table.sum(dim=-1, keepdim=True)

        g = torch.Generator(device=default_device)
        g.manual_seed(seed)
        n_samples = 0
        while True:
            sample = (
                EnglishExactTrigramTask.sample_trigram_with_at_least_one_unique
                if force_strong_signal
                else EnglishExactTrigramTask.sample_trigram
            )
            yield sample(
                seq_length,
                monogram_table,
                bigram_table,
                trigram_table,
                g,
                default_device,
            )
            n_samples += 1
            if max_length is not None and n_samples >= max_length:
                return


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
        g = torch.Generator(device=default_device)
        g.manual_seed(seed)
        n_samples = 0
        n_cs = seq_length - 3
        while True:
            tokens = torch.randperm(num_tokens, generator=g)
            (a, b), cs = tokens[:2], tokens[(2 if b_unique else 1) :]
            cs = cs[torch.randint(0, cs.size(0), (n_cs,), generator=g)]
            split_index1, split_index2 = (
                torch.randint(1, cs.size(0) + 1, (2,), generator=g).sort().values
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

    @staticmethod
    def sample_trigrams(
        a: int,
        b: int,
        num: int,
        trigram_counts_table: Float[
            Tensor, "num_tokens num_tokens num_tokens"  # noqa: F722
        ],
        *,
        avoid_a: bool = True,
        avoid_b: bool = True,
        g: torch.Generator,
    ) -> Iterable[int]:
        trigram_table = trigram_counts_table.clone()
        if avoid_a:
            trigram_table[:, :, a] = 0
        if avoid_b:
            trigram_table[:, :, b] = 0
        trigram_table = torch.where(
            trigram_table.sum(dim=-1, keepdim=True) != 0,
            trigram_table,
            trigram_table.sum(dim=0),
        )
        trigram_table = torch.where(
            trigram_table.sum(dim=-1, keepdim=True) != 0,
            trigram_table,
            trigram_table.sum(dim=0).sum(dim=0),
        )
        assert (
            trigram_table.sum(dim=-1) != 0
        ).all(), f"a: {a}, b: {b}, indices: {(trigram_table.sum(dim=-1) == 0).nonzero()}, values: {trigram_table.sum(dim=-1)[trigram_table.sum(dim=-1) == 0]}"
        trigram_table /= trigram_table.sum(dim=-1, keepdim=True)
        for _ in range(num):
            a, b = b, int(torch.multinomial(trigram_table[a, b], 1, generator=g).item())
            yield b

    @staticmethod
    def increment_zero_counts(table: Tensor) -> Tensor:
        if (table == 0).any():
            return table + 1 / (1 + table.max() * table.numel())
        else:
            return table

    @staticmethod
    def sample_ngrams_from_start(
        num: int,
        ngram_counts_table: Tensor,
        *,
        g: torch.Generator,
    ) -> Iterable[int]:
        ngram_counts_table = ABCBCEnglishTask.increment_zero_counts(ngram_counts_table)
        if (ngram_counts_table == 0).any():
            ngram_counts_table = ngram_counts_table + 1 / (
                1 + ngram_counts_table.max() * ngram_counts_table.numel()
            )
        prev = []
        for _ in range(num):
            cur_table = ngram_counts_table[tuple(prev)]
            if len(cur_table.shape) > 1:
                cur_table = cur_table.sum(dim=tuple(range(1, len(cur_table.shape))))
            cur_table = cur_table / cur_table.sum()
            next_token = int(torch.multinomial(cur_table, 1, generator=g).item())
            yield next_token
            prev.append(next_token)
            prev = prev[: len(ngram_counts_table.shape) - 1]

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
        corpus: str = DEFAULT_CORPUS,
        alpha_mix_uniform: Optional[float] = None,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        trigram_counts_table = torch.tensor(
            ngram_count_table(n=3, corpus=corpus), device=default_device
        )
        trigram_counts_table = ABCBCEnglishTask.increment_zero_counts(
            trigram_counts_table
        )
        assert trigram_counts_table.shape == (
            num_tokens,
            num_tokens,
            num_tokens,
        ), f"trigram_table.shape={trigram_counts_table.shape} != (num_tokens, num_tokens, num_tokens) = ({num_tokens}, {num_tokens}, {num_tokens})"
        if alpha_mix_uniform is not None:
            trigram_counts_table = trigram_counts_table / trigram_counts_table.sum(
                dim=-1, keepdim=True
            )
            uniform_counts_table = torch.ones_like(trigram_counts_table)
            uniform_counts_table /= uniform_counts_table.sum(dim=-1, keepdim=True)
            trigram_counts_table = (
                trigram_counts_table * (1 - alpha_mix_uniform)
                + uniform_counts_table * alpha_mix_uniform
            )
        g = torch.Generator(device=default_device)
        g.manual_seed(seed)
        n_samples = 0
        n_cs = seq_length - 3
        while True:
            n_cs1 = int(torch.randint(0, n_cs + 1, (1,), generator=g).item())
            n_cs2 = int(torch.randint(0, n_cs - n_cs1 + 1, (1,), generator=g).item())
            if torch.rand(1, generator=g) < 0.5:
                n_cs1, n_cs2 = n_cs2, n_cs1
            n_cs3 = n_cs - n_cs1 - n_cs2
            assert (
                n_cs1 + n_cs2 + n_cs3 == n_cs
            ), f"{n_cs1} + {n_cs2} + {n_cs3} != {n_cs}"
            assert n_cs1 >= 0, n_cs1
            assert n_cs2 >= 0, n_cs2
            assert n_cs3 >= 0, n_cs3
            cs1 = torch.tensor(
                list(
                    ABCBCEnglishTask.sample_ngrams_from_start(
                        int(n_cs1) + 2, trigram_counts_table, g=g
                    )
                ),
                dtype=torch.long,
                device=default_device,
            )

            cs1, a, b = cs1[:-2], int(cs1[-2]), int(cs1[-1])
            if a == b and (a_unique or b_unique):
                continue
            if (a_unique and a in cs1) or (b_unique and b in cs1):
                continue
            cs2 = torch.tensor(
                list(
                    ABCBCEnglishTask.sample_trigrams(
                        a,
                        b,
                        int(n_cs2),
                        trigram_counts_table,
                        g=g,
                        avoid_a=a_unique,
                        avoid_b=b_unique,
                    )
                ),
                dtype=torch.long,
                device=default_device,
            )

            cs3 = torch.tensor(
                list(
                    ABCBCEnglishTask.sample_trigrams(
                        a,
                        b,
                        int(n_cs3),
                        trigram_counts_table,
                        g=g,
                        avoid_a=a_unique,
                        avoid_b=b_unique,
                    )
                ),
                dtype=torch.long,
                device=default_device,
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
