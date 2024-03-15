from typing import Collection, Iterable, Optional, Sequence, Tuple
import einops
from jaxtyping import Float, Integer
import torch
from torch import Tensor
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


class ABCBCBigramTask:

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


class ABCBCEnglishBigramTask:

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
            trigram_table.sum(dim=-1) != 0, trigram_table, trigram_table.sum(dim=0)
        )
        trigram_table /= trigram_table.sum(dim=-1, keepdim=True)
        for _ in range(num):
            a, b = b, int(torch.multinomial(trigram_table[a, b], 1, generator=g).item())
            yield b

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
        corpus: str = DEFAULT_CORPUS,
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        default_device = torch.tensor([]).device
        monogram_table = torch.tensor(
            ngram_count_table(n=1, corpus=corpus), device=default_device
        )
        bigram_table = torch.tensor(
            ngram_count_table(n=2, corpus=corpus), device=default_device
        )
        trigram_counts_table = torch.tensor(
            ngram_count_table(n=3, corpus=corpus), device=default_device
        )
        monogram_table /= monogram_table.sum(dim=-1, keepdim=True)
        # zero the diagonal of the bigram table, since we want bigrams to always be distinct
        bigram_table[torch.arange(num_tokens), torch.arange(num_tokens)] = 0
        bigram_table /= bigram_table.sum(dim=-1, keepdim=True)
        assert monogram_table.shape == (
            num_tokens,
        ), f"monogram_table.shape={monogram_table.shape} != (num_tokens,) = ({num_tokens},)"
        assert bigram_table.shape == (
            num_tokens,
            num_tokens,
        ), f"bigram_table.shape={bigram_table.shape} != (num_tokens, num_tokens) = ({num_tokens}, {num_tokens})"
        assert trigram_counts_table.shape == (
            num_tokens,
            num_tokens,
            num_tokens,
        ), f"trigram_table.shape={trigram_counts_table.shape} != (num_tokens, num_tokens, num_tokens) = ({num_tokens}, {num_tokens}, {num_tokens})"
        g = torch.Generator(device=default_device)
        g.manual_seed(seed)
        n_samples = 0
        n_cs = seq_length - 3
        while True:
            a = int(torch.multinomial(monogram_table, 1, generator=g).item())
            b = int(torch.multinomial(bigram_table[a], 1, generator=g).item())
            cs = torch.tensor(
                list(
                    ABCBCEnglishBigramTask.sample_trigrams(
                        a, b, n_cs, trigram_counts_table, avoid_b=b_unique, g=g
                    )
                ),
                dtype=torch.long,
                device=default_device,
            )
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
