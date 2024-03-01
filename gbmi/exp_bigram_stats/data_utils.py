from typing import Iterable, Optional, Tuple
import einops
from jaxtyping import Float, Integer
import torch
from torch import Tensor


class ExactBigramTask:

    @staticmethod
    def loss_fn(
        logits: Float[Tensor, "batch pos num_tokens"],  # noqa: F722
        labels: Integer[Tensor, "batch pos num_tokens"],  # noqa: F722
        *,
        use_bos: bool,
        only_eos: Optional[int] = None,
    ) -> Float[Tensor, ""]:  # noqa: F722
        if use_bos:
            logits = logits[:, 1:, :]
            labels = labels[:, 1:, :]
        if only_eos is not None:
            logits = logits[:, -only_eos:, :]
            labels = labels[:, -only_eos:, :]
        logits = einops.rearrange(logits, "b p v -> (b p) v")
        labels = einops.rearrange(labels, "b p v -> (b p) v")
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
        *, seed: int, num_tokens: int, seq_length: int, max_length: int
    ) -> Iterable[Integer[Tensor, "seq_length"]]:  # noqa F821
        assert seq_length in (
            4,
            5,
        ), f"Only implemented for seq_length=4,5, not {seq_length}"
        default_device = torch.tensor([]).device
        g = torch.Generator(device=default_device)
        g.manual_seed(seed)
        n_samples = 0
        while True:
            # permute arange(num_tokens) randomly
            tokens = torch.randperm(num_tokens, generator=g)
            a, b, c = tokens[:3]
            if torch.rand(1) < 0.5:
                yield torch.tensor(
                    [a, b, c, b, c][:seq_length],
                    dtype=torch.long,
                    device=default_device,
                )
            else:
                yield torch.tensor(
                    [a, b, c, a, b][:seq_length],
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
