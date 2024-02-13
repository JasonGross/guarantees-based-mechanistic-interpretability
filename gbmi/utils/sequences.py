import math
from typing import Callable, Generic, TypeVar, Union, overload
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


import itertools

from transformer_lens import HookedTransformer

T = TypeVar("T")


def generate_all_sequences(
    n_digits: int, sequence_length: int = 2
) -> Float[Tensor, "n_seqs sequence_length"]:  # noqa: F722
    data = list(itertools.product(range(n_digits), repeat=sequence_length))
    return torch.tensor(data)


def generate_all_sequences_for_model(
    model: HookedTransformer,
) -> Float[Tensor, "n_seqs sequence_length"]:  # noqa: F722
    return generate_all_sequences(
        n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx
    )


class SequenceDataset(Dataset[Tensor]):
    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.length = self.vocab_size**self.seq_len

    def __len__(self):
        return self.length

    @overload
    def __getitem__(self, index: int) -> Float[Tensor, "seq_len"]:  # noqa: F821
        ...

    @overload
    def __getitem__(self, index: slice) -> Tensor: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Tensor, Float[Tensor, "seq_len"]]:  # noqa: F821
        # Convert the index to a sequence of numbers
        if isinstance(index, slice):
            start, stop, stride = index.indices(self.length)
            sequence = []
            for i in range(start, stop, stride):
                sequence.append(self[i])
            return torch.stack(sequence, dim=0)
        else:
            sequence = []
            for _ in range(self.seq_len):
                sequence.append(index % self.vocab_size)
                index = index // self.vocab_size
            return torch.tensor(sequence, dtype=torch.long)


class ThunkedDataset(Generic[T], Dataset[Callable[[], T]]):
    def __init__(self, dataset: Dataset[T]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, *args, **kwargs) -> Callable[[], T]:
        return lambda: self.dataset.__getitem__(*args, **kwargs)  # type: ignore


def count_sequences(
    sequence_length: int, nonmax_count: int, num_nonmax_tok_choices: int
) -> int:
    """
    Count the number of sequences of length sequence_length with nonmax_count items less than or equal to max_nonmax_tok and the remaining tokens equal to a fixed value, where order matters
    """
    total_count = 0

    for i in range(nonmax_count + 1):
        combinations = math.comb(sequence_length, i)
        token_variations = num_nonmax_tok_choices**i
        total_count += combinations * token_variations

    return total_count
