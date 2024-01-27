import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


import itertools

from transformer_lens import HookedTransformer


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

    def __getitem__(self, index: int) -> Float[Tensor, "seq_len"]:  # noqa: F821
        # Convert the index to a sequence of numbers
        sequence = []
        for _ in range(self.seq_len):
            sequence.append(index % self.vocab_size)
            index = index // self.vocab_size
        return torch.tensor(sequence, dtype=torch.long)
