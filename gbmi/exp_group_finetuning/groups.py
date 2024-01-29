from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast, Literal, Generic, TypeVar
from gbmi import utils
from gbmi.utils.sequences import generate_all_sequences
import random
import json
import torch

T = TypeVar("T")


class Group(ABC, Generic[T]):
    @abstractmethod
    def id(self) -> T:
        ...

    @abstractmethod
    def toJSON(self):
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def index(self) -> int:
        ...

    @abstractmethod
    def op(self, a: T, b: T) -> T:
        ...

    def reduce(self, xs: T) -> T:
        accumulator = self.id()
        for x in xs:
            accumulator = self.op(accumulator, x)

        return accumulator


class DihedralGroup(Group):
    def __init__(self, n: int):
        self.n = n
        self.lookup = []
        for x in range(2 * n):
            self.lookup.append([])
            for y in range(2 * n):
                j = x % 2
                if j == 0:
                    result = (y % 2 + (2 * ((x // 2 + y // 2) % n))) % (2 * n)
                else:
                    result = ((y % 2 + 1) % 2 + (2 * ((x // 2 - y // 2) % n))) % (2 * n)

                self.lookup[x].append(result)

        self.lookup = torch.tensor(self.lookup).to("cuda")

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "DihedralGroup" + str(2 * self.n)

    def size(self) -> int:
        return 2 * self.n

    def index(self) -> int:
        return self.n

    def id(self):
        return 0

    def op(self, x, y):
        rows = self.lookup[x]
        if len(rows.shape) == 1:
            rows = rows.unsqueeze(0)
        diag = torch.diag(rows[:, y])
        if len(diag.shape) == 2:
            diag = torch.squeeze(diag)
        return diag


class GLN_p(Group):
    def __init__(self, p: int):
        self.p = p
        self.matrices = generate_all_sequences(p, 4)
        self.matrices = self.matrices.reshape((self.matrices.shape[0], 2, 2)).double()
        self.invertible_matrices = self.matrices[
            ((torch.det(self.matrices)).int() % p) != 0
        ]
        self.lookup = []
        for x in range(len(self.invertible_matrices)):
            self.lookup.append([])
            for y in range(len(self.invertible_matrices)):
                self.lookup[x].append(
                    self.invertible_matrices.tolist().index(
                        (
                            (self.invertible_matrices[x] @ self.invertible_matrices[y])
                            % p
                        ).tolist()
                    )
                )
        self.lookup = torch.tensor(self.lookup).to("cuda")

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "GLN_p" + str(self.p)

    def size(self) -> int:
        return len(self.invertible_matrices)

    def index(self) -> int:
        return self.p

    def id(self):
        return self.p * (self.p - 1) ** 2 + 1

    def op(self, x, y):
        rows = self.lookup[x]
        if len(rows.shape) == 1:
            rows = rows.unsqueeze(0)
        diag = torch.diag(rows[:, y])
        if len(diag.shape) == 2:
            diag = torch.squeeze(diag)
        return diag


class PermutedCyclicGroup(Group):
    def __init__(self, n: int):
        self.n = n
        self.permutation = [(i - 1) % n for i in range(n)]
        random.shuffle(self.permutation)

        self.permutation = [(i - 1) % n for i in range(n)]
        self.lookup = []
        for x in range(self.n):
            self.lookup.append([])
            for y in range(self.n):
                self.lookup[x].append(
                    self.permutation.index(
                        (self.permutation[x] + self.permutation[y]) % self.n
                    )
                )
        self.lookup = torch.tensor(self.lookup).to("cuda")

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "CyclicGroup" + str(self.n)

    def size(self) -> int:
        return self.n

    def index(self) -> int:
        return self.n

    def id(self):
        return self.permutation.index(0)

    def op(self, x, y):
        rows = self.lookup[x]
        print(rows.shape, "rows")
        if len(rows.shape) == 1:
            rows = rows.unsqueeze(0)
        print(rows.shape)
        diag = torch.diag(rows[:, y])
        if len(diag.shape) == 2:
            diag = torch.squeeze(diag)
        print(diag.shape)
        return diag


class CyclicGroup(Group):
    def __init__(self, n: int):
        self.n = n

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "CyclicGroup" + str(self.n)

    def size(self) -> int:
        return self.n

    def index(self) -> int:
        return self.n

    def id(self):
        return 0

    def op(self, x, y):
        return (x + y) % self.n


GroupDict = {
    "PermutedCyclicGroup": PermutedCyclicGroup,
    "CyclicGroup": CyclicGroup,
    "DihedralGroup": DihedralGroup,
    "GLN_p": GLN_p,
}
cycle = CyclicGroup(5)
dihedral = DihedralGroup(4)
gln = GLN_p(2)
print(PermutedCyclicGroup(3).op(torch.tensor([1, 2]), torch.tensor([2, 0])))
