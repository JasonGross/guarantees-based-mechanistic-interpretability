from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast, Literal, Generic, TypeVar
import json
import torch

T = TypeVar("T")


class Group(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def id() -> T:
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

    @staticmethod
    @abstractmethod
    def parameternames() -> List[str]:
        ...

    @abstractmethod
    def op(self, a: T, b: T) -> T:
        ...

    def reduce(self, xs: T) -> T:
        accumulator = self.__class__.id()
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

    def parameternames() -> List[str]:
        return ["modulus"]

    def id():
        return 0

    def op(self, x, y):
        return self.lookup[x][:, y]


class CyclicGroup(Group):
    def __init__(self, n: int):
        self.n = n

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "CyclicGroup" + str(self.n)

    def size(self) -> int:
        return self.n

    def parameternames() -> List[str]:
        return ["modulus"]

    def index(self) -> int:
        return self.n

    def id():
        return 0

    def op(self, x, y):
        return (x + y) % self.n


GroupDict = {"CyclicGroup": CyclicGroup, "DihedralGroup": DihedralGroup}
cycle = CyclicGroup(5)
dihedral = DihedralGroup(4)
