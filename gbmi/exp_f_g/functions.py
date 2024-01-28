from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast, Literal, Generic, TypeVar
from gbmi.utils.sequences import generate_all_sequences
import json
import torch

T = TypeVar("T")


class Fun(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    @abstractmethod
    def toJSON(self):
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def index(self) -> int:
        ...

    @staticmethod
    @abstractmethod
    def parameternames() -> List[str]:
        ...

    @abstractmethod
    def op_1(self, a: T, b: T) -> T:
        ...

    @abstractmethod
    def op_2(self, a: T, b: T) -> T:
        ...

    @abstractmethod
    def agree_indices(self) -> List[int]:
        ...

    @abstractmethod
    def reduce_1(self, xs: T) -> T:
        ...

    @abstractmethod
    def reduce_2(self, xs: T) -> T:
        ...


class max_min(Fun):
    def __init__(self, n: int, elements: int):
        self.n = n
        self.elements = elements

        """
        self.lookup = []
        
        self.lookup = torch.tensor(self.lookup).to("cuda")
        """

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "max_min"

    def index(self) -> int:
        return self.n

    def parameternames() -> List[str]:
        return ["modulus", "elements"]

    def op_1(self, x, y):
        return torch.max(x, y)

    def op_2(self, x, y):
        return torch.min(x, y)

    def reduce_1(self, xs: T) -> T:
        accumulator = self.op_1(xs[0], xs[1])
        for x in xs[2:]:
            accumulator = self.op_1(accumulator, x)

        return accumulator

    def reduce_2(self, xs: T) -> T:
        accumulator = self.op_2(xs[0], xs[1])
        for x in xs[2:]:
            accumulator = self.op_2(accumulator, x)

        return accumulator

    def agree_indices(self):
        data = generate_all_sequences(self.n, 2 * self.elements + 1)

        op_1_results = self.reduce_1(data[:, : self.elements].T)
        op_2_results = self.reduce_2(data[:, self.elements :].T)
        l = []
        for i in range(len(op_1_results)):
            if torch.equal(op_1_results[i], op_2_results[i]):
                l.append(i)
        return torch.tensor(l)


class add_sub(Fun):
    def __init__(self, n: int, elements: int):
        self.n = n
        self.elements = elements

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def name(self) -> str:
        return "CyclicGroup" + str(self.n)

    def parameternames() -> List[str]:
        return ["modulus", "elements"]

    def index(self) -> int:
        return self.n

    def op_1(self, x, y):
        return (x + y) % self.n

    def op_2(self, x, y):
        return (x - y) % self.n

    def reduce_1(self, xs: T) -> T:
        accumulator = self.op_1(xs[0], xs[1])
        for x in xs[2:]:
            accumulator = self.op_1(accumulator, x)

        return accumulator

    def reduce_2(self, xs: T) -> T:
        accumulator = self.op_2(xs[0], xs[1])
        for x in xs[2:]:
            accumulator = self.op_2(accumulator, x)

        return accumulator

    def agree_indices(self):
        data = generate_all_sequences(self.n, 2 * self.elements + 1)

        op_1_results = self.reduce_1(data[:, : self.elements].T)
        op_2_results = self.reduce_2(data[:, self.elements :].T)
        l = []
        for i in range(len(op_1_results)):
            if torch.equal(op_1_results[i], op_2_results[i]):
                l.append(i)
        return torch.tensor(l)


FunDict = {"max_min": max_min, "add_sub": add_sub}
