from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast, Literal, Generic, TypeVar


T = TypeVar("T")


class Group(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def id() -> T:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @staticmethod
    @abstractmethod
    def parameternames() -> List[str]:
        ...

    @staticmethod
    @abstractmethod
    def op(a: T, b: T) -> T:
        ...

    @classmethod
    def reduce(cls, xs: T) -> T:
        accumulator = cls.id()
        for x in xs:
            accumulator = cls.op(accumulator, x)

        return accumulator


class CyclicGroup(Group):
    def __init__(self, n: int):
        self.n = n

    def name(self) -> str:
        return "CyclicGroup" + str(self.n)

    def size(self) -> int:
        return self.n

    def parameternames() -> List[str]:
        return ["modulus"]

    def id():
        return 0

    def op(self, x, y):
        return (x + y) % self.n


GroupDict = {"Cyclic": CyclicGroup}
cycle = CyclicGroup(5)
