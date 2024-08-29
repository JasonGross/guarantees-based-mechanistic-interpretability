from collections import OrderedDict
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def idset(args: Iterable[T]) -> Sequence[T]:
    result = OrderedDict()
    for arg in args:
        if id(arg) not in result:
            result[id(arg)] = arg
    return tuple(result.values())
