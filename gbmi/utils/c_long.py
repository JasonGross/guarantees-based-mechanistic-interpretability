from typing import Iterable, Mapping, TypeVar, Union

import numpy as np

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def too_big_for_C_long(value: T) -> bool:
    """Returns whether the value is too big for a C long."""
    return isinstance(value, int) and (
        value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max
    )


def str_if_too_big_for_C_long(value: T) -> Union[T, str]:
    """Returns the value as a string if it is too big for a C long."""
    if too_big_for_C_long(value):
        return str(value)
    return value


def str_values_if_too_big_for_C_long(d: dict[K, T]) -> dict[K, Union[str, T]]:
    """Returns a new dictionary with values as strings if they are too big for a C long."""
    return {k: str_if_too_big_for_C_long(v) for k, v in d.items()}


def str_list_values_if_any_too_big_for_C_long(
    ds: Iterable[Mapping[K, T]]
) -> list[dict[K, Union[str, T]]]:
    """Returns a new list of dictionaries with values as strings for any key for which any dict in the list is too big for a C long."""
    ds = list(ds)
    too_big_keys = set().union(
        *[{k for k, v in d.items() if too_big_for_C_long(v)} for d in ds]
    )
    return [{k: str(v) if k in too_big_keys else v for k, v in d.items()} for d in ds]
