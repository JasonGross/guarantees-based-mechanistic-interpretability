from functools import partial
from inspect import signature
from typing import Callable, Optional, List

from functorch.dim import dims, Dim
from torch import Tensor


def apply(
    f: Callable[..., Tensor],
    collect: Callable[[Tensor, Dim], Tensor],
    sizes: Optional[List[Optional[int]]] = None,
) -> Tensor:
    n_args = len(signature(f).parameters)
    if sizes is None:
        sizes = [None for _ in range(n_args)]
    dim = dims(sizes=[sizes[0]])
    if n_args == 1:
        xs = f(dim)
    else:
        xs = apply(partial(f, dim), collect, sizes[1:])
    return collect(xs, dim)


def sum(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(f, collect=lambda xs, d: xs.sum(d), sizes=sizes)


def min(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(f, collect=lambda xs, d: xs.min(d).values, sizes=sizes)


def max(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(f, collect=lambda xs, d: xs.max(d).values, sizes=sizes)


def array(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(f, collect=lambda xs, d: xs.order(d), sizes=sizes)  # type: ignore[attr-defined]
