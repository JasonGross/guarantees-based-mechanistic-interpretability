from functools import partial
from inspect import signature
from typing import Callable, Optional, List

from functorch.dim import dims, Dim
from torch import Tensor
from einops import repeat


def apply(
    f: Callable[..., Tensor],
    collect: Callable[[Tensor, Dim], Tensor],
    no_dim: Callable[[Tensor, Dim], Tensor],
    sizes: Optional[List[Optional[int]]] = None,
) -> Tensor:
    # no_dim is called if the dim we're 'iterating' over isn't in the returned expression
    n_args = len(signature(f).parameters)
    if sizes is None:
        sizes = [None for _ in range(n_args)]
    dim = dims(sizes=[sizes[0]])
    if n_args == 1:
        xs = f(dim)
    else:
        xs = apply(partial(f, dim), collect, no_dim, sizes[1:])
    if hash(dim) in [hash(i) for i in xs.dims]:
        return collect(xs, dim)
    else:
        return no_dim(xs, dim)


def sum(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(
        f,
        collect=lambda xs, d: xs.sum(d),
        no_dim=lambda xs, d: xs * d.size,
        sizes=sizes,
    )


def min(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(
        f, collect=lambda xs, d: xs.min(d).values, no_dim=lambda xs, d: xs, sizes=sizes
    )


def max(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(
        f, collect=lambda xs, d: xs.max(d).values, no_dim=lambda xs, d: xs, sizes=sizes
    )


def array(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None
) -> Tensor:
    return apply(f, collect=lambda xs, d: xs.order(d), no_dim=lambda xs, d: xs.unsqueeze(0).repeat(d.size, *[1 for _ in xs.shape]), sizes=sizes)  # type: ignore[attr-defined]
