from contextlib import contextmanager
from enum import Enum
from functools import partial, cache
from inspect import signature
from typing import Callable, Optional, List, Union, TypeVar, Generic, Set, Dict

import torch
from functorch.dim import dims, Dim
from functorch.dim import Tensor as DTensor
from torch import Tensor
from einops import repeat

from gbmi.utils.hashing import lambda_hash

TensorLike = Union[Tensor, DTensor]

# T = TypeVar("T")
#
#
# class ContextualGlobal(Generic[T]):
#     def __init__(self, val: T):
#         self.vals = [val]
#
#     @contextmanager
#     def set(self, val: T):
#         self.vals.append(val)
#         yield
#         self.vals.pop()
#
#     def get(self) -> T:
#         return self.vals[-1]

U = TypeVar("U")
V = TypeVar("V")


class ContextualCache(Generic[U, V]):
    def __init__(self) -> None:
        self.cache: Optional[Dict[U, V]] = {}

    @contextmanager
    def access(self):
        if self.cache is None:
            self.cache = {}
            yield self.cache
            self.cache = None
        else:
            yield self.cache


tensor_cache: ContextualCache[str, TensorLike] = ContextualCache()


class ConstraintTrackingTensor(DTensor):
    _constraints: Set[int]

    @staticmethod
    def add_constraint(tensor, size):
        if isinstance(tensor, ConstraintTrackingTensor):
            if hasattr(tensor, "_constraints"):
                tensor._constraints.add(size)
            else:
                tensor._constraints = {size}

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func.__name__ == "__getitem__":
            print(args[0], args[1])
            if isinstance(args[1], ConstraintTrackingTensor):
                ConstraintTrackingTensor.add_constraint(args[1], args[0].shape[0])
            elif isinstance(args[1], tuple) and isinstance(
                args[1][0], ConstraintTrackingTensor
            ):
                for size, index in zip(args[0].shape, args[1]):
                    ConstraintTrackingTensor.add_constraint(index, size)
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)


def _apply_single_dim(
    f: Callable[[int], TensorLike],
    collect: Callable[[DTensor, Dim], TensorLike],
    no_dim: Callable[[TensorLike, Dim], TensorLike],
    size: Optional[int] = None,
    device="cpu",
) -> TensorLike:
    # no_dim is called if the dim we're 'iterating' over isn't in the returned expression
    # c: Dict[str, TensorLike]
    # with tensor_cache.access() as c:
    # key = lambda_hash((f, collect, no_dim, hash(size)))
    # if key in c:
    #     return c[key]

    # if size is None:
    #     idx = ConstraintTrackingTensor(torch.tensor(0))
    #     f(idx)  # type: ignore
    #     constraints = idx._constraints
    #     if len(constraints) > 1:
    #         # TODO: name the dimension argument with the error
    #         raise ValueError(
    #             f"Error: incompatible constraints for dimension ({constraints})"
    #         )
    #     elif len(constraints) == 0:
    #         # TODO: introduce warning if we fail
    #         size = None
    #     else:
    #         size = list(constraints)[0]

    dim = dims(sizes=[size])
    if size is not None:
        idx = torch.arange(size).to(device)
        xs = f(idx[dim])  # type: ignore
    else:
        xs = f(dim)
    if isinstance(xs, DTensor) and hash(dim) in [hash(i) for i in xs.dims]:
        result = collect(xs, dim)
    else:
        result = no_dim(xs, dim)

    # c[key] = result
    return result


def _apply(
    f: Callable[..., Tensor],
    collect: Callable[[Tensor, Dim], Tensor],
    no_dim: Callable[[Tensor, Dim], Tensor],
    sizes: Optional[List[Optional[int]]] = None,
    device="cpu",
) -> Tensor:
    n_args = len(signature(f).parameters)
    if sizes is None:
        sizes = [None for _ in range(n_args)]
    assert len(sizes) == n_args

    return _apply_single_dim(
        (
            (lambda dim: _apply(partial(f, dim), collect, no_dim, sizes[1:], device))
            if len(sizes) > 1
            else f
        ),
        collect,
        no_dim,
        sizes[0],
        device,
    )


def sum(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None, device="cpu"
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.sum(d),
        no_dim=lambda xs, d: xs * d.size,
        sizes=sizes,
        device=device,
    )


def min(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None, device="cpu"
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.min(d).values,
        no_dim=lambda xs, d: xs,
        sizes=sizes,
        device=device,
    )


def max(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None, device="cpu"
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.max(d).values,
        no_dim=lambda xs, d: xs,
        sizes=sizes,
        device=device,
    )


def array(
    f: Callable[..., Tensor], sizes: Optional[List[Optional[int]]] = None, device="cpu"
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.order(d),
        no_dim=lambda xs, d: xs.unsqueeze(0).repeat(d.size, *[1 for _ in xs.shape]),
        sizes=sizes,
        device=device,
    )
