from __future__ import annotations

import hashlib
import io
import pickle
import logging
from contextlib import contextmanager
from functools import partial
from inspect import signature
from typing import Callable, Optional, List, Union, TypeVar, Generic, Set, Dict

import dill

import torch
from functorch.dim import dims, Dim
from functorch.dim import Tensor as DTensor
from torch import Tensor

TensorLike = Union[Tensor, DTensor]

U = TypeVar("U")
V = TypeVar("V")


class ContextualCache(Generic[U, V]):
    def __init__(self) -> None:
        self.cache: Optional[Dict[U, V]] = None

    @contextmanager
    def access(self):
        if self.cache is None:
            self.cache = {}
            yield self.cache
            self.cache = None
        else:
            yield self.cache


tensor_cache: ContextualCache[str, TensorLike] = ContextualCache()


def lambda_hash(thing: object) -> str:
    # TODO: speed up
    class Pickler(dill.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, DTensor):
                return pickle.loads, (
                    pickle.dumps(
                        obj.order(*obj.dims),
                    ),
                )
            if isinstance(obj, torch.Tensor):
                if "BatchedTensor" in repr(obj):
                    return pickle.loads, (
                        pickle.dumps(torch._C._functorch.get_unwrapped(obj)),
                    )
                return NotImplemented
            if isinstance(obj, Dim):
                # TODO: we're losing data here... pass int through if it exists.
                return pickle.loads, (None,)
            return NotImplemented

    def dumps(obj):
        f = io.BytesIO()
        p = Pickler(f)
        p.dump(obj)
        return f.getvalue()

    return hashlib.md5(dumps(thing)).hexdigest()


class ConstraintTrackingTensor(Tensor):
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
        args_l = list(args)
        if func.__name__ == "__getitem__":
            if isinstance(args_l[1], ConstraintTrackingTensor):
                ConstraintTrackingTensor.add_constraint(args_l[1], args_l[0].shape[0])
            elif isinstance(args_l[1], tuple) and any(
                isinstance(i, ConstraintTrackingTensor) for i in args_l[1]
            ):
                for i, (size, index) in enumerate(zip(args_l[0].shape, args_l[1])):
                    ConstraintTrackingTensor.add_constraint(index, size)

            if isinstance(args_l[0], ConstraintTrackingTensor):
                args_l[0] = torch.tensor(args_l[0])
            return torch.tensor(
                super().__torch_function__(func, types, tuple(args_l), kwargs)
            )
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, tuple(args_l), kwargs)


def _apply_single_dim(
    f: Callable[[int], TensorLike],
    collect: Callable[[DTensor, Dim], TensorLike],
    no_dim: Callable[[TensorLike, Dim], TensorLike],
    size: Optional[int] = None,
    device: Optional[str | torch.device] = None,
    use_cache: bool = True,
) -> TensorLike:
    # no_dim is called if the dim we're 'iterating' over isn't in the returned expression
    c: Dict[str, TensorLike]
    with tensor_cache.access() as c:
        if use_cache:
            key = lambda_hash((f, collect, no_dim, hash(size)))
            if key in c:
                return c[key]

        idx = ConstraintTrackingTensor(torch.tensor(0))
        reified = None
        if size is None:
            reified = f(idx)  # type: ignore
            constraints = getattr(idx, "_constraints", [])
            if len(constraints) > 1:
                # TODO: name the dimension argument with the error
                raise ValueError(
                    f"Error: incompatible constraints for dimension ({constraints})"
                )
            elif len(constraints) == 0:
                # TODO: introduce warning if we fail
                size = None
            else:
                size = list(constraints)[0]

        dim = dims(sizes=[size])
        if size is not None:
            idx = torch.arange(size)  # type: ignore
            if device is not None:
                idx = idx.to(device)  # type: ignore
            elif reified is not None:
                idx = idx.to(reified.device)  # type: ignore
            try:
                xs = f(idx[dim])  # type: ignore
            except RuntimeError as e:
                if (
                    device is not None
                    or reified is not None
                    or "same device" not in str(e)
                ):
                    raise e
                reified = f(idx)  # type: ignore
                idx = idx.to(reified.device)  # type: ignore
                xs = f(idx[dim])  # type: ignore
                # warn only if running it a second time actually works
                logging.warning(
                    f"Ran ein function twice just to get target device, try passing device={reified.device!r}"
                )
        else:
            xs = f(dim)
        if isinstance(xs, DTensor) and hash(dim) in [hash(i) for i in xs.dims]:
            result = collect(xs, dim)
        else:
            result = no_dim(xs, dim)

        if use_cache:
            c[key] = result
        return result


def _apply(
    f: Callable[..., Tensor],
    collect: Callable[[Tensor, Dim], Tensor],
    no_dim: Callable[[Tensor, Dim], Tensor],
    sizes: Optional[List[Optional[int]]] = None,
    device: Optional[str | torch.device] = None,
    use_cache: bool = True,
) -> Tensor:
    n_args = len(signature(f).parameters)
    if sizes is None:
        sizes = [None for _ in range(n_args)]
    assert len(sizes) == n_args

    return _apply_single_dim(
        (
            (
                lambda dim: _apply(
                    partial(f, dim),
                    collect,
                    no_dim,
                    sizes[1:],
                    device=device,
                    use_cache=use_cache,
                )
            )
            if len(sizes) > 1
            else f
        ),
        collect,
        no_dim,
        sizes[0],
        device=device,
        use_cache=use_cache,
    )


def sum(
    f: Callable[..., Tensor],
    sizes: Optional[List[Optional[int]]] = None,
    device: Optional[str | torch.device] = None,
    use_cache: bool = True,
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.sum(d),
        no_dim=lambda xs, d: xs * d.size,
        sizes=sizes,
        device=device,
        use_cache=use_cache,
    )


def min(
    f: Callable[..., Tensor],
    sizes: Optional[List[Optional[int]]] = None,
    device: Optional[str | torch.device] = None,
    use_cache: bool = True,
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.min(d).values,
        no_dim=lambda xs, d: xs,
        sizes=sizes,
        device=device,
        use_cache=use_cache,
    )


def max(
    f: Callable[..., Tensor],
    sizes: Optional[List[Optional[int]]] = None,
    device: Optional[str | torch.device] = None,
    use_cache: bool = True,
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.max(d).values,
        no_dim=lambda xs, d: xs,
        sizes=sizes,
        device=device,
        use_cache=use_cache,
    )


def array(
    f: Callable[..., Tensor],
    sizes: Optional[List[Optional[int]]] = None,
    device: Optional[str | torch.device] = None,
    use_cache: bool = True,
) -> Tensor:
    return _apply(
        f,
        collect=lambda xs, d: xs.order(d),
        no_dim=lambda xs, d: xs.unsqueeze(0).repeat(d.size, *[1 for _ in xs.shape]),
        sizes=sizes,
        device=device,
        use_cache=use_cache,
    )
