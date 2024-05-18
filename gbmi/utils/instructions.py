"""Some utilities for counting floating point operations"""

# %%
from __future__ import annotations
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import OrderedDict
from functools import partial, cache, cached_property
from itertools import zip_longest
from types import NoneType
from ctypes import c_uint64
from typing import (
    cast,
    Iterable,
    Sequence,
    Literal,
    Optional,
    NamedTuple,
    SupportsIndex,
    Union,
    Tuple,
    Collection,
    Callable,
    Iterator,
    Any,
    Protocol,
    TypeVar,
    overload,
)
from types import EllipsisType
import numpy as np
import torch
from torch import empty as torch_empty  # for stability under hot patching
from torch import ones as torch_ones  # for stability under hot patching
from transformer_lens import HookedTransformer
import fancy_einsum
import einops
import einops._backends
from gbmi.verification_tools.svd import compute_verify_svd_close_matrices

# %%
try:
    import cirron
    import cirron.cirron

    HAS_CIRRON = True
    PERF_WORKING = cirron.cirron.overhead["instruction_count"] > 0

except Exception as e:
    print(f"Warning: perf cpu instruction counting not available ({e})")
    HAS_CIRRON = False
    PERF_WORKING = False

    class _cirron:
        FAKE = True

        class Collector:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        class cirron:
            class Counter:
                pass

    cirron = _cirron
# from cirron import Collector
# from cirron.cirron import Counter


class PerfCounter(cirron.cirron.Counter):  # type: ignore
    @staticmethod
    def lift(f: Callable[..., int], *args, **kwargs) -> "PerfCounter":
        if hasattr(cirron, "FAKE"):
            return PerfCounter()
        apply_f = lambda attr: c_uint64(
            f(
                *[
                    getattr(c, attr).value if isinstance(c, PerfCounter) else c
                    for c in args
                ],
                **{
                    k: getattr(c, attr).value if isinstance(c, PerfCounter) else c
                    for k, c in kwargs.items()
                },
            )
        )
        return PerfCounter(
            time_enabled_ns=apply_f("time_enabled_ns"),
            instruction_count=apply_f("instruction_count"),
            branch_misses=apply_f("branch_misses"),
            page_faults=apply_f("page_faults"),
        )

    def __add__(self, other: Union[int, "PerfCounter"]) -> "PerfCounter":
        return PerfCounter.lift(int.__add__, self, other)

    def __radd__(self, other: Union[int, "PerfCounter"]) -> "PerfCounter":
        return PerfCounter.lift(int.__radd__, self, other)

    def __sub__(self, other: Union[int, "PerfCounter"]) -> "PerfCounter":
        return PerfCounter.lift(int.__sub__, self, other)

    def __rsub__(self, other: Union[int, "PerfCounter"]) -> "PerfCounter":
        return PerfCounter.lift(int.__rsub__, self, other)

    def __abs__(self) -> "PerfCounter":
        return PerfCounter.lift(int.__abs__, self)

    def __neg__(self) -> "PerfCounter":
        return PerfCounter.lift(int.__neg__, self)

    def __pos__(self) -> "PerfCounter":
        return PerfCounter.lift(int.__pos__, self)

    def __mul__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__mul__, self, other)

    def __rmul__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__rmul__, self, other)

    def __floordiv__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__floordiv__, self, other)

    def __rfloordiv__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__rfloordiv__, self, other)

    def __mod__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__mod__, self, other)

    def __rmod__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__rmod__, self, other)

    def __lshift__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__lshift__, self, other)

    def __rlshift__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__rlshift__, self, other)

    def __rshift__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__rshift__, self, other)

    def __rrshift__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__rrshift__, self, other)

    def __pow__(self, other: int) -> "PerfCounter":
        return PerfCounter.lift(int.__pow__, self, other)

    def __round__(self, *args, **kwargs) -> "PerfCounter":
        return PerfCounter.lift(int.__round__, self, *args, **kwargs)


class PerfCollector(cirron.Collector):  # type: ignore
    def __init__(self):
        super().__init__()
        self.counters = PerfCounter()


@dataclass
class InstructionCount:
    flop: int = 0
    int_op: int = 0
    branch: int = 0

    @property
    def total(self) -> int:
        return self.flop + self.int_op + self.branch

    def copy(self) -> "InstructionCount":
        return InstructionCount(flop=self.flop, int_op=self.int_op, branch=self.branch)

    def update(self, other: "InstructionCount") -> "InstructionCount":
        self.flop = other.flop
        self.int_op = other.int_op
        self.branch = other.branch
        return self

    def __add__(
        self, other: Union["InstructionCount", Literal[0]]
    ) -> "InstructionCount":
        if isinstance(other, InstructionCount):
            return InstructionCount(
                flop=self.flop + other.flop,
                int_op=self.int_op + other.int_op,
                branch=self.branch + other.branch,
            )
        assert other == 0, f"other == {other} != 0"
        return self

    __radd__ = __add__

    def __iadd__(
        self, other: Union["InstructionCount", Literal[0]]
    ) -> "InstructionCount":
        return self.update(self.__add__(other))

    def add_flop(self, flop: int = 1) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop + flop, int_op=self.int_op, branch=self.branch
        )

    def add_int_op(self, int_op: int = 1) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop, int_op=self.int_op + int_op, branch=self.branch
        )

    def add_branch(self, branch: int = 1) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop, int_op=self.int_op, branch=self.branch + branch
        )

    def __mul__(self, other: int) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop * other,
            int_op=self.int_op * other,
            branch=self.branch * other,
        )

    __rmul__ = __mul__

    def __imul__(self, other: int) -> "InstructionCount":
        return self.update(self.__mul__(other))

    def __div__(self, other: int) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop // other,
            int_op=self.int_op // other,
            branch=self.branch // other,
        )

    def __idiv__(self, other: int) -> "InstructionCount":
        return self.update(self.__div__(other))

    def __truediv__(self, other: int) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop // other,
            int_op=self.int_op // other,
            branch=self.branch // other,
        )

    def __itruediv__(self, other: int) -> "InstructionCount":
        return self.update(self.__truediv__(other))

    def __floordiv__(self, other: int) -> "InstructionCount":
        return InstructionCount(
            flop=self.flop // other,
            int_op=self.int_op // other,
            branch=self.branch // other,
        )

    def __ifloordiv__(self, other: int) -> "InstructionCount":
        return self.update(self.__floordiv__(other))

    def __str__(self) -> str:
        return f"InstructionCount(flop={self.flop}, int_op={self.int_op}, branch={self.branch})"

    def __repr__(self) -> str:
        return f"InstructionCount(flop={self.flop!r}, int_op={self.int_op!r}, branch={self.branch!r})"

    def __hash__(self) -> int:
        return hash((self.flop, self.int_op, self.branch))


_T_co = TypeVar("_T_co", covariant=True)


class _NestedSequence(Protocol[_T_co]):
    """A protocol for representing nested sequences.

    References::
        `numpy._typing._NestedSequence`
        <https://github.com/numpy/numpy/blob/main/numpy/_typing/_nested_sequence.py>
    """

    def __len__(self, /) -> int: ...
    def __getitem__(self, index: int, /) -> _T_co | _NestedSequence[_T_co]: ...
    def __contains__(self, x: object, /) -> bool: ...
    def __iter__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]: ...
    def __reversed__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]: ...
    def count(self, value: Any, /) -> int: ...
    def index(self, value: Any, /) -> int: ...


def nested_sequence_empty(seq: _NestedSequence) -> bool:
    try:
        iter(seq)
        return len(seq) == 0
    except TypeError:
        return True


def tensor_of_nested_sequence(seq: _NestedSequence) -> torch.Tensor:
    if isinstance(seq, torch.Tensor):
        return seq
    if nested_sequence_empty(seq):
        return torch.tensor(seq)
    return torch.stack([tensor_of_nested_sequence(s) for s in seq], dim=0)


def CountTensor_of_nested_sequence(seq: _NestedSequence) -> CountTensor:
    if isinstance(seq, torch.Tensor):
        return CountTensor.from_numpy(seq)
    if nested_sequence_empty(seq):
        return CountTensor(shape=())
    return CountTensor.stack([CountTensor_of_nested_sequence(s) for s in seq], dim=0)


def index_nested_sequence(seq: _NestedSequence, index: Tuple[int, ...]) -> Any:
    """Index a nested sequence."""
    for i in index:
        seq = seq[i]
    return seq


TensorIndexType = Union[
    Union[
        SupportsIndex,
        Union[None, bool, int, slice, EllipsisType, torch.Tensor],
        _NestedSequence[Union[None, bool, int, slice, EllipsisType, torch.Tensor]],
    ],
    tuple[
        Union[
            SupportsIndex,
            Union[None, bool, int, slice, EllipsisType, torch.Tensor],
            _NestedSequence[Union[None, bool, int, slice, EllipsisType, torch.Tensor]],
        ],
        ...,
    ],
]
CountTensorIndexType = Union[
    Union[
        SupportsIndex,
        Union[None, bool, int, slice, EllipsisType, "CountTensor"],
        _NestedSequence[Union[None, bool, int, slice, EllipsisType, "CountTensor"]],
    ],
    tuple[
        Union[
            SupportsIndex,
            Union[None, bool, int, slice, EllipsisType, "CountTensor"],
            _NestedSequence[Union[None, bool, int, slice, EllipsisType, "CountTensor"]],
        ],
        ...,
    ],
]
TensorOrCountTensorIndexType = Union[
    Union[
        SupportsIndex,
        Union[None, bool, int, slice, EllipsisType, "CountTensor", torch.Tensor],
        _NestedSequence[
            Union[None, bool, int, slice, EllipsisType, "CountTensor", torch.Tensor]
        ],
    ],
    tuple[
        Union[
            SupportsIndex,
            Union[None, bool, int, slice, EllipsisType, "CountTensor", torch.Tensor],
            _NestedSequence[
                Union[None, bool, int, slice, EllipsisType, "CountTensor", torch.Tensor]
            ],
        ],
        ...,
    ],
]


class count_values_indices(NamedTuple):
    values: CountTensor
    indices: CountTensor


mode: Literal["verify", "search"] = "verify"
default_sanity_check: bool = True


@contextmanager
def set_sanity_check(sanity_check: bool):
    global default_sanity_check
    old_sanity_check = default_sanity_check
    default_sanity_check = sanity_check
    try:
        yield
    finally:
        default_sanity_check = old_sanity_check


count_to_update: Optional[InstructionCount] = None


@contextmanager
def CountTensorOperations() -> Iterator[InstructionCount]:
    global count_to_update
    old_count_to_update = count_to_update
    count_to_update = InstructionCount()
    yield count_to_update
    count_to_update = old_count_to_update


def add_to_count(count: InstructionCount):
    global count_to_update
    if count_to_update is not None:
        count_to_update += count


def get_count() -> InstructionCount:
    if count_to_update is None:
        return None
    return count_to_update.copy()


@dataclass
class CountTensor:
    shape: Sequence[int]
    count: InstructionCount = InstructionCount()
    # parents: dict[int, "CountTensor"] = field(default_factory=OrderedDict)
    is_bool: bool = False

    def __post_init__(self):
        if count_to_update is None:
            self.global_count_at_creation = None
        else:
            self.global_count_at_creation = count_to_update.copy()

    def __str__(self):
        return f"CountTensor(shape={self.shape}, count={self.count}, is_bool={self.is_bool}"

    def __repr__(self):
        return f"CountTensor(shape={self.shape!r}, count={self.count!r}, is_bool={self.is_bool!r}"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # @staticmethod
    # def _parents_of_tuple(parents: Iterable["CountTensor"]) -> dict[int, "CountTensor"]:
    #     return OrderedDict((id(p), p) for p in parents)

    @staticmethod
    def from_numpy(
        x: Union[np.ndarray, torch.Tensor, int, float, bool, "CountTensor"],
        return_not_implemented: bool = True,
    ) -> "CountTensor":
        if isinstance(x, CountTensor):
            return x
        elif not isinstance(x, torch.Tensor):
            try:
                x = torch.tensor(x)
            except TypeError as e:
                if return_not_implemented:
                    return NotImplemented
                raise e
        return CountTensor(
            shape=x.shape, is_bool=x.dtype == torch.bool or x.dtype == bool
        )

    # @cached_property
    # def transitive_parents(self) -> OrderedDict[int, "CountTensor"]:
    #     all_parents = OrderedDict()
    #     pending = list(self.parents.items())
    #     while pending:
    #         idp, cur_parent = pending.pop(0)
    #         if idp in all_parents:
    #             continue
    #         all_parents[idp] = cur_parent
    #         pending.extend(list(cur_parent.parents.items()))
    #     return all_parents

    # @cached_property
    # def full_count(self) -> InstructionCount:
    #     count = self.count
    #     for p in self.transitive_parents.values():
    #         count += p.count
    #     return count

    def _unary(self, is_bool: Optional[bool] = None) -> "CountTensor":
        count = InstructionCount(flop=int(np.prod(self.shape)))
        add_to_count(count)
        return CountTensor(
            shape=self.shape,
            count=count,
            is_bool=is_bool if is_bool is not None else self.is_bool,
            # parents=CountTensor._parents_of_tuple((self,)),
        )

    def unary(self) -> "CountTensor":
        return self._unary(is_bool=None)

    def unary_bool(self) -> "CountTensor":
        return self._unary(is_bool=True)

    def unary_arith(self) -> "CountTensor":
        return self._unary(is_bool=False)

    def _binary_only_scalar(
        self, other: Union[int, float], is_bool: Optional[bool] = None
    ) -> "CountTensor":
        return self._unary(is_bool=is_bool)

    def binary_only_scalar(self, other: Union[int, float]) -> "CountTensor":
        return self._binary_only_scalar(other, is_bool=None)

    def binary_only_scalar_bool(self, other: Union[int, float]) -> "CountTensor":
        return self._binary_only_scalar(other, is_bool=True)

    def binary_only_scalar_arith(self, other: Union[int, float]) -> "CountTensor":
        return self._binary_only_scalar(other, is_bool=False)

    def _binary_only(
        self, other: "CountTensor", is_bool: Optional[bool] = None
    ) -> "CountTensor":
        if other is NotImplemented:
            return NotImplemented
        shape = torch.broadcast_shapes(self.shape, other.shape)
        assert isinstance(
            other, CountTensor
        ), f"Expected CountTensor, got {type(other)}"
        count = InstructionCount(flop=int(np.prod(shape)))
        add_to_count(count)
        return CountTensor(
            shape=shape,
            count=count,
            is_bool=(
                is_bool if is_bool is not None else (self.is_bool and other.is_bool)
            ),
            # parents=CountTensor._parents_of_tuple((self, other)),
        )

    def _binary(
        self,
        other: Union[int, float, "CountTensor", np.ndarray, torch.Tensor],
        is_bool: Optional[bool] = None,
    ) -> "CountTensor":
        if isinstance(other, CountTensor):
            return self._binary_only(other, is_bool=is_bool)
        elif hasattr(other, "shape"):
            return self._binary_only(CountTensor.from_numpy(other), is_bool=is_bool)
        return self._unary(is_bool=is_bool)

    def binary(
        self, other: Union[int, float, "CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        return self._binary(other, is_bool=None)

    def binary_bool(
        self, other: Union[int, float, "CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        return self._binary(other, is_bool=True)

    def binary_arith(
        self, other: Union[int, float, "CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        return self._binary(other, is_bool=False)

    def _fold_reduce(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
        is_bool: Optional[bool] = None,
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        shape = list(self.shape)
        if dim is None:
            count = InstructionCount(flop=int(np.prod(shape)) - 1)
            add_to_count(count)
            return CountTensor(
                shape=[],
                count=count,
                is_bool=is_bool if is_bool is not None else self.is_bool,
                # parents=CountTensor._parents_of_tuple((self,)),
            )
        shape_without_dim = list(shape)
        if not hasattr(dim, "__iter__"):
            dim = (dim,)
        shape_only_dim = []
        dim = tuple(reversed(sorted([i % len(shape) for i in dim])))
        for i in dim:
            shape_only_dim.append(shape_without_dim.pop(i))
        count = InstructionCount(
            flop=int(np.prod(shape_without_dim)) * (int(np.prod(shape_only_dim)) - 1)
        )
        add_to_count(count)
        result = CountTensor(
            shape=shape_without_dim,
            count=count,
            is_bool=is_bool if is_bool is not None else self.is_bool,
            # parents=CountTensor._parents_of_tuple((self,)),
        )
        if keepdim:
            for i in dim:
                result = result.unsqueeze(i)
        return result

    def fold_reduce(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        return self._fold_reduce(dim=dim, axis=axis, keepdim=keepdim, is_bool=None)

    def fold_reduce_bool(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        return self._fold_reduce(dim=dim, axis=axis, keepdim=keepdim, is_bool=True)

    def fold_reduce_arith(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        return self._fold_reduce(dim=dim, axis=axis, keepdim=keepdim, is_bool=False)

    def _fold_reduce_values_indices(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
        is_bool: Optional[bool] = None,
    ) -> Union["CountTensor", count_values_indices]:
        result = self._fold_reduce(dim=dim, axis=axis, keepdim=keepdim, is_bool=is_bool)
        if dim is None and axis is None:
            return result
        return count_values_indices(result, result)

    def fold_reduce_values_indices(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> Union["CountTensor", count_values_indices]:
        return self._fold_reduce_values_indices(
            dim=dim, axis=axis, keepdim=keepdim, is_bool=None
        )

    def fold_reduce_values_indices_bool(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> Union["CountTensor", count_values_indices]:
        return self._fold_reduce_values_indices(
            dim=dim, axis=axis, keepdim=keepdim, is_bool=True
        )

    def fold_reduce_values_indices_arith(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> Union["CountTensor", count_values_indices]:
        return self._fold_reduce_values_indices(
            dim=dim, axis=axis, keepdim=keepdim, is_bool=False
        )

    __add__ = binary
    __radd__ = binary
    __sub__ = binary
    __rsub__ = binary
    __mul__ = binary
    __rmul__ = binary
    __div__ = binary
    __rdiv__ = binary
    __truediv__ = binary
    __rtruediv__ = binary
    __floordiv__ = binary
    __rfloordiv__ = binary
    __mod__ = binary
    __rmod__ = binary
    __or__ = binary_bool
    __ror__ = binary_bool
    __and__ = binary_bool
    __rand__ = binary_bool
    __xor__ = binary_bool
    __rxor__ = binary_bool
    __eq__ = binary_bool  # type: ignore
    __ne__ = binary_bool  # type: ignore
    __lt__ = binary_bool
    __le__ = binary_bool
    __gt__ = binary_bool
    __ge__ = binary_bool
    __req__ = binary_bool
    __rne__ = binary_bool
    __rlt__ = binary_bool
    __rle__ = binary_bool
    __rgt__ = binary_bool
    __rge__ = binary_bool

    clone = unary
    __abs__ = unary
    __neg__ = unary
    __pos__ = unary
    __invert__ = unary_bool
    abs = __abs__
    conj = unary
    sqrt = unary_arith
    exp = unary_arith
    log = unary_arith
    log1p = unary_arith
    isnan = unary_bool
    pow = binary_only_scalar_arith
    float = unary_arith
    long = unary_arith
    bool = unary_bool

    def diag(self, diagonal: int = 0) -> "CountTensor":
        if len(self.shape) < 2:
            shape = (self.shape[0], self.shape[0])
        else:
            init_shape, (a, b) = tuple(self.shape[:-2]), self.shape[-2:]
            # r = i, c = i + diagonal if diagonal > 0
            # r = i - diagonal, c = i if diagonal < 0
            if diagonal >= 0:
                shape = init_shape + (min(a, b - diagonal),)
            else:
                shape = init_shape + (min(a - diagonal, b),)
        count = InstructionCount(flop=int(np.prod(shape)))
        add_to_count(count)
        return CountTensor(shape=shape, count=count, is_bool=self.is_bool)

    def norm(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        return (self * self).fold_reduce_arith(dim=dim, axis=axis, keepdim=keepdim)

    def tril(self, diagonal: int = 0) -> "CountTensor":
        return self.unary()

    triu = tril

    def _reshape(self, new_shape: Sequence[int]) -> "CountTensor":
        """explicitly does not check the number of elements"""
        return CountTensor(
            shape=new_shape,
            # parents=CountTensor._parents_of_tuple((self,)),
            is_bool=self.is_bool,
        )

    def reshape(self, new_shape: Sequence[int]) -> "CountTensor":
        return self._reshape(new_shape)

    def expand(self, *sizes: Union[int, Sequence[int]]) -> "CountTensor":
        if len(sizes) == 0:
            return self
        if len(sizes) == 1 and isinstance(sizes[0], Sequence):
            return self.expand(*sizes[0])
        sizes = list(sizes)
        for d, (i, sz) in zip(reversed(self.shape), reversed(list(enumerate(sizes)))):
            if sz == -1:
                sizes[i] = d
        shape = torch.broadcast_shapes(self.shape, sizes)
        return self._reshape(shape)

    def squeeze(self) -> "CountTensor":
        return self.reshape(tuple(idx for idx in self.shape if idx != 1))

    def broadcast_to(self, shape: Sequence[int]) -> "CountTensor":
        return self.expand(*shape)

    def __matmul__(
        self, other: Union["CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        assert hasattr(
            other, "shape"
        ), f"Expected CountTensor, ndarray, or Tensor, got {type(other)} ({other})"
        if not isinstance(other, CountTensor):
            other = CountTensor.from_numpy(other)
            if other is NotImplemented:
                return NotImplemented
        if len(other.shape) == 1:
            x_shape = tuple(self.shape)
            out_shape = x_shape[:-1]
        else:
            x_shape = torch.broadcast_shapes(self.shape, other.shape[:-1])
            out_shape = (*x_shape[:-1], other.shape[-1])
        # at each index, we do x_shape[-1] multiplications and x_shape[-1] - 1 additions
        count = InstructionCount(flop=int(np.prod(out_shape)) * (x_shape[-1] * 2 - 1))
        add_to_count(count)
        return CountTensor(
            shape=out_shape,
            count=count,
            is_bool=self.is_bool and other.is_bool,
            # parents=CountTensor._parents_of_tuple((self, other)),
        )

    def __rmatmul__(
        self, other: Union["CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        assert hasattr(
            other, "shape"
        ), f"Expected CountTensor, ndarray, or Tensor, got {type(other)} ({other})"
        if not isinstance(other, CountTensor):
            other = CountTensor.from_numpy(other)
        return other.__matmul__(self)

    sum = fold_reduce
    argmax = fold_reduce
    argmin = fold_reduce
    max = fold_reduce_values_indices
    min = fold_reduce_values_indices
    amin = fold_reduce
    amax = fold_reduce
    prod = fold_reduce
    any = fold_reduce_bool
    all = fold_reduce_bool

    def repeat(self, *reps: Union[Sequence[int], int]):
        if len(reps) == 1 and hasattr(reps[0], "__iter__"):
            return self.repeat(*reps[0])
        shape_reps0 = [
            (s, r)
            for s, r in zip_longest(reversed(self.shape), reversed(reps), fillvalue=1)
        ]
        shape_reps = cast(list[Tuple[int, int]], shape_reps0)
        shape = tuple(reversed([s * r for s, r in shape_reps]))
        return self._reshape(shape)

    def mean(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        if dim is not None and not hasattr(dim, "__iter__"):
            dim = (dim,)
        total = int(
            np.prod(self.shape if dim is None else [self.shape[i] for i in dim])
        )
        return self.sum(dim=dim, keepdim=keepdim) / total

    def softmax(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        adjusted = self - self.amax(dim=dim, keepdim=True)
        adjusted_exp = adjusted.exp()
        return adjusted_exp / adjusted_exp.sum(dim=dim, keepdim=True)

    def log_softmax(
        self,
        dim: Optional[Union[int, Sequence[int]]] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        adjusted = self - self.amax(dim=dim, keepdim=True)
        adjusted_log_sum_exp = adjusted.exp().sum(dim=dim, keepdim=True).log()
        return adjusted - adjusted_log_sum_exp

    def flip(self, *args, **kwargs) -> "CountTensor":
        return self.unary()

    def permute(self, *args, **kwargs) -> "CountTensor":
        return self._reshape(
            tuple(
                torch.tensor(self.shape, dtype=torch.long)
                .permute(*args, **kwargs)
                .tolist()
            )
        )

    def transpose(self, dim0: int, dim1: int) -> "CountTensor":
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return self._reshape(tuple(new_shape))

    @property
    def T(self) -> "CountTensor":
        return self._reshape(tuple(reversed(self.shape)))

    @property
    def mT(self) -> "CountTensor":
        return self.transpose(-2, -1)

    def adjoint(self) -> "CountTensor":
        return self.transpose(-2, -1).conj()

    @property
    def mH(self) -> "CountTensor":
        return self.adjoint()

    @property
    def H(self) -> "CountTensor":
        return self.mH

    @staticmethod
    def einsum(equation: str, *args: "CountTensor") -> "CountTensor":
        assert "..." not in equation, "Ellipsis not yet supported"
        lhs, rhs = equation.split("->")
        contracted_idxs = set(lhs.replace(",", "")) - set(rhs)
        lhs = lhs.split(",")
        assert len(lhs) == len(args), f"Expected {len(lhs)} arguments, got {len(args)}"
        shape_map = {}
        for arg, part in zip(args, lhs):
            assert len(part) == len(
                arg.shape
            ), f"Expected {len(part)} indices, got {len(arg.shape)}"
            for idx, size in zip(part, arg.shape):
                shape_map[idx] = torch.broadcast_shapes(
                    (size,), (shape_map.get(idx, size),)
                )[0]
        num_output_elements = int(np.prod([shape_map[idx] for idx in rhs]))
        # we multiply all elements with len(lhs) - 1 multiplications * prod of contracted indices, then sum over all contracted indices with prod contracted indices - 1 additions
        num_contracted_indices = int(
            np.prod([shape_map[idx] for idx in contracted_idxs])
        )
        # num_output_elements * ((len(lhs) - 1) * num_contracted_indices + num_contracted_indices - 1)
        # = num_output_elements * (len(lhs) * num_contracted_indices - 1)
        flops = num_output_elements * (len(lhs) * num_contracted_indices - 1)
        count = InstructionCount(flop=flops)
        add_to_count(count)
        return CountTensor(
            shape=tuple(shape_map[idx] for idx in rhs),
            count=count,
            is_bool=False,
            # parents=CountTensor._parents_of_tuple(
            #     arg for arg in args if isinstance(arg, CountTensor)
            # ),
        )

    @staticmethod
    def reduce(equation: str, *args: "CountTensor") -> "CountTensor":
        assert "..." not in equation, "Ellipsis not yet supported"
        assert "," not in equation, "Commas invalid for reduce"
        lhs, rhs = equation.split("->")
        contracted_idxs = set(lhs) - set(rhs)
        lhs = lhs.split(",")
        assert len(lhs) == len(args), f"Expected {len(lhs)} arguments, got {len(args)}"
        shape_map = {}
        for arg, part in zip(args, lhs):
            assert len(part) == len(
                arg.shape
            ), f"Expected {len(part)} indices, got {len(arg.shape)}"
            for idx, size in zip(part, arg.shape):
                shape_map[idx] = torch.broadcast_shapes(
                    (size,), (shape_map.get(idx, size),)
                )[0]
        num_output_elements = int(np.prod([shape_map[idx] for idx in rhs]))
        # we multiply all elements with len(lhs) - 1 multiplications * prod of contracted indices, then sum over all contracted indices with prod contracted indices - 1 additions
        num_contracted_indices = int(
            np.prod([shape_map[idx] for idx in contracted_idxs])
        )
        # num_output_elements * ((len(lhs) - 1) * num_contracted_indices + num_contracted_indices - 1)
        # = num_output_elements * (len(lhs) * num_contracted_indices - 1)
        flops = num_output_elements * (len(lhs) * num_contracted_indices - 1)
        count = InstructionCount(flop=flops)
        add_to_count(count)
        return CountTensor(
            shape=tuple(shape_map[idx] for idx in rhs),
            count=count,
            is_bool=False,
            # parents=CountTensor._parents_of_tuple(
            #     arg for arg in args if isinstance(arg, CountTensor)
            # ),
        )

    @staticmethod
    def fancy_einsum(equation: str, *args: "CountTensor") -> "CountTensor":
        return CountTensor.einsum(fancy_einsum.convert_equation(equation), *args)

    @staticmethod
    def fancy_reduce(equation: str, arg: "CountTensor") -> "CountTensor":
        return CountTensor.reduce(fancy_einsum.convert_equation(equation), arg)

    def where(
        self: Union["CountTensor", torch.Tensor, np.ndarray],
        x: Union["CountTensor", torch.Tensor, np.ndarray],
        y: Union["CountTensor", torch.Tensor, np.ndarray],
    ) -> "CountTensor":
        cond = self
        cond, x, y = (
            CountTensor.from_numpy(cond),
            CountTensor.from_numpy(x),
            CountTensor.from_numpy(y),
        )
        shape = torch.broadcast_shapes(cond.shape, x.shape, y.shape)
        count = InstructionCount(flop=int(np.prod(shape)))
        add_to_count(count)
        return CountTensor(
            shape=shape,
            count=count,
            is_bool=x.is_bool and y.is_bool,
            # parents=CountTensor._parents_of_tuple(cond, x, y),
        )

    @staticmethod
    def zeros(*sizes: Union[int, Sequence[int]]) -> "CountTensor":
        if len(sizes) == 0:
            return CountTensor(shape=())
        if len(sizes) == 1 and isinstance(sizes[0], Sequence):
            return CountTensor.zeros(*sizes[0])
        return CountTensor(shape=tuple(sizes))

    ones = zeros

    @staticmethod
    def zeros_like(other: "CountTensor") -> "CountTensor":
        return CountTensor.zeros(other.shape).to(other)

    ones_like = zeros_like

    @staticmethod
    def eye(n: int, m: Optional[int] = None, *, dtype=None):
        if m is None:
            m = n
        count = InstructionCount(flop=m * n)
        add_to_count(count)
        return CountTensor(
            shape=(n, m), count=count, is_bool=dtype in (bool, torch.bool)
        )

    @staticmethod
    def stack(
        tensors: Sequence[Union["CountTensor", torch.Tensor]], dim: int = 0
    ) -> "CountTensor":
        tensors = [CountTensor.from_numpy(t) for t in tensors]
        shape = list(torch.broadcast_shapes(*[t.shape for t in tensors]))
        if dim < 0:
            dim = dim + len(shape) + 1
        shape.insert(dim, len(tensors))
        return CountTensor(
            shape=tuple(shape),
            count=InstructionCount(),
            is_bool=all(t.is_bool for t in tensors),
            # parents=CountTensor._parents_of_tuple(tensors),
        )

    @staticmethod
    def cat(
        tensors: Sequence[Union["CountTensor", torch.Tensor]], dim: int = 0
    ) -> "CountTensor":
        parents = tuple(
            CountTensor.from_numpy(t) for t in tensors if tuple(t.shape) != (0,)
        )
        shapes = [list(t.shape) for t in parents]
        new_index = [sh.pop(dim) for sh in shapes]
        shape = list(torch.broadcast_shapes(*shapes))
        if dim < 0:
            dim = dim + len(shape) + 1
        shape.insert(dim, sum(new_index))
        return CountTensor(
            shape=tuple(shape),
            is_bool=all(t.is_bool for t in parents),
            count=InstructionCount(),
            # parents=CountTensor._parents_of_tuple(parents),
        )

    @staticmethod
    def accumulate_indices(
        indices: TensorOrCountTensorIndexType,
    ) -> Tuple[list["CountTensor"], TensorIndexType]:
        if isinstance(indices, CountTensor):
            return [indices], torch_ones(
                indices.shape, dtype=torch.long if not indices.is_bool else torch.bool
            )
        elif isinstance(indices, torch.Tensor):
            return [], indices
        elif isinstance(indices, tuple):
            tensors, new_indices = zip(
                *[CountTensor.accumulate_indices(idx) for idx in indices]
            )
            return [idx for idxs in tensors for idx in idxs], tuple(new_indices)
        elif hasattr(indices, "__iter__"):
            tensors, new_indices = zip(
                *[CountTensor.accumulate_indices(idx) for idx in indices]
            )
            return [idx for idxs in tensors for idx in idxs], list(new_indices)
        else:  # any(isinstance(indices, ty) for ty in [int, slice, bool, type(None)]) or hasattr(indices, "__index__"):
            return [], indices

    def parents_and_shape_of_slice(
        self, indices: TensorOrCountTensorIndexType, sanity_check: Optional[bool] = None
    ) -> Tuple[Sequence["CountTensor"], Sequence[int]]:
        if sanity_check is None:
            sanity_check = default_sanity_check
        # cheap hack
        # if isinstance(indices, slice):
        #     start, stop, stride = indices.indices(self.shape[0])
        #     shape = [int(np.ceil((stop - start) / stride))]
        #     return CountTensor(shape=shape, count=self.count)
        # if isinstance(indices, int):
        #     return CountTensor(shape=[], count=self.count)
        # if isinstance(indices, tuple):
        #     t_shapes = [idx.shape[:-1] for idx in indices if isinstance(idx, CountTensor)]
        #     init_shape = torch.broadcast_shapes(*t_shapes)
        orig_indices = indices
        if not isinstance(indices, tuple):
            indices = (indices,)
        assert not any(
            isinstance(idx, bool) for idx in indices
        ), f"Why are you doing this sort of indexing? ({indices}) ({orig_indices})"
        indices = tuple(
            CountTensor_of_nested_sequence(idx) if hasattr(idx, "__iter__") else idx
            for idx in indices
        )
        # print(indices)
        if (
            len(indices) == 1
            and isinstance(indices[0], CountTensor)
            and indices[0].is_bool
        ):
            init_shape = list(indices[0].shape)  # worst case if all true
            mid_shape = []
            post_shape = list(self.shape[len(init_shape) :])
            idx_parents = (indices[0],)
        else:
            if len(indices) > 1:
                assert not any(
                    idx.is_bool for idx in indices if isinstance(idx, CountTensor)
                ), f"Why are you doing this sort of indexing? ({indices}) ({orig_indices})"
            init_shapes = []
            mid_shape = []
            # print(self)
            post_shape = list(self.shape)
            idx_parents = []
            for remaining, idx in reversed(list(enumerate(reversed(list(indices))))):
                if idx is None:
                    mid_shape.append(1)
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(post_shape.pop(0))
                    mid_shape.append(int(np.ceil((stop - start) / stride)))
                    # print(locals())
                elif isinstance(idx, CountTensor):
                    init_shapes.append(idx.shape[:-1])
                    mid_shape.append(idx.shape[-1])
                    post_shape.pop(0)
                    idx_parents.append(idx)
                elif isinstance(idx, EllipsisType):
                    # print(f"idx={idx}, remaining={remaining}, init_shapes={init_shapes}, post_shape={post_shape}, mid_shape={mid_shape}, indices={indices}")
                    if remaining == 0:
                        mid_shape.extend(post_shape)
                        post_shape = []
                    else:
                        mid_shape.extend(post_shape[:-remaining])
                        post_shape = post_shape[-remaining:]
                    # print(f"after: idx={idx}, remaining={remaining}, init_shapes={init_shapes}, post_shape={post_shape}, mid_shape={mid_shape}, indices={indices}")
                else:
                    assert not hasattr(
                        idx, "__iter__"
                    ), f"Why is {idx} ({remaining}) iterable in {indices} ({orig_indices})"
                    post_shape.pop(0)
            init_shape = list(torch.broadcast_shapes(*init_shapes))
            idx_parents = tuple(idx_parents)
        shape = tuple(init_shape + mid_shape + post_shape)
        # print(f"shape={shape}")
        if sanity_check:
            _idx_parents, tindices = CountTensor.accumulate_indices(orig_indices)
            assert all(
                isinstance(idx, CountTensor) for idx in _idx_parents
            ), f"Expected CountTensor, got {_idx_parents} ({[type(idx) for idx in _idx_parents]})"
            assert (
                shape == torch_empty(self.shape)[tindices].shape
            ), f"{shape} != {torch_empty(self.shape)[tindices].shape} == torch.zeros({self.shape})[{tindices}].shape"
        return idx_parents, shape

    def __getitem__(
        self, indices: CountTensorIndexType, sanity_check: Optional[bool] = None
    ) -> "CountTensor":
        idx_parents, shape = self.parents_and_shape_of_slice(
            indices, sanity_check=sanity_check
        )
        return CountTensor(
            shape=shape,
            count=InstructionCount(),
            # parents=CountTensor._parents_of_tuple([self, *idx_parents]),
        )

    def __setitem__(
        self,
        indices: CountTensorIndexType,
        other: Union[float, int, "CountTensor", torch.Tensor],
        sanity_check: Optional[bool] = None,
    ) -> "CountTensor":
        idx_parents, shape = self.parents_and_shape_of_slice(
            indices, sanity_check=sanity_check
        )
        additional_count = InstructionCount(flop=int(np.prod(shape)))
        add_to_count(additional_count)
        self.count += additional_count
        self.__post_init__()
        # phantom_parent = CountTensor(
        #     shape=(),
        #     parents=CountTensor._parents_of_tuple(
        #         [*idx_parents, CountTensor.from_numpy(other)]
        #     ),
        # )
        # self.parents[id(phantom_parent)] = phantom_parent
        return self

    def gather(
        self, dim: int, index: Union["CountTensor", torch.Tensor]
    ) -> "CountTensor":
        index = CountTensor.from_numpy(index)
        return CountTensor(
            shape=index.shape,
            count=InstructionCount(),
            # parents=CountTensor._parents_of_tuple(self, index),
        )

    def unsqueeze(self, dim: int) -> "CountTensor":
        shape = list(self.shape)
        if dim < 0:
            dim = dim + len(shape) + 1
        shape.insert(dim, 1)
        return CountTensor(
            shape=shape,
            count=InstructionCount(),
            # parents=CountTensor._parents_of_tuple((self,)),
        )

    def allclose(
        self,
        other: Union[int, float, "CountTensor", np.ndarray, torch.Tensor],
        *args,
        **kwargs,
    ) -> "CountTensor":
        return ((self - other) == 0).all()

    @property
    def device(self):
        return torch.device("cpu")

    def cpu(self) -> "CountTensor":
        return self

    def item(self) -> "CountTensor":
        return self

    def to(self, *args, **kwargs) -> "CountTensor":
        if "dtype" in kwargs:
            if kwargs["dtype"] == bool or kwargs["dtype"] == torch.bool:
                return self.bool()
            return self
        if isinstance(args[0], type) or isinstance(args[0], torch.dtype):
            if args[0] == bool or args[0] == torch.bool:
                return self.bool()
            return self
        return self

    def detach(self) -> "CountTensor":
        return self

    def requires_grad_(self, *args, **kwargs) -> "CountTensor":
        return self

    def size(self, dim: Optional[int] = None) -> Union[int, Sequence[int]]:
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def dtype(self):
        if self.is_bool:
            return torch.bool
        return torch.float

    def sort(self, dim: int = -1, *args, **kwargs) -> count_values_indices:
        idxs_to_sort = list(self.shape)
        idxs_to_sort.pop(dim)
        arrays_to_sort = int(np.prod(idxs_to_sort))
        n = self.shape[dim]
        match mode:
            case "verify":
                flops_to_sort = n - 1
            case "search":
                raise NotImplementedError("Search mode not yet implemented for sort")
        count = InstructionCount(flop=arrays_to_sort * flops_to_sort)
        add_to_count(count)
        result = CountTensor(
            shape=self.shape,
            count=count,
            # parents=CountTensor._parents_of_tuple((self,)),
        )
        return count_values_indices(result, result)

    @staticmethod
    def linalg_svd(
        A: "CountTensor", full_matrices: bool = True
    ) -> Tuple["CountTensor", "CountTensor", "CountTensor"]:
        init_shape, (n, m) = tuple(A.shape[:-2]), A.shape[-2:]
        min_nm = np.min([n, m])
        if full_matrices:
            Ushape, Sshape, Vhshape = (n, n), (min_nm,), (m, m)
        else:
            Ushape, Sshape, Vhshape = (n, min_nm), (min_nm,), (min_nm, m)
        U = CountTensor(shape=init_shape + Ushape)
        S = CountTensor(shape=init_shape + Sshape)
        Vh = CountTensor(shape=init_shape + Vhshape)
        match mode:
            case "verify":
                Sdiff, Udiff, Vhdiff = compute_verify_svd_close_matrices(A, U, S, Vh)
                U.count = ((Udiff[1] - Udiff[0]).abs() == 0).count
                S.count = ((Sdiff[1] - Sdiff[0]).abs() == 0).count
                Vh.count = ((Vhdiff[1] - Vhdiff[0]).abs() == 0).count
            case "search":
                raise NotImplementedError("Search mode not yet implemented for svd")
        return U, S, Vh

    def svd(
        self, some: bool = True, compute_uv: bool = True
    ) -> Tuple["CountTensor", "CountTensor", "CountTensor"]:
        U, S, Vh = CountTensor.linalg_svd(self, full_matrices=not some)
        return U, S, Vh.mT  # TODO: use .mH when we track real vs complex

    def matrix_norm(
        self,
        ord: Union[Literal["fro", "nuc", 1, -1, 2, -2], float],
        dim: Tuple[int, int] = (-2, -1),
        keepdim: bool = False,
    ) -> "CountTensor":
        dim = tuple(i % len(self.shape) for i in dim)
        dim = (int(np.min(dim)), int(np.max(dim)))
        assert dim[0] != dim[1]
        match ord:
            case "fro":
                return (
                    self.pow(2)
                    .sum(dim=dim[1], keepdim=keepdim)
                    .sum(dim=dim[0], keepdim=keepdim)
                    .sqrt()
                )
            case "nuc" | 2 | -2:
                if dim != (-2, -1):
                    raise NotImplementedError(
                        f"General dim ({dim}) not yet implemented"
                    )
                _, S, _ = self.svd()
                match ord:
                    case "nuc":
                        S = S.sum(dim=-1)
                    case 2:
                        S = S.amax(dim=-1)
                    case -2:
                        S = S.amin(dim=-1)
                if keepdim:
                    return S.unsqueeze(dim[0]).unsqueeze(dim[1])
                return S
            case _:
                raise NotImplementedError(ord)

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.shape),
                self.count,
                self.is_bool,
                # tuple(self.parents.values()),
            )
        )


class CountTensorBackend(
    fancy_einsum.AbstractBackend, einops._backends.AbstractBackend
):
    framework_name = "CountTensor"

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, CountTensor)

    def einsum(self, equation, *operands):
        return CountTensor.einsum(equation, *operands)

    def from_numpy(self, x):
        return CountTensor.from_numpy(x)

    # def to_numpy(self, x):
    # def create_symbol(self, shape):
    # def eval_symbol(self, symbol, input_dict):
    # def arange(self, start, stop):
    # def stack_on_zeroth_dimension(self, tensors: list):
    def add_axis(self, x, new_position):
        return x.unsqueeze(new_position)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return CountTensor.cat(tensors, dim=axis)

    def is_float_type(self, x):
        return True

    def reduce(self, x, operation, reduced_axes):
        if operation == "min":
            return x.amin(dim=reduced_axes)
        elif operation == "max":
            return x.amax(dim=reduced_axes)
        elif operation == "sum":
            return x.sum(dim=reduced_axes)
        elif operation == "mean":
            return x.mean(dim=reduced_axes)
        elif operation in ("any", "all", "prod"):
            # pytorch supports reducing only one operation at a time
            for i in list(sorted(reduced_axes))[::-1]:
                x = getattr(x, operation)(dim=i)
            return x
        else:
            raise NotImplementedError("Unknown reduction ", operation)

    # def layers(self):


fancy_einsum._backends[CountTensorBackend.framework_name] = CountTensorBackend()
einops._backends._loaded_backends[CountTensorBackend.framework_name] = (
    CountTensorBackend()
)


class DefaultCountTensorWrapper:
    def __init__(
        self, mod, name: str, count_name: Optional[str] = None, static: bool = False
    ):
        self.mod = mod
        self.name = name
        self.count_name = count_name or name
        self.static = static
        self.func = getattr(mod, name)
        if isinstance(self.func, DefaultCountTensorWrapper):
            self.func = self.func.unwrap()

    def unwrap(self):
        setattr(self.mod, self.name, self.func)
        if isinstance(self.func, DefaultCountTensorWrapper):
            return self.func.unwrap()
        return self.func

    def __call__(self, arg, *args, **kwargs):
        # print(f"dispatching {self.name} to {type(arg)}")
        # if isinstance(arg, (torch.Tensor, CountTensor)) or (
        #     isinstance(arg, (tuple, list))
        #     and any(isinstance(a, (torch.Tensor, CountTensor)) for a in arg)
        # ):
        if self.static:
            return getattr(CountTensor, self.count_name)(arg, *args, **kwargs)
        else:
            return getattr(CountTensor.from_numpy(arg), self.count_name)(
                *args, **kwargs
            )
        # if hasattr(arg, self.name):
        #     return getattr(arg, self.name)(arg, *args, **kwargs)
        # return self.func(arg, *args, **kwargs)


class PatchTorch:
    _torch_is_static = {
        "where": True,
        "isnan": False,
        "allclose": False,
        "triu": False,
        "tril": False,
        "zeros": True,
        "ones": True,
        "zeros_like": True,
        "ones_like": True,
        "eye": True,
        "stack": True,
        "cat": True,
        "svd": False,
        "matmul": False,
    }
    _torch_count_name = {"matmul": "__matmul__"}
    _torch_linalg_is_static = {
        "matrix_norm": False,
        "svd": True,
    }
    _torch_linalg_count_name = {"svd": "linalg_svd"}

    def __init__(self, **kwargs: bool):
        self.torch_patches = tuple(
            name for name in PatchTorch._torch_is_static if kwargs.get(name, True)
        )
        self.torch_linalg_patches = tuple(
            name
            for name in PatchTorch._torch_linalg_is_static
            if kwargs.get(f"linalg_{name}", True)
        )

    def __enter__(self):
        for name in self.torch_patches:
            setattr(
                torch,
                name,
                DefaultCountTensorWrapper(
                    torch,
                    name,
                    count_name=PatchTorch._torch_count_name.get(name),
                    static=PatchTorch._torch_is_static[name],
                ),
            )
        for name in self.torch_linalg_patches:
            setattr(
                torch.linalg,
                name,
                DefaultCountTensorWrapper(
                    torch.linalg,
                    name,
                    count_name=PatchTorch._torch_linalg_count_name.get(name),
                    static=PatchTorch._torch_linalg_is_static[name],
                ),
            )

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.torch_patches:
            getattr(torch, name).unwrap()
        for name in self.torch_linalg_patches:
            getattr(torch.linalg, name).unwrap()


class CountHookedTransformer(HookedTransformer):
    def __init__(self, model: HookedTransformer):
        super().__init__(model.cfg)

        for mod in self.modules():
            # print(f"Patching {mod}")
            for name, value in mod.named_parameters():
                if "." not in name:
                    mod.__dict__[name] = mod._parameters[name] = CountTensor.from_numpy(
                        value
                    )

    def forward(self, *args, **kwargs):
        with PatchTorch(triu=False, tril=False):
            return super().forward(*args, **kwargs)


# ## %%
# model = HookedTransformer(
#     HookedTransformerConfig(
#         n_layers=1, d_vocab=5, d_model=6, n_ctx=6, d_head=2, attn_only=True
#     )
# )
# cmodel = CountHookedTransformer(model)
# # %%
# getattr(cmodel.embed,"W_E")
# # %%
# # CountEmbed.count_forward(model.embed, CountTensor([1, 6]))
# # %%
# for i in range(10):
#     print(cmodel(torch.zeros((i, 6))).full_count().flop)

# %%
