"""Some utilities for counting floating point operations"""

# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Sequence,
    Literal,
    Optional,
    NamedTuple,
    SupportsIndex,
    Union,
    Tuple,
    Collection,
    Iterator,
    Any,
    Protocol,
    TypeVar,
)
from types import EllipsisType
import numpy as np
import torch
from torch import empty as torch_zeros  # for stability under hot patching
from transformer_lens import HookedTransformer
import fancy_einsum
import einops
import einops._backends


@dataclass
class InstructionCount:
    flop: int = 0
    int_op: int = 0
    branch: int = 0

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


@dataclass
class CountTensor:
    shape: Sequence[int]
    count: InstructionCount = InstructionCount()
    parents: Collection["CountTensor"] = tuple()

    @staticmethod
    def from_numpy(x: Union[np.ndarray, torch.Tensor, "CountTensor"]) -> "CountTensor":
        if isinstance(x, CountTensor):
            return x
        elif hasattr(x, "shape"):
            return CountTensor(shape=x.shape)
        else:
            return CountTensor(shape=torch.tensor(x).shape)

    def _full_count(
        self,
        count: InstructionCount = InstructionCount(),
        seen: Collection["CountTensor"] = tuple(),
    ) -> Tuple[InstructionCount, Collection["CountTensor"]]:
        seen = tuple(seen) + (self,)
        for parent in self.parents:
            if any(parent is s for s in seen):
                continue
            count, seen = parent._full_count(count, seen)
        return count + self.count, seen

    def full_count(self) -> InstructionCount:
        return self._full_count()[0]

    def unary(self) -> "CountTensor":
        return CountTensor(
            shape=self.shape,
            count=InstructionCount(flop=int(np.prod(self.shape))),
            parents=(self,),
        )

    def binary_only_scalar(self, other: Union[int, float]) -> "CountTensor":
        return self.unary()

    def _binary(self, other: "CountTensor") -> "CountTensor":
        shape = torch.broadcast_shapes(self.shape, other.shape)
        assert isinstance(
            other, CountTensor
        ), f"Expected CountTensor, got {type(other)}"
        return CountTensor(
            shape=shape,
            count=InstructionCount(flop=int(np.prod(shape))),
            parents=(self, other),
        )

    def binary(
        self, other: Union[int, float, "CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        if isinstance(other, CountTensor):
            return self._binary(other)
        elif hasattr(other, "shape"):
            return self._binary(CountTensor.from_numpy(other))
        return self.unary()

    def fold_reduce(
        self,
        dim: Optional[int] = None,
        axis: Optional[int] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        shape = list(self.shape)
        if dim is None:
            return CountTensor(
                shape=[],
                count=InstructionCount(flop=int(np.prod(shape)) - 1),
                parents=(self,),
            )
        shape_without_dim = list(shape)
        shape_without_dim.pop(dim)
        return CountTensor(
            shape=shape_without_dim if not keepdim else shape,
            count=InstructionCount(
                flop=int(np.prod(shape_without_dim)) * (shape[dim] - 1)
            ),
            parents=(self,),
        )

    def fold_reduce_values_indices(
        self,
        dim: Optional[int] = None,
        axis: Optional[int] = None,
        keepdim: bool = False,
    ) -> Union["CountTensor", count_values_indices]:
        result = self.fold_reduce(dim=dim, axis=axis, keepdim=keepdim)
        if dim is None and axis is None:
            return result
        return count_values_indices(result, result)

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
    __or__ = binary
    __ror__ = binary
    __and__ = binary
    __rand__ = binary
    __xor__ = binary
    __rxor__ = binary
    __eq__ = binary
    __ne__ = binary
    __lt__ = binary
    __le__ = binary
    __gt__ = binary
    __ge__ = binary
    __req__ = binary
    __rne__ = binary
    __rlt__ = binary
    __rle__ = binary
    __rgt__ = binary
    __rge__ = binary

    __abs__ = unary
    __neg__ = unary
    __pos__ = unary
    sqrt = unary
    exp = unary
    log = unary
    log1p = unary
    isnan = unary
    pow = binary_only_scalar
    float = unary
    long = unary
    bool = unary

    def tril(self, diagonal: int = 0) -> "CountTensor":
        return self.unary()

    triu = tril

    def reshape(self, new_shape: Sequence[int]) -> "CountTensor":
        return CountTensor(shape=new_shape, parents=(self,))

    def __matmul__(
        self, other: Union["CountTensor", np.ndarray, torch.Tensor]
    ) -> "CountTensor":
        assert hasattr(
            other, "shape"
        ), f"Expected CountTensor, ndarray, or Tensor, got {type(other)} ({other})"
        if not isinstance(other, CountTensor):
            other = CountTensor.from_numpy(other)
        x_shape = torch.broadcast_shapes(self.shape, other.shape[:-1])
        # at each index, we do x_shape[-1] multiplications and x_shape[-1] - 1 additions
        out_shape = (*x_shape[:-1], other.shape[-1])
        return CountTensor(
            shape=out_shape,
            count=InstructionCount(
                flop=int(np.prod(out_shape)) * (x_shape[-1] * 2 - 1)
            ),
            parents=(self, other),
        )

    sum = fold_reduce
    argmax = fold_reduce
    argmin = fold_reduce
    max = fold_reduce_values_indices
    min = fold_reduce_values_indices
    prod = fold_reduce

    def mean(
        self,
        dim: Optional[int] = None,
        axis: Optional[int] = None,
        keepdim: bool = False,
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        total = int(np.prod(self.shape)) if dim is None else self.shape[dim]
        return self.sum(dim=dim, keepdim=keepdim) / total

    def softmax(
        self, dim: Optional[int] = None, axis: Optional[int] = None
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        adjusted = self - self.max(dim=dim, keepdim=True)
        adjusted_exp = adjusted.exp()
        return adjusted_exp / adjusted_exp.sum(dim=dim, keepdim=True)

    def log_softmax(
        self, dim: Optional[int] = None, axis: Optional[int] = None
    ) -> "CountTensor":
        if axis is not None:
            assert dim is None, "Cannot specify both dim and axis"
            dim = axis
        adjusted = self - self.max(dim=dim, keepdim=True)
        adjusted_log_sum_exp = adjusted.exp().sum(dim=dim, keepdim=True).log()
        return adjusted - adjusted_log_sum_exp

    def flip(self, *args, **kwargs) -> "CountTensor":
        return CountTensor(
            shape=torch.empty(self.shape).flip(*args, **kwargs).shape,
            parents=(self,),
        )

    def transpose(self, *args, **kwargs) -> "CountTensor":
        return CountTensor(
            shape=torch.empty(self.shape).transpose(*args, **kwargs).shape,
            parents=(self,),
        )

    @property
    def T(self) -> "CountTensor":
        return self.transpose(-2, -1)

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
        return CountTensor(
            shape=tuple(shape_map[idx] for idx in rhs),
            count=InstructionCount(flop=flops),
            parents=tuple(arg for arg in args if isinstance(arg, CountTensor)),
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
        return CountTensor(
            shape=tuple(shape_map[idx] for idx in rhs),
            count=InstructionCount(flop=flops),
            parents=tuple(arg for arg in args if isinstance(arg, CountTensor)),
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
        return CountTensor(
            shape=shape,
            count=InstructionCount(flop=int(np.prod(shape))),
            parents=(cond, x, y),
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
        return CountTensor.zeros(other.shape)

    ones_like = zeros_like

    @staticmethod
    def stack(
        tensors: Sequence[Union["CountTensor", torch.Tensor]], dim: int = 0
    ) -> "CountTensor":
        shape = list(torch.broadcast_shapes(*[t.shape for t in tensors]))
        shape.insert(dim, len(tensors))
        return CountTensor(
            shape=tuple(shape),
            count=InstructionCount(),
            parents=tuple(CountTensor.from_numpy(t) for t in tensors),
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
        shape.insert(dim, sum(new_index))
        return CountTensor(
            shape=tuple(shape),
            count=InstructionCount(),
            parents=parents,
        )

    @staticmethod
    def accumulate_indices(
        indices: TensorOrCountTensorIndexType,
    ) -> Tuple[list["CountTensor"], TensorIndexType]:
        if isinstance(indices, CountTensor):
            return [indices], torch_zeros(indices.shape, dtype=torch.long)
        elif isinstance(indices, torch.Tensor):
            return CountTensor.accumulate_indices(CountTensor.from_numpy(indices))
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

    def __getitem__(self, indices: CountTensorIndexType) -> "CountTensor":
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
        idx_parents, tindices = CountTensor.accumulate_indices(indices)
        assert all(
            isinstance(idx, CountTensor) for idx in idx_parents
        ), f"Expected CountTensor, got {idx_parents} ({[type(idx) for idx in idx_parents]})"
        return CountTensor(
            shape=torch_zeros(self.shape)[tindices].shape,
            count=InstructionCount(),
            parents=(self, *idx_parents),
        )

    def __setitem__(
        self,
        indices: CountTensorIndexType,
        other: Union[float, int, "CountTensor", torch.Tensor],
    ) -> "CountTensor":
        # cheap hack
        idx_parents, tindices = CountTensor.accumulate_indices(indices)
        assert all(
            isinstance(idx, CountTensor) for idx in idx_parents
        ), f"Expected CountTensor, got {idx_parents} ({[type(idx) for idx in idx_parents]})"
        self.count += InstructionCount(
            flop=int(np.prod(torch_zeros(self.shape)[tindices].shape))
        )
        self.parents = (self, *idx_parents, CountTensor.from_numpy(other))
        return self

    def gather(
        self, dim: int, index: Union["CountTensor", torch.Tensor]
    ) -> "CountTensor":
        index = CountTensor.from_numpy(index)
        return CountTensor(
            shape=index.shape,
            count=InstructionCount(),
            parents=(self, index),
        )

    def unsqueeze(self, dim: int) -> "CountTensor":
        shape = list(self.shape)
        shape.insert(dim, 1)
        return CountTensor(shape=shape, count=InstructionCount(), parents=(self,))

    @property
    def device(self):
        return torch.device("cpu")

    def to(self, *args, **kwargs) -> "CountTensor":
        return self

    def detach(self) -> "CountTensor":
        return self

    def requires_grad_(self, *args, **kwargs) -> "CountTensor":
        return self

    def size(self, dim: Optional[int] = None) -> Union[int, Sequence[int]]:
        if dim is None:
            return self.shape
        return self.shape[dim]

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
        result = CountTensor(
            shape=self.shape,
            count=InstructionCount(flop=arrays_to_sort * flops_to_sort),
            parents=(self,),
        )
        return count_values_indices(result, result)

    def __hash__(self) -> int:
        return hash((tuple(self.shape), self.count, tuple(self.parents)))


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
    # def add_axis(self, x, new_position):
    # def tile(self, x, repeats):
    # def concat(self, tensors, axis: int):
    def is_float_type(self, x):
        return True

    # def layers(self):


fancy_einsum._backends[CountTensorBackend.framework_name] = CountTensorBackend()
einops._backends._loaded_backends[CountTensorBackend.framework_name] = (
    CountTensorBackend()
)


class DefaultCountTensorWrapper:
    def __init__(self, mod, name, static: bool = False):
        self.mod = mod
        self.name = name
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
            return getattr(CountTensor, self.name)(arg, *args, **kwargs)
        else:
            return getattr(CountTensor.from_numpy(arg), self.name)(*args, **kwargs)
        # if hasattr(arg, self.name):
        #     return getattr(arg, self.name)(arg, *args, **kwargs)
        # return self.func(arg, *args, **kwargs)


class PatchTorch:
    _torch_is_static = {
        "where": True,
        "isnan": False,
        "triu": False,
        "tril": False,
        "zeros": True,
        "ones": True,
        "zeros_like": True,
        "ones_like": True,
        "stack": True,
        "cat": True,
    }

    def __init__(self, **kwargs: bool):
        self.torch_patches = tuple(
            name for name in PatchTorch._torch_is_static if kwargs.get(name, True)
        )

    def __enter__(self):
        for name in self.torch_patches:
            setattr(
                torch,
                name,
                DefaultCountTensorWrapper(
                    torch, name, static=PatchTorch._torch_is_static[name]
                ),
            )

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.torch_patches:
            getattr(torch, name).unwrap()


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
