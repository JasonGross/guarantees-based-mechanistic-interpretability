"""Some utilities for counting floating point operations"""

# %%
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import (
    Sequence,
    Literal,
    Optional,
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
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.components import (
    Embed,
    PosEmbed,
    Unembed,
    LayerNorm,
    Attention,
    LayerNormPre,
)
import fancy_einsum
import einops
import transformer_lens.utils


@dataclass
class InstructionCount:
    flop: int = 0
    int_op: int = 0
    branch: int = 0

    def __add__(self, other: "InstructionCount") -> "InstructionCount":
        return InstructionCount(
            flop=self.flop + other.flop,
            int_op=self.int_op + other.int_op,
            branch=self.branch + other.branch,
        )

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

    def __rmul__(self, other: int) -> "InstructionCount":
        return self.__mul__(other)

    def __str__(self) -> str:
        return f"InstructionCount(flop={self.flop}, int_op={self.int_op}, branch={self.branch})"

    def __repr__(self) -> str:
        return f"InstructionCount(flop={self.flop!r}, int_op={self.int_op!r}, branch={self.branch!r})"


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


@dataclass
class CountTensor:
    shape: Sequence[int]
    count: InstructionCount = InstructionCount()
    parents: Collection["CountTensor"] = tuple()

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

    def binary(self, other: Union[int, float, "CountTensor"]) -> "CountTensor":
        if isinstance(other, CountTensor):
            return self._binary(other)
        return self.unary()

    def fold_reduce(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> "CountTensor":
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

    __add__ = binary
    __radd__ = binary
    __sub__ = binary
    __rsub__ = binary
    __mul__ = binary
    __rmul__ = binary
    __div__ = binary
    __rdiv__ = binary

    sqrt = unary
    exp = unary
    log = unary
    log1p = unary
    isnan = unary
    pow = binary_only_scalar

    def __matmul__(self, other: "CountTensor") -> "CountTensor":
        assert isinstance(
            other, CountTensor
        ), f"Expected CountTensor, got {type(other)}"
        x_shape = torch.broadcast_shapes(self.shape, other.shape[:-1])
        # at each index, we do y_shape[-1] multiplications and y_shape[-1] - 1 additions
        return CountTensor(
            shape=(*x_shape, other.shape[-1]),
            count=InstructionCount(
                flop=int(np.prod(x_shape)) * (other.shape[-1] * 2 - 1)
            ),
            parents=(self, other),
        )

    sum = fold_reduce
    max = fold_reduce
    min = fold_reduce

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "CountTensor":
        total = int(np.prod(self.shape)) if dim is None else self.shape[dim]
        return self.sum(dim=dim, keepdim=keepdim) / total

    def softmax(self, dim: Optional[int] = None) -> "CountTensor":
        adjusted = self - self.max(dim=dim, keepdim=True)
        adjusted_exp = adjusted.exp()
        return adjusted_exp / adjusted_exp.sum(dim=dim, keepdim=True)

    def transpose(self, *args, **kwargs) -> "CountTensor":
        return CountTensor(
            shape=torch.empty(self.shape).transpose(*args, **kwargs).shape,
            parents=(self,),
        )

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
            for idx, size in enumerate(zip(part, arg.shape)):
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
            parents=args,
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
            for idx, size in enumerate(zip(part, arg.shape)):
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
            parents=args,
        )

    @staticmethod
    def fancy_einsum(equation: str, *args: "CountTensor") -> "CountTensor":
        return CountTensor.einsum(fancy_einsum.convert_equation(equation), *args)

    @staticmethod
    def fancy_reduce(equation: str, arg: "CountTensor") -> "CountTensor":
        return CountTensor.reduce(fancy_einsum.convert_equation(equation), arg)

    @staticmethod
    def where(cond: "CountTensor", x: "CountTensor", y: "CountTensor") -> "CountTensor":
        shape = torch.broadcast_shapes(cond.shape, x.shape, y.shape)
        return CountTensor(
            shape=shape,
            count=InstructionCount(flop=int(np.prod(shape))),
            parents=(cond, x, y),
        )

    @staticmethod
    def zeros_like(other: "CountTensor") -> "CountTensor":
        return CountTensor(shape=other.shape, count=InstructionCount())

    @staticmethod
    def ones_like(other: "CountTensor") -> "CountTensor":
        return CountTensor(shape=other.shape, count=InstructionCount())

    @staticmethod
    def accumulate_indices(
        indices: CountTensorIndexType,
    ) -> Tuple[list["CountTensor"], TensorIndexType]:
        if isinstance(indices, CountTensor):
            return [indices], torch.zeros(indices.shape, dtype=torch.long)
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
            shape=torch.zeros(self.shape)[tindices].shape,
            count=InstructionCount(),
            parents=(self, *idx_parents),
        )


class CountLayerNorm:
    @staticmethod
    def count_forward(mod: LayerNorm, x: CountTensor) -> CountTensor:
        x = x - x.mean(dim=-1, keepdim=True)  # [batch, pos, length]
        scale = (x.pow(2).mean(dim=-1, keepdim=True) + mod.eps).sqrt()
        x = x / scale  # [batch, pos, length]
        return x * CountTensor(mod.w.shape) + CountTensor(mod.b.shape)


class CountEmbed:
    @staticmethod
    def count_forward(mod: Embed, tokens: CountTensor) -> CountTensor:
        if mod.cfg.post_embedding_ln:
            return CountLayerNorm.count_forward(
                mod.ln, CountTensor(mod.W_E.shape)[tokens, :]
            )
        return CountTensor(mod.W_E.shape)[tokens, :]


class CountPosEmbed:
    @staticmethod
    def count_forward(mod: PosEmbed, tokens: CountTensor) -> CountTensor:
        # return
        # count = InstructionCount()
        # tokens_shape = list(tokens_shape)
        # tokens_length = tokens_shape[-1]
        # approx, ignore masking
        # return (*tokens_shape, *mod.W_pos.shape[1:]), InstructionCount()
        return CountTensor(mod.W_pos.shape)[tokens, :]


class CountUnembed:
    @staticmethod
    def count_forward(mod: Unembed, residual: CountTensor) -> CountTensor:
        # return (
        #     einsum(
        #         "batch pos d_model, d_model vocab -> batch pos vocab",
        #         residual,
        #         self.W_U,
        #     )
        #     + self.b_U
        # )
        return (residual @ CountTensor(mod.W_U.shape)) + CountTensor(mod.b_U.shape)


# class TokenTypeEmbed(nn.Module):
# class BertEmbed(nn.Module):
# class BertMLMHead(nn.Module):
class CountLayerNormPre:
    @staticmethod
    def count_forward(mod: LayerNormPre, x: CountTensor) -> CountTensor:
        x = x - x.mean(dim=-1, keepdim=True)  # [batch, pos, length]
        scale = (x.pow(2).mean(-1, keepdim=True) + mod.eps).sqrt()
        return x / scale


# class RMSNormPre(nn.Module):
# class RMSNorm(nn.Module):


class CountAttention:
    @staticmethod
    def count_forward(
        mod: Attention,
        query_input: CountTensor,
        key_input: CountTensor,
        value_input: CountTensor,
        # past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        additive_attention_mask: Optional[CountTensor] = None,
        attention_mask: Optional[CountTensor] = None,
    ) -> CountTensor:
        if mod.cfg.use_split_qkv_input or mod.cfg.use_attn_in:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"
        q = (
            CountTensor.fancy_einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                CountTensor(mod.W_Q.shape),
            )
            + CountTensor(mod.b_Q.shape)
        )  # [batch, pos, head_index, d_head]
        k = (
            CountTensor.fancy_einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                key_input,
                CountTensor(mod.W_K.shape),
            )
            + CountTensor(mod.b_K.shape)
        )  # [batch, pos, head_index, d_head]
        v = (
            CountTensor.fancy_einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                value_input,
                CountTensor(mod.W_V.shape),
            )
            + CountTensor(mod.b_V.shape)
        )  # [batch, pos, head_index, d_head]

        if mod.cfg.positional_embedding_type == "rotary":
            assert False
            # q = mod.hook_rot_q(mod.apply_rotary(q, kv_cache_pos_offset, attention_mask))
            # k = mod.hook_rot_k(
            #     mod.apply_rotary(k, 0, attention_mask)
            # )  # keys are cached so no offset

        attn_scores = (
            CountTensor.fancy_einsum(
                "batch query_pos head_index d_head, \
                    batch key_pos head_index d_head \
                    -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / mod.attn_scale
        )  # [batch, head_index, query_pos, key_pos]

        if mod.cfg.positional_embedding_type == "alibi":
            assert False
            # query_ctx = attn_scores.size(-2)
            # # The key context length is the number of positions in the past - this includes all positions in the cache
            # key_ctx = attn_scores.size(-1)

            # # only recompute when necessary to increase efficiency.
            # if mod.alibi is None or key_ctx > mod.alibi.size(-1):
            #     mod.alibi = Attention.create_alibi_bias(
            #         mod.cfg.n_heads, key_ctx, mod.cfg.device
            #     )

            # attn_scores += mod.alibi[
            #     :, :query_ctx, :key_ctx
            # ]  # [batch, head_index, query_pos, key_pos]

        if mod.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            # ignore for now
            pass
            # attn_scores = mod.apply_causal_mask(
            #     attn_scores, kv_cache_pos_offset, attention_mask
            # )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        pattern = attn_scores.softmax(dim=-1)
        pattern = CountTensor.where(
            pattern.isnan(), CountTensor.zeros_like(pattern), pattern
        )
        z = CountTensor.fancy_einsum(
            "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
            v,
            pattern,
        )  # [batch, pos, head_index, d_head]
        if not mod.cfg.use_attn_result:
            out = (
                (
                    CountTensor.fancy_einsum(
                        "batch pos head_index d_head, \
                            head_index d_head d_model -> \
                            batch pos d_model",
                        z,
                        CountTensor(mod.W_O.shape),
                    )
                )
                + CountTensor(mod.b_O.shape)
            )  # [batch, pos, d_model]
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            result = CountTensor.fancy_einsum(
                "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model",
                z,
                CountTensor(mod.W_O.shape),
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + mod.b_O
            )  # [batch, pos, d_model]
        return out


#     def apply_causal_mask(
#         mod,
#         attn_scores: Float[
#             torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"
#         ],
#         past_kv_pos_offset: int = 0,
#         attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
#     ):
#         # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
#         query_ctx_length = attn_scores.size(-2)
#         # The key context length is the number of positions in the past - this includes all positions in the cache
#         # If not caching, query_ctx_length == key_ctx_length
#         key_ctx_length = attn_scores.size(-1)

#         assert (
#             query_ctx_length + past_kv_pos_offset == key_ctx_length
#         ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."

#         # Index back to front to ensure local attention works
#         final_mask = mod.mask[
#             None, None, -query_ctx_length:, -key_ctx_length:
#         ]  # [1, 1, pos, pos]
#         if attention_mask is not None:
#             # Apply a causal mask to the attention scores considering the padding
#             einsum_str = "batch head pos offset_pos, batch offset_pos -> batch head pos offset_pos"
#             final_mask = einops.einsum(final_mask, attention_mask, einsum_str).bool()

#         return torch.where(final_mask, attn_scores, mod.IGNORE)

#     def calculate_sin_cos_rotary(
#         mod,
#         rotary_dim: int,
#         n_ctx: int,
#         base: int = 10000,
#         dtype: torch.dtype = torch.float32,
#     ) -> Tuple[
#         Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]
#     ]:
#         """
#         Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

#         Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
#         To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
#         """
#         high_precision = torch.float32 if dtype != torch.float64 else torch.float64
#         pos = torch.arange(n_ctx, dtype=high_precision)
#         dim = torch.arange(rotary_dim // 2, dtype=high_precision)

#         # A set of frequencies evenly spaced in log space
#         freq = base ** (dim / (rotary_dim / 2))
#         if mod.cfg.rotary_adjacent_pairs:
#             freq = einops.repeat(freq, "d -> (d 2)")
#         else:
#             freq = einops.repeat(freq, "d -> (2 d)")
#         # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
#         angles = pos[:, None] / freq[None, :]
#         return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

#     def rotate_every_two(
#         mod, x: Float[torch.Tensor, "... rotary_dim"]
#     ) -> Float[torch.Tensor, "... rotary_dim"]:
#         """
#         Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

#         The final axis of x must have even length.

#         GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
#         """
#         rot_x = x.clone()
#         if mod.cfg.rotary_adjacent_pairs:
#             rot_x[..., ::2] = -x[..., 1::2]
#             rot_x[..., 1::2] = x[..., ::2]
#         else:
#             n = x.size(-1) // 2
#             rot_x[..., :n] = -x[..., n:]
#             rot_x[..., n:] = x[..., :n]

#         return rot_x

#     def apply_rotary(
#         mod,
#         x: Float[torch.Tensor, "batch pos head_index d_head"],
#         past_kv_pos_offset=0,
#         attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
#     ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
#         # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
#         x_pos = x.size(1)
#         x_rot = x[..., : mod.cfg.rotary_dim]
#         x_pass = x[..., mod.cfg.rotary_dim :]
#         x_flip = mod.rotate_every_two(x_rot)

#         if attention_mask is None:
#             rotary_cos = mod.rotary_cos[
#                 None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
#             ]
#             rotary_sin = mod.rotary_sin[
#                 None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
#             ]
#             x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
#         else:
#             offset_position_ids = get_offset_position_ids(
#                 past_kv_pos_offset, attention_mask
#             )
#             mask_rotary_cos = mod.rotary_cos[offset_position_ids, None, :]
#             mask_rotary_sin = mod.rotary_sin[offset_position_ids, None, :]
#             x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

#         return torch.cat([x_rotated, x_pass], dim=-1)

#     @staticmethod
#     def create_alibi_slope(
#         n_ctx: int, device: torch.device = None
#     ) -> Float[torch.Tensor, "query key"]:
#         """Create an ALiBi Slope Matrix.

#         Create the slope matrix used in ALiBi, before it is multiplied by the head-specific scalar.

#         See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

#         Examples:

#         >>> Attention.create_alibi_slope(3)
#         tensor([[ 0.,  0.,  0.],
#                 [-1.,  0.,  0.],
#                 [-2., -1.,  0.]])

#         >>> Attention.create_alibi_slope(4)
#         tensor([[ 0.,  0.,  0.,  0.],
#                 [-1.,  0.,  0.,  0.],
#                 [-2., -1.,  0.,  0.],
#                 [-3., -2., -1.,  0.]])

#         Args:
#             n_ctx: The maximum number of tokens in a prompt.

#         Returns:
#             A tensor of shape (n_ctx, n_ctx), where the upper triangle is zero and the lower
#             triangle is decreasing by a constant slope of 1 (towards the bottom left corner).
#         """
#         # set rows as [[0,1,2...]]
#         rows = torch.arange(n_ctx, device=device).unsqueeze(0)

#         # Set cols as [[0],[1],[2]...]
#         cols = torch.arange(n_ctx, device=device).unsqueeze(1)

#         # Use broadcasting to create the desired lower triangular part of the matrix
#         slope_matrix = rows - cols

#         # Use the clamp method to set all positive values (upper right triangle) to
#         return slope_matrix.clamp(max=0).to(torch.float32)

#     @staticmethod
#     def create_alibi_multipliers(
#         n_heads: int, device: torch.device = None
#     ) -> Float[torch.Tensor, "head_idx"]:
#         """Create the ALiBi Scalar Multipliers for each Head.

#         For n heads, the set of multipliers (m) is the geometric sequence that starts at 2^(-8/n), and
#         uses that same value as its ratio. For example, with 8 heads the values would be [1/(2^1),
#         1/(2^2), ... , 1/(2^8)]. With 16 heads the values would be [1/(2^0.5), 1/(2^1), ... , 1/(2^8)].

#         See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

#         Examples:

#         >>> Attention.create_alibi_multipliers(8)
#         tensor([0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039])

#         >>> Attention.create_alibi_multipliers(16)
#         tensor([0.7071, 0.5000, 0.3536, 0.2500, 0.1768, 0.1250, 0.0884, 0.0625, 0.0442, 0.0312,
#                 0.0221, 0.0156, 0.0110, 0.0078, 0.0055, 0.0039])

#         Args:
#             n_heads: The number of heads in a layer.
#             device: The device to create the tensor on.

#         Returns:
#             A tensor of shape (n_heads,) containing the scalar multiplier for each head.
#         """
#         # Calculate the starting value
#         start = 2 ** (-8 / n_heads)

#         # Generate the indices [0, 1, ..., n_heads-1]
#         indices = torch.arange(n_heads, device=device)

#         # Compute the multipliers, with the starting value being the same as the ratio
#         multipliers = start * (start**indices)

#         return multipliers

#     @staticmethod
#     def create_alibi_bias(
#         n_heads: int, n_ctx: int, device: torch.device = None
#     ) -> Float[torch.Tensor, "head_idx query key"]:
#         """Create the ALiBi Bias for all Heads.

#         Calculate the ALiBi bias (https://arxiv.org/pdf/2108.12409.pdf) for all heads in a layer.

#         The broad idea behind ALiBi is to remove the positional encoding from the original transformer
#         model, and instead apply a bias to each attention score. This bias is proportional to the
#         distance between the query and key (i.e. it encourage paying less attention to more distant
#         tokens), and is added to the attention scores before the softmax. It is used in models such as
#         Bloom.

#         Examples:

#         >>> Attention.create_alibi_bias(2, 4, torch.device('cpu'))
#         tensor([[[ 0.0000,  0.0000,  0.0000,  0.0000],
#             [-0.0625,  0.0000,  0.0000,  0.0000],
#             [-0.1250, -0.0625,  0.0000,  0.0000],
#             [-0.1875, -0.1250, -0.0625,  0.0000]],
#             [[ 0.0000,  0.0000,  0.0000,  0.0000],
#             [-0.0039,  0.0000,  0.0000,  0.0000],
#             [-0.0078, -0.0039,  0.0000,  0.0000],
#             [-0.0117, -0.0078, -0.0039,  0.0000]]])

#         Args:
#             n_heads: The number of heads in a layer.
#             n_ctx: The maximum number of tokens in a prompt.
#             device: The device to create the tensor on.

#         Returns:
#             The ALiBi bias that should be added to the attention scores before the softmax.
#         """
#         # Create the slope matrix
#         slope: Float[torch.Tensor, "query key"] = Attention.create_alibi_slope(
#             n_ctx, device
#         )

#         # Create the scalar multiplier for each head.
#         multipliers: Float[torch.Tensor, "head_idx"] = (
#             Attention.create_alibi_multipliers(n_heads, device)
#         )

#         # The ALiBi bias is then m * slope_matrix
#         alibi_bias = torch.einsum("ij,k->kij", slope, multipliers)

#         return alibi_bias


# # MLP Layers
# class MLP(nn.Module):
#     def __init__(mod, cfg: Union[Dict, HookedTransformerConfig]):
#         super().__init__()
#         if isinstance(cfg, Dict):
#             cfg = HookedTransformerConfig.from_dict(cfg)
#         mod.cfg = cfg
#         mod.W_in = nn.Parameter(
#             torch.empty(mod.cfg.d_model, mod.cfg.d_mlp, dtype=cfg.dtype)
#         )
#         mod.b_in = nn.Parameter(torch.zeros(mod.cfg.d_mlp, dtype=cfg.dtype))
#         mod.W_out = nn.Parameter(
#             torch.empty(mod.cfg.d_mlp, mod.cfg.d_model, dtype=cfg.dtype)
#         )
#         mod.b_out = nn.Parameter(torch.zeros(mod.cfg.d_model, dtype=cfg.dtype))

#         mod.hook_pre = HookPoint()  # [batch, pos, d_mlp]
#         mod.hook_post = HookPoint()  # [batch, pos, d_mlp]

#         if mod.cfg.act_fn == "relu":
#             mod.act_fn = F.relu
#         elif mod.cfg.act_fn == "gelu":
#             mod.act_fn = F.gelu
#         elif mod.cfg.act_fn == "silu":
#             mod.act_fn = F.silu
#         elif mod.cfg.act_fn == "gelu_new":
#             mod.act_fn = gelu_new
#         elif mod.cfg.act_fn == "gelu_fast":
#             mod.act_fn = gelu_fast
#         elif mod.cfg.act_fn == "solu_ln":
#             mod.act_fn = solu
#             # Hook taken between activation and layer norm
#             mod.hook_mid = HookPoint()  # [batch, pos, d_mlp]
#             if mod.cfg.normalization_type == "LN":
#                 mod.ln = LayerNorm(mod.cfg, mod.cfg.d_mlp)
#             else:
#                 mod.ln = LayerNormPre(mod.cfg)

#         else:
#             raise ValueError(f"Invalid activation function name: {mod.cfg.act_fn}")

#     def forward(
#         mod, x: Float[torch.Tensor, "batch pos d_model"]
#     ) -> Float[torch.Tensor, "batch pos d_model"]:
#         # Technically, all these einsums could be done with a single matmul, but this is more readable.
#         pre_act = mod.hook_pre(
#             einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, mod.W_in)
#             + mod.b_in
#         )  # [batch, pos, d_mlp]
#         if not mod.cfg.act_fn.endswith("_ln"):
#             post_act = mod.hook_post(mod.act_fn(pre_act))  # [batch, pos, d_mlp]
#         else:
#             mid_act = mod.hook_mid(mod.act_fn(pre_act))  # [batch, pos, d_mlp]
#             post_act = mod.hook_post(mod.ln(mid_act))
#         return (
#             einsum(
#                 "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
#                 post_act,
#                 mod.W_out,
#             )
#             + mod.b_out
#         )


# # TODO
# # not sure whether to fold this into MLP or not
# class GatedMLP(nn.Module):
#     """
#     The equation of a gated MLP:
#     pre = x @ W_gate
#     pre_linear = x @ W_in
#     post = Gelu(pre) * (pre_linear) + b_in
#     mlp_out = post @ W_out + b_out

#     In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
#     """

#     def __init__(mod, cfg: Union[Dict, HookedTransformerConfig]):
#         super().__init__()
#         if isinstance(cfg, Dict):
#             cfg = HookedTransformerConfig.from_dict(cfg)
#         mod.cfg = cfg
#         mod.W_in = nn.Parameter(
#             torch.empty(mod.cfg.d_model, mod.cfg.d_mlp, dtype=cfg.dtype)
#         )
#         mod.W_gate = nn.Parameter(
#             torch.empty(mod.cfg.d_model, mod.cfg.d_mlp, dtype=cfg.dtype)
#         )
#         mod.b_in = nn.Parameter(torch.zeros(mod.cfg.d_mlp, dtype=cfg.dtype))
#         mod.W_out = nn.Parameter(
#             torch.empty(mod.cfg.d_mlp, mod.cfg.d_model, dtype=cfg.dtype)
#         )
#         mod.b_out = nn.Parameter(torch.zeros(mod.cfg.d_model, dtype=cfg.dtype))

#         # hook on gate output but before act_fn
#         mod.hook_pre = HookPoint()  # [batch, pos, d_mlp]
#         # hook on the linear component of the input
#         mod.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
#         # hook on act_fn(gate_output) * W_in(x) + b_in
#         mod.hook_post = HookPoint()  # [batch, pos, d_mlp]

#         if mod.cfg.act_fn == "relu":
#             mod.act_fn = F.relu
#         elif mod.cfg.act_fn == "gelu":
#             mod.act_fn = F.gelu
#         elif mod.cfg.act_fn == "silu":
#             mod.act_fn = F.silu
#         elif mod.cfg.act_fn == "gelu_new":
#             mod.act_fn = gelu_new
#         elif mod.cfg.act_fn == "gelu_fast":
#             mod.act_fn = gelu_fast
#         elif mod.cfg.act_fn == "solu_ln":
#             mod.act_fn = solu
#             # Hook taken between activation and layer norm
#             mod.hook_mid = HookPoint()  # [batch, pos, d_mlp]
#             if mod.cfg.normalization_type == "LN":
#                 mod.ln = LayerNorm(mod.cfg, mod.cfg.d_mlp)
#             else:
#                 mod.ln = LayerNormPre(mod.cfg)

#         else:
#             raise ValueError(f"Invalid activation function name: {mod.cfg.act_fn}")

#     def forward(
#         mod, x: Float[torch.Tensor, "batch pos d_model"]
#     ) -> Float[torch.Tensor, "batch pos d_model"]:
#         # Technically, all these einsums could be done with a single matmul, but this is more readable.
#         pre_act = mod.hook_pre(
#             einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, mod.W_gate)
#         )  # [batch, pos, d_mlp]
#         if not mod.cfg.act_fn.endswith("_ln"):
#             pre_linear = mod.hook_pre_linear(
#                 einsum(
#                     "batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, mod.W_in
#                 )
#             )
#             post_act = mod.hook_post(
#                 (mod.act_fn(pre_act) * pre_linear) + mod.b_in
#             )  # [batch, pos, d_mlp]
#         else:
#             mid_act = mod.hook_mid(mod.act_fn(pre_act))  # [batch, pos, d_mlp]
#             post_act = mod.hook_post(mod.ln(mid_act))
#         return (
#             einsum(
#                 "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
#                 post_act,
#                 mod.W_out,
#             )
#             + mod.b_out
#         )


# # Transformer Block
# class TransformerBlock(nn.Module):
#     def __init__(mod, cfg: Union[Dict, HookedTransformerConfig], block_index):
#         super().__init__()
#         if isinstance(cfg, Dict):
#             cfg = HookedTransformerConfig.from_dict(cfg)
#         mod.cfg = cfg
#         if mod.cfg.normalization_type == "LN":
#             mod.ln1 = LayerNorm(cfg)
#             if not mod.cfg.attn_only:
#                 mod.ln2 = LayerNorm(cfg)
#         elif mod.cfg.normalization_type == "LNPre":
#             # We've folded in LayerNorm weights, so just need the center + scale parts
#             mod.ln1 = LayerNormPre(cfg)
#             if not mod.cfg.attn_only:
#                 mod.ln2 = LayerNormPre(cfg)
#         elif mod.cfg.normalization_type == "RMS":
#             mod.ln1 = RMSNorm(cfg)
#             if not mod.cfg.attn_only:
#                 mod.ln2 = RMSNorm(cfg)
#         elif mod.cfg.normalization_type == "RMSPre":
#             mod.ln1 = RMSNormPre(cfg)
#             if not mod.cfg.attn_only:
#                 mod.ln2 = RMSNormPre(cfg)
#         elif mod.cfg.normalization_type is None:
#             mod.ln1 = nn.Identity()
#             if not mod.cfg.attn_only:
#                 mod.ln2 = nn.Identity()
#         else:
#             logging.warning(
#                 f"Invalid normalization_type passed in {mod.cfg.normalization_type}"
#             )

#         if not mod.cfg.use_local_attn:
#             mod.attn = Attention(cfg, "global", block_index)
#         else:
#             assert mod.cfg.attn_types is not None
#             attn_type = mod.cfg.attn_types[block_index]
#             mod.attn = Attention(cfg, attn_type, block_index)
#         if not mod.cfg.attn_only:
#             if mod.cfg.gated_mlp:
#                 mod.mlp = GatedMLP(cfg)
#             else:
#                 mod.mlp = MLP(cfg)

#         mod.hook_attn_in = HookPoint()  # [batch, pos, n_heads, d_model]
#         mod.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
#         mod.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
#         mod.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]
#         mod.hook_mlp_in = HookPoint()  # [batch, pos, d_model]

#         mod.hook_attn_out = HookPoint()  # [batch, pos, d_model]
#         mod.hook_mlp_out = HookPoint()  # [batch, pos, d_model]

#         mod.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
#         if not mod.cfg.attn_only and not mod.cfg.parallel_attn_mlp:
#             mod.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
#         mod.hook_resid_post = HookPoint()  # [batch, pos, d_model]

#     def forward(
#         mod,
#         resid_pre: Float[torch.Tensor, "batch pos d_model"],
#         shortformer_pos_embed: Optional[
#             Float[torch.Tensor, "batch pos d_model"]
#         ] = None,
#         past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
#         attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
#     ) -> Float[torch.Tensor, "batch pos d_model"]:
#         """A single Transformer block.

#         Args:
#             resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
#             cache (HookedTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
#             shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See HookedTransformerConfig for details. Defaults to None.
#             attention_mask (torch.Tensor, optional): The attention mask for padded tokens. Defaults to None.

#         Returns:
#             _type_: _description_
#         """
#         resid_pre = mod.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

#         def add_head_dimension(
#             tensor: Float[torch.Tensor, "batch pos d_model"],
#             clone_tensor=True,
#             # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
#         ):
#             repeated_tensor = einops.repeat(
#                 tensor,
#                 "batch pos d_model -> batch pos n_heads d_model",
#                 n_heads=mod.cfg.n_heads,
#             )
#             if clone_tensor:
#                 return repeated_tensor.clone()
#             else:
#                 return repeated_tensor

#         if mod.cfg.use_attn_in or mod.cfg.use_split_qkv_input:
#             # We're adding a head dimension
#             attn_in = add_head_dimension(resid_pre, clone_tensor=False)
#             if shortformer_pos_embed is not None:
#                 shortformer_pos_embed = add_head_dimension(shortformer_pos_embed)
#         else:
#             attn_in = resid_pre

#         if mod.cfg.use_attn_in:
#             attn_in = mod.hook_attn_in(attn_in.clone())

#         if mod.cfg.use_split_qkv_input:
#             query_input = mod.hook_q_input(attn_in.clone())
#             key_input = mod.hook_k_input(attn_in.clone())
#             value_input = mod.hook_v_input(attn_in.clone())
#         else:
#             query_input = attn_in
#             key_input = attn_in
#             value_input = attn_in

#         attn_out = mod.hook_attn_out(
#             # hook the residual stream states that are used to calculate the
#             # queries, keys and values, independently.
#             # Then take the layer norm of these inputs, and pass these to the attention module.
#             mod.attn(
#                 query_input=mod.ln1(query_input)
#                 + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
#                 key_input=mod.ln1(key_input)
#                 + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
#                 value_input=mod.ln1(value_input),
#                 past_kv_cache_entry=past_kv_cache_entry,
#                 attention_mask=attention_mask,
#             )
#         )  # [batch, pos, d_model]
#         if not mod.cfg.attn_only and not mod.cfg.parallel_attn_mlp:
#             resid_mid = mod.hook_resid_mid(
#                 resid_pre + attn_out
#             )  # [batch, pos, d_model]
#             mlp_in = (
#                 resid_mid
#                 if not mod.cfg.use_hook_mlp_in
#                 else mod.hook_mlp_in(resid_mid.clone())
#             )
#             normalized_resid_mid = mod.ln2(mlp_in)
#             mlp_out = mod.hook_mlp_out(
#                 mod.mlp(normalized_resid_mid)
#             )  # [batch, pos, d_model]
#             resid_post = mod.hook_resid_post(
#                 resid_mid + mlp_out
#             )  # [batch, pos, d_model]
#         elif mod.cfg.parallel_attn_mlp:
#             # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
#             # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
#             normalized_resid_pre_2 = mod.ln2(
#                 resid_pre
#                 if not mod.cfg.use_hook_mlp_in
#                 else mod.hook_mlp_in(resid_pre.clone())
#             )
#             mlp_out = mod.hook_mlp_out(
#                 mod.mlp(normalized_resid_pre_2)
#             )  # [batch, pos, d_model]
#             resid_post = mod.hook_resid_post(
#                 resid_pre + attn_out + mlp_out
#             )  # [batch, pos, d_model]
#         else:
#             resid_post = mod.hook_resid_post(
#                 resid_pre + attn_out
#             )  # [batch, pos, d_model]
#         return resid_post


# class BertBlock(nn.Module):
#     """
#     BERT Block. Similar to the TransformerBlock, except that the LayerNorms are applied after the attention and MLP, rather than before.
#     """

#     def __init__(mod, cfg: HookedTransformerConfig):
#         super().__init__()
#         mod.cfg = cfg

#         mod.attn = Attention(cfg)
#         mod.ln1 = LayerNorm(cfg)
#         mod.mlp = MLP(cfg)
#         mod.ln2 = LayerNorm(cfg)

#         mod.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
#         mod.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
#         mod.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]

#         mod.hook_attn_out = HookPoint()  # [batch, pos, d_model]
#         mod.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
#         mod.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
#         mod.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
#         mod.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
#         mod.hook_resid_post = HookPoint()  # [batch, pos, d_model]
#         mod.hook_normalized_resid_post = HookPoint()  # [batch, pos, d_model]

#     def forward(
#         mod,
#         resid_pre: Float[torch.Tensor, "batch pos d_model"],
#         additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
#     ):
#         resid_pre = mod.hook_resid_pre(resid_pre)

#         query_input = resid_pre
#         key_input = resid_pre
#         value_input = resid_pre

#         if mod.cfg.use_split_qkv_input:

#             def add_head_dimension(tensor):
#                 return einops.repeat(
#                     tensor,
#                     "batch pos d_model -> batch pos n_heads d_model",
#                     n_heads=mod.cfg.n_heads,
#                 ).clone()

#             query_input = mod.hook_q_input(add_head_dimension(query_input))
#             key_input = mod.hook_k_input(add_head_dimension(key_input))
#             value_input = mod.hook_v_input(add_head_dimension(value_input))

#         attn_out = mod.hook_attn_out(
#             mod.attn(
#                 query_input,
#                 key_input,
#                 value_input,
#                 additive_attention_mask=additive_attention_mask,
#             )
#         )
#         resid_mid = mod.hook_resid_mid(resid_pre + attn_out)

#         mlp_in = (
#             resid_mid
#             if not mod.cfg.use_hook_mlp_in
#             else mod.hook_mlp_in(resid_mid.clone())
#         )
#         normalized_resid_mid = mod.ln1(mlp_in)
#         mlp_out = mod.hook_mlp_out(mod.mlp(normalized_resid_mid))
#         resid_post = mod.hook_resid_post(normalized_resid_mid + mlp_out)
#         normalized_resid_post = mod.hook_normalized_resid_post(mod.ln2(resid_post))

#         return normalized_resid_post


# # %%
# model = HookedTransformer(
#     HookedTransformerConfig(
#         n_layers=1, d_vocab=5, d_model=6, n_ctx=6, d_head=2, attn_only=True
#     )
# )
# # %%
# CountEmbed.count_forward(model.embed, CountTensor([1, 6]))
# # %%


# # class CountHookedTransformer:

# #     @staticmethod
# #     def count_input_to_embed(
# #         model: HookedTransformer,
# #         input_shape: Sequence[int],
# #         prepend_bos: Optional[Union[bool, None]] = None,
# #         padding_side: Optional[
# #             Union[Literal["left", "right"], None]
# #         ] = None,
# #     ) -> Tuple[Tuple[
# #         Sequence[int],  # residual
# #         Optional[Sequence[int]],  # tokens
# #         Optional[Sequence[int]],  # shortformer_pos_embed
# #         Optional[Sequence[int]],  # attention_mask [batch pos]
# #     ], InstructionCount]:
# #         count = InstructionCount()
# #         tokens_shape = input_shape
# #         if len(tokens_shape) == 1:
# #             # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
# #             tokens_shape = (1, *tokens_shape)
# #         if (
# #             model.tokenizer and model.tokenizer.padding_side == "left"
# #         ):
# #             if prepend_bos is None:
# #                 prepend_bos = model.cfg.default_prepend_bos
# #             # Approximation for:
# #             # attention_mask = transformer_lens.utils.get_attention_mask(
# #             #     mod.tokenizer, tokens, prepend_bos
# #             # )
# #             count += InstructionCount(flop=int(np.prod(list(tokens_shape))))
# #             attention_mask_shape = tokens_shape
# #         else:
# #             # We separate this case from for computational efficiency.
# #             attention_mask = None

# #         pos_offset = 0
# #         embed = mod.hook_embed(mod.embed(tokens))  # [batch, pos, d_model]
# #         if mod.cfg.positional_embedding_type == "standard":
# #             pos_embed = mod.hook_pos_embed(
# #                 mod.pos_embed(tokens, pos_offset, attention_mask)
# #             )  # [batch, pos, d_model]
# #             residual = embed + pos_embed  # [batch, pos, d_model]
# #             shortformer_pos_embed = None
# #         elif mod.cfg.positional_embedding_type == "shortformer":
# #             # If we're using shortformer style attention, we don't add the positional embedding to
# #             # the residual stream. See HookedTransformerConfig for details
# #             pos_embed = mod.hook_pos_embed(
# #                 mod.pos_embed(tokens, pos_offset, attention_mask)
# #             )  # [batch, pos, d_model]
# #             residual = embed
# #             shortformer_pos_embed = pos_embed
# #         elif mod.cfg.positional_embedding_type == "rotary":
# #             # Rotary doesn't use positional embeddings, instead they're applied when dot producting
# #             # keys and queries. See HookedTransformerConfig for details
# #             residual = embed
# #             shortformer_pos_embed = None
# #         elif mod.cfg.positional_embedding_type == "alibi":
# #             # ALiBi does not add positional embeddings to word embeddings,instead it biases QK attention scores.
# #             residual = embed
# #             shortformer_pos_embed = None
# #         else:
# #             raise ValueError(
# #                 f"Invalid positional_embedding_type passed in {mod.cfg.positional_embedding_type}"
# #             )
# #         return residual, tokens, shortformer_pos_embed, attention_mask


# #     @staticmethod
# #     def count_forward(model: HookedTransformer, arg_shape: Sequence[int],
# #                                return_type: Optional[str] = "logits",
# #         loss_per_token: Optional[bool] = False,
# #         prepend_bos: Optional[Union[bool, None]] = None,
# #         padding_side: Optional[Literal["left", "right"]] = None,
# #         start_at_layer: Optional[int] = None,
# #         tokens: Optional[Sequence[int]] = None,
# #         shortformer_pos_embed: Optional[Sequence[int]] = None,
# #         attention_mask: Optional[Sequence[int]] = None,  # [batch pos]
# #         stop_at_layer: Optional[int] = None) -> InstructionCount:
# #         """
# #         Count the number of operations in the forward pass of the model.

# #         Args:
# #             model: The model to count the operations of.

# #         Returns:
# #             The number of floating point operations, integer operations, and branch operations.
# #         """
# #         with transformer_lens.utils.LocallyOverridenDefaults(
# #                 model, prepend_bos=prepend_bos, padding_side=padding_side
# #             ):
# #             if len(arg_shape) == 1:
# #                 arg_shape = (1, *arg_shape) # add batch dimension

# #             if start_at_layer is None:
# #                 start_at_layer = 0

# #             blocks_and_idxs = list(zip(range(model.cfg.n_layers), model.blocks))
# #             for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:  # type: ignore
# #                 # Note that each block includes skip connections, so we don't need
# #                 # residual + block(residual)
# #                 # If we're using multiple GPUs, we need to send the residual and shortformer_pos_embed to the correct GPU
# #                 residual = residual.to(devices.get_device_for_block_index(i, mod.cfg))
# #                 if shortformer_pos_embed is not None:
# #                     shortformer_pos_embed = shortformer_pos_embed.to(
# #                         devices.get_device_for_block_index(i, mod.cfg)
# #                     )

# #                 residual = block(
# #                     residual,
# #                     # Cache contains a list of HookedTransformerKeyValueCache objects, one for each
# #                     # block
# #                     past_kv_cache_entry=past_kv_cache[i]
# #                     if past_kv_cache is not None
# #                     else None,
# #                     shortformer_pos_embed=shortformer_pos_embed,
# #                     attention_mask=attention_mask,
# #                 )  # [batch, pos, d_model]

# #             if stop_at_layer is not None:
# #                 # When we stop at an early layer, we end here rather than doing further computation
# #                 return residual

# #             if mod.cfg.normalization_type is not None:
# #                 residual = mod.ln_final(residual)  # [batch, pos, d_model]
# #             if return_type is None:
# #                 return None
# #             else:
# #                 logits = mod.unembed(residual)  # [batch, pos, d_vocab]
# #                 if return_type == "logits":
# #                     return logits
# #                 else:
# #                     assert (
# #                         tokens is not None
# #                     ), "tokens must be passed in if return_type is 'loss' or 'both'"
# #                     loss = mod.loss_fn(logits, tokens, per_token=loss_per_token)
# #                     if return_type == "loss":
# #                         return loss
# #                     elif return_type == "both":
# #                         return Output(logits, loss)
# #                     else:
# #                         logging.warning(f"Invalid return_type passed in: {return_type}")
# #                         return None


# #     )
# #     return sum(
# #         count_flops(module, config)
# #         for module in model.modules()
# #         if isinstance(module, torch.nn.Linear)
# #     )

# # %%
