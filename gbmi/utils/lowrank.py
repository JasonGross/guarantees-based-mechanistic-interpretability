from __future__ import annotations
from torch import Tensor
import torch
from jaxtyping import Float
from typing import TypeVar, Union, Optional, overload
import plotly.express as px
from transformer_lens import FactoredMatrix


T = TypeVar("T")


def _via_tensor(attr: str):
    def delegate(self: LowRankTensor, *args, **kwargs):
        return getattr(self.AB, attr)(*args, **kwargs)

    if hasattr(Tensor, attr):
        reference = getattr(Tensor, attr)
        for docattr in ("__doc__", "__name__"):
            if hasattr(reference, docattr):
                setattr(delegate, docattr, getattr(reference, docattr))
    return delegate


def _merge_check_params(
    *args: dict[str, T], merge_tol=max, merge_equal_nan=bool.__or__
) -> dict[str, T]:
    result = {}
    for arg in args:
        for key in arg:
            v = arg[key]
            if key not in result:
                result[key] = v
            elif key in ("rtol", "atol"):
                result[key] = merge_tol(result[key], v)
            elif key in ("equal_nan",):
                result[key] = merge_equal_nan(result[key], v)
            elif isinstance(result[key], dict) and isinstance(v, dict):
                result[key] = _merge_check_params(result[key], v)
            elif result[key] != v:
                raise ValueError(
                    f"Cannot merge check parameters {result[key]} and {v} for key {key}"
                )
    return result


# TODO: Drop wrapper around FactoredMatrix at some point
class LowRankTensor(FactoredMatrix):
    @overload
    def __init__(
        self,
        A: Tensor,
        B: Tensor,
        *,
        check: Union[bool, dict] = False,
        show: bool = True,
        checkparams: Optional[dict] = None,
    ): ...
    @overload
    def __init__(
        self,
        A: FactoredMatrix,
        *,
        check: Union[bool, dict] = False,
        show: bool = True,
        checkparams: Optional[dict] = None,
    ): ...

    def __init__(
        self,
        A: Union[Tensor, FactoredMatrix],
        B: Optional[Tensor] = None,
        *,
        check: Union[bool, dict] = False,
        show: bool = True,
        checkparams: Optional[dict] = None,
    ):
        if isinstance(A, FactoredMatrix):
            assert B is None, "B must not be provided if A is a FactoredMatrix"
            assert not isinstance(A, LowRankTensor), "Cannot pass a LowRankTensor as A"
            return self.__init__(
                A.A, A.B, check=check, show=show, checkparams=checkparams
            )
        assert B is not None, "B must be provided if A is not a FactoredMatrix"
        if A.ndim == 1:
            A = A[:, None]
        if B.ndim == 1:
            B = B[None, :]
        super().__init__(A, B)
        self._check = bool(check)
        self._checkparams = (
            checkparams
            if checkparams is not None
            else check if isinstance(check, dict) else {}
        )
        self._show = show

    def setcheckparams(self, **kwargs) -> LowRankTensor:
        self._checkparams = kwargs
        return self

    @torch.no_grad()
    def check(
        self,
        other: Union[Tensor, LowRankTensor, FactoredMatrix],
        show: Optional[bool] = None,
        descr: Optional[str] = None,
        renderer: Optional[str] = None,
        **kwargs,
    ) -> bool:
        if show is None:
            show = self._show
        full_kwargs = dict(self._checkparams)
        full_kwargs.update(kwargs)
        if isinstance(other, LowRankTensor) or isinstance(other, FactoredMatrix):
            other = other.AB
        if torch.allclose(self.AB, other, **full_kwargs):
            return True
        descr = "" if descr is None else " " + descr
        if show:
            px.imshow(self.numpy(), title=f"self{descr}").show(renderer=renderer)
            px.imshow(other.cpu().numpy(), title=f"other{descr}").show(
                renderer=renderer
            )
            px.imshow(
                (self - other).abs().cpu().numpy(),
                title=f"difference{descr} ({self._checkparams})",
            ).show(renderer=renderer)
        return False

    @torch.no_grad()
    def maybe_check(
        self,
        other: Union[Tensor, LowRankTensor, FactoredMatrix],
        show: Optional[bool] = None,
        descr: Optional[str] = None,
        renderer: Optional[str] = None,
        **kwargs,
    ) -> bool:
        return (
            self.check(other, show=show, descr=descr, renderer=renderer, **kwargs)
            if self._check
            else True
        )

    def params(self):
        return dict(
            check=self._check,
            show=self._show,
            checkparams=self._checkparams,
        )

    def _mergeparams(self, other):
        if isinstance(other, LowRankTensor):
            return _merge_check_params(self.params(), other.params())
        else:
            return self.params()

    @property
    def T(self) -> LowRankTensor:
        return LowRankTensor(super().T, **self.params())  # type: ignore

    def __matmul__(self, other: Union[Tensor, LowRankTensor, FactoredMatrix]):
        result = super().__matmul__(other)
        if not isinstance(result, LowRankTensor):
            result = LowRankTensor(result, **self._mergeparams(other))  # type: ignore
        if self._check:
            assert result.check(self.AB @ other, descr="matmul")
        return result

    def __rmatmul__(self, other: Union[Tensor, LowRankTensor]):
        result = super().__rmatmul__(other)
        if not isinstance(result, LowRankTensor):
            result = LowRankTensor(result, **self._mergeparams(other))  # type: ignore
        if self._check:
            assert result.check(other @ self.AB, descr="matmul")
        return result

    @torch.no_grad()
    def numpy(self):
        return self.AB.cpu().numpy()

    __add__ = _via_tensor("__add__")
    __radd__ = _via_tensor("__radd__")
    __sub__ = _via_tensor("__sub__")
    __rsub__ = _via_tensor("__rsub__")
