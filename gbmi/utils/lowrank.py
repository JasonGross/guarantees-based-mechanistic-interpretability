from torch import Tensor
import torch
from jaxtyping import Float
from typing import Union, Optional
import plotly.express as px


class LowRankTensor(Tensor):
    def __init__(
        self, u: Tensor, v: Tensor, check: Union[bool, dict] = False, show: bool = True
    ):
        if u.ndim == 1:
            u = u[:, None]
        if v.ndim == 1:
            v = v[None, :]
        assert (
            u.shape[-1] == v.shape[-2]
        ), f"u.shape[-1] must equal v.shape[-2]; u.shape={u.shape}; v.shape={v.shape}"
        super().__init__(u @ v)
        self._u = u
        self._v = v
        self._check = check
        self._show = show

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    def totensor(self) -> Tensor:
        return self.u @ self.v

    @torch.no_grad()
    def check(
        self,
        other: Tensor,
        show: Optional[bool] = None,
        descr: Optional[str] = None,
        renderer: Optional[str] = None,
        **kwargs,
    ) -> bool:
        if show is None:
            show = self._show
        full_kwargs = dict(self._check) if isinstance(self._check, dict) else {}
        full_kwargs.update(kwargs)
        if torch.allclose(self.totensor(), other, **full_kwargs):
            return True
        descr = "" if descr is None else " " + descr
        if show:
            px.imshow(self.detach().numpy(), title=f"self{descr}").show(
                renderer=renderer
            )
            px.imshow(other.detach().numpy(), title=f"other{descr}").show(
                renderer=renderer
            )
            px.imshow(
                (self - other).abs().detach().numpy(), title=f"difference{descr}"
            ).show(renderer=renderer)
        return False

    def __matmul__(self, other: Tensor):
        if isinstance(other, LowRankTensor):
            # prefer to keep the dimensions of stored matrices as low as possible
            u, mid, v = self.u, self.v @ other.u, other.v
            if len(mid.shape) <= 1:
                if u.shape[-1] <= v.shape[-2]:
                    v = mid @ v
                else:
                    u = u @ mid
            elif mid.shape[-2] <= mid.shape[-1]:
                v = mid @ v
            else:
                u = u @ mid
        else:
            u, v = self.u, self.v @ other
        result = LowRankTensor(u, v, check=self._check, show=self._show)
        if self._check:
            assert result.check(self.totensor() @ other, descr="matmul")
        return result

    def __rmatmul__(self, other: Tensor):
        if isinstance(other, LowRankTensor):
            # prefer to keep the dimensions of stored matrices as low as possible
            u, mid, v = other.u, other.v @ self.u, self.v
            if len(mid.shape) <= 1:
                if u.shape[-1] <= v.shape[-2]:
                    v = mid @ v
                else:
                    u = u @ mid
            elif mid.shape[-2] <= mid.shape[-1]:
                v = mid @ v
            else:
                u = u @ mid
        else:
            u, v = other @ self.u, self.v
        result = LowRankTensor(u, v, check=self._check, show=self._show)
        if self._check:
            assert result.check(other @ self.totensor(), descr="matmul")
        return result

    def __repr__(self):
        return f"LowRankTensor(u={self.u!r}, v={self.v!r})"

    def __str__(self):
        return f"LowRankTensor(u={self.u}, v={self.v})"
