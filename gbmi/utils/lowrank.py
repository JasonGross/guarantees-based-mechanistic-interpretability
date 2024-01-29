from torch import Tensor
from jaxtyping import Float
from typing import Union


class LowRankTensor(Tensor):
    def __init__(self, u: Tensor, v: Tensor, check: Union[bool, dict] = False):
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

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    def check(self, show: bool = False, **kwargs):
        full_kwargs = dict(self._check) if isinstance(self._check, dict) else {}
        full_kwargs.update(kwargs)

    #     assert torch.allclose(m1, m2, **kwargs), [
    #     px.imshow(m1).show(),
    #     px.imshow(m2).show(),
    #     px.imshow((m1 - m2).abs()).show(),
    # ]
    #     if self._check:
    #         return self.allclose(self._check, atol=atol, rtol=rtol)
    #     else:
    #         return True

    # def __matmul__(self, other: Tensor) -> LowRankTensor:
    #     if isinstance(other, LowRankTensor):
    #         u, mid, v = self.u, self.v @ other.u, other.v
    #         if u.shape
    #         return LowRankTensor(self.u @ other.u, other.v @ self.v)
    #     else:
    #         return super().__matmul__(other)

    # def __repr__(self):
    #     return f"LowRankTensor(u={self.u!r}, v={self.v!r})"

    # def __str__(self):
    #     return f"LowRankTensor(u={self.u}, v={self.v})"
