# Copy of https://github.com/TransformerLensOrg/TransformerLens/blob/5a374ec4b33cec6281b37494175d14f06c75dcfd/transformer_lens/FactoredMatrix.py with some changes that haven't yet been upstreamed, to work better with CountTensor
# This file is licensed under https://github.com/TransformerLensOrg/TransformerLens/blob/main/LICENSE
# MIT License

# Copyright (c) 2022 TransformerLensOrg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Factored Matrix.

Utilities for representing a matrix as a product of two matrices, and for efficient calculation of
eigenvalues, norm and SVD.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Union, overload

import torch
import transformer_lens.utils as utils
from jaxtyping import Float


class FactoredMatrix:
    """
    Class to represent low rank factored matrices, where the matrix is represented as a product of two matrices. Has utilities for efficient calculation of eigenvalues, norm and SVD.
    """

    def __init__(
        self,
        A: Float[torch.Tensor, "... ldim mdim"],  # noqa: F722
        B: Float[torch.Tensor, "... mdim rdim"],  # noqa: F722
    ):
        self.A = A
        self.B = B
        assert self.A.size(-1) == self.B.size(
            -2
        ), f"Factored matrix must match on inner dimension, shapes were a: {self.A.shape}, b:{self.B.shape}"
        self.ldim = self.A.size(-2)
        self.rdim = self.B.size(-1)
        self.mdim = self.B.size(-2)
        self.has_leading_dims = (self.A.ndim > 2) or (self.B.ndim > 2)
        self.shape = torch.broadcast_shapes(self.A.shape[:-2], self.B.shape[:-2]) + (
            self.ldim,
            self.rdim,
        )
        self.A = self.A.broadcast_to(self.shape[:-2] + (self.ldim, self.mdim))
        self.B = self.B.broadcast_to(self.shape[:-2] + (self.mdim, self.rdim))

    @overload
    def __matmul__(
        self,
        other: Union[
            Float[torch.Tensor, "... rdim new_rdim"],  # noqa: F722
            "FactoredMatrix",
        ],
    ) -> "FactoredMatrix": ...

    @overload
    def __matmul__(  # type: ignore
        self,
        other: Float[torch.Tensor, "rdim"],  # noqa: F821
    ) -> Float[torch.Tensor, "... ldim"]: ...  # noqa: F722

    def __matmul__(
        self,
        other: Union[
            Float[torch.Tensor, "... rdim new_rdim"],  # noqa: F722
            Float[torch.Tensor, "rdim"],  # noqa: F821
            "FactoredMatrix",
        ],
    ) -> Union["FactoredMatrix", Float[torch.Tensor, "... ldim"]]:  # noqa: F722
        if isinstance(other, FactoredMatrix):
            return (self @ other.A) @ other.B
        else:
            if other.ndim < 2:
                # It's a vector, so we collapse the factorisation and just return a vector
                # Squeezing/Unsqueezing is to preserve broadcasting working nicely
                return (self.A @ (self.B @ other.unsqueeze(-1))).squeeze(-1)
            else:
                assert (
                    other.size(-2) == self.rdim
                ), f"Right matrix must match on inner dimension, shapes were self: {self.shape}, other:{other.shape}"
                if self.rdim > self.mdim:
                    return FactoredMatrix(self.A, self.B @ other)
                else:
                    return FactoredMatrix(self.AB, other)

    @overload
    def __rmatmul__(  # type: ignore
        self,
        other: Union[
            Float[torch.Tensor, "... new_rdim ldim"],  # noqa: F722
            "FactoredMatrix",
        ],
    ) -> "FactoredMatrix": ...

    @overload
    def __rmatmul__(  # type: ignore
        self,
        other: Float[torch.Tensor, "ldim"],  # noqa: F821
    ) -> Float[torch.Tensor, "... rdim"]: ...  # noqa: F722

    def __rmatmul__(  # type: ignore
        self,
        other: Union[
            Float[torch.Tensor, "... new_rdim ldim"],  # noqa: F722
            Float[torch.Tensor, "ldim"],  # noqa: F821
            "FactoredMatrix",
        ],
    ) -> Union["FactoredMatrix", Float[torch.Tensor, "... rdim"]]:  # noqa: F722
        if isinstance(other, FactoredMatrix):
            return other.A @ (other.B @ self)
        else:
            assert (
                other.size(-1) == self.ldim
            ), f"Left matrix must match on inner dimension, shapes were self: {self.shape}, other:{other.shape}"
            if other.ndim < 2:
                # It's a vector, so we collapse the factorisation and just return a vector
                return ((other.unsqueeze(-2) @ self.A) @ self.B).squeeze(-2)
            elif self.ldim > self.mdim:
                return FactoredMatrix(other @ self.A, self.B)
            else:
                return FactoredMatrix(other, self.AB)

    def __mul__(self, scalar: Union[int, float, torch.Tensor]) -> FactoredMatrix:
        """
        Left scalar multiplication. Scalar multiplication distributes over matrix multiplication, so we can just multiply one of the factor matrices by the scalar.
        """
        if isinstance(scalar, torch.Tensor):
            assert (
                scalar.numel() == 1
            ), f"Tensor must be a scalar for use with * but was of shape {scalar.shape}. For matrix multiplication, use @ instead."
        return FactoredMatrix(self.A * scalar, self.B)

    def __rmul__(self, scalar: Union[int, float, torch.Tensor]) -> FactoredMatrix:  # type: ignore
        """
        Right scalar multiplication. For scalar multiplication from the right, we can reuse the __mul__ method.
        """
        return self * scalar

    @property
    def AB(self) -> Float[torch.Tensor, "*leading_dims ldim rdim"]:  # noqa: F722
        """The product matrix - expensive to compute, and can consume a lot of GPU memory"""
        return self.A @ self.B

    @property
    def BA(self) -> Float[torch.Tensor, "*leading_dims rdim ldim"]:  # noqa: F722
        """The reverse product. Only makes sense when ldim==rdim"""
        assert (
            self.rdim == self.ldim
        ), f"Can only take ba if ldim==rdim, shapes were self: {self.shape}"
        return self.B @ self.A

    @property
    def T(self) -> FactoredMatrix:
        return FactoredMatrix(self.B.transpose(-2, -1), self.A.transpose(-2, -1))

    @lru_cache(maxsize=None)
    def svd(
        self,
    ) -> Tuple[
        Float[torch.Tensor, "*leading_dims ldim mdim"],  # noqa: F722
        Float[torch.Tensor, "*leading_dims mdim"],  # noqa: F722
        Float[torch.Tensor, "*leading_dims rdim mdim"],  # noqa: F722
    ]:
        """
        Efficient algorithm for finding Singular Value Decomposition, a tuple (U, S, Vh) for matrix M st S is a vector and U, Vh are orthogonal matrices, and U @ S.diag() @ Vh.T == M

        (Note that Vh is given as the transpose of the obvious thing)
        """
        Ua, Sa, Vha = torch.svd(self.A)
        Ub, Sb, Vhb = torch.svd(self.B)
        middle = Sa[..., :, None] * utils.transpose(Vha) @ Ub * Sb[..., None, :]
        Um, Sm, Vhm = torch.svd(middle)
        U = Ua @ Um
        Vh = Vhb @ Vhm
        S = Sm
        return U, S, Vh

    @property
    def U(self) -> Float[torch.Tensor, "*leading_dims ldim mdim"]:  # noqa: F722
        return self.svd()[0]

    @property
    def S(self) -> Float[torch.Tensor, "*leading_dims mdim"]:  # noqa: F722
        return self.svd()[1]

    @property
    def Vh(self) -> Float[torch.Tensor, "*leading_dims rdim mdim"]:  # noqa: F722
        return self.svd()[2]

    @property
    def eigenvalues(self) -> Float[torch.Tensor, "*leading_dims mdim"]:  # noqa: F722
        """Eigenvalues of AB are the same as for BA (apart from trailing zeros), because if BAv=kv ABAv = A(BAv)=kAv, so Av is an eigenvector of AB with eigenvalue k."""
        return torch.linalg.eig(self.BA).eigenvalues

    def _convert_to_slice(self, sequence: Union[Tuple, List], idx: int) -> Tuple:
        """
        e.g. if sequence = (1, 2, 3) and idx = 1, return (1, slice(2, 3), 3). This only edits elements if they are ints.
        """
        if isinstance(idx, int):
            sequence = list(sequence)
            if isinstance(sequence[idx], int):
                sequence[idx] = slice(sequence[idx], sequence[idx] + 1)
            sequence = tuple(sequence)

        return sequence

    def __getitem__(self, idx: Union[int, Tuple]) -> FactoredMatrix:
        """Indexing - assumed to only apply to the leading dimensions."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        length = len([i for i in idx if i is not None])
        if length <= len(self.shape) - 2:
            return FactoredMatrix(self.A[idx], self.B[idx])
        elif length == len(self.shape) - 1:
            idx = self._convert_to_slice(idx, -1)
            return FactoredMatrix(self.A[idx], self.B[idx[:-1]])
        elif length == len(self.shape):
            idx = self._convert_to_slice(idx, -1)
            idx = self._convert_to_slice(idx, -2)
            return FactoredMatrix(
                self.A[idx[:-1]], self.B[idx[:-2] + (slice(None), idx[-1])]
            )
        else:
            raise ValueError(
                f"{idx} is too long an index for a FactoredMatrix with shape {self.shape}"
            )

    def norm(self) -> Float[torch.Tensor, "*leading_dims"]:  # noqa: F821
        """
        Frobenius norm is sqrt(sum of squared singular values)
        """
        return self.S.pow(2).sum(-1).sqrt()

    def __repr__(self):
        return f"FactoredMatrix: Shape({self.shape}), Hidden Dim({self.mdim})"

    def make_even(self) -> FactoredMatrix:
        """
        Returns the factored form of (U @ S.sqrt().diag(), S.sqrt().diag() @ Vh) where U, S, Vh are the SVD of the matrix. This is an equivalent factorisation, but more even - each half has half the singular values, and orthogonal rows/cols
        """
        return FactoredMatrix(
            self.U * self.S.sqrt()[..., None, :],
            self.S.sqrt()[..., :, None] * utils.transpose(self.Vh),
        )

    def get_corner(self, k=3):
        return utils.get_corner(self.A[..., :k, :] @ self.B[..., :, :k], k)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def collapse_l(
        self,
    ) -> Float[torch.Tensor, "*leading_dims mdim rdim"]:  # noqa: F722
        """
        Collapses the left side of the factorization by removing the orthogonal factor (given by self.U). Returns a (..., mdim, rdim) tensor
        """
        return self.S[..., :, None] * utils.transpose(self.Vh)

    def collapse_r(
        self,
    ) -> Float[torch.Tensor, "*leading_dims ldim mdim"]:  # noqa: F722
        """
        Analogous to collapse_l, returns a (..., ldim, mdim) tensor
        """
        return self.U * self.S[..., None, :]

    def unsqueeze(self, k: int) -> FactoredMatrix:
        return FactoredMatrix(self.A.unsqueeze(k), self.B.unsqueeze(k))

    @property
    def pair(
        self,
    ) -> Tuple[
        Float[torch.Tensor, "*leading_dims ldim mdim"],  # noqa: F722
        Float[torch.Tensor, "*leading_dims mdim rdim"],  # noqa: F722
    ]:
        return (self.A, self.B)
