from typing import Union, Sequence, Tuple
import torch
from jaxtyping import Float
from torch import Tensor


def compute_verify_svd_close_matrices(
    A: Float[Tensor, "... m n"],  # noqa: F722
    U: Float[Tensor, "... m min(m,n)"],  # noqa: F722
    S: Float[Tensor, "... min(m,n)"],  # noqa: F722
    Vh: Float[Tensor, "... m min(m,n)"],  # noqa: F722
) -> Sequence[Tuple[Tensor, Tensor]]:
    # matrices match
    result = []
    m, n = A.shape[-2:]
    min_mn = min(m, n)
    U = U[..., m, :min_mn]
    S = S[..., :min_mn]
    Vh = Vh[..., :min_mn, n]
    if m <= n:
        U = U * S.unsqueeze(-2)
    else:
        Vh = Vh * S.unsqueeze(-1)
    USVh = U @ Vh
    result.append((A, USVh))

    # vectors are orthogonal:
    for m in (U, Vh):
        mH = m.mH
        if m.shape[-2] <= m.shape[-1]:
            mmH = mH @ m
        else:
            mmH = m @ mH
        assert mmH.shape[-2:] == (
            min_mn,
            min_mn,
        ), f"mmH.shape[-2:] == {mmH.shape[-2:]} != ({min_mn}, {min_mn})"
        result.append((mmH, torch.eye(min_mn).to(mmH)))

    return result


def verify_svd(
    A: Float[Tensor, "... m n"],  # noqa: F722
    U: Float[Tensor, "... m min(m,n)"],  # noqa: F722
    S: Float[Tensor, "... min(m,n)"],  # noqa: F722
    Vh: Float[Tensor, "... m min(m,n)"],  # noqa: F722
    **allclose_kwargs,
) -> bool:
    return all(
        torch.allclose(a, b, **allclose_kwargs)
        for (a, b) in compute_verify_svd_close_matrices(A, U, S, Vh)
    )
