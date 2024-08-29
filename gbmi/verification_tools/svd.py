from typing import Sequence, Tuple, Union

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
    # print(f"m={m}, n={n}, U={U}, S={S}, Vh={Vh}")
    U = U[..., :m, :min_mn]
    S = S[..., :min_mn]
    Vh = Vh[..., :min_mn, :n]
    # print(f"m={m}, n={n}, U={U}, S={S}, Vh={Vh}")
    if m <= n:
        # S2 = S.unsqueeze(-2)
        # print(f"m={m}, n={n}, S={S}, U={U}, S2={S2}, U * S2 = {U * S2}")
        U = U * S.unsqueeze(-2)
    else:
        # S2 = S.unsqueeze(-1)
        # print(f"m={m}, n={n}, S={S}, Vh={Vh}, S2={S2}, Vh * S2 = {Vh * S2}")
        Vh = Vh * S.unsqueeze(-1)
    USVh = U @ Vh
    result.append((A, USVh))

    # vectors are orthogonal:
    for mat in (U, Vh):
        matH = mat.mH
        if mat.shape[-2] <= mat.shape[-1]:
            matmatH = mat @ matH
        else:
            matmatH = matH @ mat
        assert matmatH.shape[-2:] == (
            min_mn,
            min_mn,
        ), f"matmatH.shape[-2:] == {matmatH.shape[-2:]} != ({min_mn}, {min_mn})"
        result.append((matmatH, torch.eye(min_mn).to(matmatH)))

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
