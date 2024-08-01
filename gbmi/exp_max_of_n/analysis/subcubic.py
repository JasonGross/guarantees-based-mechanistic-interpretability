from typing import Union, Optional, Tuple
import time
from functools import reduce
import torch
from tqdm.auto import tqdm
from jaxtyping import Float, Integer
from torch import Tensor
from transformer_lens import HookedTransformer
from gbmi.verification_tools.l1h1 import (
    all_EVOU,
    all_PVOU,
)
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.exp_max_of_n.analysis.quadratic import (
    W_EP_direction_for_tricks,
    find_min_gaps,
    find_EKQE_error_directions,
    W_EP_direction_for_tricks_kwargs,
)
from gbmi.exp_max_of_n.verification.subcubic import decompose_EQKE_error


@torch.no_grad()
def find_min_gaps_with_EQKE(
    model: HookedTransformer,
    *,
    EVOU: Optional[Float[Tensor, "d_vocab_k d_vocab_out"]] = None,  # noqa: F722
    PVOU: Optional[Float[Tensor, "n_ctx d_vocab_out"]] = None,  # noqa: F722
    W_EP: Optional[Float[Tensor, "d_vocab_q d_model"]] = None,  # noqa: F722
    W_U: Optional[Float[Tensor, "d_model d_vocab_out"]] = None,  # noqa: F722
    key_direction: Optional[Tensor] = None,
    query_direction: Optional[Tensor] = None,
    second_key_direction: Optional[Tensor] = None,
    second_query_direction: Optional[Tensor] = None,
    W_Q_U: Optional[Tensor] = None,
    W_K_U: Optional[Tensor] = None,
    sanity_check: bool = True,
    atol: float = 1e-4,
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
    # svd_EUPU: bool = False,
    attn_scale: Optional[Union[Float[Tensor, ""], float]] = None,  # noqa: F722
    position: Optional[int] = None,
    leave: Optional[bool] = None,
    desc: Optional[str] = None,
    sub_pbar: Optional[tqdm] = None,
    pbar: Optional[tqdm] = None,
    record_time: bool = False,
) -> Union[
    Tuple[
        Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"],  # noqa: F722
        float,
    ],
    Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"],  # noqa: F722
]:
    if pbar is not None:
        pbar.update(1)
    duration = 0.0
    start = time.time()
    if EVOU is None:
        EVOU = all_EVOU(model)
    if PVOU is None:
        PVOU = all_PVOU(model)
    if W_EP is None:
        W_EP = model.W_E + model.W_pos.mean(dim=0, keepdim=True)
    if W_U is None:
        W_U = model.W_U
    if attn_scale is None:
        attn_scale = model.blocks[0].attn.attn_scale
    assert attn_scale is not None

    (
        EQKE_query_key,
        EQKE_pos_err,
        (EQKE_err_upper_bound, EQKE_err_matrices),
    ) = decompose_EQKE_error(
        model,
        key_direction=key_direction,
        query_direction=query_direction,
        second_key_direction=second_key_direction,
        second_query_direction=second_query_direction,
        W_Q_U=W_Q_U,
        W_K_U=W_K_U,
        sanity_check=sanity_check,
        atol=atol,
        tricks=tricks,
    )

    cur_EQKE = EQKE_query_key + 0.0  # convert to tensor from low-rank

    W_EP_direction = W_EP_direction_for_tricks(W_EP=W_EP, W_U=W_U, tricks=tricks)
    # cur_EUPU_low_rank = EUPU_lowrank if svd_EUPU else None
    # cur_EUPU_high_rank = torch.zeros_like(EUPU) if svd_EUPU else EUPU
    # cur_EUPU_max_err = torch.tensor(0) if not svd_EUPU else EUPU_err_upper_bound

    result = find_min_gaps(
        EQKE=cur_EQKE,
        EQKE_err_upper_bound=EQKE_err_upper_bound,
        EQKE_pos_err=EQKE_pos_err,
        EVOU=EVOU,
        PVOU=PVOU,
        tricks=tricks,
        attn_scale=attn_scale,
        position=position,
        leave=leave,
        desc=desc,
        pbar=sub_pbar,
        W_EP=W_EP,
        W_U=W_U,
        W_EP_direction=W_EP_direction,
    )
    duration += time.time() - start
    if record_time:
        return result, duration
    return result


@torch.no_grad()
def find_min_gaps_with_EQKE_kwargs(model: HookedTransformer):
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"] = all_EVOU(model)  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
    W_EP: Float[Tensor, "d_vocab_q d_model"] = (  # noqa: F722
        model.W_E + model.W_pos.mean(dim=0, keepdim=True)
    )
    W_U: Float[Tensor, "d_model d_vocab_out"] = model.W_U  # noqa: F722
    attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa: F722
        0
    ].attn.attn_scale
    return {
        "EVOU": EVOU,
        "PVOU": PVOU,
        "W_EP": W_EP,
        "W_U": W_U,
        "attn_scale": attn_scale,
    }


@torch.no_grad()
def find_proof_shared(model: HookedTransformer) -> dict:
    shared_proof_search_duration = 0.0
    start = time.time()
    W_EP_direction_kwargs = W_EP_direction_for_tricks_kwargs(model)
    find_min_gaps_kwargs = find_min_gaps_with_EQKE_kwargs(model)
    size_and_query_directions_kwargs = find_EKQE_error_directions(model)
    shared_proof_search_duration += time.time() - start
    return {
        "W_EP_direction_kwargs": W_EP_direction_kwargs,
        "find_min_gaps_kwargs": find_min_gaps_kwargs,
        "size_and_query_directions_kwargs": size_and_query_directions_kwargs,
        "shared_proof_search_duration": shared_proof_search_duration,
    }


@torch.no_grad()
def find_proof(
    model: HookedTransformer,
    tricks: LargestWrongLogitQuadraticConfig,
    *,
    W_EP_direction_kwargs,
    find_min_gaps_kwargs,
    size_and_query_directions_kwargs,
    shared_proof_search_duration: float = 0,
    record_time: bool = False,
    **find_min_gaps_with_EQKE_extra_kwargs,
) -> Union[dict, Tuple[dict, float]]:
    proof_search_duration: float
    min_gaps, proof_search_duration = find_min_gaps_with_EQKE(
        model=model,
        **find_min_gaps_kwargs,  # type: ignore
        **size_and_query_directions_kwargs,
        tricks=tricks,
        **find_min_gaps_with_EQKE_extra_kwargs,
        record_time=True,
    )
    proof_search_duration += shared_proof_search_duration
    start = time.time()
    W_EP_direction = W_EP_direction_for_tricks(**W_EP_direction_kwargs, tricks=tricks)
    proof_search_duration += time.time() - start

    result = {
        "W_EP_direction": W_EP_direction,
        "min_gaps": min_gaps,
        "tricks": tricks,
        **size_and_query_directions_kwargs,
    }
    if record_time:
        return result, proof_search_duration
    return result
