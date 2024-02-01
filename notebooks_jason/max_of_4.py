# %%
from __future__ import annotations

# %%
import importlib
import gbmi.exp_max_of_n.analysis
import gbmi.analysis_tools.decomp
import gbmi.verification_tools.decomp
import gbmi.utils.lowrank
import gbmi.exp_max_of_n.analysis
import gbmi.exp_max_of_n.plot
import gbmi.exp_max_of_n.verification
import gbmi.utils
import gbmi.utils.memoshelve
import gbmi.utils.sequences

importlib.reload(gbmi.exp_max_of_n.plot)
importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.analysis_tools.decomp)
importlib.reload(gbmi.verification_tools.decomp)
importlib.reload(gbmi.utils.lowrank)
importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.utils)
importlib.reload(gbmi.exp_max_of_n.verification)
importlib.reload(gbmi.utils.memoshelve)
importlib.reload(gbmi.utils.sequences)
# %%
import dataclasses
from collections import defaultdict
from typing import Callable, ClassVar, Collection, Literal, Optional, Tuple, Union
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils.sequences import count_sequences
from gbmi.utils.lowrank import LowRankTensor
from gbmi.utils.memoshelve import memoshelve
from gbmi.exp_max_of_n.analysis import (
    find_second_singular_contributions,
    find_size_and_query_direction,
)
from gbmi.exp_max_of_n.plot import display_basic_interpretation
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    IterableDatasetCfg,
    MaxOfN,
    MaxOfNTrainingWrapper,
    train_or_load_model,
)
from gbmi.model import Config, RunData
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from jaxtyping import Float, Integer, Bool
from torch import Tensor
import plotly.express as px
from transformer_lens import HookedTransformerConfig, HookedTransformer
from gbmi.utils import default_device, dropnan
from gbmi.utils.memocache import Memoize
from gbmi.utils.sequences import (
    SequenceDataset,
    ThunkedDataset,
    generate_all_sequences_for_model,
)
import shelve
from gbmi.verification_tools.decomp import factor_contribution

from gbmi.verification_tools.general import EU_PU
from gbmi.verification_tools.l1h1 import (
    all_EQKE,
    all_EQKP,
    all_EVOU,
    all_PVOU,
    all_attention_scores,
)
from gbmi.verification_tools.utils import complexity_of
from gbmi.utils.hashing import get_hash_ascii

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
# %%
DISPLAY_PLOTS: bool = not IN_COLAB  # @param {type:"boolean"}
# %%
cfg = Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            act_fn=None,
            attn_only=True,
            d_head=32,
            d_mlp=None,
            d_model=32,
            d_vocab=64,
            device="cpu",
            n_ctx=2,
            n_heads=1,
            n_layers=1,
            normalization_type=None,
            seed=613947648,
        ),
        zero_biases=True,
        use_log1p=True,
        use_end_of_sequence=False,
        seq_len=4,
        train_dataset_cfg=IterableDatasetCfg(pick_max_first=False),
        test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
    ),
    deterministic=True,
    seed=123,
    batch_size=128,
    train_for=(3000, "steps"),
)
cfg_hash = get_hash_ascii(cfg)
# %%
runtime, model = train_or_load_model(cfg, force="load")
# %%
training_wrapper = MaxOfNTrainingWrapper(cfg, model)
# training_wrapper.run_batch = Memoize(training_wrapper.run_batch, name=f"{__file__}.training_wrapper.run_batch", use_pandas=False, use_shelf=True)  # type: ignore
# %% [markdown]
# # Brute Force Proof
# %%
all_tokens_dataset = SequenceDataset(
    seq_len=model.cfg.n_ctx, vocab_size=model.cfg.d_vocab
)
# %%
batch_size = 4096  # 16_384 # 8182
# Resetting the DataLoader without shuffle for consistent processing
# data_loader = DataLoader(all_tokens_dataset, batch_size=batch_size, shuffle=False)

# Variables to accumulate total loss and accuracy
total_loss = 0.0
total_accuracy = 0.0
total_samples = 0

brute_force_proof_deterministic: bool = True  # @param {type:"boolean"}


# loop for computing overall loss and accuracy
@torch.no_grad()
def _run_batch_loss_accuracy(i: int, batch_size: int) -> Tuple[float, float, int]:
    batch = all_tokens_dataset[i : i + batch_size]
    size = batch.shape[0]
    batch.to(default_device(deterministic=brute_force_proof_deterministic))
    loss, accuracy = training_wrapper.run_batch(
        batch, return_accuracy=True, log_output=False
    )
    loss = loss.item()
    return loss, accuracy, size


# , get_hash_mem=(lambda x:x[0]), get_hash=str
with memoshelve(
    _run_batch_loss_accuracy,
    filename=f"{__file__}.run_batch_loss_accuracy-{cfg_hash}-{brute_force_proof_deterministic}",
)() as run_batch_loss_accuracy:
    for i in tqdm(range(0, len(all_tokens_dataset), batch_size)):
        loss, accuracy, size = run_batch_loss_accuracy(i, batch_size)  # type: ignore
        # Accumulate loss and accuracy
        total_loss += loss * size
        total_accuracy += accuracy * size
        total_samples += size

# Calculate average loss and accuracy
average_loss = total_loss / total_samples
average_accuracy = total_accuracy / total_samples
# %%
print(f"Brute force proof:")
print(f"Model Accuracy: {average_accuracy * 100}%")
print(
    f"Number Correct Sequences: {int(round(average_accuracy * all_tokens_dataset.length))}"
)
print(
    f"Number Incorrect Sequences: {all_tokens_dataset.length - int(round(average_accuracy * all_tokens_dataset.length))}"
)
print(f"Model Loss: {average_loss}")
# %% [markdown]
# Complexity: $$\mathcal{O}(\text{d\_vocab}^\text{n\_ctx} \cdot \text{n\_ctx} \cdot \text{d\_vocab} \cdot \text{d\_model})$$
# (batch size * number tensors in each sequence * cost of most expensive vector-matrix-multiplication)
# %%
# # %% [markdown]
# # Brute Force Proof (with Memoization)
# TODO: Didn't want to implement it yet
# It's should be possible to get
# $$\mathcal{O}(\text{d\_vocab}^\text{n\_ctx} \cdot \left( \text{n\_ctx} \cdot \text{d\_model} + \text{d\_model} \cdot \text{d\_vocab}\right))$$
# (batch size * (sequence length * longest vector operated on per position + cost of unembed))
#
# Alex and Soufiane also tell me it's possible to do very clever caching and get this down to something like $\mathcal{O}(\text{vocab}^\text{n\_ctx} \cdot \text{d\_model}^2)$
# # %% [markdown]
# # # Important Sequences Only By Convexity
# # If we can iterate over the query tokens and the max tokens and the non-max tokens and do a forward pass for each, we can bound loss by convexity.
# %% [markdown]
# ## Convexity Property
#
# **Lemma**: For a single attention head, it suffices to consider sequences with at most two distinct tokens.
#
# Note that we are comparing sequences by pre-final-layernorm-scaling gap between the logit of the minimum token and the logit of any other fixed token.
# Layernorm scaling is non-linear, but if we only care about accuracy and not log-loss, then we can ignore it (neither scaling nor softmax changes which logit is the largest).
#
# **Proof sketch**:
# We show that any sequence with three token values, $x < y < z$, is strictly dominated either by a sequence with just $x$ and $y$ or a sequence with just $x$ and $z$.
#
# Suppose we have $k$ copies of $x$, $n$ copies of $y$, and $\ell - k - n$ copies of $z$, the attention scores are $s_x$, $s_y$, and $s_z$, and the differences between the logit of $x$ and our chosen comparison logit (as computed by the OV circuit for each token) are $v_x$, $v_y$, and $v_z$.
# Then the difference in logit between $x$ and the comparison token is
# $$\left(k e^{s_x} v_x + n e^{s_y} v_y + (\ell - k - n)e^{s_z}v_z \right)\left(k e^{s_x} + n e^{s_y} + (\ell - k - n)e^{s_z}\right)^{-1}$$
# Rearrangement gives
# $$\left(\left(k e^{s_x} v_x + (\ell - k) e^{s_z} v_z\right) + n \left(e^{s_y} v_y - e^{s_z}v_z\right) \right)\left(\left(k e^{s_x} + (\ell - k) e^{s_z}\right) + n \left(e^{s_y} - e^{s_z}\right)\right)^{-1}$$
# This is a fraction of the form $\frac{a + bn}{c + dn}$.  Taking the derivative with respect to $n$ gives $\frac{bc - ad}{(c + dn)^2}$.  Noting that $c + dn$ cannot equal zero for any valid $n$, we get the the derivative never changes sign.  Hence our logit difference is maximized either at $n = 0$ or at $n = \ell - k$, and the sequence with just two values dominates the one with three.
#
# This proof generalizes straightforwardly to sequences with more than three values.
#
# Similarly, this proof shows that, when considering only a single attention head, it suffices to consider sequences of $\ell$ copies of the minimum token and sequences with one copy of the minimum token and $\ell - 1$ copies of the non-minimum token, as intermediate values are dominated by the extremes.
# %%
# # %%
# # build a map of max_tok -> query_tok -> non_max_tok -> sequence_count, max_loss, accuracy
# batch_size = 4096  # 16_384 # 8182
# loss_accuracy_cubic_memcache = {}

# run_batch_cubic_shelf_name = f"{__file__}.run_batch_cubic_shelf"
# results = {}
# with shelve.open(run_batch_cubic_shelf_name) as shelf:
#     with torch.no_grad():
#         for i in tqdm(range(0, len(all_tokens_dataset), batch_size)):
#             try:
#                 mapping = loss_accuracy_cubic_memcache[(i, batch_size)]
#             except KeyError:
#                 key = f"{i}_{batch_size}"
#                 try:
#                     mapping = loss_accuracy_cubic_memcache[(i, batch_size)] = shelf[key]
#                 except KeyError:
#                     batch = all_tokens_dataset[i : i + batch_size]
#                     # Mask to identify rows with duplicates
#                     mask = torch.tensor([row.unique().size(0) < 4 for row in batch])
#                     batch = batch[mask]
#                     batch.to(default_device(deterministic=True))
#                     q_toks = batch[:, -1]
#                     max_non_qtoks = batch[:, :-1].max(dim=-1).values
#                     min_non_qtoks = batch[:, :-1].min(dim=-1).values
#                     mapping = {}
#                     for q_tok in set(q_toks.tolist()):
#                         for max_non_qtok in set(max_non_qtoks.tolist()):
#                             for min_non_qtok in set(min_non_qtoks.tolist()):
#                                 subbatch = batch[
#                                     (q_toks == q_tok)
#                                     & (max_non_qtoks == max_non_qtok)
#                                     & (min_non_qtoks == min_non_qtok)
#                                 ]
#                                 if subbatch.shape[0] == 0:
#                                     continue
#                                 size = subbatch.shape[0]
#                                 loss, accuracy = training_wrapper.run_batch(
#                                     subbatch, return_accuracy=True, log_output=False
#                                 )
#                                 loss = loss.item()
#                                 mapping[(q_tok, max_non_qtok, min_non_qtok)] = (
#                                     loss,
#                                     accuracy,
#                                     size,
#                                 )
#                     loss_accuracy_cubic_memcache[(i, batch_size)] = shelf[key] = mapping
#             for key, (loss, accuracy, size) in mapping.items():
#                 cur_loss, cur_accuracy, cur_size = results.get(key, (0, True, 0))
#                 results[key] = (
#                     max(cur_loss, loss),
#                     cur_accuracy and (int(round(accuracy * size)) == size),
#                     cur_size + size,
#                 )


# %% [markdown]
# # Cubic proof
# Target Complexity: $$\mathcal{O}(\text{d\_vocab}^3 \cdot \text{n\_ctx}^2)$$
#
# This will leave us enough room to compute out all the matrices.
# We will get to run computations on $\text{d\_vocab}^2$ sequences, iterating over the query and max tokens.
# %%
EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = EU_PU(model)  # noqa: F722
print(f"Complexity of EU_PU: {complexity_of(EU_PU)}")  # O(d_vocab^2 * d_model)
EVOU: Float[Tensor, "d_vocab d_vocab_out"] = all_EVOU(model)  # noqa: F722
print(f"Complexity of EVOU: {complexity_of(all_EVOU)}")  # O(d_vocab^2 * d_model)
PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
print(f"Complexity of PVOU: {complexity_of(all_PVOU)}")  # O(n_ctx * d_vocab * d_model)
# EPQKEP: Float[
#     Tensor, "n_ctx_k d_vocab_q d_vocab_k"  # noqa: F722
# ] = all_attention_scores(model)
# print(
#     f"Complexity of (E+P[-1])QKEP: {complexity_of(all_attention_scores)}"
# )  # O(d_vocab^2 * d_model * n_ctx)
EQKE: Float[Tensor, "d_vocab_q d_vocab_k"] = all_EQKE(model)  # noqa: F722
print(f"Complexity of EQKE: {complexity_of(all_EQKE)}")  # O(d_vocab^2 * d_model)
EQKP: Float[Tensor, "d_vocab_q n_ctx_k"] = all_EQKP(model)  # noqa: F722
print(f"Complexity of EQKP: {complexity_of(all_EQKP)}")  # O(d_vocab * d_model * n_ctx)


# %%
# for q_tok in tqdm(range(model.cfg.d_vocab)):
#     for max_non_qtok in range(model.cfg.d_vocab):
#         for min_non_qtok in range(model.cfg.d_vocab):
# TODO COMPUTATION
# START HERE
# %%
@torch.no_grad()
def compute_min_softmaxed_right_attention_cubic_simple(
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    EQKP: Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa F722
        0
    ].attn.attn_scale,
    position: Optional[int] = None,
) -> Float[
    Tensor,
    "attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
]:
    # TODO: return both min and max attention to query so the proof goes through
    r"""
    Computes the min post-softmax attention (pessimized over sequence orderings) paid to the maximum token (attn=0) and
    the min paid to the query token (attn=1) and the max paid to the query token (attn=2):
        - by each possible value of the query token,
        - for each possible value of the max token,
        - for each possible value of the nonmax token,
        - for each number of copies of the non max token

    Basically, this attempts to lower bound the attention that is paid to the max token and the query token, by
    pessimising over the order of the non-query tokens.

    Note that we could do a bit of a better job than this by taking in min_gap and a pre-computed extreme_attention
    matrix, if we wanted to.

    Time Complexity: O(d_vocab^3 * n_ctx^2)

    Preconditions:
        . attn_scale is correct for the model (\sqrt(d_head) by default)
        . EQKE[q, k] is the attention paid from query q to key token k
        . EQKP[q, p] is the attention paid from query q to key position p
    Postconditions:
        \forall w \in {0,1,2}, q, m, k, n_copies_nonmax: ("w" for which)
          if q > m or k > m: return[w, q, m, k, n_copies_nonmax] = nan
                (That is, the answer is undefined if the query token is greater than the max token, or if the non-max
                token is greater than the max token)
          elif m = k and n_copies_nonmax != 0: return[w, q, m, k, n_copies_nonmax] = nan
                (That is, the answer is undefined if the non-max token is equal to the max token and there are non-zero
                copies of non-max tokens)
          elif q != m and n_copies_nonmax >= n_ctx - 1: return[w, q, m, k, n_copies_nonmax] = nan
                (That is, the answer is undefined if the query token is not equal to the max token and there are n_ctx
                - 1 or more copies of the non-max token, because then the max token would be missing)
          else: amongst all permutations of [the non-query-tokens in] the sequence with query q, n_copies_nonmax copies
                of k, and all other tokens equal to m:
                return[0, q, m, k, n_copies_nonmax] <= post-softmax attention paid to max token m
                return[1, q, m, k, n_copies_nonmax] <= post-softmax attention paid to query token q <= return[2, q, m, k, n_copies_nonmax]

    """
    d_vocab, n_ctx = EQKE.shape[-1], EQKP.shape[-1]
    result = torch.zeros((2, d_vocab, d_vocab, d_vocab, n_ctx)).to(EQKE) + float("nan")
    tmp = torch.zeros((n_ctx,)).to(EQKE)
    # constants for indices so we don't have 0 and 1 floating around
    w_max = 0
    w_qry_min = 1
    w_qry_max = 2
    # we sort EQKP so that higher-attention positions are at the back, so we can put the max token at the front.
    EQKP, EQKPm1 = EQKP[:, :-1].sort(dim=-1).values, EQKP[:, -1]

    for max_tok in tqdm(range(d_vocab), desc="max_tok", position=position):
        for q_tok in range(max_tok + 1):
            tmp[-1] = EQKE[q_tok, q_tok] + EQKPm1[q_tok]
            for k_tok in range(max_tok + 1):
                if k_tok == max_tok:
                    if q_tok == max_tok:
                        # only max tok, so we pay 100% attention to it
                        result[w_max, q_tok, max_tok, k_tok, 0] = 1
                        result[w_qry_min, q_tok, max_tok, k_tok, 0] = 0
                        result[w_qry_max, q_tok, max_tok, k_tok, 0] = 0
                        continue
                    tmp[:-1] = EQKP[q_tok] + EQKE[q_tok, k_tok]
                    tmp_sm = (tmp / attn_scale).softmax(dim=-1)
                    result[w_max, q_tok, max_tok, k_tok, 0] = tmp_sm[:-1].sum()
                    result[w_qry_min, q_tok, max_tok, k_tok, 0] = tmp_sm[-1]
                    result[w_qry_max, q_tok, max_tok, k_tok, 0] = tmp_sm[-1]
                    continue
                for n_copies_nonmax in range(n_ctx):
                    n_copies_max_nonquery = n_ctx - n_copies_nonmax - 1
                    if q_tok != max_tok and n_copies_nonmax >= n_ctx - 1:
                        continue
                    tmp[:-1] = EQKP[q_tok]

                    tmp[:n_copies_max_nonquery] += EQKE[q_tok, max_tok]
                    tmp[n_copies_max_nonquery:-1] += EQKE[
                        q_tok, k_tok
                    ]  # attention paid to non-max tokens other than in the query position
                    tmp_sm = (tmp / attn_scale).softmax(dim=-1)
                    result[w_max, q_tok, max_tok, k_tok, n_copies_nonmax] = tmp_sm[
                        :n_copies_max_nonquery
                    ].sum() + (tmp_sm[-1] if q_tok == max_tok else 0)
                    result[w_qry_min, q_tok, max_tok, k_tok, n_copies_nonmax] = result[
                        w_qry_max, q_tok, max_tok, k_tok, n_copies_nonmax
                    ] = (tmp_sm[-1] if q_tok != max_tok else 0)
    return result


# @torch.no_grad()
# def compute_extrema_softmaxed_attention_cubic(
#     EQKE: Float[Tensor, "n_ctx_k d_vocab_q d_vocab_max"],  # noqa: F722
#     attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa F722
#         0
#     ].attn.attn_scale,
# ) -> Float[
#     Tensor,
#     "order=5 attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
# ]:
#     r"""
#     Computes the extreme post-softmax attention paid to the maximum (attn=0), non-maximum (attn=1), and query (attn=2) tokens by each query token, for each number of copies of the non max token.
#     Note that we could do a bit of a better job than this by taking in min_gap and a pre-computed extreme_attention matrix, if we wanted to.

#     The order determines which extreme we're at acconding to positional encoding:
#     0: max > nonmax
#     1: max > query > nonmax

#     min_gap is used only to determine when the result should be nan

#     Complexity: O(d_vocab^2 * n_ctx^2)

#     Preconditions:
#         Define:
#         EQKE[q, p, k] := (W_E[q] + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E[k] + W_pos[p]).T
#         Then we demand:
#         . \forall q, m, p1, p2, k:
#           if ((q == m) or (q <= m - min_gap[q, m])) and (k <= m - min_gap[q, m]):
#             min_right_attention[q, m] + EQKE_pos_error[q, p1] - EKQE_pos_error[q, p2]
#             <= EQKE[q, p1, m] - EQKE[q, p2, k]
#     Postconditions:
#         \forall q, m, n_copies_nonmax:
#           if q > m: return[q, m, n_copies_nonmax] = nan
#           elif m - min_gap[q, m] < q < m: return[q, m, n_copies_nonmax] = nan
#           elif m < min_gap[q, m] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
#           elif m < min_gap[q, m]: return[q, m, 0] = nan
#           else: return[q, m, n_copies_nonmax] <= post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m, and all other tokens <= m - min_gap[q, m]
#     """


# %%
# min_gap = 1
min_right_attention_softmaxed_cubic: Float[
    Tensor,
    "attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
] = compute_min_softmaxed_right_attention_cubic_simple(
    EQKE=EQKE,
    EQKP=EQKP,
    attn_scale=model.blocks[0].attn.attn_scale,
)
print(
    f"Complexity of compute_min_softmaxed_right_attention_cubic_simple: {complexity_of(compute_min_softmaxed_right_attention_cubic_simple)}"
)  # O(d_vocab^3 * n_ctx^2)
# print(
#     (min_right_attention[~min_right_attention.isnan()] > err_upper_bound).sum().item()
# )
# min_right_attention_softmaxed = compute_min_softmaxed_right_attention(
#     min_right_attention - err_upper_bound, EQKE_pos_err, min_gap=1
# )


# %%
@torch.no_grad()
def compute_largest_wrong_logit_cubic(
    min_softmaxed_right_attention: Float[
        Tensor,
        "attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
    ],
    *,
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    # permitted_nonmax_tokens: Optional[Bool[
    #     Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    # ]] = None,
) -> Float[
    Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
]:
    r"""
        Computes the largest difference between the wrong logit and the right logit for each query token, max token, nonmax
        token, and number of copies of the non-max token.

        Complexity: O(d_vocab^3 * n_ctx^2)

        Preconditions:
            min_softmaxed_right_attention satisfies the postcondition of compute_min_softmaxed_right_attention_cubic_simple
                (a lower bound on the post-softmax attention paid to the max token and the query token,
                by each possible value of the query token,
                for each possible value of the max token,
                for each possible value of the non-max token,
                for each number of copies of the non max token)
            EUPU = (W_E + W_pos[-1]) @ W_U
            EVOU = W_E @ W_V @ W_O @ W_U
            PVOU = W_pos @ W_V @ W_O @ W_U
        Postconditions:
            \forall q, m, k, n_copies_nonmax, x:
              if q > m or k > m: return[q, m, k, n_copies_nonmax] = nan
              elif m = k and n_copies_nonmax != 0: return[q, m, k, n_copies_nonmax] = nan
              elif q != m and n_copies_nonmax >= n_ctx - 1: return[q, m, k, n_copies_nonmax] = nan
                (That is, in these cases, the answer is undefined because the query token is greater than the max token, or
                there isn't enough room for the non-max token or the max token in the sequence.)
              else: for all sequences with query q, max token m, n_copies_nonmax [non-query] copies of k (and the rest of
                the non-query tokens equal to m), we have:
                return[q, m, k, n_copies_nonmax] <= model(sequence)[-1, x] - model(sequence)[-1, m]
                That is, we return a lower bound on the difference between the wrong logit and the right logit for this
                combination of query token, max token, non-max token, and number of copies of the non-max token.

    The main idea here is that by pessimizing over the positional attention independently of the embedding attention, we can
        later use a convexity argument for the embedding attention.
    """
    # if permitted_nonmax_tokens is None:
    #     permitted_nonmax_tokens = torch.ones_like(min_softmaxed_right_attention[0], dtype=torch.bool)
    results = torch.zeros_like(min_softmaxed_right_attention[0, :, :, :, :]) + float(
        "nan"
    )
    _, d_vocab, _, _, n_ctx = min_softmaxed_right_attention.shape
    w_max = 0
    w_qry_min = 1
    w_qry_max = 2
    for max_tok in range(d_vocab):
        # center PVOU according to max token, O(d_vocab * n_ctx)
        PVOU -= PVOU[:, max_tok].unsqueeze(-1)
        # center EUPU according to max token, O(d_vocab^2)
        EUPU -= EUPU[:, max_tok].unsqueeze(-1)
        # center EVOU according to max token, O(d_vocab^2)
        EVOU -= EVOU[:, max_tok].unsqueeze(-1)
        # to make convexity go through, we need to consider a larger region of phase space bounded by points where
        # positional attention is independent of token attention.
        # Here we pessimize over positional attention, assuming that we pay 100% of attention to the worst position
        PVOU_pessimized: Float[Tensor, "d_vocab_out"] = PVOU.max(  # noqa: F821
            dim=0
        ).values

        # handle the case with only the max token
        logits_only_max: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
            EUPU[max_tok, :] + EVOU[max_tok, :] + PVOU_pessimized
        )
        logits_only_max -= logits_only_max[max_tok].item()
        logits_only_max[max_tok] = float(
            "-inf"
        )  # so we can max the logits across the non-max tokens
        results[max_tok, max_tok, 0] = logits_only_max.max().item()

        # now handle the cases with only the query token and n_ctx - 1 copies of the max token
        for q_tok in range(max_tok):
            cur_min_right_attention = min_softmaxed_right_attention[
                :, q_tok, max_tok, max_tok, 0
            ]
            # N.B. because EVOU[q_tok, max_tok] == 0 by centering above, we just take the maximum attention paid to the query
            logits_only_q_and_max: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
                EUPU[q_tok, :]
                + PVOU_pessimized
                + EVOU[max_tok, :] * cur_min_right_attention[w_max]
                + EVOU[q_tok, :] * cur_min_right_attention[w_qry_max]
            )
            logits_only_q_and_max -= logits_only_q_and_max[max_tok].item()
            logits_only_q_and_max[max_tok] = float("-inf")
            results[q_tok, max_tok, max_tok, 0] = logits_only_q_and_max.max().item()

        # precompose pessimization for EUPU over output logit, so we have enough compute budget
        EUPU_tmp: Float[
            Tensor, "d_vocab_q d_vocab_out"  # noqa: F722
        ] = EUPU.detach().clone()
        EUPU_tmp[:, max_tok] = float("-inf")
        EUPU_per_query_pessimized: Float[
            Tensor, "d_vocab_q"  # noqa: F821
        ] = EUPU_tmp.max(dim=-1).values

        # Ditto for EVOU
        # distribute PVOU over EVOU, to avoid premature pessimization
        # TODO: mean+diff
        EPVOU_tmp: Float[Tensor, "d_vocab_k d_vocab_out"] = (  # noqa: F722
            EVOU + PVOU_pessimized
        )
        EPVOU_tmp[:, max_tok] = float("-inf")
        EPVOU_per_key_pessimized: Float[
            Tensor, "d_vocab_k"  # noqa: F821
        ] = EPVOU_tmp.max(dim=-1).values

        # now handle the cases with at least one non-max non-query token
        for nonmax_tok in range(max_tok):
            for n_copies_nonmax in range(1, n_ctx):
                # distribute PVOU over EVOU, to avoid premature pessimization

                # pessimize over the thing we're not supposed to be paying attention to (w.r.t. the token that is non-max that we're paying attention)
                # maximum added to the wrong logit from paying attention to the wrong thing
                wrong_attention_logits: Float[
                    Tensor, ""  # noqa: F722
                ] = EPVOU_per_key_pessimized[nonmax_tok]

                # pessimize also over the thing we are paying attention to
                right_attention_wrong_logits: Float[
                    Tensor, ""  # noqa: F722
                ] = EPVOU_per_key_pessimized[max_tok]

                for q_tok in range(max_tok + 1):
                    if q_tok != max_tok and n_copies_nonmax >= n_ctx - 1:
                        continue
                    query_wrong_logits: Float[
                        Tensor, ""  # noqa: F722
                    ] = EPVOU_per_key_pessimized[q_tok]
                    right_attn = min_softmaxed_right_attention[
                        w_max, q_tok, max_tok, nonmax_tok, n_copies_nonmax
                    ]
                    q_attn_min, q_attn_max = (
                        min_softmaxed_right_attention[
                            w_qry_min, q_tok, max_tok, nonmax_tok, n_copies_nonmax
                        ],
                        min_softmaxed_right_attention[
                            w_qry_max, q_tok, max_tok, nonmax_tok, n_copies_nonmax
                        ],
                    )
                    wrong_attn_minquery = 1 - right_attn - q_attn_min
                    wrong_attn_maxquery = 1 - right_attn - q_attn_max
                    results[q_tok, max_tok, nonmax_tok, n_copies_nonmax] = (
                        EUPU_per_query_pessimized[q_tok]
                        + right_attn * right_attention_wrong_logits
                        + torch.max(
                            q_attn_min * query_wrong_logits
                            + wrong_attn_minquery * wrong_attention_logits,
                            q_attn_max * query_wrong_logits
                            + wrong_attn_maxquery * wrong_attention_logits,
                        )
                    ).item()
    return results


# %%
# min_right_attention_softmaxed_cubic: Float[
#     Tensor,
#     "attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
# ] = compute_min_softmaxed_right_attention_cubic_simple(
#     EQKE=EQKE,
#     EQKP=EQKP,
#     attn_scale=model.blocks[0].attn.attn_scale,
# )
print(
    f"Complexity of compute_min_softmaxed_right_attention_cubic_simple: {complexity_of(compute_min_softmaxed_right_attention_cubic_simple)}"
)  # O(d_vocab^3 * n_ctx^2)
EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = EU_PU(model)  # noqa: F722
print(f"Complexity of EU_PU: {complexity_of(EU_PU)}")  # O(d_vocab^2 * d_model)
EVOU: Float[Tensor, "d_vocab d_vocab_out"] = all_EVOU(model)  # noqa: F722
print(f"Complexity of EVOU: {complexity_of(all_EVOU)}")  # O(d_vocab^2 * d_model)
PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
print(f"Complexity of PVOU: {complexity_of(all_PVOU)}")  # O(n_ctx * d_vocab * d_model)
largest_wrong_logit_cubic: Float[
    Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
] = compute_largest_wrong_logit_cubic(
    min_right_attention_softmaxed_cubic,
    EUPU=EUPU,
    EVOU=EVOU,
    PVOU=PVOU,
)
print(
    f"Complexity of compute_largest_wrong_logit_cubic: {complexity_of(compute_largest_wrong_logit_cubic)}"
)  # O(d_vocab^3 * n_ctx^2)


# %%
# @torch.no_grad()
# def find_min_gaps(
#     *,
#     EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
#     EQKE_err_upper_bound: Union[float, Float[Tensor, ""]],  # noqa: F722
#     EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
#     EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa: F722
#     EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
#     PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
#     tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
#     position: Optional[int] = None,
# ) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
#     """
#     Run the argument across all possible min_gaps, and return the min_gap that works for each query token and max token.

#     Since here we are finding the argument/proof rather than verifying it, the complexity does not matter.
#     """
#     d_vocab_q, d_vocab_k = EQKE.shape
#     n_ctx, d_vocab_out = PVOU.shape
#     min_gaps = torch.ones((d_vocab_q, d_vocab_k, n_ctx), dtype=torch.long)
#     for min_gap in tqdm(list(reversed(range(1, d_vocab_k))), position=position):
#         min_right_attention = compute_min_right_attention_quadratic(
#             EQKE, min_gap=min_gap
#         )
#         min_right_attention_softmaxed = compute_min_softmaxed_right_attention_quadratic(
#             min_right_attention - EQKE_err_upper_bound, EQKE_pos_err, min_gap=min_gap
#         )
#         largest_wrong_logit = compute_largest_wrong_logit_quadratic(
#             min_right_attention_softmaxed,
#             EUPU=EUPU,
#             EVOU=EVOU,
#             PVOU=PVOU,
#             min_gap=min_gap,
#             tricks=tricks,
#         )
#         # if the largest wrong logit is negative, then this gap works
#         min_gaps[largest_wrong_logit < 0] = min_gap

#     return min_gaps


@torch.no_grad()
def count_correct_sequences_cubic(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    ],
) -> int:
    d_vocab_q, d_vocab_max, d_vocab_nonmax, n_ctx = largest_wrong_logit.shape
    correct_count = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            for n_copies_nonmax in range(n_ctx):
                if q_tok > max_tok or (
                    q_tok != max_tok and n_copies_nonmax >= n_ctx - 1
                ):
                    continue
                if n_copies_nonmax == 0:
                    correct_count += 1
                else:
                    cur_largest_wrong_logit = largest_wrong_logit[
                        q_tok, max_tok, :max_tok, n_copies_nonmax
                    ]  # consider wrong logits only when non-max token is less than max token
                    num_nonmax_tok_choices = cur_largest_wrong_logit[
                        ~cur_largest_wrong_logit.isnan() & (cur_largest_wrong_logit < 0)
                    ].size(0)
                    # cur_largest_wrong_logit < 0 -> the model gets it right (and the non-nan just ensures its valid)
                    # N.B. Here, n_copies_nonmax does NOT include the query token
                    correct_count += count_sequences(
                        n_ctx - 1, n_copies_nonmax, num_nonmax_tok_choices
                    )
                    # consider the space where there's one dimension for each input token that controls "do we use this or not"
                    # we consider a subspace where "did the model get it right" is convex in this space
                    # i.e. if you get it for non-max token = x and non-max token = y, you get it for all sequences
                    # that contain any combination of x and y (note that there can only be 1, 2, or 3 non-max tokens in
                    # the sequence (where 3 occurs only when the query token is the max token))
                    # You can extend this to 3 non-max token choices by induction

    return correct_count


def compute_accuracy_lower_bound_from_cubic(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    ],
) -> Tuple[float, Tuple[int, int]]:
    """
    returns correct_count / total_sequences, (correct_count, total_sequences)
    """
    d_vocab_q, d_vocab_max, _, n_ctx = largest_wrong_logit.shape
    correct_count = count_correct_sequences_cubic(largest_wrong_logit)
    total_sequences = d_vocab_max**n_ctx
    return correct_count / total_sequences, (correct_count, total_sequences)


# %%
# EQKE: Float[Tensor, "d_vocab_q d_vocab_k"] = all_EQKE(model)  # noqa: F722
print(f"Complexity of EQKE: {complexity_of(all_EQKE)}")  # O(d_vocab^2 * d_model)
# EQKP: Float[Tensor, "d_vocab_q n_ctx_k"] = all_EQKP(model)  # noqa: F722
print(f"Complexity of EQKP: {complexity_of(all_EQKP)}")  # O(d_vocab * d_model * n_ctx)
# min_right_attention_softmaxed_cubic: Float[
#     Tensor,
#     "attn=3 d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_copies_nonmax",  # noqa: F722
# ] = compute_min_softmaxed_right_attention_cubic_simple(
#     EQKE=EQKE,
#     EQKP=EQKP,
#     attn_scale=model.blocks[0].attn.attn_scale,
# )
print(
    f"Complexity of compute_min_softmaxed_right_attention_cubic_simple: {complexity_of(compute_min_softmaxed_right_attention_cubic_simple)}"
)  # O(d_vocab^3 * n_ctx^2)
EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = EU_PU(model)  # noqa: F722
print(f"Complexity of EU_PU: {complexity_of(EU_PU)}")  # O(d_vocab^2 * d_model)
EVOU: Float[Tensor, "d_vocab d_vocab_out"] = all_EVOU(model)  # noqa: F722
print(f"Complexity of EVOU: {complexity_of(all_EVOU)}")  # O(d_vocab^2 * d_model)
PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
print(f"Complexity of PVOU: {complexity_of(all_PVOU)}")  # O(n_ctx * d_vocab * d_model)
# largest_wrong_logit_cubic: Float[
#     Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
# ] = compute_largest_wrong_logit_cubic(
#     min_right_attention_softmaxed_cubic,
#     EUPU=EUPU,
#     EVOU=EVOU,
#     PVOU=PVOU,
# )
print(
    f"Complexity of compute_largest_wrong_logit_cubic: {complexity_of(compute_largest_wrong_logit_cubic)}"
)  # O(d_vocab^3 * n_ctx^2)
accuracy_bound_cubic, (
    correct_count_cubic,
    total_sequences,
) = compute_accuracy_lower_bound_from_cubic(largest_wrong_logit_cubic)
print(
    f"Accuracy lower bound: {accuracy_bound_cubic} ({correct_count_cubic} correct sequences of {total_sequences})"
)

# # %%


# %% [markdown]
# # Plots
# %%
if DISPLAY_PLOTS:
    display_basic_interpretation(model)


# %% [markdown]
# # Size-Query analysis
#
# We find the size direction and the query direction, and approximate the QK computation using only these vectors.  Then we'll look at the error terms.
#
# We compute as follows:
# $$
# \begin{align*}
# \overline{W_\text{pos}} & := W_\text{pos}\text{.mean}(\text{dim}=0) \\
# \widetilde{E_q} & := W_E + W_\text{pos}[-1] \\
# \widetilde{E_k} & := W_E + \overline{W_\text{pos}} \\
# \text{EQKE}_p
# & := \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \widetilde{E_q}W_QW_K^T(W_{\text{pos}}[p] - \overline{W_\text{pos}})^T \\
# & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \text{EQKE\_pos\_err}
# \end{align*}
# $$
# We can decompose $\widetilde{E_k}$ as a sum of a rank 1 matrix in the given key direction and a matrix orthogonal to the key direction, say $E_k = E_{k,\text{key}} + E_{k,\text{key}}^\perp$.
# We can decompose $\widetilde{E_q}$ as a sum of a rank 1 matrix in the given query direction and a matrix orthogonal to the query direction, say $E_q = E_{q,\text{query}} + E_{q,\text{query}}^\perp$.
# We can decompose $E_{k,\text{key}}^\perp$, $E_{q,\text{query}}^\perp$, $W_Q$, and $W_K$ as sums of rank 1 matrices in the second key direction, second query direction, W\_Q\_U, and W\_K\_U, respectively.
# $$
# \begin{align*}
# E_{k,\text{key}}^\perp & = E_{k,\text{key},\text{second}} + E_{k,\text{key},\text{second}}^\perp \\
# E_{q,\text{query}}^\perp & = E_{q,\text{query},\text{second}} + E_{q,\text{query},\text{second}}^\perp \\
# W_Q & = W_{Q,\text{U}} + W_{Q,\text{U}}^\perp \\
# W_K & = W_{K,\text{U}} + W_{K,\text{U}}^\perp
# \end{align*}
# $$
# Then we can write
# $$
# \begin{align*}
# \text{EQKE}_p - \text{EQKE\_pos\_err}
# & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T \\
# & = E_{k,\text{key}}W_QW_K^T E_{q,\text{query}}^T \\
# & \phantom{{}={}}{} + E_{k,\text{key}}W_QW_K^T {E_{q,\text{query}}^\perp}^T + E_{k,\text{key}}^\perp W_QW_K^T E_{q,\text{query}}^T \\
# & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}W_QW_K^T E_{q,\text{query},\text{second}}^T \\
# & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}W_QW_K^T {E_{q,\text{query},\text{second}}^\perp}^T + E_{k,\text{key},\text{second}}^\perp W_QW_K^T E_{q,\text{query},\text{second}}^T \\
# & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}W_{K,\text{U}}^T {E_{q,\text{query},\text{second}}^\perp}^T \\
# & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}^\perp W_{K,\text{U}}^T {E_{q,\text{query},\text{second}}^\perp}^T + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}} {W_{K,\text{U}}^\perp}^T {E_{q,\text{query},\text{second}}^\perp}^T \\
# & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}^\perp {W_{K,\text{U}}^\perp}^T {E_{q,\text{query},\text{second}}^\perp}^T
# \end{align*}
# $$
# Except for the last line, all of these components are rank 1 matrices, and we can compute them efficiently.
# We compute an upper bound on what the final component can contribute to differences in elements in the same row:
# Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt{2} \sigma_1(M)$.
# This is the value we return, computing an upper bound on the first singular value by multiplying the first singular values of each matrix.
# %%
# This stuff is old, probably delete
# Define
# - $W_{E,\text{qerr}}=(W_E - \text{query} W_E)$
# - $W_{E,\text{kerr}}=(W_E - \text{size} W_E)$
# - $\overline{W_{\text{pos}}} = W_{\text{pos}}.\text{mean}(\text{dim}=0)$
# - $W_{\text{pos},\text{err}} = W_{\text{pos}} - \overline{W_\text{pos}}$
# $$\begin{align*}
# & \phantom{{}={}} (W_E + W_\text{pos}[-1])W_Q W_K^T (W_E + W_\text{pos})^T \\
# & = (\text{query} W_E + W_\text{pos}[-1] + W_{E,\text{qerr}})W_Q W_K^T (\text{size}W_E + \overline{W_\text{pos}} + W_{E,\text{kerr}} + W_{\text{pos},\text{err}})^T \\
# & = (\text{query} W_E + W_\text{pos}[-1])W_Q W_K^T(\text{size}W_E + \overline{W_\text{pos}}) \\
# & \phantom{{}={}}{} + W_{E,\text{qerr}}W_Q W_K^T (W_{E,\text{kerr}} + W_{\text{pos},\text{err}})^T \\
# & \phantom{{}={}}{} + (\text{query} W_E + W_\text{pos}[-1])W_Q W_K^T (W_{E,\text{kerr}} + W_{\text{pos},\text{err}})^T \\
# & \phantom{{}={}}{} + W_{E,\text{qerr}}W_Q W_K^T (\text{size}W_E + \overline{W_\text{pos}})^T \\
# \end{align*}$$
# %%
# %%
@torch.no_grad()
def decompose_EQKE_error(
    model: HookedTransformer,
    *,
    key_direction: Tensor,
    query_direction: Tensor,
    second_key_direction: Tensor,
    second_query_direction: Tensor,
    W_Q_U: Tensor,
    W_K_U: Tensor,
    sanity_check: bool = True,
    atol: float = 1e-4,
) -> Tuple[
    Tuple[
        Float[LowRankTensor, "d_vocab_q d_vocab_k"],  # noqa: F722
        Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    ],
    Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    Tuple[
        Float[Tensor, ""],  # noqa: F722
        Tuple[
            Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
            Float[Tensor, "d_model d_model"],  # noqa: F722
            Float[Tensor, "d_model d_model"],  # noqa: F722
            Float[Tensor, "d_model d_vocab_k"],  # noqa: F722
        ],
    ],
]:
    r"""
    Returns:
        ((EQKE_query_key, err_accumulator), EQKE_pos_err, (remaining_error_upper_bound, four matrices whose product is the exact remaining error))
    where
        EQKE_query_key is the rank 1 approximation of the query-key contribution to the EQKE matrix
        err_accumulator is the sum of the efficiently-computable (O(d_vocab^2)) error terms
        EQKE_pos_err is the contribution of the position embeddings to the error
        remaining_error_upper_bound is a bound on the maximum difference between two elements in the same row of EQKE

    Note that EQKE is actually computed as (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0, keepdim=True)).T

    Complexity: O(d_vocab * (d_vocab + d_model * n_ctx) + d_vocab * d_model^2)

    The d_model^2 term comes from having to do SVD to compute remaining_error_upper_bound

    Preconditions:
        (none)
    Postconditions:
        Define err := EQKE - (EQKE_query_key + err_accumulator)
        Then we guarantee:
        . max_{i,j} err_{r, i} - err_{r, j} <= remaining_error_upper_bound
        . EQKE_pos_err[p] := (W_E + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_pos[p] - W_pos.mean(dim=0, keepdim=True)).T

    EQKE_query_key uses key_direction and query_direction for the rank 1 approximation

    We compute as follows:
    $$
    \begin{align*}
    \overline{W_\text{pos}} & := W_\text{pos}\text{.mean}(\text{dim}=0) \\
    \widetilde{E_q} & := W_E + W_\text{pos}[-1] \\
    \widetilde{E_k} & := W_E + \overline{W_\text{pos}} \\
    \text{EQKE}_p
    & := \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \widetilde{E_q}W_QW_K^T(W_{\text{pos}}[p] - \overline{W_\text{pos}})^T \\
    & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \text{EQKE\_pos\_err}
    \end{align*}
    $$
    We can decompose $\widetilde{E_k}$ as a sum of a rank 1 matrix in the given key direction and a matrix orthogonal to the key direction, say $E_k = E_{k,\text{key}} + E_{k,\text{key}}^\perp$.
    We can decompose $\widetilde{E_q}$ as a sum of a rank 1 matrix in the given query direction and a matrix orthogonal to the query direction, say $E_q = E_{q,\text{query}} + E_{q,\text{query}}^\perp$.
    We can decompose $E_{k,\text{key}}^\perp$, $E_{q,\text{query}}^\perp$, $W_Q$, and $W_K$ as sums of rank 1 matrices in the second key direction, second query direction, W\_Q\_U, and W\_K\_U, respectively.
    $$
    \begin{align*}
    E_{k,\text{key}}^\perp & = E_{k,\text{key},\text{second}} + E_{k,\text{key},\text{second}}^\perp \\
    E_{q,\text{query}}^\perp & = E_{q,\text{query},\text{second}} + E_{q,\text{query},\text{second}}^\perp \\
    W_Q & = W_{Q,\text{U}} + W_{Q,\text{U}}^\perp \\
    W_K & = W_{K,\text{U}} + W_{K,\text{U}}^\perp
    \end{align*}
    $$
    Then we can write
    $$
    \begin{align*}
    \text{EQKE}_p - \text{EQKE\_pos\_err}
    & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T \\
    & = E_{k,\text{key}}W_QW_K^T E_{q,\text{query}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key}}W_QW_K^T {E_{q,\text{query}}^\perp}^T + E_{k,\text{key}}^\perp W_QW_K^T E_{q,\text{query}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}W_QW_K^T E_{q,\text{query},\text{second}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}W_QW_K^T {E_{q,\text{query},\text{second}}^\perp}^T + E_{k,\text{key},\text{second}}^\perp W_QW_K^T E_{q,\text{query},\text{second}}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}W_{K,\text{U}}^T {E_{q,\text{query},\text{second}}^\perp}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}^\perp W_{K,\text{U}}^T {E_{q,\text{query},\text{second}}^\perp}^T + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}} {W_{K,\text{U}}^\perp}^T {E_{q,\text{query},\text{second}}^\perp}^T \\
    & \phantom{{}={}}{} + E_{k,\text{key},\text{second}}^\perp W_{Q,\text{U}}^\perp {W_{K,\text{U}}^\perp}^T {E_{q,\text{query},\text{second}}^\perp}^T
    \end{align*}
    $$
    Except for the last line, all of these components are rank 1 matrices, and we can compute them efficiently.
    We compute an upper bound on what the final component can contribute to differences in elements in the same row:
    Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt{2} \sigma_1(M)$.
    This is the value we return, computing an upper bound on the first singular value by multiplying the first singular values of each matrix.

    Note that the first component is returned as EQKE_query_key, the middle components are accumulated in err_accumulator.
    """
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )

    W_E_pos_k = W_E + W_pos.mean(dim=0)[None, :]
    W_pos_err = W_pos - W_pos.mean(dim=0)[None, :]
    W_E_pos_q = W_E + W_pos[-1][None, :]
    W_E_key, W_E_key_err = factor_contribution(
        W_E_pos_k, key_direction.squeeze(), sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_key.setcheckparams(atol=atol)
    W_E_query, W_E_query_err = factor_contribution(
        W_E_pos_q, query_direction.squeeze(), sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_query.setcheckparams(atol=atol)
    EQKE_query_key = (W_E_query @ W_Q[0, 0]) @ (
        W_K[0, 0].T @ W_E_key.T
    )  # O(d_vocab * d_vocab)
    err_accumulator = torch.zeros_like(EQKE_query_key.totensor())  # O(d_vocab^2)
    EQKE_query_cross_err = (
        (W_E_query @ W_Q[0, 0]) @ W_K[0, 0].T
    ) @ W_E_key_err.T  # O(d_vocab * d_model)
    err_accumulator += EQKE_query_cross_err
    EQKE_err_cross_key = W_E_query_err @ (
        W_Q[0, 0] @ (W_K[0, 0].T @ W_E_key.T)
    )  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_cross_key

    # This is a differently-shaped error term, and will be treated separately
    EQKE_pos_err = W_E_pos_q @ (
        W_Q[0, 0] @ (W_K[0, 0].T @ W_pos_err.T)
    )  # O(d_vocab * d_model * n_ctx)

    # We'd like a faster way to estimate the quantity (EQKE_err_err_check.max(dim=-1) - EQKE_err_err_check.min(dim=-1)).max()
    # The naive computation is O(d_vocab^2 * d_model), and we can only get this down to O(d_vocab * d_model^2) by using SVD
    # To improve our error bounds a bit, first we again peel off the leading singular values
    W_E_second_key, W_E_key_err2 = factor_contribution(
        W_E_key_err, second_key_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_second_key.setcheckparams(atol=1e-4)
    (
        W_E_second_query,
        W_E_query_err2,
    ) = factor_contribution(
        W_E_query_err, second_query_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_second_query.setcheckparams(atol=1e-4)
    EQKE_err_second_query_key = (W_E_second_query @ W_Q[0, 0]) @ (
        W_K[0, 0].T @ W_E_second_key.T
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_second_query_key
    EQKE_err_second_query_cross_err = (
        (W_E_second_query @ W_Q[0, 0]) @ W_K[0, 0].T
    ) @ W_E_key_err2.T  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_second_query_cross_err
    EQKE_err_err_cross_second_key = W_E_query_err2 @ (
        W_Q[0, 0] @ (W_K[0, 0].T @ W_E_second_key.T)
    )  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_err_cross_second_key

    # Now we peel off the first singular vectors of W_Q and W_K
    W_Q_rank1, W_Q_err = factor_contribution(
        W_Q[0, 0], W_Q_U.squeeze(), sanity_check=sanity_check
    )  # O(d_model * d_model)
    W_Q_rank1.setcheckparams(atol=1e-4)
    W_K_rank1, W_K_err = factor_contribution(
        W_K[0, 0], W_K_U.squeeze(), sanity_check=sanity_check
    )  # O(d_model * d_model)
    W_K_rank1.setcheckparams(atol=1e-4)

    EQKE_err_err_err__first_singular = (W_E_query_err2 @ W_Q_rank1) @ (
        W_K_rank1.T @ W_E_key_err2.T
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_err_err__first_singular

    EQKE_err_err_err__Q_cross_err = (
        (W_E_query_err2 @ W_Q_rank1) @ W_K_err.T
    ) @ W_E_key_err2.T  # O(d_vocab * d_voacb)
    err_accumulator += EQKE_err_err_err__Q_cross_err
    EQKE_err_err_err__err_cross_K = W_E_query_err2 @ (
        W_Q_err @ (W_K_rank1.T @ W_E_key_err2.T)
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_err_err__err_cross_K

    # We would like a faster way to compute EQKE_err_err_err__err_cross_err
    # unfortunately, we can only get this down to O(d_vocab * d_model^2) by using SVD

    # take the product of the first signular values in each matrix to get a bound on the singular value of the product
    prod_max_singular = torch.tensor(
        [
            torch.linalg.matrix_norm(m, ord=2)
            for m in (W_E_query_err2, W_Q_err, W_K_err, W_E_key_err2)
        ]
    ).prod()
    # since \sigma_1(M) = \sup_x \| M x \| / \|x\|, considering vectorswith one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is sqrt(2) * \sigma_1(M)
    return (
        (EQKE_query_key, err_accumulator),
        EQKE_pos_err,
        (
            prod_max_singular * np.sqrt(2),
            (W_E_query_err2, W_Q_err, W_K_err.T, W_E_key_err2.T),
        ),
    )


# %%
(
    size_direction,
    query_direction,
    size_query_singular_value,
) = find_size_and_query_direction(model)
(second_key_direction, second_key_singular_value), (
    second_query_direction,
    second_query_singular_value,
) = find_second_singular_contributions(model, size_direction, query_direction)
(W_Q_U, W_Q_S, W_Q_Vh), (W_Q_contrib, W_Q_err) = split_svd_contributions(
    model.W_Q[0, 0]
)
(W_K_U, W_K_S, W_K_Vh), (W_K_contrib, W_K_err) = split_svd_contributions(
    model.W_K[0, 0]
)
(
    (EQKE_query_key, err_accumulator),
    EQKE_pos_err,
    (err_upper_bound, (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)),
) = decompose_EQKE_error(
    model,
    key_direction=size_direction,
    query_direction=query_direction,
    second_key_direction=second_key_direction,
    second_query_direction=second_query_direction,
    W_Q_U=W_Q_U,
    W_K_U=W_K_U,
    sanity_check=True,
)

# %% [markdown]
# # more plots
# %%
if DISPLAY_PLOTS:
    px.imshow(EQKE_query_key.numpy(), title="EQKE_query_key").show()
    px.imshow(err_accumulator.numpy(), title="err_accumulator").show()
    px.imshow(EQKE_pos_err.numpy(), title="EQKE_pos_err").show()
    px.imshow((W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T).numpy()).show()
print(f"err_upper_bound: {err_upper_bound}")


# %%
@torch.no_grad()
def compute_min_right_attention_quadratic(
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
) -> Float[Tensor, "d_vocab_q d_vocab_max"]:  # noqa: F722
    r"""
    Computes a tensor of minimum right attention (more attention paid to the max than to a single instance of a non-max token at least min_gap less than the max token) for each query token and each max token
    When the query token is larger than the max token, the matrix holds nan.

    Complexity: O(d_vocab^2)

    Preconditions:
        (none)
    Postconditions:
        \forall q, m:
          if q > m: return[q, m] = nan
          elif m - min_gap[q, m] < q < m: return[q, m] = nan
          elif m < min_gap[q, m]: return[q, m] = 0
          else: return[q, m] = EQKE[q, m] - \max_{k <= m - min_gap[q, m]} EQKE[q, k]
    """
    result = torch.zeros_like(EQKE)
    for q_tok in range(EQKE.shape[0]):
        running_maxes = torch.zeros_like(EQKE[q_tok])
        for max_tok in range(EQKE.shape[1]):
            cur_min_gap = (
                min_gap
                if isinstance(min_gap, int)
                else int(min_gap[q_tok, max_tok].item())
            )
            if max_tok > 0:
                running_maxes[max_tok] = max(
                    running_maxes[max_tok - 1].item(), EQKE[q_tok, max_tok].item()
                )
            if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                result[q_tok, max_tok] = float("nan")
            elif max_tok < cur_min_gap:
                result[q_tok, max_tok] = 0
            else:
                result[q_tok, max_tok] = (
                    EQKE[q_tok, max_tok] - running_maxes[max_tok - cur_min_gap]
                )
    return result


# %%
@torch.no_grad()
def compute_min_softmaxed_right_attention_quadratic(
    min_right_attention: Float[Tensor, "d_vocab_q d_vocab_max"],  # noqa: F722
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
    attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa: F722
        0
    ].attn.attn_scale,
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"]:  # noqa: F722
    r"""
    Computes the minimum post-softmax attention paid to the maximum token by each query token, for each number of copies of a non-max token.

    min_gap is used only to determine when the result should be nan

    Complexity: O(d_vocab^2 * n_ctx^2)

    Preconditions:
        . attn_scale is correct for the model
        Define:
        EQKE[q, p, k] := (W_E[q] + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E[k] + W_pos[p]).T
        Then we demand:
        . \forall q, m, p1, p2, k:
          if ((q == m) or (q <= m - min_gap[q, m])) and (k <= m - min_gap[q, m]):
            min_right_attention[q, m] + EQKE_pos_error[q, p1] - EKQE_pos_error[q, p2]
            <= EQKE[q, p1, m] - EQKE[q, p2, k]
    Postconditions:
        \forall q, m, n_copies_nonmax:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m]: return[q, m, 0] = nan
          else: return[q, m, n_copies_nonmax] <= post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m (including possibly the query token), and all other tokens <= m - min_gap[q, m]
    """
    n_ctx = EQKE_pos_err.shape[-1]
    result = torch.zeros(tuple(min_right_attention.shape) + (n_ctx,))
    tmp = torch.zeros((n_ctx,))
    EQKE_pos_err -= EQKE_pos_err[:, -1].unsqueeze(
        -1
    )  # softmax is invariant to adding a constant to all inputs, so we offset by the attention paid to the query position
    EQKE_pos_err = EQKE_pos_err[:, :-1].sort(dim=-1).values
    for q_tok in range(min_right_attention.shape[0]):
        for max_tok in range(min_right_attention.shape[1]):
            cur_min_gap = (
                min_gap
                if isinstance(min_gap, int)
                else int(min_gap[q_tok, max_tok].item())
            )
            if max_tok < q_tok:
                result[q_tok, max_tok] = float("nan")
                continue
            for n_copies_nonmax in range(n_ctx):
                if n_copies_nonmax == 0 and max_tok != q_tok:
                    result[q_tok, max_tok, n_copies_nonmax] = float("nan")
                    continue
                if max_tok < cur_min_gap and n_copies_nonmax != 0:
                    result[q_tok, max_tok, n_copies_nonmax] = float("nan")
                    continue
                if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                    result[q_tok, max_tok, n_copies_nonmax] = float("nan")
                    continue
                tmp[:-1] = EQKE_pos_err[q_tok]
                tmp[-1] = 0
                if n_copies_nonmax == n_ctx - 1 and max_tok == q_tok:
                    # max tok in the query position, so we handle this case specially
                    tmp[-1] += min_right_attention[q_tok, max_tok]
                    tmp = (tmp / attn_scale).softmax(dim=-1)
                    result[q_tok, max_tok, n_copies_nonmax] = tmp[-1]
                else:
                    # put the max tokens in the least favored slots, where attention is lowest
                    n_copies_max = n_ctx - n_copies_nonmax
                    tmp[:n_copies_max] += min_right_attention[q_tok, max_tok]
                    tmp = (tmp / attn_scale).softmax(dim=-1)
                    result[q_tok, max_tok, n_copies_nonmax] = (
                        tmp[:n_copies_max].sum().item()
                    )
    return result


# %%
min_gap = 1
min_right_attention = compute_min_right_attention_quadratic(
    EQKE_query_key + err_accumulator, min_gap=20
)
print(
    (min_right_attention[~min_right_attention.isnan()] > err_upper_bound).sum().item()
)
min_right_attention_softmaxed = compute_min_softmaxed_right_attention_quadratic(
    min_right_attention - err_upper_bound,
    EQKE_pos_err,
    min_gap=1,
    attn_scale=model.blocks[0].attn.attn_scale,
)

# %% [markdown]
# ## The average+diff trick
#
# (Stealing notation from Aryan)
#
# Suppose we have quantities $f_{x,y}$ and $g_{y,z}$ and we want to pessimize (WLOG, suppose minimize) the quantity $f_{x,y} + g_{y,z}$ over $x$, $y$, and $z$ in time less than $\mathcal{O}(n_x n_y n_z)$, say we allow $\mathcal{O}(n_x n_y + n_y n_z + n_x n_z)$.
#
# We can of course say
# $$\min_{x,y} f_{x,y} + \min_{y, z} g_{y,z} \le f_{x,y} + g_{y,z}$$
# But we can do better!
#
# Note that
# $$f_{x,y} = \mathbb{E}_x f_{x,y} + (f_{x,y} - \mathbb{E}_x f_{x,y})$$
#
# Suppose that $f_{x,y}$ varies much less over $x$ than it does over $y$, and much less than $h_{y,z}$ varies over either of $y$ and $z$.
# This will make the following bonud a good approximation, though the bound is sound even without this assumption.
# We can write
# $$
# \begin{align*}
# f_{x,y} + g_{y,z}
# & \ge \min_{x,y,z} [f_{x,y} + g_{y,z}] \\
# & = \min_{x,y,z} [\mathbb{E}_x f_{x,y} + g_{y,z} + f_{x,y} - \mathbb{E}_x f_{x,y}] \\
# & \ge \min_{x,y,z} [\mathbb{E}_x f_{x,y} + g_{y,z}] + \min_{x,y,z}[f_{x,y} - \mathbb{E}_x f_{x,y}] \\
# & = \min_{y,z} [\mathbb{E}_x f_{x,y} + g_{y,z}] + \min_{x,y}[f_{x,y} - \mathbb{E}_x f_{x,y}]
# \end{align*}
# $$


# %%
# TODO: find the worse bounds without all the tricks


# %%
@torch.no_grad()
def compute_largest_wrong_logit_quadratic(
    min_softmaxed_right_attention: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    *,
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    r"""
    Computes the largest gap between the wrong logit and the right logit for each query token, max token, and number of copies of a non-max token.

    Complexity: O(d_vocab^2 * n_ctx^2)

    Preconditions:
        EUPU = (W_E + W_pos[-1]) @ W_U
        EVOU = W_E @ W_V @ W_O @ W_U
        PVOU = W_pos @ W_V @ W_O @ W_U
        \forall q, m, n_copies_nonmax:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m]: return[q, m, 0] = nan
          else: return[q, m, n_copies_nonmax] <= post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m, and all other tokens <= m - min_gap[q, m]
    Postconditions:
        \forall q, m, n_copies_nonmax, x:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m]: return[q, m, 0] = nan
          else: for all sequences with query q, max token m, n_copies_nonmax tokens not equal to m (including the query when the query is not equal to m), and all tokens either equal to m or less than or equal to m - min_gap[q, m], we have:
            return[q, m, n_copies_nonmax] <= model(sequence)[-1, x] - model(sequence)[-1, m]
    """
    results = torch.zeros_like(min_softmaxed_right_attention) + float("nan")
    d_vocab_q, d_vocab_max, n_ctx = min_softmaxed_right_attention.shape
    EVOU_max_logit_diff: Float[Tensor, "d_vocab_k"] = (  # noqa: F821
        EVOU.max(dim=-1).values - EVOU.min(dim=-1).values
    )  # for when we're paying attention to the wrong token

    # EUPU is too expensive to center with respect to the max token
    # so we split it
    # this one we can center with respect to the max token
    EUPU_mean_query: Float[Tensor, "d_vocab_out"]  # noqa: F821
    # this one we pessimize over the wrong token
    EUPU_per_query_max_gap: Float[Tensor, "d_vocab_q"]  # noqa: F821
    EUPU_mean_query, EUPU_per_query_max_gap = tricks.split_EUPU(EUPU)
    # center EVOU with respect to the diagonal, so it's convenient for the max token

    EVOU -= EVOU.diag()[:, None]
    for max_tok in range(d_vocab_max):
        # center PVOU according to max token, O(d_vocab * n_ctx)
        PVOU -= PVOU[:, max_tok].unsqueeze(-1)

        # Pessimization over position:
        # relax to PVOU attention being indepenent of EVOU attention, and also relax to it being possible to pay 100% attention to one PVOU position (this is reasonable, the gap in pre-softmax attention between adjacent tokens is like 20, 1e-20 is essentially 0 in float32.  EDIT: except I forgot to divide by attn_scale when doing this, attn_scale is 5.7, 20/5.7 is 3.5, exp(3.5) is 0.03, so it's not quite 100% attention.  Probably still pretty small)
        cur_PVOU: Float[Tensor, "d_vocab_out"] = PVOU.max(dim=0).values  # noqa: F821

        # center EUPU according to max token, O(d_vocab)
        EUPU_mean_query -= EUPU_mean_query[max_tok].item()

        # handle the case with only the max token
        # here we can use EUPU exactly
        logits_only_max: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
            EUPU[max_tok, :] + EVOU[max_tok, :] + cur_PVOU
        )
        logits_only_max -= logits_only_max[max_tok].item()
        logits_only_max[max_tok] = float(
            "-inf"
        )  # so we can max the logits across the non-max tokens
        results[max_tok, max_tok, 0] = logits_only_max.max().item()

        # now handle the cases with at least one non-max token
        cur_min_gap = (
            min_gap
            if isinstance(min_gap, int)
            else int(min_gap[: max_tok + 1, max_tok].min().item())
        )
        if max_tok < cur_min_gap:
            # we must have only the max token
            continue
        # query-independent logits from the skip connection / PVOU independence
        logits: Float[Tensor, "d_vocab_out"] = EUPU_mean_query + cur_PVOU  # noqa: F821
        assert logits[max_tok] == 0  # sanity check from centering above
        # pessimize over the thing we're not supposed to be paying attention to (w.r.t. the token that is non-max that we're paying attention)
        # maximum added to the wrong logit from paying attention to the wrong thing
        wrong_attention_logits: Float[Tensor, ""] = EVOU_max_logit_diff[  # noqa: F722
            : max_tok - cur_min_gap + 1
        ].max()
        # exact logits from paying attention to the right thing
        right_attention_logits: Float[Tensor, "d_vocab_out"] = EVOU[  # noqa: F821
            max_tok
        ]
        assert right_attention_logits[max_tok] == 0  # sanity check from centering above
        right_attention_logits_tmp = right_attention_logits.detach().clone()
        right_attention_logits_tmp[max_tok] = float("-inf")
        # maximum added to the wrong logit from paying attention to the right thing
        right_attention_logits_max: Float[
            Tensor, ""  # noqa: F722
        ] = right_attention_logits_tmp.max()
        for n_copies_nonmax in range(1, n_ctx):
            # First we combine query-independent logit information, then we reduce over output tokens and loop over queries
            # drop the nan values where the query token is invalid given the number of copies and the max token
            average_right_attention: Float[Tensor, ""]  # noqa: F722
            right_attention_adjustment: Float[Tensor, "d_vocab_q"]
            (
                average_right_attention,
                right_attention_adjustment,
            ) = tricks.split_min_softmaxed_right_attention(
                min_softmaxed_right_attention[:, max_tok, n_copies_nonmax],
                max_tok=max_tok,
            )
            cur_copies_logits = (
                logits + average_right_attention * right_attention_logits
            )
            assert (
                cur_copies_logits[max_tok] == 0
            ), f"cur_copies_logits[{max_tok}] == {cur_copies_logits[max_tok]} != 0"  # sanity check from centering above
            cur_copies_logits[max_tok] = float("-inf")
            # find maximum wrong logit
            average_wrong_logit = (
                cur_copies_logits.max()
                + (1 - average_right_attention) * wrong_attention_logits
            )
            for q_tok in range(max_tok + 1):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok])
                )
                if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                    continue
                cur_extra_right_attention = (
                    min_softmaxed_right_attention[q_tok, max_tok, n_copies_nonmax]
                    - average_right_attention
                )
                results[q_tok, max_tok, n_copies_nonmax] = (
                    average_wrong_logit + EUPU_per_query_max_gap[q_tok]
                )
                # add attention correction factor on right attention
                results[q_tok, max_tok, n_copies_nonmax] += (
                    cur_extra_right_attention * right_attention_logits_max
                )
                # add attention correction factor on right attention, using += - instead of -= to make the negation more obvious
                results[q_tok, max_tok, n_copies_nonmax] += (
                    -cur_extra_right_attention * wrong_attention_logits
                )
    return results


# %%
min_gap = 20
min_right_attention = compute_min_right_attention_quadratic(
    EQKE_query_key + err_accumulator, min_gap=min_gap
)
print(
    f"Complexity of compute_min_right_attention_quadratic: {complexity_of(compute_min_right_attention_quadratic)}"
)  # O(d_vocab^2)
print(
    (min_right_attention[~min_right_attention.isnan()] > err_upper_bound).sum().item()
)
min_right_attention_softmaxed = compute_min_softmaxed_right_attention_quadratic(
    min_right_attention - err_upper_bound,
    EQKE_pos_err,
    min_gap=min_gap,
    attn_scale=model.blocks[0].attn.attn_scale,
)
print(
    f"Complexity of compute_min_softmaxed_right_attention: {complexity_of(compute_min_softmaxed_right_attention_quadratic)}"
)  # O(d_vocab^2 * n_ctx^2)
EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = EU_PU(model)  # noqa: F722
print(f"Complexity of EU_PU: {complexity_of(EU_PU)}")  # O(d_vocab^2 * d_model)
EVOU: Float[Tensor, "d_vocab d_vocab_out"] = all_EVOU(model)  # noqa: F722
print(f"Complexity of EVOU: {complexity_of(all_EVOU)}")  # O(d_vocab^2 * d_model)
PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
print(f"Complexity of PVOU: {complexity_of(all_PVOU)}")  # O(n_ctx * d_vocab * d_model)
largest_wrong_logit: Float[
    Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
] = compute_largest_wrong_logit_quadratic(
    min_right_attention_softmaxed, EUPU=EUPU, EVOU=EVOU, PVOU=PVOU, min_gap=min_gap
)
print(
    f"Complexity of compute_largest_wrong_logit_quadratic: {complexity_of(compute_largest_wrong_logit_quadratic)}"
)  # O(d_vocab^2 * n_ctx^2)


# %%
@torch.no_grad()
def find_min_gaps(
    *,
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    EQKE_err_upper_bound: Union[float, Float[Tensor, ""]],  # noqa: F722
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
    attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa: F722
        0
    ].attn.attn_scale,
    position: Optional[int] = None,
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    """
    Run the argument across all possible min_gaps, and return the min_gap that works for each query token and max token.

    Since here we are finding the argument/proof rather than verifying it, the complexity does not matter.
    """
    d_vocab_q, d_vocab_k = EQKE.shape
    n_ctx, d_vocab_out = PVOU.shape
    min_gaps = torch.ones((d_vocab_q, d_vocab_k, n_ctx), dtype=torch.long)
    for min_gap in tqdm(list(reversed(range(1, d_vocab_k))), position=position):
        min_right_attention = compute_min_right_attention_quadratic(
            EQKE, min_gap=min_gap
        )
        min_right_attention_softmaxed = compute_min_softmaxed_right_attention_quadratic(
            min_right_attention - EQKE_err_upper_bound,
            EQKE_pos_err,
            min_gap=min_gap,
            attn_scale=attn_scale,
        )
        largest_wrong_logit = compute_largest_wrong_logit_quadratic(
            min_right_attention_softmaxed,
            EUPU=EUPU,
            EVOU=EVOU,
            PVOU=PVOU,
            min_gap=min_gap,
            tricks=tricks,
        )
        # if the largest wrong logit is negative, then this gap works
        min_gaps[largest_wrong_logit < 0] = min_gap

    return min_gaps


# %%
@torch.no_grad()
def count_correct_sequences(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
) -> int:
    d_vocab_q, d_vocab_max, n_ctx = largest_wrong_logit.shape
    correct_count = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok].item())
                )
                # use not to also catch nans
                if (
                    (not largest_wrong_logit[q_tok, max_tok, n_copies_nonmax] < 0)
                    or q_tok > max_tok
                    or (q_tok != max_tok and n_copies_nonmax == 0)
                    or (q_tok != max_tok and max_tok - q_tok < cur_min_gap)
                ):
                    continue
                if n_copies_nonmax == 0:
                    correct_count += 1
                elif q_tok == max_tok and n_copies_nonmax == n_ctx - 1:
                    correct_count += 1
                elif q_tok != max_tok and n_copies_nonmax == 1:
                    correct_count += 1
                else:
                    # N.B. Here, n_copies_nonmax DOES include the query token when it's not equal to max_tok
                    nonmax_pre_query_count = (
                        n_copies_nonmax - 1 if q_tok != max_tok else n_copies_nonmax
                    )
                    # count the number of sequences of length n_ctx - 1 with nonmax_pre_query_count tokens less than or equal to max_tok - cur_min_gap and the remaining tokens equal to max_tok, where order matters
                    correct_count += count_sequences(
                        n_ctx - 1, nonmax_pre_query_count, max_tok - cur_min_gap
                    )
    return correct_count


def compute_accuracy_lower_bound_from(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
) -> Tuple[float, Tuple[int, int]]:
    """
    returns correct_count / total_sequences, (correct_count, total_sequences)
    """
    d_vocab_q, d_vocab_max, n_ctx = largest_wrong_logit.shape
    correct_count = count_correct_sequences(largest_wrong_logit, min_gap=min_gap)
    total_sequences = d_vocab_max**n_ctx
    return correct_count / total_sequences, (correct_count, total_sequences)


# %%
try_all_configurations: bool = True  # @param {type:"boolean"}
use_tricks: bool = True  # @param {type:"boolean"}
if try_all_configurations:
    all_configs = list(enumerate_dataclass_values(LargestWrongLogitQuadraticConfig))
elif use_tricks:
    all_configs = [LargestWrongLogitQuadraticConfig()]
else:
    all_configs = [LargestWrongLogitQuadraticConfig.OFF]
with memoshelve(
    (
        lambda cfg: (
            cfg,
            find_min_gaps(
                EQKE=EQKE_query_key + err_accumulator,
                EQKE_err_upper_bound=err_upper_bound,
                EQKE_pos_err=EQKE_pos_err,
                EUPU=EUPU,
                EVOU=EVOU,
                PVOU=PVOU,
                tricks=cfg,
                attn_scale=model.blocks[0].attn.attn_scale,
                position=1,
            ),
        )
    ),
    filename=f"{__file__}.find_min_gaps-{cfg_hash}",
)() as find_min_gaps_for:
    min_gaps_list = [
        find_min_gaps_for(cfg)
        for cfg in tqdm(
            all_configs,
            position=0,
            desc="trick cfg",
        )
    ]

# %%
for tricks, min_gaps in min_gaps_list:
    print(f"========================================\nTricks: {tricks}")
    min_gap = min_gaps.max(dim=-1).values
    use_exact_error = False
    err_exact = W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
    min_right_attention = compute_min_right_attention_quadratic(
        EQKE_query_key + err_accumulator + (err_exact if use_exact_error else 0),
        min_gap=min_gap,
    )
    print(
        f"Complexity of compute_min_right_attention_quadratic: {complexity_of(compute_min_right_attention_quadratic)}"
    )  # O(d_vocab^2)
    print(
        (min_right_attention[~min_right_attention.isnan()] > err_upper_bound)
        .sum()
        .item()
    )
    min_right_attention_softmaxed = compute_min_softmaxed_right_attention_quadratic(
        min_right_attention - (err_upper_bound if not use_exact_error else 0),
        EQKE_pos_err,
        min_gap=min_gap,
        attn_scale=model.blocks[0].attn.attn_scale,
    )
    print(
        f"Complexity of compute_min_softmaxed_right_attention: {complexity_of(compute_min_softmaxed_right_attention_quadratic)}"
    )  # O(d_vocab^2 * n_ctx^2)
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = EU_PU(model)  # noqa: F722
    print(f"Complexity of EU_PU: {complexity_of(EU_PU)}")  # O(d_vocab^2 * d_model)
    EVOU: Float[Tensor, "d_vocab d_vocab_out"] = all_EVOU(model)  # noqa: F722
    print(f"Complexity of EVOU: {complexity_of(all_EVOU)}")  # O(d_vocab^2 * d_model)
    PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
    print(
        f"Complexity of PVOU: {complexity_of(all_PVOU)}"
    )  # O(n_ctx * d_vocab * d_model)
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ] = compute_largest_wrong_logit_quadratic(
        min_right_attention_softmaxed,
        EUPU=EUPU,
        EVOU=EVOU,
        PVOU=PVOU,
        min_gap=min_gap,
        tricks=tricks,
    )
    print(
        f"Complexity of compute_largest_wrong_logit_quadratic: {complexity_of(compute_largest_wrong_logit_quadratic)}"
    )  # O(d_vocab^2 * n_ctx^2)
    accuracy_bound, (
        correct_count,
        total_sequences,
    ) = compute_accuracy_lower_bound_from(largest_wrong_logit, min_gap=min_gap)
    print(
        f"Accuracy lower bound: {accuracy_bound} ({correct_count} correct sequences of {total_sequences})"
    )

# %%
