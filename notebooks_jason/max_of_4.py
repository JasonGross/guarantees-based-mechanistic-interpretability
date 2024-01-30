# %%
import importlib
import gbmi.exp_max_of_n.analysis
import gbmi.analysis_tools.decomp
import gbmi.verification_tools.decomp
import gbmi.utils.lowrank
import gbmi.exp_max_of_n.analysis
import gbmi.exp_max_of_n.plot
import gbmi.utils

importlib.reload(gbmi.exp_max_of_n.plot)
importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.analysis_tools.decomp)
importlib.reload(gbmi.verification_tools.decomp)
importlib.reload(gbmi.utils.lowrank)
importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.utils)
# %%
from collections import defaultdict
from typing import Tuple, Union
import math
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
from gbmi.utils.lowrank import LowRankTensor
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
from jaxtyping import Float, Integer
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
from gbmi.verification_tools.l1h1 import all_EVOU, all_PVOU, all_attention_scores
from gbmi.verification_tools.utils import complexity_of
from gbmi.utils.hashing import get_hash_ascii

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
run_batch_shelf_name = f"{__file__}.run_batch-{cfg_hash[:8]}_shelf"
# %%
loss_accuracy_memcache = {run_batch_shelf_name: {}}
# %%
batch_size = 4096  # 16_384 # 8182
# Resetting the DataLoader without shuffle for consistent processing
# data_loader = DataLoader(all_tokens_dataset, batch_size=batch_size, shuffle=False)

# Variables to accumulate total loss and accuracy
total_loss = 0.0
total_accuracy = 0.0
total_samples = 0

# loop for computing overall loss and accuracy
with shelve.open(run_batch_shelf_name) as shelf:
    with torch.no_grad():
        for i in tqdm(range(0, len(all_tokens_dataset), batch_size)):
            try:
                loss, accuracy, size = loss_accuracy_memcache[run_batch_shelf_name][
                    (i, batch_size)
                ]
            except KeyError:
                key = f"{i}_{batch_size}"
                try:
                    loss, accuracy, size = loss_accuracy_memcache[run_batch_shelf_name][
                        (i, batch_size)
                    ] = shelf[key]
                except KeyError:
                    batch = all_tokens_dataset[i : i + batch_size]
                    size = batch.shape[0]
                    batch.to(default_device(deterministic=True))
                    loss, accuracy = training_wrapper.run_batch(
                        batch, return_accuracy=True, log_output=False
                    )
                    loss = loss.item()
                    loss_accuracy_memcache[run_batch_shelf_name][
                        (i, batch_size)
                    ] = shelf[key] = (
                        loss,
                        accuracy,
                        size,
                    )

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
    f"Number Incorrect Sequences: {int(round(average_accuracy * all_tokens_dataset.length))}"
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
EPQKEP: Float[
    Tensor, "n_ctx_k d_vocab_q d_vocab_k"  # noqa: F722
] = all_attention_scores(model)
print(
    f"Complexity of (E+P[-1])QKEP: {complexity_of(all_attention_scores)}"
)  # O(d_vocab^2 * d_model * n_ctx)
# %%
# for q_tok in tqdm(range(model.cfg.d_vocab)):
#     for max_non_qtok in range(model.cfg.d_vocab):
#         for min_non_qtok in range(model.cfg.d_vocab):
# TODO COMPUTATION

# %% [markdown]
# # Plots
# %%
display_basic_interpretation(model)


# %% [markdown]
# # Size-Query analysis
#
# We find the size direction and the query direction, and approximate the QK computation using only these vectors.  Then we'll look at the error terms.
#
# We compute as follows:
# $$
# \begin{align*}
# \overline{W_\text{pos}} & := W_\text{pos}\text{.mean}(\text{dim}=0)
# \widetilde{E_q} & := W_E + W_\text{pos}[-1] \\
# \widetilde{E_k} & := W_E + \overline{W_\text{pos}} \\
# \text{EQKE}_p
# & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \widetilde{E_q}W_QW_K^T(W_{\text{pos}}[p] - \overline{W_\text{pos}})^T \\
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
# Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt(2) \sigma_1(M)$.
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

    EQKE_query_key uses key_direction and query_direction for the rank 1 approximation

    We compute as follows:
    $$
    \begin{align*}
    \overline{W_\text{pos}} & := W_\text{pos}\text{.mean}(\text{dim}=0)
    \widetilde{E_q} & := W_E + W_\text{pos}[-1] \\
    \widetilde{E_k} & := W_E + \overline{W_\text{pos}} \\
    \text{EQKE}_p
    & = \widetilde{E_q}W_QW_K^T \widetilde{E_k}^T + \widetilde{E_q}W_QW_K^T(W_{\text{pos}}[p] - \overline{W_\text{pos}})^T \\
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
    Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt(2) \sigma_1(M)$.
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
    EQKQ_err_err_err__err_cross_K = W_E_query_err2 @ (
        W_Q_err @ (W_K_rank1.T @ W_E_key_err2.T)
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKQ_err_err_err__err_cross_K

    # We would like a faster way to compute EQKQ_err_err_err__err_cross_err
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
    """
    Computes a tensor of minimum right attention (more attention paid to the max than to a single instance of a non-max token at least min_gap less than the max token) for each query token and each max token
    When the query token is larger than the max token, the matrix holds nan.

    Complexity: O(d_vocab^2)
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
def compute_min_softmaxed_right_attention(
    min_right_attention: Float[Tensor, "d_vocab_q d_vocab_max"],  # noqa: F722
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx"]:  # noqa: F722
    """
    Computes the minimum post-softmax attention paid to the maximum token by each query token, for each number of copies of a non-max token.

    min_gap is used only to determine when the result should be nan

    Complexity: O(d_vocab^2 * n_ctx^2)
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
                    tmp = tmp.softmax(dim=-1)
                    result[q_tok, max_tok, n_copies_nonmax] = tmp[-1]
                else:
                    # put the max tokens in the least favored slots, where attention is lowest
                    n_copies_max = n_ctx - n_copies_nonmax
                    tmp[:n_copies_max] += min_right_attention[q_tok, max_tok]
                    tmp = tmp.softmax(dim=-1)
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
min_right_attention_softmaxed = compute_min_softmaxed_right_attention(
    min_right_attention - err_upper_bound, EQKE_pos_err, min_gap=1
)


# %%
@torch.no_grad()
def compute_largest_wrong_logit_quadratic(
    min_softmaxed_right_attention: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    EUPU: Float[Tensor, "d_vocab_q d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    min_gap: Union[int, Integer[Tensor, "d_vocab_q d_vocab_max"]] = 1,  # noqa: F722
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    """
    Computes the largest gap between the wrong logit and the right logit for each query token, max token, and number of copies of a non-max token.

    Complexity: O(d_vocab^2 * n_ctx^2)
    """
    results = torch.zeros_like(min_softmaxed_right_attention) + float("nan")
    d_vocab_q, d_vocab_max, n_ctx = min_softmaxed_right_attention.shape
    EVOU_max_gap: Float[Tensor, "d_vocab_k"] = (  # noqa: F821
        EVOU.max(dim=-1).values - EVOU.min(dim=-1).values
    )  # for when we're paying attention to the wrong token
    EUPU_mean_query: Float[Tensor, "d_vocab_out"] = EUPU.mean(  # noqa: F821
        dim=0
    )  # this one we can center with respect to the max token
    EUPU_per_query: Float[Tensor, "d_vocab_q d_vocab_out"] = (  # noqa: F722
        EUPU - EUPU_mean_query[None, :]
    )  # this one is too expensive to center with respect to the max token
    EUPU_per_query_max_gap: Float[Tensor, "d_vocab_q"] = (  # noqa: F821
        EUPU_per_query.max(dim=-1).values - EUPU_per_query.min(dim=-1).values
    )
    # center EVOU with respect to the diagonal, so it's convenient for the max token
    EVOU -= EVOU.diag()[:, None]
    for max_tok in range(d_vocab_max):
        # center PVOU according to max token, O(d_vocab * n_ctx)
        PVOU -= PVOU[:, max_tok].unsqueeze(-1)
        # relax to PVOU attention being indepenent of EVOU attention, and also relax to it being possible to pay 100% attention to one PVOU position (this is reasonable, the gap in pre-softmax attention between adjacent tokens is like 20, 1e-20 is essentially 0 in float32)
        cur_PVOU: Float[Tensor, "d_vocab_out"] = PVOU.max(dim=0).values  # noqa: F821
        # center EUPU according to max token, O(d_vocab)
        EUPU_mean_query -= EUPU_mean_query[max_tok].item()

        # handle the case with only the max token
        logits: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
            EUPU[max_tok, :] + EVOU[max_tok, :] + cur_PVOU
        )
        logits -= logits[max_tok].item()
        logits[max_tok] = float(
            "-inf"
        )  # so we can max the logits across the non-max tokens
        results[max_tok, max_tok, 0] = logits.max().item()

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
        # maximum added to the wrong logit from paying attention to the wrong thing
        wrong_attention_logits: Float[Tensor, ""] = EVOU_max_gap[  # noqa: F722
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
            average_right_attention = dropnan(
                min_softmaxed_right_attention[:, max_tok, n_copies_nonmax]
            ).mean()
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
min_right_attention_softmaxed = compute_min_softmaxed_right_attention(
    min_right_attention - err_upper_bound, EQKE_pos_err, min_gap=min_gap
)
print(
    f"Complexity of compute_min_softmaxed_right_attention: {complexity_of(compute_min_softmaxed_right_attention)}"
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
    min_right_attention_softmaxed, EUPU, EVOU, PVOU, min_gap=min_gap
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
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    """
    Run the argument across all possible min_gaps, and return the min_gap that works for each query token and max token.

    Since here we are finding the argument/proof rather than verifying it, the complexity does not matter.
    """
    d_vocab_q, d_vocab_k = EQKE.shape
    n_ctx, d_vocab_out = PVOU.shape
    min_gaps = torch.ones((d_vocab_q, d_vocab_k, n_ctx), dtype=torch.long)
    for min_gap in tqdm(list(reversed(range(1, d_vocab_k)))):
        min_right_attention = compute_min_right_attention_quadratic(
            EQKE, min_gap=min_gap
        )
        min_right_attention_softmaxed = compute_min_softmaxed_right_attention(
            min_right_attention - EQKE_err_upper_bound, EQKE_pos_err, min_gap=min_gap
        )
        largest_wrong_logit = compute_largest_wrong_logit_quadratic(
            min_right_attention_softmaxed,
            EUPU=EUPU,
            EVOU=EVOU,
            PVOU=PVOU,
            min_gap=min_gap,
        )
        # if the largest wrong logit is negative, then this gap works
        min_gaps[largest_wrong_logit < 0] = min_gap

    return min_gaps


# %%
def count_sequences(
    sequence_length: int, nonmax_count: int, max_nonmax_tok: int
) -> int:
    """
    Count the number of sequences of length sequence_length with nonmax_count items less than or equal to max_nonmax_tok and the remaining tokens equal to a fixed value, where order matters
    """
    total_count = 0

    for i in range(nonmax_count + 1):
        combinations = math.comb(sequence_length, i)
        token_variations = max_nonmax_tok**i
        total_count += combinations * token_variations

    return total_count


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
min_gaps = find_min_gaps(
    EQKE=EQKE_query_key + err_accumulator,
    EQKE_err_upper_bound=err_upper_bound,
    EQKE_pos_err=EQKE_pos_err,
    EUPU=EUPU,
    EVOU=EVOU,
    PVOU=PVOU,
)

# %%
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
    (min_right_attention[~min_right_attention.isnan()] > err_upper_bound).sum().item()
)
min_right_attention_softmaxed = compute_min_softmaxed_right_attention(
    min_right_attention - (err_upper_bound if not use_exact_error else 0),
    EQKE_pos_err,
    min_gap=min_gap,
)
print(
    f"Complexity of compute_min_softmaxed_right_attention: {complexity_of(compute_min_softmaxed_right_attention)}"
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
    min_right_attention_softmaxed, EUPU, EVOU, PVOU, min_gap=min_gap
)
print(
    f"Complexity of compute_largest_wrong_logit_quadratic: {complexity_of(compute_largest_wrong_logit_quadratic)}"
)  # O(d_vocab^2 * n_ctx^2)
accuracy_bound, (correct_count, total_sequences) = compute_accuracy_lower_bound_from(
    largest_wrong_logit, min_gap=min_gap
)
print(
    f"Accuracy lower bound: {accuracy_bound} ({correct_count} correct sequences of {total_sequences})"
)

# %%
