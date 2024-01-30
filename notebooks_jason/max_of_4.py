# %%
import importlib
import gbmi.exp_max_of_n.analysis
import gbmi.analysis_tools.decomp
import gbmi.verification_tools.decomp
import gbmi.utils.lowrank
import gbmi.exp_max_of_n.analysis

importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.analysis_tools.decomp)
importlib.reload(gbmi.verification_tools.decomp)
importlib.reload(gbmi.utils.lowrank)
importlib.reload(gbmi.exp_max_of_n.analysis)
# %%
from collections import defaultdict
from typing import Tuple
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
from jaxtyping import Float
from torch import Tensor
import plotly.express as px
from transformer_lens import HookedTransformerConfig, HookedTransformer
from gbmi.utils import default_device
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
import importlib
import gbmi.exp_max_of_n.plot

importlib.reload(gbmi.exp_max_of_n.plot)
gbmi.exp_max_of_n.plot.display_basic_interpretation(model)
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
            (W_E_query_err2, W_Q_err, W_K_err, W_E_key_err2),
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
    (err_upper_bound, (W_E_query_err2, W_Q_err, W_K_err, W_E_key_err2)),
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

# %%
