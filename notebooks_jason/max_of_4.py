# %%
from collections import defaultdict
from gbmi.analysis_tools.decomp import analyze_svd
from gbmi.exp_max_of_n.analysis import find_size_and_query_direction
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

from gbmi.verification_tools.general import EU_PU
from gbmi.verification_tools.l1h1 import all_EVOU, all_PVOU, all_attention_scores
from gbmi.verification_tools.utils import complexity_of

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
        train_dataset_cfg=IterableDatasetCfg(),
        test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
    ),
    deterministic=True,
    seed=123,
    batch_size=128,
    train_for=(3000, "steps"),
)
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
run_batch_shelf_name = f"{__file__}.run_batch_shelf"
# %%
loss_accuracy_memcache = {}
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
                loss, accuracy, size = loss_accuracy_memcache[(i, batch_size)]
            except KeyError:
                key = f"{i}_{batch_size}"
                try:
                    loss, accuracy, size = loss_accuracy_memcache[
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
                    loss_accuracy_memcache[(i, batch_size)] = shelf[key] = (
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
# & \phantom{{}={}}{} + W_{E,\text{qerr}})W_Q W_K^T (\text{size}W_E + \overline{W_\text{pos}})^T \\
# \end{align*}$$
# %%
(
    size_direction,
    query_direction,
    size_query_singular_value,
) = find_size_and_query_direction(model)
# %%
import gbmi.analysis_tools.decomp

importlib.reload(gbmi.analysis_tools.decomp)
with torch.no_grad():
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )

    W_E_query = query_direction @ W_E
    W_E_query_reflect_alt = torch.stack(
        [W_E_query * (row @ W_E_query) for row in W_E], dim=0
    )
    W_E_query_reflect = (W_E @ W_E_query[:, None]) @ W_E_query[None, :]
    assert torch.allclose(W_E_query_reflect, W_E_query_reflect_alt)
    px.imshow(
        (W_E + W_pos[-1][None, :])
        @ W_Q[0, 0]
        @ W_K[0, 0].T
        @ (W_E + W_pos.mean(dim=0)[None, :]).T
    ).show()
    px.imshow(
        (W_E_query_reflect + W_pos[-1][None, :])
        @ W_Q[0, 0]
        @ W_K[0, 0].T
        @ (W_E + W_pos.mean(dim=0)[None, :]).T
    ).show()
    px.imshow(
        (W_E - W_E_query_reflect)
        @ W_Q[0, 0]
        @ W_K[0, 0].T
        @ (W_E + W_pos.mean(dim=0)[None, :]).T
    ).show()
    # print(W_E_query.shape, W_E_query_reflect.shape, W_E_query_reflect_alt.shape)
    # px.imshow(W_E_query_reflect - W_E_query_reflect_alt).show()

    # W_E_err = torch.stack([row - W_E_query * (row @ W_E_query) for row in W_E], dim=0)

    # W_E_from_query = query_direction[:, None] @ W_E_query[None, :]
    # W_E_size = (size_direction @ (W_E + W_pos.mean(dim=0)[None, :])) @ W_K[0, 0] @ W_Q[0, 0].T
    # # compute matrix of W_E_query @ W_E_size
    # W_E_query_W_E_size = W_E_query[:, None] @ W_E_size[None, :]
    # px.imshow(W_E).show()
    # px.imshow(W_E_from_query).show()
    # px.imshow(W_E - W_E_from_query).show()
    # gbmi.analysis_tools.decomp.analyze_svd(W_E, scale_by_singular_value=False)
    # U, S, Vh = torch.linalg.svd(W_Q[0,0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)[None, :]).T)
    # gbmi.analysis_tools.decomp.analyze_svd(W_Q[0,0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)[None, :]).T, scale_by_singular_value=False)
    # for i, s in enumerate(S):
    #     if i < U.shape[1]:
    #         U[:, i] *= s
    # for i in range(len(S), U.shape[1]):
    #     U[:, i] = 0
    # gbmi.analysis_tools.decomp.analyze_svd(W_E @ U, scale_by_singular_value=False)
    # gbmi.analysis_tools.decomp.analyze_svd(W_E @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E).T, scale_by_singular_value=True)
    # gbmi.analysis_tools.decomp.analyze_svd(W_E @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos[-1][None, :]).T, scale_by_singular_value=False)


# W_E_qerr = W_E - (query_direction @ W_E @ W_Q @ W_K.T @ (size_direction @ (W_E + W_pos.mean(dim=0)[None, :])))
# px.imshow(W_E).show()
# %%
