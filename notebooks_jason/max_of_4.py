# %%
import importlib
import gbmi.exp_max_of_n.analysis
import gbmi.analysis_tools.decomp
import gbmi.verification_tools.decomp
import gbmi.utils.lowrank

importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.analysis_tools.decomp)
importlib.reload(gbmi.verification_tools.decomp)
importlib.reload(gbmi.utils.lowrank)
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
# %%
import importlib
import gbmi.analysis_tools.decomp
import gbmi.verification_tools.decomp

importlib.reload(gbmi.verification_tools.decomp)
importlib.reload(gbmi.analysis_tools.decomp)
from gbmi.verification_tools.decomp import factor_contribution

sanity_check: bool = True


@torch.no_grad()
def assert_allclose_or_show(m1: Tensor, m2: Tensor, **kwargs):
    assert torch.allclose(m1, m2, **kwargs), [
        px.imshow(m1).show(),
        px.imshow(m2).show(),
        px.imshow((m1 - m2).abs()).show(),
    ]


with torch.no_grad():
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )

    W_E_pos_k = W_E + W_pos.mean(dim=0)[None, :]
    W_pos_err = W_pos - W_pos.mean(dim=0)[None, :]
    W_E_pos_q = W_E + W_pos[-1][None, :]
    W_E_size, W_E_size_err = factor_contribution(
        W_E_pos_k, size_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_size.setcheckparams(atol=1e-6)
    W_E_query, W_E_query_err = factor_contribution(
        W_E_pos_q, query_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_query.setcheckparams(atol=1e-6)
    EQKE_query_size = LowRankTensor(
        query_direction * np.sqrt(size_query_singular_value),
        size_direction * np.sqrt(size_query_singular_value),
        check=sanity_check,
        show=True,
    )  # O(d_vocab * d_vocab)
    if sanity_check:
        assert EQKE_query_size.check(W_E_query @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size.T)
    err_accumulator = torch.zeros_like(EQKE_query_size.totensor())  # O(d_vocab^2)
    EQKE_query_cross_err = (
        (W_E_query @ W_Q[0, 0]) @ W_K[0, 0].T
    ) @ W_E_size_err.T  # O(d_vocab * d_model)
    err_accumulator += EQKE_query_cross_err
    EQKE_err_cross_size = W_E_query_err @ (
        W_Q[0, 0] @ (W_K[0, 0].T @ W_E_size.T)
    )  # O(d_vocab * d_model)
    # %%
    err_accumulator += EQKE_err_cross_size
    if sanity_check:
        assert_allclose_or_show(
            EQKE_err_cross_size,
            W_E_query_err @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size.T,
            atol=1e-6,
        )

    # This is a differently-shaped error term, and will be treated separately
    EQKE_pos_err = W_E_pos_q @ (
        W_Q[0, 0] @ (W_K[0, 0].T @ W_pos_err.T)
    )  # O(d_vocab * d_model * n_ctx)

    # We'd like a faster way to estimate the quantity (EQKE_err_err_check.max(dim=-1) - EQKE_err_err_check.min(dim=-1)).max()
    # The naive computation is O(d_vocab^2 * d_model), and we can only get this down to O(d_vocab * d_model^2) by using SVD
    # To improve our error bounds a bit, first we again peel off the leading singular values
    second_key_direction_alt, (W_E_second_key, W_E_size_err2) = factor_contribution(
        W_E_size_err, second_key_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    second_query_direction_alt, (
        W_E_second_query,
        W_E_query_err2,
    ) = factor_contribution(
        W_E_query_err, second_query_direction, sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    EQKE_err_second_query_key_singular_value = (
        (second_query_direction @ W_E_query_err @ W_Q[0, 0])
        @ (second_key_direction @ W_E_size_err @ W_K[0, 0])
    ).item()  # O(d_vocab * d_model)
    EQKE_err_second_query_key = (
        second_query_direction[:, None]
        @ second_key_direction[None, :]
        * EQKE_err_second_query_key_singular_value
    )  # O(d_vocab * d_vocab)
    if sanity_check:
        assert_allclose_or_show(
            EQKE_err_second_query_key,
            W_E_second_query @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_second_key.T,
        )
    err_accumulator += EQKE_err_second_query_key
    EQKE_err_second_query_cross_err = (
        second_query_direction[:, None]
        @ (second_query_direction_alt @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T)[
            None, :
        ]
    )  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_second_query_cross_err
    if sanity_check:
        assert_allclose_or_show(
            EQKE_err_second_query_cross_err,
            W_E_second_query @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T,
            atol=1e-4,
        )
    EQKE_err_err_cross_second_key = (
        W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ second_key_direction_alt
    )[:, None] @ second_key_direction[
        None, :
    ]  # O(d_vocab * d_model)
    err_accumulator += EQKE_err_err_cross_second_key
    if sanity_check:
        assert_allclose_or_show(
            EQKE_err_err_cross_second_key,
            W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_second_key.T,
            atol=1e-6,
        )

    # Now we peel off the first singular vectors of W_Q and W_K
    # W_Q_direction_alt, (W_Q_low_rank, W_Q_err) = factor_contribution(
    #     W_Q[0, 0], W_Q_U.squeeze(), sanity_check=sanity_check
    # )  # O(d_vocab * d_vocab)
    # W_K_direction_alt, (W_K_low_rank, W_K_err) = factor_contribution(
    #     W_K[0, 0], W_K_U.squeeze(), sanity_check=sanity_check
    # )  # O(d_vocab * d_vocab)
    EQKE_err_err_err__first_singular = (
        (W_E_query_err2 @ W_Q_U)
        @ (W_Q_Vh @ W_K_Vh.T * W_Q_S * W_K_S)
        @ (W_E_size_err2 @ W_K_U).T
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_err_err__first_singular
    if sanity_check:
        assert_allclose_or_show(
            EQKE_err_err_err__first_singular,
            W_E_query_err2
            @ (W_Q_U @ W_Q_Vh * W_Q_S)
            @ (W_K_U @ W_K_Vh * W_K_S).T
            @ W_E_size_err2.T,
            atol=1e-6,
        )

    W_Q_err = W_Q[0, 0] - W_Q_U @ W_Q_Vh * W_Q_S
    W_K_err = W_K[0, 0] - W_K_U @ W_K_Vh * W_K_S

    EQKE_err_err_err__Q_cross_err = (W_E_query_err2 @ W_Q_U) @ (
        (W_Q_S * W_Q_Vh @ W_K_err.T) @ W_E_size_err2.T
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKE_err_err_err__Q_cross_err
    if sanity_check:
        assert_allclose_or_show(
            EQKE_err_err_err__Q_cross_err,
            W_E_query_err2 @ (W_Q_U @ W_Q_Vh * W_Q_S) @ W_K_err.T @ W_E_size_err2.T,
            atol=1e-6,
        )
    EQKQ_err_err_err__err_cross_K = W_E_query_err2 @ (
        W_Q_err @ (W_K_S * W_K_Vh.T @ (W_K_U.T @ W_E_size_err2.T))
    )  # O(d_vocab * d_vocab)
    err_accumulator += EQKQ_err_err_err__err_cross_K
    if sanity_check:
        assert_allclose_or_show(
            EQKQ_err_err_err__err_cross_K,
            W_E_query_err2 @ W_Q_err @ (W_K_U @ W_K_Vh * W_K_S).T @ W_E_size_err2.T,
            atol=1e-6,
        )

    # We would like a faster way to compute EQKQ_err_err_err__err_cross_err
    if sanity_check:
        EQKQ_err_err_err__err_cross_err_check = (
            W_E_query_err2 @ W_Q_err @ W_K_err.T @ W_E_size_err2.T
        )
    error = W_E_query_err2 @ W_Q_err @ W_K_err.T @ W_E_size_err2.T
    # px.imshow(error, title="error ≈ EQKE", labels={"x":"key token", "y":"query token"}).show(renderer="png")
    px.imshow(error).show(renderer="png")
    analyze_svd(error, renderer="png")
    for m in (W_E_query_err2, W_Q_err, W_K_err.T, W_E_size_err2.T):
        analyze_svd(m, renderer="png", scale_by_singular_value=False)
    print((error.max(dim=-1).values - error.min(dim=-1).values).max())
    print(error.abs().max())
    print(torch.linalg.matrix_norm(error, ord=2))
    print(
        [
            torch.linalg.matrix_norm(m, ord=2).item()
            for m in (W_E_query_err2, W_Q_err, W_K_err, W_E_size_err2)
        ]
    )
    print(
        torch.prod(
            torch.tensor(
                [
                    torch.linalg.matrix_norm(m, ord=2).item()
                    for m in (W_E_query_err2, W_Q_err, W_K_err, W_E_size_err2)
                ]
            )
        )
    )
    print(
        [
            (m @ m.T).trace().sqrt().item()
            for m in (W_E_query_err2, W_Q_err, W_K_err, W_E_size_err2)
        ]
    )
    print(
        torch.prod(
            torch.tensor(
                [
                    (m @ m.T).trace().sqrt().item()
                    for m in (W_E_query_err2, W_Q_err, W_K_err, W_E_size_err2)
                ]
            )
        )
    )

    # px.imshow(W_Q_err).show()

    # EQKE_query_size = (
    #     query_direction[:, None] @ size_direction[None, :] * size_query_singular_value
    # )  # O(d_vocab * d_vocab)
    # if sanity_check:
    #     EQKE_check = W_E_query @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size.T
    #     assert torch.allclose(EQKE_check, EQKE_query_size), [
    #         px.imshow(EQKE_check).show(),
    #         px.imshow(EQKE_query_size).show(),
    #         px.imshow((EQKE_query_size - EQKE_check).abs()).show(),
    #     ]

    # EQKE_err_err_err_first_singular

    # EQKE_err_err_err_first_singular =

    # px.imshow(err_accumulator).show()
    # px.imshow(W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T).show()
    # print(((W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T).max(dim=-1).values - (W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T).min(dim=-1).values).max())
    # print((W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T).abs().max())
    # error = W_E_query_err2 @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err2.T
    # # analyze_svd(error)
    # print(torch.linalg.matrix_norm(error, ord=2))
    # print([torch.linalg.matrix_norm(m, ord=2) for m in (W_E_query_err2, W_Q[0, 0], W_K[0, 0], W_E_size_err2)])

# %%
# HERE
#     EQKE_err_err_query_second_key_second = (second_query_direction,)
#     second_query_singular_value,

#     analyze_svd(W_E_query_err, scale_by_singular_value=False)
#     analyze_svd(W_E_size_err, scale_by_singular_value=False)
#     analyze_svd(W_E_pos_k, scale_by_singular_value=False)
#     analyze_svd(W_E_pos_q, scale_by_singular_value=False)
#     analyze_svd(W_Q[0, 0], scale_by_singular_value=False)
#     analyze_svd(W_K[0, 0], scale_by_singular_value=False)
#     _, SEq, _ = torch.linalg.svd(W_E_query_err)
#     _, SEs, _ = torch.linalg.svd(W_E_size_err)
#     _,
#     if sanity_check:
#         EQKE_err_err_check = (
#             W_E_query_err @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_err.T
#         )  # O(d_vocab^2 * d_model)
#     # EQKE_query_cross_err =
#     # EQKE_query_cross_err =
#     # EQKE_err_cross_size
#     # EQKQ_err_err =

#     # %%

#     W_E_size = size_direction @ W_E_pos_k
#     W_E_size = W_E_size / W_E_size.norm(dim=-1, keepdim=True)

#     W_E_size_reflect_alt = torch.stack(
#         [W_E_size * (row @ W_E_size) for row in W_E_pos_k], dim=0
#     )
#     W_E_size_reflect = (W_E_pos_k @ W_E_size[:, None]) @ W_E_size[None, :]
#     # torch.set_printoptions(threshold=5000, precision=10)
#     # print(f"W_E_pos_k={W_E_pos_k};\nW_E_pos_q={W_E_pos_q};\nW_Q={W_Q};\nW_K={W_K}")
#     # torch.set_printoptions() # reset display
#     # print([row @ W_E_size for row in W_E_pos_k - W_E_size_reflect_alt])
#     assert torch.allclose(W_E_size_reflect, W_E_size_reflect_alt)
#     W_E_query = query_direction @ W_E_pos_q
#     W_E_query = W_E_query / W_E_query.norm(dim=-1, keepdim=True)
#     W_E_query_reflect_alt = torch.stack(
#         [W_E_query * (row @ W_E_query) for row in W_E_pos_q], dim=0
#     )
#     W_E_query_reflect = (W_E_pos_q @ W_E_query[:, None]) @ W_E_query[None, :]
#     assert torch.allclose(W_E_query_reflect, W_E_query_reflect_alt)
#     W_E_q_err = W_E_pos_q - W_E_query_reflect
#     W_E_k_err = W_E_pos_k - W_E_size_reflect
#     (W_E_q_err_q, W_E_q_err_s, W_E_q_err_k), (
#         W_E_q_err_contrib,
#         W_E_q_err_resid,
#     ) = gbmi.analysis_tools.decomp.split_svd_contributions(W_E_q_err)
#     (W_E_k_err_k, W_E_k_err_s, W_E_k_err_q), (
#         W_E_k_err_contrib,
#         W_E_k_err_resid,
#     ) = gbmi.analysis_tools.decomp.split_svd_contributions(W_E_k_err)
#     (W_Q_q, W_Q_s, W_Q_k), (
#         W_Q_contrib,
#         W_Q_resid,
#     ) = gbmi.analysis_tools.decomp.split_svd_contributions(W_Q[0, 0])
#     (W_K_k, W_K_s, W_K_q), (
#         W_K_contrib,
#         W_K_resid,
#     ) = gbmi.analysis_tools.decomp.split_svd_contributions(W_K[0, 0])
#     matrices = (
#         ("E_q_err", W_E_q_err_resid),
#         ("E_k_err", W_E_k_err_resid),
#         ("Q", W_Q_resid),
#         ("K", W_K_resid),
#     )
#     print(
#         "(val - contrib).abs().max():        "
#         + ", ".join(f"{n}: {m.abs().max().item():2.4f}" for n, m in matrices)
#     )
#     print(
#         "(val - contrib).matrix_norm(ord=2): "
#         + ", ".join(
#             f"{n}: {torch.linalg.matrix_norm(m, ord=2).item():2.4f}"
#             for n, m in matrices
#         )
#     )
#     print(
#         "(diff.T @ diff).trace().sqrt():     "
#         + ", ".join(f"{n}: {(m.T @ m).trace().sqrt().item():2.4f}" for n, m in matrices)
#     )
#     _, S, _ = torch.linalg.svd(W_E_q_err_resid)
#     print((S.norm() ** 2, (W_E_q_err_resid.T @ W_E_q_err_resid).trace()))
#     W_E_q_err_resid_s, _ = W_E_q_err_resid.sort(dim=0)
#     W_E_q_err_resid_ss, _ = W_E_q_err_resid_s.sort(dim=1)
#     analyze_svd(W_E_q_err_resid, scale_by_singular_value=False)
#     analyze_svd(W_E_q_err_resid_s, scale_by_singular_value=False)
#     analyze_svd(W_E_q_err_resid_ss, scale_by_singular_value=False)

#     # px.imshow(W_E_q_err_contrib).show()
#     # px.imshow(W_E_q_err_resid).show()
#     # analyze_svd(W_E_q_err_resid, scale_by_singular_value=False)

#     # analyze_svd(W_Q, descr="Q", scale_by_singular_value=False)
#     # analyze_svd(W_K, descr="K", scale_by_singular_value=False)
#     if False:
#         px.imshow(W_E_pos_q @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_pos_k.T).show()
#         px.imshow(W_E_pos_q @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_reflect.T).show()
#         px.imshow(
#             W_E_pos_q @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E_pos_k - W_E_size_reflect).T
#         ).show()
#     if True:
#         px.imshow(
#             W_E_pos_q @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_pos_k.T, title="EQKE"
#         ).show()
#         px.imshow(
#             W_E_query_reflect @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_reflect.T,
#             title="queryQKsize",
#         ).show()
#         px.imshow(
#             W_E_query_reflect @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_k_err.T
#             + W_E_q_err @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_reflect.T
#             + W_E_q_err @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E_pos_k - W_E_size_reflect).T,
#             title="allerror",
#         ).show()
#         px.imshow(
#             W_E_query_reflect @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_k_err.T,
#             title="queryQKerror",
#         ).show()
#         px.imshow(
#             W_E_q_err @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_size_reflect.T,
#             title="errorQKsize",
#         ).show()
#         px.imshow(
#             W_E_q_err @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_k_err.T,
#             title="errorQKerror",
#         ).show()
#         analyze_svd(
#             W_E_q_err @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_k_err.T,
#             scale_by_singular_value=False,
#         )

#     # mat = W_E_pos_q @ W_Q[0, 0] @ W_K[0, 0].T @ W_E_pos_k.T
#     # U, S, Vh = torch.linalg.svd(mat)
#     # U = U * S[None, : U.shape[1]].sqrt()
#     # Vh = Vh * S[: Vh.shape[0], None].sqrt()
#     # signs = torch.sign(U.mean(dim=-1))
#     # U[:, 0] *= signs[0]
#     # Vh[0, :] *= signs[0]
#     # U[:, 1] *= signs[1]
#     # Vh[1, :] *= signs[1]
#     # U2 = U.clone()
#     # U2[:, 1:] = 0
#     # Vh2 = Vh.clone()
#     # Vh2[1:, :] = 0
#     # px.imshow(U2).show()
#     # px.imshow(Vh2).show()
#     # px.imshow(U2 @ Vh2).show()
#     # px.imshow(mat - U2 @ Vh2).show()
#     # U3, S3, V3 = torch.linalg.svd(mat - U2 @ Vh2)
#     # analyze_svd(mat - U2 @ Vh2, scale_by_singular_value=True)

#     # W_E_query = query_direction @ W_E
#     # W_E_query_reflect_alt = torch.stack(
#     #     [W_E_query * (row @ W_E_query) for row in W_E], dim=0
#     # )
#     # W_E_query_reflect = (W_E @ W_E_query[:, None]) @ W_E_query[None, :]
#     # assert torch.allclose(W_E_query_reflect, W_E_query_reflect_alt)
#     # px.imshow(
#     #     (W_E + W_pos[-1][None, :])
#     #     @ W_Q[0, 0]
#     #     @ W_K[0, 0].T
#     #     @ (W_E + W_pos.mean(dim=0)[None, :]).T
#     # ).show()
#     # px.imshow(
#     #     (W_E_query_reflect + W_pos[-1][None, :])
#     #     @ W_Q[0, 0]
#     #     @ W_K[0, 0].T
#     #     @ (W_E + W_pos.mean(dim=0)[None, :]).T
#     # ).show()
#     # px.imshow(
#     #     (W_E - W_E_query_reflect)
#     #     @ W_Q[0, 0]
#     #     @ W_K[0, 0].T
#     #     @ (W_E + W_pos.mean(dim=0)[None, :]).T
#     # ).show()
#     # # print(W_E_query.shape, W_E_query_reflect.shape, W_E_query_reflect_alt.shape)
#     # # px.imshow(W_E_query_reflect - W_E_query_reflect_alt).show()

#     # # W_E_err = torch.stack([row - W_E_query * (row @ W_E_query) for row in W_E], dim=0)

#     # # W_E_from_query = query_direction[:, None] @ W_E_query[None, :]
#     # # W_E_size = (size_direction @ (W_E + W_pos.mean(dim=0)[None, :])) @ W_K[0, 0] @ W_Q[0, 0].T
#     # # # compute matrix of W_E_query @ W_E_size
#     # # W_E_query_W_E_size = W_E_query[:, None] @ W_E_size[None, :]
#     # # px.imshow(W_E).show()
#     # # px.imshow(W_E_from_query).show()
#     # # px.imshow(W_E - W_E_from_query).show()
#     # # gbmi.analysis_tools.decomp.analyze_svd(W_E, scale_by_singular_value=False)
#     # # U, S, Vh = torch.linalg.svd(W_Q[0,0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)[None, :]).T)
#     # # gbmi.analysis_tools.decomp.analyze_svd(W_Q[0,0] @ W_K[0, 0].T @ (W_E + W_pos.mean(dim=0)[None, :]).T, scale_by_singular_value=False)
#     # # for i, s in enumerate(S):
#     # #     if i < U.shape[1]:
#     # #         U[:, i] *= s
#     # # for i in range(len(S), U.shape[1]):
#     # #     U[:, i] = 0
#     # # gbmi.analysis_tools.decomp.analyze_svd(W_E @ U, scale_by_singular_value=False)
#     # # gbmi.analysis_tools.decomp.analyze_svd(W_E @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E).T, scale_by_singular_value=True)
#     # # gbmi.analysis_tools.decomp.analyze_svd(W_E @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E + W_pos[-1][None, :]).T, scale_by_singular_value=False)


# # W_E_qerr = W_E - (query_direction @ W_E @ W_Q @ W_K.T @ (size_direction @ (W_E + W_pos.mean(dim=0)[None, :])))
# # px.imshow(W_E).show()
# # %%

# %%
