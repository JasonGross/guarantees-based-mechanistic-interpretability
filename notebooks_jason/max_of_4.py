# %%
from __future__ import annotations

# %%
import importlib
import gbmi.analysis_tools.plot
import gbmi.exp_max_of_n.analysis
import gbmi.analysis_tools.decomp
import gbmi.verification_tools.decomp
import gbmi.utils.lowrank
import gbmi.exp_max_of_n.plot
import gbmi.exp_max_of_n.train
import gbmi.exp_max_of_n.verification
import gbmi.utils
import gbmi.utils.memoshelve
import gbmi.utils.sequences
import gbmi.analysis_tools.utils
import gbmi.utils.latex_export
import gbmi.utils.images

importlib.reload(gbmi.analysis_tools.plot)
importlib.reload(gbmi.exp_max_of_n.plot)
importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.analysis_tools.decomp)
importlib.reload(gbmi.verification_tools.decomp)
importlib.reload(gbmi.utils.lowrank)
importlib.reload(gbmi.analysis_tools.utils)
importlib.reload(gbmi.exp_max_of_n.analysis)
importlib.reload(gbmi.exp_max_of_n.train)
importlib.reload(gbmi.utils)
importlib.reload(gbmi.exp_max_of_n.verification)
importlib.reload(gbmi.utils.memoshelve)
importlib.reload(gbmi.utils.sequences)
importlib.reload(gbmi.utils.latex_export)
importlib.reload(gbmi.utils.images)
# %%
import traceback
import sys
import re
import time
from functools import reduce
from PIL import Image
import io
import dataclasses
import math
from scipy import stats
from contextlib import contextmanager
from collections import defaultdict
import tikzplotly
from typing import (
    Callable,
    ClassVar,
    Collection,
    Literal,
    Sequence,
    Optional,
    Tuple,
    Union,
    List,
    Iterator,
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from gbmi.exp_max_of_n.plot import (
    scatter_attention_difference_vs_gap,
    hist_attention_difference_over_gap,
    hist_EVOU_max_minus_diag_logit_diff,
)
from gbmi.analysis_tools.plot import hist_EVOU_max_logit_diff, weighted_histogram
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
from gbmi.analysis_tools.utils import pm_round, pm_mean_std
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils.sequences import count_sequences
from gbmi.utils.lowrank import LowRankTensor
import gbmi.utils.ein as ein
import gbmi.utils.images as image_utils
from gbmi.utils.images import trim_plotly_figure
from gbmi.utils.memoshelve import memoshelve
from gbmi.utils.latex_export import to_latex_defs
from gbmi.exp_max_of_n.analysis import (
    find_second_singular_contributions,
    find_size_and_query_direction,
)
from gbmi.exp_max_of_n.plot import display_basic_interpretation
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    IterableDatasetCfg,
    MaxOfN,
    MaxOfNDataModule,
    MaxOfNTrainingWrapper,
    train_or_load_model,
)
from gbmi.model import Config, RunData
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import numpy as np
from jaxtyping import Float, Integer, Bool
from torch import Tensor
import plotly.express as px
from transformer_lens import HookedTransformerConfig, HookedTransformer
from pathlib import Path
from gbmi.utils import default_device, dropnan, shuffle_tensors, shuffle_tensor
from gbmi.utils.memocache import Memoize
from gbmi.utils.sequences import (
    SequenceDataset,
    ThunkedDataset,
    generate_all_sequences_for_model,
)
import shelve
from gbmi.verification_tools.decomp import (
    factor_contribution,
    max_row_diffs_per_dim,
    bound_max_row_diff_by_SVD,
)

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
RENDERER: Optional[str] = "png"  # @param ["png", None]
cache_dir = Path(__file__).parent / ".cache"
cache_dir.mkdir(exist_ok=True)
compute_expensive_average_across_many_models: bool = True  # @param {type:"boolean"}
LATEX_FIGURE_PATH = Path(__file__).with_suffix("") / "figures"
LATEX_FIGURE_PATH.mkdir(exist_ok=True, parents=True)
LATEX_VALUES_PATH = Path(__file__).with_suffix("") / "perf-numbers.tex"
LATEX_VALUES_PATH.parent.mkdir(exist_ok=True, parents=True)
# %%
latex_values: dict[str, Union[int, float, str]] = {}
latex_figures: dict[str, go.Figure] = {}
# %%
# hack around newlines of black formatting
seeds = (
    sorted(
        set(
            map(
                int,
                "50,104,123,519,742,913,1185,1283,1412,1490,1681,1696,1895,1951,2236,2306,2345,2549,2743,2773,3175,3254,3284,4157,4305,4430,4647,4729,4800,4810,5358,5615,5781,5928,6082,6155,6159,6204,6532,6549,6589,6910,7098,7238,7310,7467,7790,7884,8048,8299,8721,8745,8840,8893,9132,9134,9504,9816,10248,11124,11130,11498,11598,11611,12141,12287,12457,12493,12552,12561,13036,13293,13468,13654,13716,14095,14929,15043,15399,15622,15662,16069,16149,16197,16284,17080,17096,17194,17197,18146,18289,18668,19004,19093,19451,19488,19538,19917,20013,20294,20338,20415,20539,20751,20754,20976,21317,21598,22261,22286,22401,22545,23241,23367,23447,23633,23696,24144,24173,24202,24262,24438,24566,25516,26278,26374,26829,26932,27300,27484,27584,27671,27714,28090,28716,28778,29022,29052,29110,29195,29565,29725,29726,30371,30463,30684,30899,31308,32103,32374,32382".split(
                    ","
                ),
            )
        )
    )
    if compute_expensive_average_across_many_models
    else []
)
cfgs = {
    seed: Config(
        experiment=MaxOfN(
            model_config=HookedTransformerConfig(
                act_fn=None,
                attn_only=True,
                d_head=32,
                d_mlp=None,
                d_model=32,
                d_vocab=64,
                device="cpu",
                n_ctx=4,
                n_heads=1,
                n_layers=1,
                normalization_type=None,
            ),
            zero_biases=True,
            use_log1p=True,
            use_end_of_sequence=False,
            seq_len=4,
            optimizer="AdamW",
            optimizer_kwargs={"lr": 0.001, "betas": (0.9, 0.999)},
            train_dataset_cfg=IterableDatasetCfg(pick_max_first=False),
            test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
        ),
        deterministic=True,
        seed=seed,
        batch_size=128,
        train_for=(3000, "steps"),
    )
    for seed in [123] + list(seeds)
}
cfg_hashes = {seed: get_hash_ascii(cfg) for seed, cfg in cfgs.items()}
cfg_hashes_for_filename = {
    seed: cfg_hash.replace("/", "__SLASH__") for seed, cfg_hash in cfg_hashes.items()
}
datamodules = {seed: MaxOfNDataModule(cfg) for seed, cfg in cfgs.items()}
# %%
with memoshelve(
    train_or_load_model,
    filename=cache_dir / f"{Path(__file__).name}.train_or_load_model",
    get_hash=get_hash_ascii,
)() as memo_train_or_load_model:
    runtime_models = {}
    for seed, cfg in tqdm(cfgs.items()):
        try:
            runtime_models[seed] = memo_train_or_load_model(cfg, force="load")
        except Exception as e:
            print(f"Error loading model for seed {seed}: {e}")
# %%
training_wrappers = {
    seed: MaxOfNTrainingWrapper(cfgs[seed], model)
    for seed, (_runtime, model) in runtime_models.items()
}
# training_wrapper.run_batch = Memoize(training_wrapper.run_batch, name=f"{__file__}.training_wrapper.run_batch", use_pandas=False, use_shelf=True)  # type: ignore
# %% [markdown]
# # Training stats
# %%
train_total_loss = {}
train_total_accuracy = {}
train_total_samples = {}
train_measurement_deterministic: bool = False  # @param {type:"boolean"}
train_average_loss = {}
train_average_accuracy = {}
dataloader_iter: Iterator


# loop for computing overall loss and accuracy
@torch.no_grad()
def _run_train_batch_loss_accuracy(
    seed: int, i: int, batch_size: int
) -> Tuple[float, float, int]:
    xs, ys = next(dataloader_iter)
    device = default_device(deterministic=train_measurement_deterministic)
    xs.to(device)
    ys.to(device)
    loss, accuracy = training_wrappers[seed].run_batch((xs, ys), log_output=False)
    loss = loss.item()
    return loss, accuracy, batch_size


for seedi, seed in enumerate(tqdm(runtime_models.keys(), desc="seed", position=0)):
    leave = seedi == len(runtime_models.keys()) - 1
    train_total_loss[seed] = 0.0
    train_total_accuracy[seed] = 0.0
    train_total_samples[seed] = 0

    datamodule = datamodules[seed]
    datamodule.setup("train")
    dataloader = datamodule.train_dataloader()
    dataloader_iter = iter(dataloader)
    with memoshelve(
        _run_train_batch_loss_accuracy,
        filename=cache_dir
        / f"{Path(__file__).name}.run_batch_loss_accuracy-{cfg_hashes_for_filename[seed]}-{train_measurement_deterministic}",
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    )() as run_batch_loss_accuracy:
        for i in tqdm(
            range(0, len(dataloader)),
            desc=f"batches",
            position=1,
            leave=leave,
        ):
            loss, accuracy, size = run_batch_loss_accuracy(seed, i, cfg.batch_size)  # type: ignore
            # Accumulate loss and accuracy
            train_total_loss[seed] += loss * size
            train_total_accuracy[seed] += accuracy * size
            train_total_samples[seed] += size

    # Calculate average loss and accuracy
    train_average_loss[seed] = train_total_loss[seed] / train_total_samples[seed]
    train_average_accuracy[seed] = (
        train_total_accuracy[seed] / train_total_samples[seed]
    )
# %%
num_seeds = len(train_average_loss)
avg_train_average_loss = sum(train_average_loss.values()) / num_seeds
avg_train_average_accuracy = sum(train_average_accuracy.values()) / num_seeds
std_dev_train_average_loss = float(np.std(list(train_average_loss.values())))
std_dev_train_average_accuracy = float(np.std(list(train_average_accuracy.values())))
latex_values["NumSeeds"] = num_seeds
latex_values["AvgTrainAccuracyFloat"] = avg_train_average_accuracy
latex_values["StdDevTrainAccuracyFloat"] = std_dev_train_average_accuracy
latex_values["AvgTrainLossFloat"] = avg_train_average_loss
latex_values["StdDevTrainLossFloat"] = std_dev_train_average_loss
print(f"Overall Training stats ({num_seeds} training runs):")
print(
    f"Model Accuracy: ({pm_round(avg_train_average_accuracy * 100, std_dev_train_average_accuracy * 100)})%"
)
print(f"Model Loss: {pm_round(avg_train_average_loss, std_dev_train_average_loss)}")

# %%
# import sys
# sys.exit(0)
# %%
seed = 123
cfg = cfgs[seed]
cfg_hash = cfg_hashes[seed]
cfg_hash_for_filename = cfg_hashes_for_filename[seed]
runtime, model = runtime_models[seed]
training_wrapper = training_wrappers[seed]
latex_values["seed"] = seed
latex_values["ModelSeed"] = cfg.experiment.model_config.seed
latex_values["TrainAccuracyFloat"] = train_average_accuracy[seed]
latex_values["TrainLossFloat"] = train_average_accuracy[seed]
# %%
print(f"Training stats:")
print(f"Model Accuracy: {train_average_accuracy[seed] * 100}%")
print(f"Model Loss: {train_average_loss[seed]}")

# %% [markdown]
# # Brute Force Proof
# %%
all_tokens_dataset = SequenceDataset(
    seq_len=model.cfg.n_ctx, vocab_size=model.cfg.d_vocab
)
# %%
batch_size = 4096  # 16_384 # 8182
latex_values["BruteForceBatchSize"] = batch_size
latex_values["BruteForceNumBatches"] = int(
    math.ceil(len(all_tokens_dataset) / batch_size)
)
# Resetting the DataLoader without shuffle for consistent processing
# data_loader = DataLoader(all_tokens_dataset, batch_size=batch_size, shuffle=False)

# Variables to accumulate total loss and accuracy
total_loss = 0.0
total_accuracy = 0.0
total_samples = 0
all_incorrect_sequences = []

brute_force_proof_deterministic: bool = True  # @param {type:"boolean"}
latex_values["BruteForceCPU"] = brute_force_proof_deterministic


# loop for computing overall loss and accuracy
@torch.no_grad()
def _run_batch_loss_accuracy(
    i: int, batch_size: int, return_incorrect_sequences: bool = True
) -> Union[Tuple[float, float, int], Tuple[Tuple[float, float, int], Tensor]]:
    batch = all_tokens_dataset[i : i + batch_size]
    size = batch.shape[0]
    device = default_device(deterministic=brute_force_proof_deterministic)
    batch.to(device)
    labels = training_wrapper.config.experiment.get_ground_truth(batch)
    xs, ys, y_preds = training_wrapper.compute_batch((batch, labels), device=device)
    loss = training_wrapper.loss_fn(
        y_preds, ys, log_softmax=training_wrapper.log_softmax
    ).item()
    full_accuracy = training_wrapper.acc_fn_per_seq(y_preds, ys)
    accuracy = full_accuracy.float().mean().item()
    if return_incorrect_sequences:
        return (loss, accuracy, size), xs[~full_accuracy]
    return loss, accuracy, size


with memoshelve(
    _run_batch_loss_accuracy,
    filename=cache_dir
    / f"{Path(__file__).name}.run_batch_loss_accuracy-{cfg_hash_for_filename}-{brute_force_proof_deterministic}",
    get_hash_mem=(lambda x: x[0]),
    get_hash=str,
)() as run_batch_loss_accuracy:
    for i in tqdm(range(0, len(all_tokens_dataset), batch_size)):
        (loss, accuracy, size), incorrect_sequences = run_batch_loss_accuracy(i, batch_size)  # type: ignore
        # Accumulate loss and accuracy
        total_loss += loss * size
        total_accuracy += accuracy * size
        total_samples += size
        all_incorrect_sequences.append(incorrect_sequences)

# Calculate average loss and accuracy
average_loss = total_loss / total_samples
average_accuracy = total_accuracy / total_samples
incorrect_sequences = torch.cat(all_incorrect_sequences, dim=0)
num_correct_sequences = int(round(average_accuracy * all_tokens_dataset.length))
num_incorrect_sequences = all_tokens_dataset.length - num_correct_sequences
latex_values["BruteForceLossFloat"] = average_loss
latex_values["BruteForceAccuracyFloat"] = average_accuracy
latex_values["BruteForceNumCorrect"] = num_correct_sequences
latex_values["BruteForceNumIncorrect"] = num_incorrect_sequences
# %%
print(f"Brute force proof:")
print(f"Model Accuracy: {average_accuracy * 100}%")
print(f"Number Correct Sequences: {num_correct_sequences}")
print(f"Number Incorrect Sequences: {num_incorrect_sequences}")
print(f"Model Loss: {average_loss}")
# %%
fraction_of_incorrect_sequences_by_max = []
count_of_incorrect_sequences_by_query = []
for tok in range(model.cfg.d_vocab_out):
    cur_sequence_count = (
        1 if tok == 0 else (tok + 1) ** model.cfg.n_ctx - tok**model.cfg.n_ctx
    )
    fraction_of_incorrect_sequences_by_max.append(
        incorrect_sequences[incorrect_sequences.max(dim=-1).values == tok].shape[0]
        / cur_sequence_count
    )
    count_of_incorrect_sequences_by_query.append(
        incorrect_sequences[incorrect_sequences[:, -1] == tok].shape[0]
    )

# %% [markdown]
# Complexity: $$\mathcal{O}(\text{d\_vocab}^\text{n\_ctx} \cdot \text{n\_ctx} \cdot \text{d\_vocab} \cdot \text{d\_model})$$
# (batch size * number tensors in each sequence * cost of most expensive vector-matrix-multiplication)
# # %% [markdown]
# # # Brute Force Proof Standalone
# %%
# brute_force_use_gpu: bool = True #@param {type:"boolean"}
# brute_force_device = "cuda" if brute_force_use_gpu and torch.cuda.is_available() else "cpu"
# def generate_all_sequences_in_batches(d_vocab: int, n_ctx: int, per_batch: int = 2, postfix: Optional[List[int]] = None):
#     if postfix is None: postfix = []
#     if n_ctx <= per_batch:
#         sequences = torch.cartesian_prod(*[torch.arange(d_vocab) for _ in range(n_ctx)])
#         # append broadcasted postfix if not none
#         if len(postfix):
#             sequences = torch.cat([sequences, torch.tensor(postfix).expand(sequences.shape[0], -1)], dim=-1)
#         yield sequences
#     else:
#         for last_tok in range(d_vocab):
#             for sequences in generate_all_sequences_in_batches(d_vocab, n_ctx - 1, per_batch=per_batch, postfix=[last_tok] + postfix):
#                 yield sequences
# with shelve.open(cache_dir / f"{Path(__file__).name}.brute-force-standalone-{cfg_hash_for_filename}-{brute_force_device}") as shelf:

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
# This is a fraction of the form $\frac{a + bn}{c + dn}$.  Taking the derivative with respect to $n$ gives $\frac{bc - ad}{(c + dn)^2}$.  Noting that $c + dn$ cannot equal zero for any valid $n$, we get that the derivative never changes sign.  Hence our logit difference is maximized either at $n = 0$ or at $n = \ell - k$, and the sequence with just two values dominates the one with three.
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

# %% [markdown]
# # Mathematical Proof for Cubic
#
# Precompute:
# $$\begin{align*}\
# E_q &:= W_E[q] + W_\text{pos}[-1] \
# && \mathcal{O}(\text{d\_vocab} \cdot \text{d\_model}) \\
# \text{EQKE}[q,k] &:= E_q W_Q W_K^T (W_E[k])^T \
# && \mathcal{O}(\text{d\_vocab}^2 \cdot \text{d\_model}) \\
# \text{EQKP}[q,p] &:= E_q W_Q W_K^T (W_\text{pos}[p])^T \
# && \mathcal{O}(\text{d\_vocab} \cdot \text{d\_model} \cdot \text{n\_ctx}) \\
# \text{EUPU}[q] &:= E_q W_U \
# && \mathcal{O}(\text{d\_vocab}^2 \cdot \text{d\_model}) \\
# \text{EVOU}[k] &:= W_E[k] W_V W_O W_U \
# && \mathcal{O}(\text{d\_vocab}^2 \cdot \text{d\_model}) \\
# \text{PVOU}[p] &:= W_\text{pos}[p] W_V W_O W_U \
# && \mathcal{O}(\text{d\_vocab} \cdot \text{d\_model} \cdot \text{n\_ctx}) \\
# \end{align*}$$
# For a sequence $$\mathbf{x}$$, define
# $$\begin{align*}\
# \alpha(x_{-1},x_{i_k},i_k) &:= \frac{1}{\sqrt{\text{d\_head}}}(\text{EQKE}[x_{-1},x_{i_k}] + \text{EQKP}[x_{-1},i_k]) \\
# \mathbf{y}(\mathbf{x}) & :=\frac{1}{\sum_i e^{\alpha(x_{-1},x_i,i)}}\left[\sum_{i=0}^{n-1} e^{\alpha(x_{-1},x_i,i)}(\text{EVOU}[x_i] + \text{PVOU}[i]) \right] + \text{EUPU}[x_{-1}] \\
# \end{align*}$$
# We are interested in counting up the sequences with maximum token $m$ for which we have for all $t \neq m$ that
# $$y_t(\mathbf{x}) - y_m(\mathbf{x}) < 0$$
# Define
# $$w := \argmax_{t\text{ s.t. }t\neq m} y_t(\mathbf{x}) - y_m(\mathbf{x})$$
# Then
# $$\begin{align*}\
# &y_w(\mathbf{x}) - y_m(\mathbf{x})\\
# &= \max_{t \ne m}y_t(\mathbf{x}) - y_m(\mathbf{x}) \\
# &= \max_{t \ne m}\left(\frac{1}{\sum_i e^{\alpha(x_{-1},x_i,i)}}\left[\sum_{i=0}^{n-1} e^{\alpha(x_{-1},x_i,i)}(\text{EVOU}[x_i, t] - \text{EVOU}[x_i, m] + \text{PVOU}[i, t] - \text{PVOU}[i, m]) \right] + \text{EUPU}[x_{-1}, t] + \text{EUPU}[x_{-1}, m]\right) \\
# &= \max_{t \ne m}\left(\frac{1}{\sum_i e^{\alpha(x_{-1},x_i,i)}}\left[\sum_{i=0}^{n-1} e^{\alpha(x_{-1},x_i,i)}(\text{EVOU}[x_i, t] - \text{EVOU}[x_i, m] + \text{PVOU}[i, t] - \text{PVOU}[i, m]) \right] + \text{EUPU}[x_{-1}, t] + \text{EUPU}[x_{-1}, m]\right) \\
# \end{align*}$$


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
    result = torch.zeros((3, d_vocab, d_vocab, d_vocab, n_ctx)).to(EQKE) + float("nan")
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
              elif m = k and n_copies_nonmax != 0: return[q, m, m, n_copies_nonmax] = nan
              elif m != k and n_copies_nonmax == 0: return[q, m, k, 0] = nan
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
        PVOU = PVOU - PVOU[:, max_tok].unsqueeze(-1)
        # center EUPU according to max token, O(d_vocab^2)
        EUPU = EUPU - EUPU[:, max_tok].unsqueeze(-1)
        # center EVOU according to max token, O(d_vocab^2)
        EVOU = EVOU - EVOU[:, max_tok].unsqueeze(-1)
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
        results[max_tok, max_tok, max_tok, 0] = logits_only_max.max().item()

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
        EUPU_tmp: Float[Tensor, "d_vocab_q d_vocab_out"] = (  # noqa: F722
            EUPU.detach().clone()
        )
        EUPU_tmp[:, max_tok] = float("-inf")
        EUPU_per_query_pessimized: Float[Tensor, "d_vocab_q"] = EUPU_tmp.max(
            dim=-1
        ).values  # noqa: F821

        # Ditto for EVOU
        # distribute PVOU over EVOU, to avoid premature pessimization
        # TODO: mean+diff
        EPVOU_tmp: Float[Tensor, "d_vocab_k d_vocab_out"] = (  # noqa: F722
            EVOU + PVOU_pessimized
        )
        EPVOU_tmp[:, max_tok] = float("-inf")
        EPVOU_per_key_pessimized: Float[Tensor, "d_vocab_k"] = (  # noqa: F821
            EPVOU_tmp.max(dim=-1).values
        )

        # now handle the cases with at least one non-max non-query token
        for nonmax_tok in range(max_tok):
            for n_copies_nonmax in range(1, n_ctx):
                # distribute PVOU over EVOU, to avoid premature pessimization

                # pessimize over the thing we're not supposed to be paying attention to (w.r.t. the token that is non-max that we're paying attention)
                # maximum added to the wrong logit from paying attention to the wrong thing
                wrong_attention_logits: Float[Tensor, ""] = (  # noqa: F72
                    EPVOU_per_key_pessimized[nonmax_tok]
                )

                # pessimize also over the thing we are paying attention to
                right_attention_wrong_logits: Float[Tensor, ""] = (  # noqa: F722
                    EPVOU_per_key_pessimized[max_tok]
                )

                for q_tok in range(max_tok + 1):
                    if q_tok != max_tok and n_copies_nonmax >= n_ctx - 1:
                        continue
                    query_wrong_logits: Float[Tensor, ""] = (  # noqa: F722
                        EPVOU_per_key_pessimized[q_tok]
                    )
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
            # if the largest wrong logit is positive when the sequence is all max tokens, then pessimizing over position is not adequate for the convexity argument, so we skip these sequences.
            # in practice, we lose 6**4 == 1296 sequences this way, which is 0.0077% of the total
            largest_wrong_logit_in_only_max_sequences = largest_wrong_logit[
                max_tok, max_tok, max_tok, :
            ]
            largest_wrong_logit_in_only_max_sequences = (
                largest_wrong_logit_in_only_max_sequences[
                    ~largest_wrong_logit_in_only_max_sequences.isnan()
                ]
            )
            if largest_wrong_logit_in_only_max_sequences.item() > 0:
                # we did not account for these sequences in convexity
                continue
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
@torch.no_grad()
def count_unaccounted_for_by_cubic_convexity_sequences(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
    ],
) -> Tuple[int, Integer[Tensor, "num_wrong_toks"]]:  # noqa: F821
    """Computes the number of sequences that we are leaving on the table by pessimizing over position, and returns the query tokens"""
    d_vocab_q, d_vocab_max, _, n_ctx = largest_wrong_logit.shape
    unaccounted_for = ein.array(
        lambda max_tok: largest_wrong_logit[max_tok, max_tok, max_tok, 0],
        sizes=[d_vocab_q],
    )
    assert not unaccounted_for.isnan().any(), f"unaccounted_for: {unaccounted_for}"
    unaccounted_for_toks = torch.arange(d_vocab_q)[unaccounted_for > 0]
    unaccounted_for_count = (
        ((unaccounted_for_toks + 1) ** n_ctx - unaccounted_for_toks**n_ctx).sum().item()
    )
    assert isinstance(
        unaccounted_for_count, int
    ), f"unaccounted_for_count: {unaccounted_for_count} ({type(unaccounted_for_count)})"
    return unaccounted_for_count, unaccounted_for_toks.long()


# %%
starttime = time.time()
prooftime = 0.0
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
# largest_wrong_logit_cubic: Float[
#     Tensor, "d_vocab_q d_vocab_max d_vocab_nonmax n_ctx_nonmax_copies"  # noqa: F722
# ] = compute_largest_wrong_logit_cubic(
#     min_right_attention_softmaxed_cubic,
#     EUPU=EUPU,
#     EVOU=EVOU,
#     PVOU=PVOU,
# )
accuracy_bound_cubic, (
    correct_count_cubic,
    total_sequences,
) = compute_accuracy_lower_bound_from_cubic(largest_wrong_logit_cubic)
print(
    f"Accuracy lower bound: {accuracy_bound_cubic} ({correct_count_cubic} correct sequences of {total_sequences})"
)
prooftime += time.time() - starttime
print(f"Proof time: {prooftime}s")
cubic_dropped_sequences, wrong_toks_full_attention = (
    count_unaccounted_for_by_cubic_convexity_sequences(largest_wrong_logit_cubic)
)
cubic_dropped_sequences_frac = cubic_dropped_sequences / total_sequences
print(
    f"Note that we are leaving {cubic_dropped_sequences} sequences on the floor, which is {cubic_dropped_sequences_frac * 100}% of the total ({wrong_toks_full_attention.tolist()})"
)
assert (
    cubic_dropped_sequences
    == (wrong_toks_full_attention.max().item() + 1) ** model.cfg.n_ctx
), f"LaTeX will be wrong in 4-results.tex: {cubic_dropped_sequences} != ({wrong_toks_full_attention.max().item()} + 1) ** {model.cfg.n_ctx}, {wrong_toks_full_attention.tolist()}"
latex_values["CubicLargestWrongTokenFullAttention"] = (
    wrong_toks_full_attention.max().item()
)
latex_values["CubicAccuracyFloat"] = accuracy_bound_cubic
latex_values["CubicCorrectCount"] = correct_count_cubic
latex_values["CubicProofTimeFloat"] = prooftime
latex_values["CubicDroppedSequences"] = cubic_dropped_sequences
latex_values["CubicDroppedSequencesFracFloat"] = cubic_dropped_sequences_frac

# # %%


# %% [markdown]
# # Plots
# %%
if DISPLAY_PLOTS:
    figs = display_basic_interpretation(
        model, include_uncentered=True, renderer=RENDERER
    )
    latex_figures["EQKE"] = figs["EQKE"]
    latex_figures["EVOU"] = figs["EVOU"]
    latex_figures["EVOU-centered"] = figs["EVOU-centered"]
    latex_figures["EQKP"] = figs["EQKP"]
    latex_figures["EQKE-SVD"] = figs["EQKE Attention SVD"]
    del figs["EQKE Attention SVD"]
    EUPU_keys = [k for k in figs.keys() if k.startswith("irrelevant_")]
    assert len(EUPU_keys) == 1, f"EUPU_keys: {EUPU_keys}"
    latex_figures["EUPU"] = figs[EUPU_keys[0]]
    del figs[EUPU_keys[0]]
    latex_figures["PVOU"] = figs["irrelevant"]
    del figs["irrelevant"]
    unused_keys = [k for k in figs if k not in latex_figures]
    if unused_keys:
        print(f"Unused keys: {unused_keys}")


# %%
# for slides
@torch.no_grad()
def make_FAR_slides_plots(
    model: HookedTransformer,
    OV_colorscale="Picnic_r",
    QK_colorscale="Plasma",
    renderer=None,
):
    W_E, W_pos, W_U, W_V, W_O, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_U,
        model.W_V[0, 0],
        model.W_O[0, 0],
        model.W_Q[0, 0],
        model.W_K[0, 0],
    )
    EPq = W_E + W_pos[-1]
    EPk = W_E + W_pos.mean(dim=0)
    Pk = W_pos - W_pos.mean(dim=0)
    EPU = EPq @ W_U
    EVOU = EPk @ W_V @ W_O @ W_U
    EVOU_centered = EVOU - EVOU.diag()[:, None]
    PVOU = Pk @ W_V @ W_O @ W_U
    EQKE = EPq @ W_Q @ W_K.T @ EPk.T
    EQKP = EPq @ W_Q @ W_K.T @ Pk.T
    OV_zmax = np.max(
        [EVOU.abs().max().item(), PVOU.abs().max().item(), EPU.abs().max().item()]
    )
    QK_zmax = np.max([EQKE.abs().max().item(), EQKP.abs().max().item()])
    for m, title, colorscale, zmax, labels in (
        (
            EPU,
            "EPU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "query token t<sub>i</sub>"},
        ),
        (
            EVOU,
            "EVOU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "key token t<sub>j</sub>"},
        ),
        (
            PVOU,
            "PVOU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "position j"},
        ),
        (
            EQKE,
            "EQKE",
            QK_colorscale,
            QK_zmax,
            {"x": "key token t<sub>k</sub>", "y": "query token t<sub>q</sub>"},
        ),
        (
            EQKP,
            "EQKP",
            QK_colorscale,
            QK_zmax,
            {"x": "key position k", "y": "query token t<sub>q</sub>"},
        ),
    ):
        fig = px.imshow(
            m,
            title=title,
            color_continuous_scale=colorscale,
            color_continuous_midpoint=0,
            zmin=-zmax,
            zmax=zmax,
            labels=labels,
        )
        fig.show(renderer)
        # remove title
        fig.update_layout(title_text="")
        fig.update(layout_coloraxis_showscale=False)
        # crop whitespace
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show(renderer)


## %%
if DISPLAY_PLOTS:
    make_FAR_slides_plots(model, renderer=RENDERER)


# %%
# for slides
@torch.no_grad()
def make_better_slides_plots_00(
    model: HookedTransformer,
    OV_colorscale: str = "Picnic_r",
    QK_colorscale: str = "Plasma",
    renderer: Optional[str] = None,
) -> dict[str, go.Figure]:
    W_E, W_pos, W_U, W_V, W_O, W_Q, W_K = (
        model.W_E.cpu(),
        model.W_pos.cpu(),
        model.W_U.cpu(),
        model.W_V[0, 0].cpu(),
        model.W_O[0, 0].cpu(),
        model.W_Q[0, 0].cpu(),
        model.W_K[0, 0].cpu(),
    )
    attn_scale = model.blocks[0].attn.attn_scale
    EPq = W_E + W_pos[-1]
    EPk = W_E + W_pos.mean(dim=0)
    Pk = W_pos - W_pos.mean(dim=0)
    EPU = EPq @ W_U
    EVOU = EPk @ W_V @ W_O @ W_U
    EVOU_centered = EVOU - EVOU.diag()[:, None]
    PVOU = Pk @ W_V @ W_O @ W_U
    EQKE = EPq @ W_Q @ W_K.T @ EPk.T / attn_scale
    EQKP = EPq @ W_Q @ W_K.T @ Pk.T / attn_scale
    OV_zmax = np.max(
        [EVOU.abs().max().item(), PVOU.abs().max().item(), EPU.abs().max().item()]
    )
    QK_zmax = np.max([EQKE.abs().max().item(), EQKP.abs().max().item()])
    results = {}
    for key, zmax, colorscale in (
        ("OV", OV_zmax, OV_colorscale),
        ("QK", QK_zmax, QK_colorscale),
    ):
        results[f"{key}-colorbar"] = fig = go.Figure(
            data=go.Heatmap(
                z=[[0]],
                colorscale=colorscale,
                showscale=True,
                zmin=-zmax,
                zmax=zmax,
                zmid=0,
                colorbar=dict(x=0),
            )
        )
        fig.add_trace(
            go.Heatmap(
                z=[[0]],
                colorscale="Picnic_r",
                showscale=False,
                zmin=-zmax,
                zmax=zmax,
                zmid=0,
            )
        )
        fig.update_layout(
            width=75,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig.show(renderer)
    for m, title, colorscale, zmax, labels in (
        (
            EPU,
            "EPU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "query token t<sub>i</sub>"},
        ),
        (
            EVOU,
            "EVOU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "key token t<sub>j</sub>"},
        ),
        (
            PVOU,
            "PVOU",
            OV_colorscale,
            OV_zmax,
            {"x": "output logit", "y": "position j"},
        ),
        (
            EQKE,
            "EQKE",
            QK_colorscale,
            QK_zmax,
            {"x": "key token t<sub>k</sub>", "y": "query token t<sub>q</sub>"},
        ),
        (
            EQKP,
            "EQKP",
            QK_colorscale,
            QK_zmax,
            {"x": "key position k", "y": "query token t<sub>q</sub>"},
        ),
    ):
        key = title
        results[key] = fig = px.imshow(
            m,
            title=title,
            color_continuous_scale=colorscale,
            color_continuous_midpoint=0,
            zmin=-zmax,
            zmax=zmax,
            labels=labels,
        )
        fig.show(renderer)
        # remove title
        fig.update_layout(title_text="")
        fig.update(layout_coloraxis_showscale=False)
        # crop whitespace
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        trim_plotly_figure(fig)
        fig.show(renderer)
    return results


## %%
if DISPLAY_PLOTS:
    figs = make_better_slides_plots_00(model, renderer=RENDERER)
    for k, fig in figs.items():
        latex_figures[f"Decomposition-{k}"] = fig

# %% [markdown]
# # Back of the envelope math for sub-cubic
# %%
if DISPLAY_PLOTS:
    latex_figures["EVOU-hist-max-row-diff"] = hist_EVOU_max_logit_diff(
        model, renderer=RENDERER
    )
    for duplicate_by_sequence_count in [False, True]:
        key = "EVOU-hist-min-above-diag"
        if duplicate_by_sequence_count:
            key += "-dup-by-seq-count"
        latex_figures[key] = hist_EVOU_max_minus_diag_logit_diff(
            model,
            duplicate_by_sequence_count=duplicate_by_sequence_count,
            renderer=RENDERER,
        )


# %%
if DISPLAY_PLOTS:
    scatter_attention_difference_vs_gap(model, renderer="png")
    for duplicate_by_sequence_count in [False, True]:
        hist_attention_difference_over_gap(
            model,
            duplicate_by_sequence_count=duplicate_by_sequence_count,
            renderer=RENDERER,
        )


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
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
) -> Tuple[
    Tuple[
        Float[LowRankTensor, "d_vocab_q d_vocab_k"],  # noqa: F722
        Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    ],
    Float[Tensor, "d_vocab_q n_ctx_k"],  # noqa: F722
    Tuple[
        Union[Float[Tensor, ""], Float[Tensor, "d_vocab_q"]],  # noqa: F722
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
        remaining_error_upper_bound is a bound on the maximum difference between two elements in the same row of the remaining error of EQKE, and may be either a float or a tensor indexed by query token, depending on the configuration of tricks

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
    Note that the first component is returned as EQKE_query_key, the middle components are accumulated in err_accumulator.

    Except for the last line, all of these components are rank 1 matrices, and we can compute them efficiently.
    In the default case, we use the svd method for the final component:
    We compute an upper bound on what the final component can contribute to differences in elements in the same row:
    Since $\sigma_1(M) = \sup_x \| M x \| / \|x\|$, considering vectors with one 1, one -1, and zero elsewhere, the maximum difference between elements in a row is $\sqrt{2} \sigma_1(M)$.
    This is the value we return, computing an upper bound on the first singular value by multiplying the first singular values of each matrix.

    If tricks.attention_error_handling is "max_diff", then we instead compute the maximum difference in a more clever way.
    $$\begin{align*}
    &\max_{r,i,j} (AB)_{r,i} - (AB)_{r,j} \\
    &= \max_{r,i,j} \sum_k \left(A_{r,k} B_{k,i} - A_{r,k} B_{k,j}\right) \\
    &= \max_{r,i,j} \sum_k A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
    &\le \max_r \sum_k \max_{i,j} A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
    &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ \min_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
    &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ -\max_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
    &= \max_r \sum_k \left|A_{r,k}\max_{i,j}  \left(B_{k,i} - B_{k,j}\right)\right| \\
    &= \max_r \sum_k \left|A_{r,k}\right|\max_{i,j}  \left(B_{k,i} - B_{k,j}\right) \\
    \end{align*}$$
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
    err_accumulator = torch.zeros_like(EQKE_query_key.AB)  # O(d_vocab^2)
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

    return (
        (EQKE_query_key, err_accumulator),
        EQKE_pos_err,
        (
            tricks.bound_attention_error(
                W_E_query_err2, W_Q_err, W_K_err.T, W_E_key_err2.T
            ),
            (W_E_query_err2, W_Q_err, W_K_err.T, W_E_key_err2.T),
        ),
    )


# %%
(
    size_direction,
    query_direction,
    size_query_singular_value,
), _ = find_size_and_query_direction(model)
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
@torch.no_grad()
def display_EQKE_SVD_analysis(
    model: HookedTransformer, renderer: Optional[str] = None
) -> dict[str, go.Figure]:
    (
        size_direction,
        query_direction,
        size_query_singular_value,
    ), _ = find_size_and_query_direction(model)
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


if DISPLAY_PLOTS:
    px.imshow(
        (
            EQKE_query_key
            + err_accumulator
            + W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
        ).numpy(),
        color_continuous_scale="plasma",
        color_continuous_midpoint=0,
        title="EQKE",
        labels={"x": "key token", "y": "query token"},
    ).show(RENDERER)
    px.imshow(
        EQKE_query_key.numpy(),
        title="EQKE<sub>1</sub>",
        color_continuous_scale="plasma",
        color_continuous_midpoint=0,
    ).show(RENDERER)
    px.imshow(
        err_accumulator.numpy(),
        title="err_accumulator",
        color_continuous_scale="plasma",
        color_continuous_midpoint=0,
    ).show(RENDERER)
    px.imshow(
        (EQKE_query_key + err_accumulator).numpy(),
        title="EQKE<sub>2</sub>",
        color_continuous_scale="plasma",
        color_continuous_midpoint=0,
    ).show(RENDERER)
    px.imshow(
        EQKE_pos_err.numpy(),
        title="(W<sub>E</sub> + W<sub>pos</sub>[-1])W<sub>Q</sub>W<sub>K</sub><sup>T</sup>(W<sub>pos</sub> - 𝔼<sub>p</sub>W<sub>pos</sub>[p])<sup>T</sup>",
    ).show(RENDERER)
    zmax = (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T).abs().max().item()
    px.imshow(
        (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T).numpy(),
        title="EQKE_err",
        labels={"x": "key token", "y": "query token"},
        color_continuous_midpoint=0,
        zmax=zmax,
        zmin=-zmax,
    ).show(RENDERER)
    analyze_svd(
        (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T),
        descr="EQKE_err",
        renderer=RENDERER,
    )
    analyze_svd(
        (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T),
        descr="EQKE_err",
        colorscale="plasma",
        renderer=RENDERER,
    )
    s1 = torch.linalg.matrix_norm(
        (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T), ord=2
    )
    print(f"σ₁(EQKE_err)√2 = {s1}√2 = {s1*np.sqrt(2)}")
    ss = [
        torch.linalg.matrix_norm(m, ord=2).item()
        for m in (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)
    ]
    print(f"singular values: {ss}")
    print(f"√2∏σ₁ = {np.prod(ss)}√2 = {np.prod(ss)*np.sqrt(2)}")
    for m, s in (
        (W_E_query_err2, "E<sub>q,2</sub><sup>⟂</sup>"),
        (W_Q_err, "Q<sup>⟂</sup>"),
        (W_K_errT, "K<sup>⟂</sup>"),
        (W_E_key_err2T, "E<sub>k,2</sub><sup>⟂</sup>"),
    ):
        fig = px.imshow(
            m.numpy(),
            title=s,
            color_continuous_midpoint=0,
            zmax=zmax,
            zmin=-zmax,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show(RENDERER)
        analyze_svd(
            m,
            scale_by_singular_value=False,
            descr=s,
            colorscale="plasma",
            renderer=RENDERER,
        )
    sf1 = torch.linalg.matrix_norm(
        (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T), ord="fro"
    )
    print(f"σf₁(EQKE_err)√2 = {sf1}√2 = {sf1*np.sqrt(2)}")
    sfs = [
        torch.linalg.matrix_norm(m, ord="fro").item()
        for m in (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)
    ]
    print(f"singular fro values: {sfs}")
    print(f"√2∏σf₁ = {np.prod(sfs)}√2 = {np.prod(sfs)*np.sqrt(2)}")
print(f"err_upper_bound: {err_upper_bound}")

# %%
if DISPLAY_PLOTS:
    zmax = (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T).abs().max().item()
    uvs = []
    ss = []
    for m, s in (
        (W_E_query_err2, "E<sub>q,2</sub><sup>⟂</sup>"),
        (W_Q_err, "Q<sup>⟂</sup>"),
        (W_K_errT, "K<sup>⟂</sup>"),
        (W_E_key_err2T, "E<sub>k,2</sub><sup>⟂</sup>"),
        (W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T, "EQKE_err"),
    ):
        U, S, Vh = torch.linalg.svd(m)
        U = U[:, : S.shape[0]] * S[None, : U.shape[1]].sqrt()
        Vh = Vh[: S.shape[0], :] * S[: Vh.shape[0], None].sqrt()
        uvs.extend(((U, f"{s} U"), (Vh, f"{s} Vh")))
        ss.append((S, s))
    pre_uvs = uvs[:]
    num_subplots = len(pre_uvs)
    fig = make_subplots(rows=1, cols=num_subplots, horizontal_spacing=0.02)
    for i, (mv, us) in enumerate(pre_uvs, start=1):
        fig.add_trace(
            go.Heatmap(z=mv, zmin=-zmax, zmax=zmax, colorbar=None, showscale=False),
            row=1,
            col=i,
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].scaleanchor = "x1"  # type: ignore
            fig.layout[axis].scaleratio = 1  # type: ignore
    fig.update_layout(height=600, width=300 * num_subplots, plot_bgcolor="white")
    fig.show(RENDERER)
    # for mv, us in uvs:
    #     fig = px.imshow(
    #         mv.numpy(),
    #         title="",
    #         color_continuous_midpoint=0,
    #         zmax=zmax,
    #         zmin=-zmax,
    #     )
    #     # fig.update_traces(colorbar=None)
    #     fig.update_xaxes(showticklabels=False)
    #     fig.update_yaxes(showticklabels=False)
    #     fig.update(layout_coloraxis_showscale=False)
    #     fig.show(RENDERER)
    num_subplots = len(ss)
    fig = make_subplots(rows=1, cols=num_subplots)  # , horizontal_spacing=0.02)
    for i, (s, st) in enumerate(ss, start=1):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(s)), y=s, mode="lines+markers", line=dict(color="blue")
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        height=300,  # Adjust as needed
        width=300 * num_subplots,  # Adjust width to fit all subplots side by side
        showlegend=False,  # Optionally hide the legend if it's not needed
    )
    fig.update_xaxes(showticklabels=False)
    # fig.update_yaxes(showticklabels=False)
    fig.show(RENDERER)
    # fig = px.line(s.numpy(), title=f"{st} singular values").show(RENDERER)
    # analyze_svd(m, scale_by_singular_value=True, descr=s, colorscale="plasma", renderer=RENDERER)

# %% [markdown]

# %%
# random resampling of EQKE_err
with torch.no_grad():
    ms = (
        (W_E_query_err2, "E<sub>q,2</sub><sup>⟂</sup>"),
        (W_Q_err, "Q<sup>⟂</sup>"),
        (W_K_errT, "K<sup>⟂</sup>"),
        (W_E_key_err2T, "E<sub>k,2</sub><sup>⟂</sup>"),
    )
    if DISPLAY_PLOTS:
        for m, s in ms:
            m_numpy = m.flatten().numpy()
            edges = np.histogram_bin_edges(m_numpy, bins="auto")
            counts, _ = np.histogram(m_numpy, bins=edges)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            pdf_values = stats.norm.pdf(
                bin_centers, loc=m.mean().item(), scale=m.std().item()
            )
            pdf_scaled = pdf_values * m.numel() * np.diff(edges)
            fig = px.histogram(
                {"": m_numpy},
                nbins=len(edges) - 1,
                title=s,
                labels={"variable": "", "value": "matrix element value"},
            )
            fig.add_scatter(
                x=bin_centers,
                y=pdf_scaled,
                mode="lines",
                name=f"𝒩({pm_round(m.mean().item(), m.std().item(), sep=', ')})",
            )
            fig.show(RENDERER)
    # what if we randomize the order of all matrices without replacement?
    torch.manual_seed(1234)
    nsamples = 100
    row_diffs = []
    max_row_diffs = []
    for _ in range(nsamples):
        ms_no_replacement = [shuffle_tensor(m) for m, s in ms]
        result = reduce(torch.matmul, ms_no_replacement)
        row_diffs.extend(result.max(dim=-1).values - result.min(dim=-1).values)
        max_row_diffs.append(
            (result.max(dim=-1).values - result.min(dim=-1).values).max().item()
        )
    row_diffs = torch.stack(row_diffs)
    max_row_diffs = torch.tensor(max_row_diffs)
    print(f"max row diff (n = {nsamples}): {pm_mean_std(max_row_diffs)}")
    # print(f"row diff: {pm_mean_std(row_diffs)}")
    # sampling from normal
    row_diffs = []
    max_row_diffs = []
    for _ in range(nsamples):
        ms_normal = [torch.randn_like(m) * m.std() + m.mean() for m, s in ms]
        result = reduce(torch.matmul, ms_normal)
        row_diffs.extend(result.max(dim=-1).values - result.min(dim=-1).values)
        max_row_diffs.append(
            (result.max(dim=-1).values - result.min(dim=-1).values).max().item()
        )
    row_diffs = torch.stack(row_diffs)
    max_row_diffs = torch.tensor(max_row_diffs)
    m_descr = ", ".join(
        f"𝒩({pm_round(m.mean().item(), m.std().item(), sep=', ')})" for m, s in ms
    )
    print(f"max row diff (n = {nsamples}, m ~ {m_descr}): {pm_mean_std(max_row_diffs)}")
    # print(f"row diff: {pm_mean_std(row_diffs)}")


# %%
@torch.no_grad()
def decompose_EUPU_error_via_svd(
    model: HookedTransformer,
    *,
    W_E_U: Tensor,
    W_U_U: Tensor,
    sanity_check: bool = True,
    atol: float = 1e-4,
) -> Tuple[
    Float[LowRankTensor, "d_vocab d_vocab_out"],  # noqa: F722
    Tuple[
        Float[Tensor, ""],  # noqa: F722
        Tuple[
            Float[Tensor, "d_vocab d_model"],  # noqa: F722
            Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
        ],
    ],
]:
    r"""
    Returns:
        (EUPU_lowrank, (remaining_error_upper_bound, two matrices whose product is the exact remaining error))
    where
        EU is the rank 1 approximation of (W_E + W_pos[-1]) @ W_U
        remaining_error_upper_bound is a bound on the maximum difference between two elements in the same row of the remaining error in EU


    Complexity: O(d_vocab * (d_vocab + d_model) + d_vocab * d_model^2)

    The d_model^2 term comes from having to do SVD to compute remaining_error_upper_bound

    Preconditions:
        (none)
    Postconditions:
        EUPU_lowrank := (W_E_U + W_pos[-1]) @ W_U_U.T
        Define err := (W_E + W_pos[-1]) @ W_U - EU_lowrank
        Then we guarantee:
        . max_{i,j} err_{r, i} - err_{r, j} <= remaining_error_upper_bound
    """
    W_E, W_pos, W_U = (
        model.W_E,
        model.W_pos,
        model.W_U,
    )

    W_E_via_U, W_E_err = factor_contribution(
        W_E + W_pos[-1], W_E_U.squeeze(), sanity_check=sanity_check
    )  # O(d_vocab * d_model)
    W_E_via_U.setcheckparams(atol=atol)
    W_U_via_U, W_U_err = factor_contribution(
        W_U, W_U_U.squeeze(), sanity_check=sanity_check
    )  # O(d_model * d_vocab_out)
    W_U_via_U.setcheckparams(atol=atol)
    EU_lowrank = W_E_via_U @ W_U_via_U  # O(d_vocab * d_vocab_out)

    return (
        EU_lowrank,
        bound_max_row_diff_by_SVD(W_E_err, W_U_err),  # type: ignore
    )


# %%
if DISPLAY_PLOTS:
    analyze_svd(model.W_E @ model.W_U, descr="W_E @ W_U", renderer=RENDERER)
    analyze_svd(
        model.W_E, descr="W_E", scale_by_singular_value=False, renderer=RENDERER
    )
    analyze_svd(
        model.W_U, descr="W_U", scale_by_singular_value=False, renderer=RENDERER
    )

# %%
(W_E_U, W_E_S, W_E_Vh), (W_E_contrib, W_E_err) = split_svd_contributions(model.W_E)
(W_U_U, W_U_S, W_U_Vh), (W_U_contrib, W_U_err) = split_svd_contributions(model.W_U)
(
    EUPU_lowrank,
    (EUPU_err_upper_bound, (EUPU_W_EP_err, EUPU_W_U_err)),
) = decompose_EUPU_error_via_svd(
    model,
    W_E_U=W_E_U,
    W_U_U=W_U_U,
    sanity_check=True,
)


# %%
@torch.no_grad()
def compute_min_right_attention_quadratic(
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"]  # noqa: F722
    ] = 1,
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"]:  # noqa: F722
    r"""
    Computes a tensor of minimum right attention (more attention paid to the max than to a single instance of a non-max token at least min_gap less than the max token) for each query token and each max token
    When the query token is larger than the max token, the matrix holds nan.

    Complexity: O(d_vocab^2 n_ctx)

    Preconditions:
        (none)
    Postconditions:
        \forall q, m, n_copies_nonmax:
          if q > m: return[q, m, :] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[q, m, n_copies_nonmax] = 0
          else: return[q, m, n_copies_nonmax] = EQKE[q, m] - \max_{k <= m - min_gap[q, m, n_copies_nonmax]} EQKE[q, k]
    """
    n_ctx = min_gap.shape[-1] if not isinstance(min_gap, int) else 1
    result = torch.zeros((EQKE.shape[0], EQKE.shape[1], n_ctx)).to(EQKE.device)
    for q_tok in range(EQKE.shape[0]):
        running_maxes = torch.zeros((EQKE[q_tok].shape[0], n_ctx)).to(EQKE.device)
        for max_tok in range(EQKE.shape[1]):
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
                if max_tok > 0:
                    running_maxes[max_tok, n_copies_nonmax] = max(
                        running_maxes[max_tok - 1, n_copies_nonmax].item(),
                        EQKE[q_tok, max_tok].item(),
                    )
                if max_tok != q_tok and (max_tok - q_tok < cur_min_gap):
                    result[q_tok, max_tok, n_copies_nonmax] = float("nan")
                elif max_tok < cur_min_gap:
                    result[q_tok, max_tok, n_copies_nonmax] = 0
                else:
                    result[q_tok, max_tok, n_copies_nonmax] = (
                        EQKE[q_tok, max_tok]
                        - running_maxes[max_tok - cur_min_gap, n_copies_nonmax]
                    )
    return result


# %%
@torch.no_grad()
def count_unaccounted_for_by_gap(
    min_gap: Integer[Tensor, "d_vocab_q d_vocab_max n_ctx"],  # noqa: F722
    collapse_n_ctx: bool = False,
) -> int:
    """Computes the number of sequences that we are leaving on the table by using gaps"""
    d_vocab_q, d_vocab_max, n_ctx = min_gap.shape
    unaccounted_for: int = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            if q_tok > max_tok:
                continue
            if collapse_n_ctx:
                gaps = min_gap[q_tok, max_tok]
                gap = gaps[~gaps.isnan()].max().long().item()
                if q_tok == max_tok:
                    if max_tok < gap:
                        unaccounted_for += (max_tok + 1) ** (n_ctx - 1)
                    else:
                        unaccounted_for += (1 + max_tok) ** (n_ctx - 1) - (
                            1 + (max_tok - gap + 1)
                        ) ** (n_ctx - 1)
                else:
                    if max_tok < gap:
                        unaccounted_for += (max_tok + 1) ** (n_ctx - 1) - max_tok ** (
                            n_ctx - 1
                        )
                    else:
                        unaccounted_for += (
                            (max_tok + 1) ** (n_ctx - 1) - max_tok ** (n_ctx - 1)
                        ) - (
                            (1 + (max_tok - gap + 1)) ** (n_ctx - 1)
                            - (max_tok - gap + 1) ** (n_ctx - 1)
                        )
            else:
                for n_copies_nonmax in range(n_ctx):
                    if n_copies_nonmax == n_ctx - 1 and max_tok != q_tok:
                        continue
                    gap = min_gap[q_tok, max_tok, n_copies_nonmax].long().item()
                    unaccounted_for += (
                        max_tok**n_copies_nonmax
                        - (1 + max_tok - gap) ** n_copies_nonmax
                    ) * math.comb(n_ctx - 1, n_copies_nonmax)
    return unaccounted_for


# %%
@torch.no_grad()
def compute_min_softmaxed_right_attention_quadratic(
    min_right_attention: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"  # noqa: F722
    ],
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"]  # noqa: F722
    ] = 1,
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
        . \forall q, m, p1, p2, k, n:
          if ((q == m) or (q <= m - min_gap[q, m, n])) and (k <= m - min_gap[q, m, n]):
            min_right_attention[q, m, n] + EQKE_pos_error[q, p1] - EKQE_pos_error[q, p2]
            <= EQKE[q, p1, m] - EQKE[q, p2, k]
    Postconditions:
        \forall q, m, n_copies_nonmax:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[q, m, 0] = nan
          else: return[q, m, n_copies_nonmax] <= post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m (including possibly the query token), and all other tokens <= m - min_gap[q, m, n_copies_nonmax]
    """
    n_ctx = EQKE_pos_err.shape[-1]
    min_right_attention = min_right_attention.expand(-1, -1, n_ctx)
    result = torch.zeros_like(min_right_attention)
    tmp = torch.zeros((n_ctx,))
    EQKE_pos_err = EQKE_pos_err - EQKE_pos_err[:, -1].unsqueeze(
        -1
    )  # softmax is invariant to adding a constant to all inputs, so we offset by the attention paid to the query position; this lets us uniformly fill in 0 for the attention paid to the query position further down, without it interfering with sorting
    EQKE_pos_err = EQKE_pos_err[:, :-1].sort(dim=-1).values
    for q_tok in range(min_right_attention.shape[0]):
        for max_tok in range(min_right_attention.shape[1]):
            if max_tok < q_tok:
                result[q_tok, max_tok] = float("nan")
                continue
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
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
                    tmp[-1] += min_right_attention[q_tok, max_tok, n_copies_nonmax]
                    tmp = (tmp / attn_scale).softmax(dim=-1)
                    result[q_tok, max_tok, n_copies_nonmax] = tmp[-1]
                else:
                    # put the max tokens in the least favored slots, where attention is lowest
                    n_copies_max = n_ctx - n_copies_nonmax
                    tmp[:n_copies_max] += min_right_attention[
                        q_tok, max_tok, n_copies_nonmax
                    ]
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
# This will make the following bound a good approximation, though the bound is sound even without this assumption.
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
#
# Example for how this helps with small variation:
#
# Take any function $k(y)$ and then take
# $$
# \begin{align*}
# f_{x,y} & := k(y) + \varepsilon_1(x, y) \\
# g_{y,z} & := -k(y) + \varepsilon_2(y, z)
# \end{align*}
# $$
# Then we have
# $$
# \begin{align*}
# \min_{x,y,z}[f_{x,y} + g_{y,z}] & = \min_{x,y,z}[\varepsilon_1(x, y) + \varepsilon_2(y, z)] \\
# \min{x,y}f_{x,y} + \min_{y,z}g_{y,z}
# & = \min_{y}k(y) + \min_{y} -k(y) + \min_{x,y}\varepsilon_1(x, y) + \min_{y,z}\varepsilon_2(y, z) \\
# & = \min_{y}k(y) - \max_{y} k(y) + \min_{x,y}\varepsilon_1(x, y) + \min_{y,z}\varepsilon_2(y, z) \\
# \min{x,y}[f_{x,y} - \mathbb{E}_x f_{x,y}] + \min_{y,z}[g_{y,z} + \mathbb{E}_x f_{x,y}]
# & = \min_{x,y}\varepsilon_1(x, y) + \min_{y,z}[\varepsilon_2(y, z) + \mathbb{E}_x\varepsilon_1(x, y)]
# \end{align*}
# $$
# If $\varepsilon_1$ and $\varepsilon_2$ are small compared to $\min_y k(y) - \max_y k(y)$, then using $\mathbb{E}_x f_{x,y}$ gives a much better bound.
#
# Note, though, that this could be a worse bound if the assumption of small variation does not hold.


# %% [markdown]
# ## Bounding the largest diff within a row of a product of matrices
#
# Suppose we have matrices $A$, $B$ and we want to compute
# $$\begin{align*}
# &\max_{r,i,j} (AB)_{r,i} - (AB)_{r,j} \\
# &= \max_{r,i,j} \sum_k \left(A_{r,k} B_{k,i} - A_{r,k} B_{k,j}\right) \\
# &= \max_{r,i,j} \sum_k A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
# &\le \max_r \sum_k \max_{i,j} A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
# &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ \min_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
# &= \max_r \sum_k A_{r,k}\begin{cases} \max_{i,j}  \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} \ge 0 \\ -\max_{i,j} \left(B_{k,i} - B_{k,j}\right) & \text{if }A_{r,j} <0 \end{cases} \\
# &= \max_r \sum_k \left|A_{r,k}\max_{i,j}  \left(B_{k,i} - B_{k,j}\right)\right| \\
# &= \max_r \sum_k \left|A_{r,k}\right|\max_{i,j}  \left(B_{k,i} - B_{k,j}\right) \\
# \end{align*}$$
# %% [markdown]
# ## Fusing Bounding the largest diff within a row of a product of matrices with Mean+Diff
#
# Suppose we have matrices $A$, $B$ and we want to compute
# $$\begin{align*}
# &\max_{r,i,j} (AB)_{r,i} - (AB)_{r,j} \\
# &= \max_{r,i,j} \sum_k \left(A_{r,k} B_{k,i} - A_{r,k} B_{k,j}\right) \\
# &= \max_{r,i,j} \sum_k A_{r,k} \left(B_{k,i} - B_{k,j}\right) \\
# &= \max_{r,i,j} \sum_k \left(\mathbb{E}_rA_{r,k} + \left(A_{r,k} - \mathbb{E}_rA_{r,k}\right)\right) \left(B_{k,i} - B_{k,j}\right) \\
# &= \max_{i,j} \left(\sum_k \mathbb{E}_rA_{r,k}\left(B_{k,i} - B_{k,j}\right) + \max_r \sum_k \left(A_{r,k} - \mathbb{E}_rA_{r,k}\right) \left(B_{k,i} - B_{k,j}\right)\right) \\
# &\le \left(\max_{i,j} \sum_k \mathbb{E}_rA_{r,k}\left(B_{k,i} - B_{k,j}\right)\right) + \max_r \sum_k \max_{i,j}\left(A_{r,k} - \mathbb{E}_rA_{r,k}\right) \left(B_{k,i} - B_{k,j}\right) \\
# &\le \left(\max_{i,j} \sum_k \mathbb{E}_rA_{r,k}\left(B_{k,i} - B_{k,j}\right)\right) + \max_r \sum_k \left|A_{r,k} - \mathbb{E}_rA_{r,k}\right| \max_{i,j}\left(B_{k,i} - B_{k,j}\right) \\
# \end{align*}$$
# %%
# # %% [markdown]
# # \[\begin{align*}
# # \text{max\_diff}[q, m, c]
# #  &= \max_{k \ne m} (\text{EVOU}[q, k] - \text{EVOU}[q, m]) \text{attn}[q, m, c] + \max_{0 \le n \le m - \text{min\_gap}[q, m, c]} (\text{EVOU}[q, m - n] - \text{EVOU}[q, m]) \text{attn}[q, m - n, c] \\
# # \end{align*}\]


# %%
@torch.no_grad()
def compute_largest_wrong_logit_quadratic(
    min_softmaxed_right_attention: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    *,
    W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
    W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]  # noqa: F722
    ] = 1,
    W_EP_direction: Optional[Float[Tensor, "d_model"]] = None,  # noqa F722
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
) -> Float[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    r"""
    Computes the largest gap between the wrong logit and the right logit for each query token, max token, and number of copies of a non-max token.

    Complexity: O(d_vocab^2 * n_ctx^2) (+ d_vocab * d_model^2 if W_EP_direction is None and tricks.EUPU_handling == "svd_query+max_diff")

    Preconditions:
        W_EP := W_E + W_pos[-1]
        EVOU = W_E @ W_V @ W_O @ W_U
        PVOU = W_pos @ W_V @ W_O @ W_U
        \forall q, m, n_copies_nonmax:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[q, m, 0] = nan
          else: return[q, m, n_copies_nonmax] <= post-softmax attention paid to max token m amongst all sequences with query q, n_ctx - n_copies_nonmax tokens equal to m, and all other tokens <= m - min_gap[q, m, n_copies_nonmax]
    Postconditions:
        \forall q, m, n_copies_nonmax, x:
          if q > m: return[q, m, n_copies_nonmax] = nan
          elif m - min_gap[q, m, n_copies_nonmax] < q < m: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax] and n_copies_nonmax != 0: return[q, m, n_copies_nonmax] = nan
          elif m < min_gap[q, m, n_copies_nonmax]: return[q, m, 0] = nan
          else: for all sequences with query q, max token m, n_copies_nonmax tokens not equal to m (including the query when the query is not equal to m), and all tokens either equal to m or less than or equal to m - min_gap[q, m, n_copies_nonmax], we have:
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
    EUPU_mean_query, EUPU_per_query_max_gap = tricks.split_EPU(
        W_EP=W_EP, W_U=W_U, W_EP_mean_query=W_EP_direction
    )

    # center EVOU with respect to the diagonal, so it's convenient for the max token
    EVOU = EVOU - EVOU.diag()[:, None]
    for max_tok in range(d_vocab_max):
        # center PVOU according to max token, O(d_vocab * n_ctx)
        PVOU = PVOU - PVOU[:, max_tok].unsqueeze(-1)
        # center EUPU_mean_query according to max token, O(d_vocab)
        EUPU_mean_query = EUPU_mean_query - EUPU_mean_query[max_tok].item()

        # Pessimization over position:
        # relax to PVOU attention being indepenent of EVOU attention, and also relax to it being possible to pay 100% attention to one PVOU position (this is reasonable, the gap in pre-softmax attention between adjacent tokens is like 20, 1e-20 is essentially 0 in float32.  EDIT: except I forgot to divide by attn_scale when doing this, attn_scale is 5.7, 20/5.7 is 3.5, exp(3.5) is 0.03, so it's not quite 100% attention.  Probably still pretty small)
        cur_PVOU: Float[Tensor, "d_vocab_out"] = PVOU.max(dim=0).values  # noqa: F821

        # handle the case with only the max token
        # here we can use EUPU exactly
        logits_only_max: Float[Tensor, "d_vocab_out"] = (  # noqa: F821
            W_EP[max_tok, :] @ W_U + EVOU[max_tok, :] + cur_PVOU
        )
        # O(d_model * d_vocab)
        logits_only_max -= logits_only_max[max_tok].item()
        logits_only_max[max_tok] = float(
            "-inf"
        )  # so we can max the logits across the non-max tokens
        results[max_tok, max_tok, 0] = logits_only_max.max().item()

        # now handle the cases with at least one non-max token
        # query-independent logits from the skip connection / PVOU independence
        logits: Float[Tensor, "d_vocab_out"] = EUPU_mean_query + cur_PVOU  # noqa: F821
        assert logits[max_tok] == 0  # sanity check from centering above
        # exact logits from paying attention to the right thing
        right_attention_logits: Float[Tensor, "d_vocab_out"] = EVOU[  # noqa: F821
            max_tok
        ]
        assert right_attention_logits[max_tok] == 0  # sanity check from centering above
        right_attention_logits_tmp = right_attention_logits.detach().clone()
        right_attention_logits_tmp[max_tok] = float("-inf")
        # maximum added to the wrong logit from paying attention to the right thing
        right_attention_logits_max: Float[Tensor, ""] = (  # noqa: F722
            right_attention_logits_tmp.max()
        )
        for n_copies_nonmax in range(1, n_ctx):
            cur_min_gap = (
                min_gap
                if isinstance(min_gap, int)
                else int(min_gap[: max_tok + 1, max_tok, n_copies_nonmax].min().item())
            )
            if max_tok < cur_min_gap:
                # we must have only the max token
                continue
            # pessimize over the thing we're not supposed to be paying attention to (w.r.t. the token that is non-max that we're paying attention)
            # maximum added to the wrong logit from paying attention to the wrong thing
            wrong_attention_logits: Float[Tensor, ""] = (  # noqa: F722
                EVOU_max_logit_diff[: max_tok - cur_min_gap + 1].max()
            )

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
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
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
def W_EP_direction_for_tricks(
    *,
    W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
    W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
    tricks: Optional[LargestWrongLogitQuadraticConfig] = None,
) -> Optional[Float[Tensor, "d_model"]]:  # noqa F722
    if (
        tricks is None or tricks.EUPU_handling == "svd_query+max_diff"
    ):  # the only one that makes use of the direction
        U, _, Vh = torch.linalg.svd(W_EP @ model.W_U)
        W_EP_svd_query = U[:, 0] @ W_EP
        W_EP_mean_query = W_EP.mean(dim=0)
        if ((W_EP - W_EP_svd_query) @ model.W_U).norm(dim=-1).mean() > (
            (W_EP + W_EP_svd_query) @ model.W_U
        ).norm(dim=-1).mean():
            # svd got the sign wrong :-/
            W_EP_svd_query = -W_EP_svd_query
        return W_EP_svd_query
    return None


# %%
W_EP: Float[Tensor, "d_vocab d_model"] = (  # noqa: F722
    (model.W_E + model.W_pos.mean(dim=0, keepdim=True)).detach().clone()
)
W_U: Float[Tensor, "d_model d_vocab_out"] = model.W_U.detach().clone()  # noqa: F722
W_EP_svd_query = W_EP_direction_for_tricks(W_EP=W_EP, W_U=W_U)
W_EP_mean_query = W_EP.mean(dim=0)


# %%
min_gap = 20
min_right_attention = compute_min_right_attention_quadratic(
    EQKE_query_key + err_accumulator, min_gap=min_gap
)
print(
    f"Complexity of compute_min_right_attention_quadratic: {complexity_of(compute_min_right_attention_quadratic)}"
)  # O(d_vocab^2)
print(
    (min_right_attention[~min_right_attention.isnan()] > err_upper_bound.min())
    .sum()
    .item()
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
    min_right_attention_softmaxed,
    W_EP=W_EP,
    W_U=W_U,
    EVOU=EVOU,
    PVOU=PVOU,
    min_gap=min_gap,
)
print(
    f"Complexity of compute_largest_wrong_logit_quadratic: {complexity_of(compute_largest_wrong_logit_quadratic)}"
)  # O(d_vocab^2 * n_ctx^2)


# %%
@torch.no_grad()
def find_min_gaps(
    *,
    EQKE: Float[Tensor, "d_vocab_q d_vocab_k"],  # noqa: F722
    EQKE_err_upper_bound: Union[
        float, Float[Tensor, ""], Float[Tensor, "d_vocab_q"]  # noqa: F722
    ],
    EQKE_pos_err: Float[Tensor, "d_vocab_q n_ctx"],  # noqa: F722
    attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa: F722
        0
    ].attn.attn_scale,
    position: Optional[int] = None,
    **compute_largest_wrong_logit_quadratic_kwargs,
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    """
    Run the argument across all possible min_gaps, and return the min_gap that works for each query token and max token.

    Since here we are finding the argument/proof rather than verifying it, the complexity does not matter.
    """
    d_vocab_q, d_vocab_k = EQKE.shape
    n_ctx, d_vocab_out = PVOU.shape
    min_gaps = torch.ones((d_vocab_q, d_vocab_k, n_ctx), dtype=torch.long)
    if not isinstance(EQKE_err_upper_bound, Tensor):
        EQKE_err_upper_bound = torch.tensor(EQKE_err_upper_bound)
    if EQKE_err_upper_bound.ndim < 1:
        EQKE_err_upper_bound = EQKE_err_upper_bound[None]
    for min_gap in tqdm(list(reversed(range(1, d_vocab_k))), position=position):
        min_right_attention: Float[
            Tensor, "d_vocab_q d_vocab_max n_ctx_copies_nonmax"  # noqa: F722
        ] = compute_min_right_attention_quadratic(EQKE, min_gap=min_gap)
        min_right_attention_softmaxed = compute_min_softmaxed_right_attention_quadratic(
            min_right_attention - EQKE_err_upper_bound[:, None, None],
            EQKE_pos_err,
            min_gap=min_gap,
            attn_scale=attn_scale,
        )
        largest_wrong_logit = compute_largest_wrong_logit_quadratic(
            min_right_attention_softmaxed,
            min_gap=min_gap,
            **compute_largest_wrong_logit_quadratic_kwargs,
        )
        # if the largest wrong logit is negative, then this gap works
        min_gaps[largest_wrong_logit < 0] = min_gap

    return min_gaps


# %%
@torch.no_grad()
def find_min_gaps_with_EQKE(
    model: HookedTransformer,
    *,
    key_direction: Tensor,
    query_direction: Tensor,
    second_key_direction: Tensor,
    second_query_direction: Tensor,
    W_Q_U: Tensor,
    W_K_U: Tensor,
    EVOU: Float[Tensor, "d_vocab_k d_vocab_out"],  # noqa: F722
    PVOU: Float[Tensor, "n_ctx d_vocab_out"],  # noqa: F722
    W_EP: Float[Tensor, "d_vocab_q d_model"],  # noqa: F722
    W_U: Float[Tensor, "d_model d_vocab_out"],  # noqa: F722
    sanity_check: bool = True,
    atol: float = 1e-4,
    tricks: LargestWrongLogitQuadraticConfig = LargestWrongLogitQuadraticConfig(),
    use_exact_EQKE: bool = False,
    # svd_EUPU: bool = False,
    attn_scale: Union[Float[Tensor, ""], float] = model.blocks[  # noqa: F722
        0
    ].attn.attn_scale,
    position: Optional[int] = None,
) -> Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]:  # noqa: F722
    (
        (EQKE_query_key, err_accumulator),
        EQKE_pos_err,
        (err_upper_bound, (W_E_query_err2, W_Q_err, W_K_errT, W_E_key_err2T)),
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

    err_exact = W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
    cur_EQKE = EQKE_query_key + err_accumulator + (err_exact if use_exact_EQKE else 0)
    EQKE_err_upper_bound = torch.tensor(0) if use_exact_EQKE else err_upper_bound

    W_EP_direction = W_EP_direction_for_tricks(W_EP=W_EP, W_U=W_U, tricks=tricks)
    # cur_EUPU_low_rank = EUPU_lowrank if svd_EUPU else None
    # cur_EUPU_high_rank = torch.zeros_like(EUPU) if svd_EUPU else EUPU
    # cur_EUPU_max_err = torch.tensor(0) if not svd_EUPU else EUPU_err_upper_bound

    return find_min_gaps(
        EQKE=cur_EQKE,
        EQKE_err_upper_bound=EQKE_err_upper_bound,
        EQKE_pos_err=EQKE_pos_err,
        EVOU=EVOU,
        PVOU=PVOU,
        tricks=tricks,
        attn_scale=attn_scale,
        position=position,
        W_EP=W_EP,
        W_U=W_U,
        W_EP_direction=W_EP_direction,
    )


# %%
@torch.no_grad()
def count_correct_sequences(
    largest_wrong_logit: Float[
        Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
    ],
    min_gap: Union[
        int, Integer[Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"]  # noqa: F722
    ] = 1,
) -> int:
    d_vocab_q, d_vocab_max, n_ctx = largest_wrong_logit.shape
    correct_count = 0
    for q_tok in range(d_vocab_q):
        for max_tok in range(d_vocab_max):
            # if the largest wrong logit is positive when the sequence is all max tokens, then pessimizing over position is not adequate for the convexity argument, so we skip these sequences.
            # in practice, we lose 6**4 == 1296 sequences this way, which is 0.0077% of the total
            largest_wrong_logit_in_only_max_sequences = largest_wrong_logit[
                max_tok, max_tok, 0
            ]
            largest_wrong_logit_in_only_max_sequences = (
                largest_wrong_logit_in_only_max_sequences[
                    ~largest_wrong_logit_in_only_max_sequences.isnan()
                ]
            )
            if largest_wrong_logit_in_only_max_sequences.item() > 0:
                # we did not account for these sequences in convexity
                continue
            for n_copies_nonmax in range(n_ctx):
                cur_min_gap = (
                    min_gap
                    if isinstance(min_gap, int)
                    else int(min_gap[q_tok, max_tok, n_copies_nonmax].item())
                )
                # use not to also catch nans
                if (
                    (not largest_wrong_logit[q_tok, max_tok, n_copies_nonmax] < 0)
                    or q_tok > max_tok
                    or (q_tok != max_tok and n_copies_nonmax == 0)
                    or (q_tok != max_tok and max_tok - q_tok < cur_min_gap)
                    or (max_tok == 0 and n_copies_nonmax > 0)
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
# %% [markdown]
# # Sub-cubic Proofs
# %%
# err_exact = W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
min_gaps_lists = {}
with torch.no_grad():
    for use_exact_EQKE in (False, True):
        # for svd_EUPU in (False, True):
        descr = "exact-EQKE" if use_exact_EQKE else ""
        with memoshelve(
            (
                lambda cfg: (
                    cfg,
                    find_min_gaps_with_EQKE(
                        model=model,
                        key_direction=size_direction,
                        query_direction=query_direction,
                        second_key_direction=second_key_direction,
                        second_query_direction=second_query_direction,
                        W_Q_U=W_Q_U,
                        W_K_U=W_K_U,
                        EVOU=EVOU,
                        PVOU=PVOU,
                        W_EP=W_EP,
                        W_U=W_U,
                        tricks=cfg,
                        use_exact_EQKE=use_exact_EQKE,
                        attn_scale=model.blocks[0].attn.attn_scale,
                        position=1,
                    ),
                )
            ),
            # cache={},
            filename=cache_dir
            / f"{Path(__file__).name}.find_min_gaps-{descr}-{cfg_hash_for_filename}",
        )() as find_min_gaps_for:
            min_gaps_lists[use_exact_EQKE] = [
                find_min_gaps_for(cfg)
                for cfg in tqdm(
                    all_configs,
                    position=0,
                    desc="trick cfg",
                )
            ]

        for tricks, min_gaps in min_gaps_lists[use_exact_EQKE]:
            print(f"==========={descr}=============================\nTricks: {tricks}")
            starttime = time.time()
            prooftime = 0.0
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
                tricks=tricks,
            )
            print(
                f"Complexity of decompose_EQKE_error: {complexity_of(decompose_EQKE_error)}"
            )
            try:
                print(f"err_upper_bound: {err_upper_bound.item()}")
            except Exception:
                # print(f"err_upper_bound: {err_upper_bound}")
                print(f"err_upper_bound.max(): {err_upper_bound.max()}")

            if use_exact_EQKE:
                print(f"Complexity of using exact EQKE: O(d_vocab^2 d_model)")
                err_exact = W_E_query_err2 @ W_Q_err @ W_K_errT @ W_E_key_err2T
                cur_EQKE = EQKE_query_key + err_accumulator + err_exact
                EQKE_err_upper_bound = torch.tensor(0)
            else:
                print(f"Complexity of using approximate EQKE: O(d_vocab^2)")
                cur_EQKE = EQKE_query_key + err_accumulator
                EQKE_err_upper_bound = err_upper_bound

            W_EP: Float[Tensor, "d_vocab d_model"] = (  # noqa: F722
                (model.W_E + model.W_pos.mean(dim=0, keepdim=True)).detach().clone()
            )
            W_U: Float[Tensor, "d_model d_vocab_out"] = (  # noqa: F722
                model.W_U.detach().clone()
            )

            prooftime += time.time() - starttime
            # this is not part of the proof checking; the proof is correct regardless of what value is returned, so we don't count the complexity
            W_EP_direction = W_EP_direction_for_tricks(
                W_EP=W_EP, W_U=W_U, tricks=tricks
            )
            starttime = time.time()

            min_right_attention = compute_min_right_attention_quadratic(
                cur_EQKE,
                min_gap=min_gaps,
            )
            print(
                f"Complexity of compute_min_right_attention_quadratic: {complexity_of(compute_min_right_attention_quadratic)}"
            )  # O(d_vocab^2)
            if not isinstance(EQKE_err_upper_bound, Tensor):
                EQKE_err_upper_bound = torch.tensor(EQKE_err_upper_bound)
            if EQKE_err_upper_bound.ndim < 1:
                EQKE_err_upper_bound = EQKE_err_upper_bound[None]
            print(
                (
                    (min_right_attention > EQKE_err_upper_bound[:, None, None])[
                        ~min_right_attention.isnan()
                    ]
                )
                .sum()
                .item()
            )

            min_right_attention_softmaxed = (
                compute_min_softmaxed_right_attention_quadratic(
                    min_right_attention - EQKE_err_upper_bound[:, None, None],
                    EQKE_pos_err,
                    min_gap=min_gaps,
                    attn_scale=model.blocks[0].attn.attn_scale,
                )
            )
            print(
                f"Complexity of compute_min_softmaxed_right_attention: {complexity_of(compute_min_softmaxed_right_attention_quadratic)}"
            )  # O(d_vocab^2 * n_ctx^2)
            # EUPU: Float[Tensor, "d_vocab_q d_vocab_out"] = EU_PU(model)  # noqa: F722
            # print(f"Complexity of EU_PU: {complexity_of(EU_PU)}")  # O(d_vocab^2 * d_model)
            EVOU: Float[Tensor, "d_vocab d_vocab_out"] = all_EVOU(model)  # noqa: F722
            print(
                f"Complexity of EVOU: {complexity_of(all_EVOU)}"
            )  # O(d_vocab^2 * d_model)
            PVOU: Float[Tensor, "n_ctx d_vocab_out"] = all_PVOU(model)  # noqa: F722
            print(
                f"Complexity of PVOU: {complexity_of(all_PVOU)}"
            )  # O(n_ctx * d_vocab * d_model)
            largest_wrong_logit: Float[
                Tensor, "d_vocab_q d_vocab_max n_ctx_nonmax_copies"  # noqa: F722
            ] = compute_largest_wrong_logit_quadratic(
                min_right_attention_softmaxed,
                W_EP=W_EP,
                W_U=W_U,
                EVOU=EVOU,
                PVOU=PVOU,
                min_gap=min_gaps,
                W_EP_direction=W_EP_direction,
                tricks=tricks,
            )
            print(
                f"Complexity of compute_largest_wrong_logit_quadratic: {complexity_of(compute_largest_wrong_logit_quadratic)}"
            )  # O(d_vocab^2 * n_ctx^2)
            accuracy_bound, (
                correct_count,
                total_sequences,
            ) = compute_accuracy_lower_bound_from(largest_wrong_logit, min_gap=min_gaps)
            print(
                f"Accuracy lower bound: {accuracy_bound} ({correct_count} correct sequences of {total_sequences})"
            )
            prooftime += time.time() - starttime
            print(f"Proof time: {prooftime}s")
            left_behind = count_unaccounted_for_by_gap(min_gaps, collapse_n_ctx=False)
            print(
                f"We leave on the floor {left_behind} sequences ({left_behind / total_sequences:.2%})"
            )

        if DISPLAY_PLOTS:
            d_vocab_q, d_vocab_max, n_ctx_nonmax_copies = min_gaps_lists[
                use_exact_EQKE
            ][0][1].shape
            weights = torch.zeros(
                (d_vocab_q, d_vocab_max, n_ctx_nonmax_copies), dtype=torch.long
            )
            for q_tok in range(d_vocab_q):
                for max_tok in range(d_vocab_max):
                    for n_copies_nonmax in range(n_ctx_nonmax_copies):
                        if (
                            (q_tok > max_tok)
                            or (
                                n_copies_nonmax == n_ctx_nonmax_copies - 1
                                and max_tok != q_tok
                            )
                            or (max_tok == 0 and n_copies_nonmax > 0)
                        ):
                            continue
                        if max_tok == 0:
                            assert q_tok == max_tok
                            assert n_copies_nonmax == 0
                            weights[q_tok, max_tok, n_copies_nonmax] = 1
                        weights[q_tok, max_tok, n_copies_nonmax] = (
                            max_tok - 1
                        ) ** n_copies_nonmax * math.comb(
                            model.cfg.n_ctx - 1, n_copies_nonmax
                        )
            for _, v in min_gaps_lists[use_exact_EQKE]:
                v = v.flatten().detach().cpu()
                if v.max().item() == 1:
                    print(f"All gaps are: {set(v.numpy())}")
                    continue
                try:
                    weighted_histogram(
                        v.numpy(),
                        weights.flatten().detach().numpy(),
                        labels={"x": "gap", "y": "count * # sequences"},
                        num_bins=v.max().item(),
                    ).show(RENDERER)
                except Exception as e:
                    etype, value, tb = sys.exc_info()
                    if value is None:
                        traceback.print_exception(e)
                    else:
                        for line in traceback.TracebackException(
                            type(value), value, tb, capture_locals=True
                        ).format():
                            print(line, file=sys.stderr)


# %%
# @title export LaTeX figures
@contextmanager
def texify_title(fig: go.Figure, show: bool = False, renderer=None):
    orig_title = fig.layout.title.text  # type: ignore
    new_title = None
    if orig_title is not None and (
        any(ch in orig_title for ch in "𝔼x̄") or r"$\mathbb{E}$" in orig_title
    ):
        print(f"Replacing 𝔼 in {orig_title}...")
        new_title = (
            orig_title.replace("𝔼", r"\mathbb{E}")
            .replace("x̄", r"\overline{x}")
            .replace("±", r"\pm ")
            .replace("σ", r"\sigma ")
        )
        for word in (
            "None",
            "dim",
            "OV",
            "EQKE",
            "EVOU",
            ".diag",
            " (weighted by sequence count)",
            " (excluding diagonal)",
            "; range",
            "max",
            "min",
        ):
            new_title = new_title.replace(word, r"\text{%s}" % word)
        new_title = re.sub(r"<sub>([^<]*)</sub>", r"_{\1}", new_title)
        new_title = re.sub(r"<sup>([^<]*)</sup>", r"^{\1}", new_title)
        new_title = new_title.replace("{pos}", r"{\text{pos}}")
        lines = new_title.split("<br>")
        if len(lines) > 1 and ":=" not in lines[0]:
            lines = [r"\text{%s}" % lines[0]] + lines[1:]
        elif ": " in lines[0]:
            lines = lines[0].split(": ")
            lines = [r"\text{%s: }%s" % (lines[0], ": ".join(lines[1:]))]
        new_title = r"\\".join(lines)
        new_title = f"${new_title}$"
        print(new_title)
    try:
        if new_title is not None:
            fig.update_layout(title_text=new_title)
            if show:
                fig.show(renderer)
        yield fig
    finally:
        if new_title is not None:
            fig.update_layout(title_text=orig_title)


for k, fig in latex_figures.items():
    fig.update_layout(font_family="Computer Modern")  # Use LaTeX fonts
    with texify_title(fig) as fig:
        if True or any(isinstance(trace, go.Heatmap) for trace in fig.data):
            for ext in (".pdf", ".svg"):
                p = LATEX_FIGURE_PATH / f"{k}{ext}"
                print(f"Saving {p}...")
                fig.write_image(p)
        else:
            p = LATEX_FIGURE_PATH / f"{k}.tex"
            print(f"Saving {p}...")
            tikzplotly.save(p, fig)
            print(fig.to_dict())
# %%
with open(LATEX_VALUES_PATH, "w") as f:
    f.write(to_latex_defs(latex_values))
# %%
