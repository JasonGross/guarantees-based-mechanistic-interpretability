# %%
from __future__ import annotations

# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    print("Not in IPython, not loading autoreload")
# %%
#!sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
# %%
import traceback
import sys
import os
import time
import subprocess
import pandas as pd
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import math
import matplotlib
from typing import (
    Optional,
    Tuple,
    Union,
    Iterator,
)

from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils.memoshelve import memoshelve
from gbmi.exp_max_of_n.train import (
    IterableDatasetCfg,
    MaxOfN,
    MaxOfNDataModule,
    MaxOfNTrainingWrapper,
    train_or_load_model,
)
from gbmi.model import Config
import torch
from tqdm.auto import tqdm
import numpy as np
from torch import Tensor
from transformer_lens import HookedTransformerConfig
from pathlib import Path
from gbmi.utils import default_device
from gbmi.utils.sequences import (
    SequenceDataset,
)

from gbmi.utils.hashing import get_hash_ascii
import gbmi.utils.git as git
import gbmi.exp_max_of_n.verification.cubic as cubic
import gbmi.exp_max_of_n.verification.subcubic as subcubic
import gbmi.exp_max_of_n.analysis.quadratic as analysis_quadratic
import gbmi.exp_max_of_n.analysis.subcubic as analysis_subcubic


# %%
cache_dir = Path(__file__).parent / ".cache"
cache_dir.mkdir(exist_ok=True)
compute_expensive_average_across_many_models: bool = True  # @param {type:"boolean"}
TRAIN_CSV_PATH = Path(__file__).with_suffix("") / "all-models-train-values.csv"
TRAIN_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
csv_path = Path(__file__).with_suffix("") / "all-models-brute-force-values.csv"
csv_path.parent.mkdir(exist_ok=True, parents=True)
CUBIC_CSV_PATH = Path(__file__).with_suffix("") / "all-models-cubic-values.csv"
CUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_CSV_PATH = Path(__file__).with_suffix("") / "all-models-subcubic-values.csv"
SUBCUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
PYTHON_VERSION_PATH = (
    Path(__file__).with_suffix("") / "all-models-values-python-version.txt"
)
PYTHON_VERSION_PATH.parent.mkdir(exist_ok=True, parents=True)
TORCH_VERSION_PATH = (
    Path(__file__).with_suffix("") / "all-models-values-torch-version.txt"
)
TORCH_VERSION_PATH.parent.mkdir(exist_ok=True, parents=True)
GIT_DIFF_PATH = Path(__file__).with_suffix("") / "all-models-values-git-diff-info.diff"
GIT_DIFF_PATH.parent.mkdir(exist_ok=True, parents=True)
GIT_SHA_PATH = Path(__file__).with_suffix("") / "all-models-values-git-sha.txt"
GIT_SHA_PATH.parent.mkdir(exist_ok=True, parents=True)
GIT_SHA_SHORT_PATH = (
    Path(__file__).with_suffix("") / "all-models-values-git-sha-short.txt"
)
GIT_SHA_SHORT_PATH.parent.mkdir(exist_ok=True, parents=True)
N_THREADS: Optional[int] = 64
SHARED_CACHE_STEM = Path(__file__).name.replace("_all_models", "")
# %%
for name, (args, kwargs) in [
    ("lscpu", (("lscpu",), {})),
    ("cat-proc-cpuinfo", (("cat", "/proc/cpuinfo"), {})),
    ("lspci-vga", (("lspci | grep -i vga",), dict(shell=True))),
    ("nvidia-smi", (("nvidia-smi",), {})),
]:
    try:
        print(f"Running {name}...")
        result = subprocess.check_output(args, **kwargs).decode()
    except Exception as e:
        print(f"Error running {name}: {e}")
    else:
        with open(Path(__file__).with_suffix("") / f"{name}.txt", "w") as f:
            f.write(result)

with open(GIT_DIFF_PATH, "w") as f:
    f.write(git.get_diff())

with open(GIT_SHA_PATH, "w") as f:
    f.write(git.get_head_sha(short=False))

with open(GIT_SHA_SHORT_PATH, "w") as f:
    f.write(git.get_head_sha(short=True))

with open(PYTHON_VERSION_PATH, "w") as f:
    f.write(sys.version)

with open(TORCH_VERSION_PATH, "w") as f:
    f.write(torch.__version__)
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
datamodules = {seed: MaxOfNDataModule(cfg) for seed, cfg in cfgs.items()}
cfg_hashes_for_filename = {
    seed: f"{seed}_{cfg_hashes[seed].replace('/', '__SLASH__')}"
    for seed, cfg in cfgs.items()
}
# %%
with memoshelve(
    train_or_load_model,
    filename=cache_dir / f"{SHARED_CACHE_STEM}.train_or_load_model",
    get_hash=get_hash_ascii,
)() as memo_train_or_load_model:
    runtime_models = {}

    def _handle_memo_train_or_load_model(arg):
        seed, cfg = arg
        try:
            runtime_models[seed] = memo_train_or_load_model(cfg, force="load")
        except Exception as e:
            print(f"Error loading model for seed {seed}: {e}")

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(_handle_memo_train_or_load_model, tqdm(cfgs.items()))

# %%
training_wrappers = {
    seed: MaxOfNTrainingWrapper(cfgs[seed], model)
    for seed, (_runtime, model) in runtime_models.items()
}


# training_wrapper.run_batch = Memoize(training_wrapper.run_batch, name=f"{__file__}.training_wrapper.run_batch", use_pandas=False, use_shelf=True)  # type: ignore
# %%
def update_csv_with_rows(
    csv_path: Path,
    new_data: list[dict[str, Union[float, int, str]]],
    columns: list[str],
    *,
    subset: str = "seed",
):
    if os.path.exists(csv_path):
        results = pd.read_csv(csv_path)

    new_df = pd.DataFrame(new_data, columns=columns)
    if results.empty:
        results = new_df
    else:
        results = pd.concat([results, new_df], ignore_index=True).drop_duplicates(
            subset=subset, keep="last"
        )
    results.to_csv(csv_path, index=False)


def update_csv(
    csv_path: Path,
    data: dict[int, dict[str, Union[float, int, str]]],
    columns: list[str],
    *,
    subset: str = "seed",
):
    new_data = [data[seed] for seed in sorted(data.keys())]
    update_csv_with_rows(csv_path, new_data, columns, subset=subset)


# %% [markdown]
# # Training stats
# %%
train_total_loss = {}
train_total_accuracy = {}
train_total_samples = {}
train_measurement_deterministic: bool = False  # @param {type:"boolean"}
train_average_loss = {}
train_average_accuracy = {}


# loop for computing overall loss and accuracy
@torch.no_grad()
def _run_train_batch_loss_accuracy(
    seed: int, i: int, batch_size: int, *, dataloader_iter: Iterator
) -> Tuple[float, float, int]:
    xs, ys = next(dataloader_iter)
    device = default_device(deterministic=train_measurement_deterministic)
    xs.to(device)
    ys.to(device)
    loss, accuracy = training_wrappers[seed].run_batch((xs, ys), log_output=False)
    loss = loss.item()
    return loss, accuracy, batch_size


def train_seed(seed: int, *, pbar: tqdm):
    train_total_loss[seed] = 0.0
    train_total_accuracy[seed] = 0.0
    train_total_samples[seed] = 0

    datamodule = datamodules[seed]
    datamodule.setup("train")
    dataloader = datamodule.train_dataloader()
    dataloader_iter = iter(dataloader)
    with memoshelve(
        partial(_run_train_batch_loss_accuracy, dataloader_iter=dataloader_iter),
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.run_batch_loss_accuracy-{cfg_hashes_for_filename[seed]}-{train_measurement_deterministic}",
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    )() as run_batch_loss_accuracy:
        for i in range(0, len(dataloader)):
            loss, accuracy, size = run_batch_loss_accuracy(seed, i, cfgs[seed].batch_size)  # type: ignore
            # Accumulate loss and accuracy
            train_total_loss[seed] += loss * size
            train_total_accuracy[seed] += accuracy * size
            train_total_samples[seed] += size
            pbar.update(1)

    # Calculate average loss and accuracy
    train_average_loss[seed] = train_total_loss[seed] / train_total_samples[seed]
    train_average_accuracy[seed] = (
        train_total_accuracy[seed] / train_total_samples[seed]
    )


def _handle_train_seed(seed: int, *, pbar: tqdm):
    try:
        return train_seed(seed, pbar=pbar)
    except Exception as e:
        print(f"Error training seed {seed}: {e}")
        traceback.print_exc()


for datamodule in datamodules.values():
    datamodule.setup("train")

total_batches = sum(
    len(datamodules[seed].train_dataloader()) for seed in runtime_models.keys()
)

with tqdm(total=total_batches, desc="batches for training", position=0) as pbar:
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(partial(_handle_train_seed, pbar=pbar), runtime_models.keys())

# %%
# load csv
train_columns = ["seed", "loss", "accuracy", "model-seed", "dataset-seed"]
if os.path.exists(TRAIN_CSV_PATH):
    train_results = pd.read_csv(TRAIN_CSV_PATH)
else:
    train_results = pd.DataFrame(columns=train_columns)

train_data = {
    seed: {
        "seed": seed,
        "loss": train_average_loss[seed],
        "accuracy": train_average_accuracy[seed],
        "model-seed": cfgs[seed].experiment.model_config.seed,
        "dataset-seed": datamodules[seed].dataset_seed,
    }
    for seed in runtime_models.keys()
}

update_csv(TRAIN_CSV_PATH, train_data, columns=train_columns)

# %% [markdown]
# # Brute Force Proof
# %%
brute_force_columns = [
    "seed",
    "loss",
    "accuracy",
    "num_correct",
    "num_incorrect",
    "cpu",
    "duration",
]
if os.path.exists(csv_path):
    brute_force_results = pd.read_csv(csv_path)
else:
    brute_force_results = pd.DataFrame(columns=brute_force_columns)

brute_force_proof_deterministic: bool = True  # @param {type:"boolean"}

batch_size = 4096  # 16_384 # 8182

unknown_seeds = set(runtime_models.keys()) - set(brute_force_results["seed"])
known_seeds = set(runtime_models.keys()) - unknown_seeds
brute_force_data = {
    seed: brute_force_results[brute_force_results["seed"] == seed].iloc[0].to_dict()
    for seed in known_seeds
}


def get_brute_force_for(seed: int, *, pbar: tqdm):
    cfg = cfgs[seed]
    cfg_hash = cfg_hashes[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    runtime, model = runtime_models[seed]
    training_wrapper = training_wrappers[seed]
    assert cfg.experiment.model_config.seed is not None
    all_tokens_dataset = SequenceDataset(
        seq_len=model.cfg.n_ctx, vocab_size=model.cfg.d_vocab
    )
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    total_duration = 0.0
    # all_incorrect_sequences = []

    # loop for computing overall loss and accuracy
    @torch.no_grad()
    def _run_batch_loss_accuracy(
        i: int, batch_size: int, return_incorrect_sequences: bool = True
    ) -> Tuple[
        Union[Tuple[float, float, int], Tuple[Tuple[float, float, int], Tensor]], float
    ]:
        batch = all_tokens_dataset[i : i + batch_size]
        size = batch.shape[0]
        device = default_device(deterministic=brute_force_proof_deterministic)
        batch.to(device)
        duration = 0.0
        start = time.time()
        labels = training_wrapper.config.experiment.get_ground_truth(batch)
        xs, ys, y_preds = training_wrapper.compute_batch((batch, labels), device=device)
        loss = training_wrapper.loss_fn(
            y_preds, ys, log_softmax=training_wrapper.log_softmax
        ).item()
        full_accuracy = training_wrapper.acc_fn_per_seq(y_preds, ys)
        accuracy = full_accuracy.float().mean().item()
        duration += time.time() - start
        if return_incorrect_sequences:
            return ((loss, accuracy, size), xs[~full_accuracy]), duration
        return (loss, accuracy, size), duration

    with memoshelve(
        _run_batch_loss_accuracy,
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.run_batch_loss_accuracy-{cfg_hash_for_filename}-{brute_force_proof_deterministic}",
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    )() as run_batch_loss_accuracy:
        for i in range(0, len(all_tokens_dataset), batch_size):
            ((loss, accuracy, size), incorrect_sequences), duration = run_batch_loss_accuracy(i, batch_size)  # type: ignore
            total_duration += duration
            # Accumulate loss and accuracy
            start = time.time()
            total_loss += loss * size
            total_accuracy += accuracy * size
            total_samples += size
            total_duration += time.time() - start
            # all_incorrect_sequences.append(incorrect_sequences)
            pbar.update(batch_size)

    # Calculate average loss and accuracy
    average_loss = total_loss / total_samples
    average_accuracy = total_accuracy / total_samples
    # incorrect_sequences = torch.cat(all_incorrect_sequences, dim=0)
    num_correct_sequences = int(round(average_accuracy * all_tokens_dataset.length))
    num_incorrect_sequences = all_tokens_dataset.length - num_correct_sequences

    row = {
        "seed": seed,
        "cpu": brute_force_proof_deterministic,
        "loss": average_loss,
        "accuracy": average_accuracy,
        "num_correct": num_correct_sequences,
        "num_incorrect": num_incorrect_sequences,
        "duration": total_duration,
    }
    return row


def _handle_brute_force_for(seed: int, *, pbar: tqdm):
    try:
        brute_force_data[seed] = get_brute_force_for(seed, pbar=pbar)
    except Exception as e:
        print(f"Error computing brute force proof for seed {seed}: {e}")
        traceback.print_exc()


lengths = [
    len(
        SequenceDataset(
            seq_len=runtime_models[seed][1].cfg.n_ctx,
            vocab_size=runtime_models[seed][1].cfg.d_vocab,
        )
    )
    for seed in runtime_models.keys()
]

total_batches = sum(
    length - length % batch_size + (batch_size if length % batch_size != 0 else 0)
    for length in lengths
)

with tqdm(total=total_batches, desc="batches for brute force", position=0) as pbar:
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(partial(_handle_brute_force_for, pbar=pbar), runtime_models.keys())

update_csv(csv_path, brute_force_data, columns=brute_force_columns)

# %% [markdown]
# # Cubic proof

# %%
cubic_columns = [
    "seed",
    "accuracy-bound",
    "correct-count-lower-bound",
    "duration-proof-search",
    "duration",
]
if os.path.exists(CUBIC_CSV_PATH):
    cubic_results = pd.read_csv(CUBIC_CSV_PATH)
else:
    cubic_results = pd.DataFrame(columns=cubic_columns)

unknown_seeds = set(runtime_models.keys()) - set(cubic_results["seed"])
known_seeds = set(runtime_models.keys()) - unknown_seeds
cubic_data = {
    seed: cubic_results[cubic_results["seed"] == seed].iloc[0].to_dict()
    for seed in known_seeds
}


def get_cubic_row(seed: int) -> dict:
    cfg = cfgs[seed]
    cfg_hash = cfg_hashes[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    runtime, model = runtime_models[seed]
    training_wrapper = training_wrappers[seed]
    assert cfg.experiment.model_config.seed is not None

    # loop for computing overall loss and accuracy
    @torch.no_grad()
    def _find_proof() -> Tuple[dict, float]:
        start = time.time()
        cubic_proof_args = cubic.find_proof(model)
        duration = time.time() - start
        return cubic_proof_args, duration

    with memoshelve(
        _find_proof,
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.cubic_find_proof-{cfg_hash_for_filename}",
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    )() as find_proof:
        cubic_proof_args, duration_proof_search = find_proof()

    with memoshelve(
        partial(cubic.verify_proof, model),
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.cubic_verify_proof-{cfg_hash_for_filename}",
        get_hash_mem=(lambda x: 0),
        get_hash=(lambda x: 0),
    )() as verify_proof:
        cubic_proof_results = verify_proof(cubic_proof_args)

    # largest_wrong_logit_cubic = cubic_proof_results["largest_wrong_logit"]
    return {
        "seed": seed,
        "accuracy-bound": cubic_proof_results["accuracy_lower_bound"],
        "correct-count-lower-bound": cubic_proof_results["correct_count_lower_bound"],
        "duration-proof-search": duration_proof_search,
        "duration": cubic_proof_results["prooftime"],
    }


def _handle_cubic(seed: int, *, pbar: tqdm):
    try:
        cubic_data[seed] = get_cubic_row(seed)
        pbar.update(1)
    except Exception as e:
        print(f"Error computing cubic proof for seed {seed}: {e}")
        traceback.print_exc()


with tqdm(
    total=len(runtime_models.keys()), desc="batches for cubic", position=0
) as pbar:
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(partial(_handle_cubic, pbar=pbar), runtime_models.keys())

update_csv(CUBIC_CSV_PATH, cubic_data, columns=cubic_columns)

# %% [markdown]
# # Sub-cubic Proofs
# %%
try_all_configurations: bool = True  # @param {type:"boolean"}
use_tricks: bool = True  # @param {type:"boolean"}
if try_all_configurations:
    all_configs = list(enumerate_dataclass_values(LargestWrongLogitQuadraticConfig))
elif use_tricks:
    all_configs = [LargestWrongLogitQuadraticConfig()]
else:
    all_configs = [LargestWrongLogitQuadraticConfig.OFF]
# %%
# %%
subcubic_columns = [
    "seed",
    "accuracy-bound",
    "duration-proof-search",
    "duration",
    "tricks",
    "err-upper-bound",
    "err-upper-bound-is-max",
    "total-sequences",
    "dropped-sequences",
    "dropped-sequences-frac",
    "most-gap-below-value",
    "most-gap-below-value-frac",
    "most-gap-below-value-num-std",
    "max-gap",
]
if os.path.exists(SUBCUBIC_CSV_PATH):
    subcubic_results = pd.read_csv(SUBCUBIC_CSV_PATH)
else:
    subcubic_results = pd.DataFrame(columns=subcubic_columns)

unknown_seeds = set(runtime_models.keys()) - set(subcubic_results["seed"])
known_seeds = set(runtime_models.keys()) - unknown_seeds
subcubic_data = {
    seed: subcubic_results[subcubic_results["seed"] == seed].iloc[0].to_dict()
    for seed in known_seeds
}


@torch.no_grad()
def try_all_proofs_subcubic(seed: int, *, pbar: tqdm) -> list[dict]:
    cfg = cfgs[seed]
    cfg_hash = cfg_hashes[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    runtime, model = runtime_models[seed]
    training_wrapper = training_wrappers[seed]
    assert cfg.experiment.model_config.seed is not None

    min_gaps_lists = {}

    rows = []

    shared_proof_search_duration = 0.0
    start = time.time()
    W_EP_direction_kwargs = analysis_quadratic.W_EP_direction_for_tricks_kwargs(model)
    find_min_gaps_kwargs = analysis_subcubic.find_min_gaps_with_EQKE_kwargs(model)
    size_and_query_directions_kwargs = analysis_quadratic.find_EKQE_error_directions(
        model
    )
    shared_proof_search_duration += time.time() - start
    for use_exact_EQKE in (False, True):
        # for svd_EUPU in (False, True):
        descr = "exact-EQKE" if use_exact_EQKE else ""
        filedescr = "-exact-EQKE--" if use_exact_EQKE else ""
        latexdescr = "ExactEQKE" if use_exact_EQKE else ""
        with memoshelve(
            (
                lambda cfg: (
                    cfg,
                    *analysis_subcubic.find_min_gaps_with_EQKE(
                        model=model,
                        **find_min_gaps_kwargs,  # type: ignore
                        **size_and_query_directions_kwargs,
                        tricks=cfg,
                        use_exact_EQKE=use_exact_EQKE,
                        pbar=pbar,
                        record_time=True,
                    ),
                )
            ),
            # cache={},
            filename=cache_dir
            / f"{SHARED_CACHE_STEM}.find_min_gaps-{descr}-{cfg_hash_for_filename}",
        )() as find_min_gaps_for:
            min_gaps_lists[use_exact_EQKE] = [
                find_min_gaps_for(cfg)
                for cfg in all_configs
                if cfg.attention_error_handling == "max_diff_exact"
                or not use_exact_EQKE  # don't bother with other ways of handling attention when we're just going to be using exact attention error handling anyway
            ]

        for tricks, min_gaps, proof_search_duration in min_gaps_lists[use_exact_EQKE]:
            proof_search_duration += shared_proof_search_duration
            # print(
            #     f"==========={descr}=============================\nTricks: {tricks}"
            # )
            # this is not part of the proof checking; the proof is correct regardless of what value is returned, so we don't count the complexity
            start = time.time()
            W_EP_direction = analysis_quadratic.W_EP_direction_for_tricks(
                **W_EP_direction_kwargs, tricks=tricks
            )
            proof_search_duration += time.time() - start
            proof_results = subcubic.verify_proof(
                model,
                W_EP_direction=W_EP_direction,
                **size_and_query_directions_kwargs,  # type: ignore
                use_exact_EQKE=use_exact_EQKE,
                min_gaps=min_gaps,
                tricks=tricks,
                sanity_check=False,
            )
            err_upper_bound = proof_results["err_upper_bound"]
            prooftime = proof_results["prooftime"]
            accuracy_bound = proof_results["accuracy_lower_bound"]
            total_sequences = proof_results["total_sequences"]
            left_behind = proof_results["left_behind"]

            try:
                # err_upper_bound_key = f"SubcubicErrUpperBound{tricks.transform_description(tricks.attention_error_handling, latex=True)}Float"
                err_upper_bound_value = err_upper_bound.item()
                err_upper_bound_is_max = False
                # print(f"err_upper_bound: {err_upper_bound_value}")
            except Exception:
                # print(f"err_upper_bound: {err_upper_bound}")
                # err_upper_bound_key = f"SubcubicErrUpperBoundMax{tricks.transform_description(tricks.attention_error_handling, latex=True)}Float"
                err_upper_bound_value = err_upper_bound.max().item()
                err_upper_bound_is_max = True
                # print(f"err_upper_bound.max(): {err_upper_bound_value}")

            # if DISPLAY_PLOTS:
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

                v = min_gaps.flatten().detach().cpu()
                mean = np.average(v.numpy(), weights=weights.flatten().numpy())
                std = np.average(
                    (v - mean).numpy() ** 2,
                    weights=weights.flatten().numpy(),
                )
                num_std = 1.5
                most_below_value = int(math.ceil(mean + num_std * std))
                frac_below = (
                    weights.flatten()[v <= most_below_value].sum() / weights.sum()
                ).item()

                row = {
                    "seed": seed,
                    "accuracy-bound": accuracy_bound,
                    "duration-proof-search": proof_search_duration,
                    "duration": prooftime,
                    "tricks": latexdescr + tricks.short_description(latex=True),
                    "err-upper-bound": err_upper_bound_value,
                    "err-upper-bound-is-max": err_upper_bound_is_max,
                    "total-sequences": total_sequences,
                    "dropped-sequences": left_behind,
                    "dropped-sequences-frac": left_behind / total_sequences,
                    "most-gap-below-value": most_below_value,
                    "most-gap-below-value-frac": frac_below,
                    "most-gap-below-value-num-std": num_std,
                    "max-gap": v.max().item(),
                }

                rows.append(row)
                pbar.update(1)
    return rows


def _handle_subcubic(seed: int, *, pbar: tqdm):
    try:
        subcubic_data[seed] = try_all_proofs_subcubic(seed, pbar=pbar)
    except Exception as e:
        print(f"Error computing subcubic proof for seed {seed}: {e}")
        traceback.print_exc()


total_count = sum(
    (1 + model.cfg.d_vocab)
    * sum(
        2 if cfg.attention_error_handling == "max_diff_exact" else 1
        for cfg in all_configs
    )
    for _runtime, model in runtime_models.values()
)

with tqdm(total=total_count, desc="configurations for subcubic", position=0) as pbar:
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(partial(_handle_subcubic, pbar=pbar), runtime_models.keys())

new_data = []
for seed in sorted(subcubic_data.keys()):
    new_data.extend(subcubic_data[seed])

update_csv_with_rows(csv_path, new_data, subcubic_columns)
