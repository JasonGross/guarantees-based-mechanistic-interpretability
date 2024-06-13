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
#!sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super pdfcrop optipng pngcrush
# %%
import traceback
import gc
import csv
from collections import defaultdict
import random
import sys
import os
import re
from contextlib import contextmanager
import time
import subprocess
import pandas as pd
from itertools import chain
from functools import partial, reduce
from concurrent.futures import ThreadPoolExecutor
import math
import matplotlib
import scipy.stats as stats
from typing import (
    Any,
    Literal,
    Optional,
    Tuple,
    Union,
    Iterator,
    Callable,
)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.figure
import seaborn as sns
import tikzplotly
import tikzplotlib
import matplotlib
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
from gbmi.analysis_tools.utils import (
    pm_round,
    pm_mean_std,
    data_summary,
    data_summary_percentiles,
)
from gbmi.analysis_tools.plot import scatter
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils.lowrank import LowRankTensor
import gbmi.utils.ein as ein
import gbmi.utils.images as image_utils
from gbmi.utils.images import trim_plotly_figure
from gbmi.utils.memoshelve import memoshelve
from gbmi.utils.latex_export import (
    to_latex_defs,
    latex_values_of_counter,
    latex_values_of_instruction_count,
    format_float_full_precision,
)
from gbmi.exp_max_of_n.analysis import (
    find_second_singular_contributions,
    find_size_and_query_direction,
)
from gbmi.exp_max_of_n.plot import display_basic_interpretation
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
from jaxtyping import Float, Integer
from torch import Tensor
import pandas as pd
import plotly.express as px
from transformer_lens import HookedTransformerConfig, HookedTransformer
from pathlib import Path
from gbmi.utils import default_device, shuffle_tensor
from gbmi.utils.sequences import (
    SequenceDataset,
)
from gbmi.verification_tools.decomp import (
    factor_contribution,
    bound_max_row_diff_by_SVD,
)

from gbmi.verification_tools.general import EU_PU
from gbmi.verification_tools.l1h1 import (
    all_EQKE,
    all_EQKP,
    all_EVOU,
    all_PVOU,
)
from gbmi.verification_tools.utils import complexity_of
from gbmi.utils.hashing import get_hash_ascii
import gbmi.utils.git as git
import gbmi.exp_max_of_n.verification.cubic as cubic
import gbmi.exp_max_of_n.verification.subcubic as subcubic
import gbmi.exp_max_of_n.verification.quadratic as quadratic
import gbmi.exp_max_of_n.analysis.quadratic as analysis_quadratic
import gbmi.exp_max_of_n.analysis.subcubic as analysis_subcubic
import gbmi.utils.instructions as instructions
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
from gbmi.verification_tools.l1h1 import all_EVOU, all_PVOU
from gbmi.verification_tools.general import EU_PU
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
import gbmi.exp_max_of_n.verification.quadratic as quadratic
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils.memoshelve import memoshelve, uncache as memoshelve_uncache
from gbmi.analysis_tools.plot import (
    EVOU_max_logit_diff,
)
from gbmi.exp_max_of_n.plot import (
    EVOU_max_minus_diag_logit_diff,
    attention_difference_over_gap,
    make_better_slides_plots_00,
    display_EQKE_SVD_analysis,
)
from gbmi.exp_max_of_n.analysis import (
    find_second_singular_contributions,
    find_size_and_query_direction,
)

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
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
from gbmi.utils import default_device
from gbmi.utils.sequences import (
    SequenceDataset,
)
from gbmi.utils.latex_export import (
    to_latex_defs,
    latex_values_of_counter,
    latex_values_of_instruction_count,
)
from gbmi.exp_max_of_n.plot import (
    scatter_attention_difference_vs_gap,
    hist_attention_difference_over_gap,
    hist_EVOU_max_minus_diag_logit_diff,
)
from gbmi.analysis_tools.plot import (
    hist_EVOU_max_logit_diff,
    weighted_histogram,
    Colorscale,
    colorscale_to_cmap,
    imshow,
    line,
)
from gbmi.utils import default_device, dropnan, shuffle_tensors, shuffle_tensor
from gbmi.utils.gc import PeriodicGarbageCollector
from gbmi.utils.hashing import get_hash_ascii
import gbmi.utils.git as git
import gbmi.exp_max_of_n.verification.cubic as cubic
import gbmi.exp_max_of_n.verification.subcubic as subcubic
import gbmi.exp_max_of_n.analysis.quadratic as analysis_quadratic
import gbmi.exp_max_of_n.analysis.subcubic as analysis_subcubic
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
import gbmi.utils.ein as ein
import gbmi.utils.instructions as instructions
from gbmi.utils.instructions import (
    InstructionCount,
    CountTensor,
    PatchTorch,
    CountHookedTransformer,
    PerfCounter,
    PerfCollector,
    int_or_value,
    CountTensorOperations,
    PERF_WORKING,
)

# %%
parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default="50,104,123,519,742,913,1185,1283,1412,1490,1681,1696,1895,1951,2236,2306,2345,2549,2743,2773,3175,3254,3284,4157,4305,4430,4647,4729,4800,4810,5358,5615,5781,5928,6082,6155,6159,6204,6532,6549,6589,6910,7098,7238,7310,7467,7790,7884,8048,8299,8721,8745,8840,8893,9132,9134,9504,9816,10248,11124,11130,11498,11598,11611,12141,12287,12457,12493,12552,12561,13036,13293,13468,13654,13716,14095,14929,15043,15399,15622,15662,16069,16149,16197,16284,17080,17096,17194,17197,18146,18289,18668,19004,19093,19451,19488,19538,19917,20013,20294,20338,20415,20539,20751,20754,20976,21317,21598,22261,22286,22401,22545,23241,23367,23447,23633,23696,24144,24173,24202,24262,24438,24566,25516,26278,26374,26829,26932,27300,27484,27584,27671,27714,28090,28716,28778,29022,29052,29110,29195,29565,29725,29726,30371,30463,30684,30899,31308,32103,32374,32382",
    help="Comma-separated list of seeds to use",
)
parser.add_argument(
    "-j", dest="n_threads", type=int, default=1, help="number of threads"
)
parser.add_argument(
    "--no-perf",
    action="store_const",
    const=True,
    default=None,
    help="Forcibly disable perf",
)
parser.add_argument(
    "--ignore-csv",
    action="store_const",
    const=True,
    default=None,
    help="Recompute seeds that appear in csvs",
)
parser.add_argument(
    "--plots",
    action=BooleanOptionalAction,
    default=True,
    help="Include plots",
)
cli_args = parser.parse_args(None if ipython is None else ["--ignore-csv"])
# %%
cache_dir = Path(__file__).parent / ".cache"
cache_dir.mkdir(exist_ok=True)
OVERWRITE_CSV_FROM_CACHE: bool = not cli_args.ignore_csv  # @param {type:"boolean"}
compute_expensive_average_across_many_models: bool = True  # @param {type:"boolean"}
TRAIN_CSV_PATH = Path(__file__).with_suffix("") / "all-models-train-values.csv"
TRAIN_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
BRUTE_FORCE_CSV_PATH = (
    Path(__file__).with_suffix("") / "all-models-brute-force-values.csv"
)
BRUTE_FORCE_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
CUBIC_CSV_PATH = Path(__file__).with_suffix("") / "all-models-cubic-values.csv"
CUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_CSV_PATH = Path(__file__).with_suffix("") / "all-models-subcubic-values.csv"
SUBCUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_ANALYSIS_CSV_PATH = (
    Path(__file__).with_suffix("") / "all-models-subcubic-analysis-values.csv"
)
SUBCUBIC_ANALYSIS_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
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
LATEX_VALUES_PATH = Path(__file__).with_suffix("") / "all-models-values.tex"
LATEX_VALUES_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_VALUES_DATATABLE_PATH = (
    Path(__file__).with_suffix("") / "all-models-all-values.csv"
)
LATEX_VALUES_DATATABLE_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_FIGURE_PATH = Path(__file__).with_suffix("") / "figures"
LATEX_FIGURE_PATH.mkdir(exist_ok=True, parents=True)
LATEX_TIKZPLOTLIB_PREAMBLE_PATH = (
    Path(__file__).with_suffix("") / "tikzplotlib-preamble.tex"
)
LATEX_TIKZPLOTLIB_PREAMBLE_PATH.parent.mkdir(exist_ok=True, parents=True)
SHARED_CACHE_STEM = Path(__file__).name.replace("_all_models", "")
N_THREADS: Optional[int] = cli_args.n_threads
DISPLAY_PLOTS: bool = False  # @param {type:"boolean"}
SAVE_PLOTS: bool = cli_args.plots
RENDERER: Optional[str] = "png"  # @param ["png", None]
PLOT_WITH: Literal["plotly", "matplotlib"] = (  # @param ["plotly", "matplotlib"]
    "matplotlib"
)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams[
    "text.latex.preamble"
] = r"""\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xfrac}
\usepackage{lmodern}
\providecommand{\dmodel}{\ensuremath{d_{\mathrm{model}}}}
\providecommand{\dhead}{\ensuremath{d_{\mathrm{head}}}}
\providecommand{\dvocab}{\ensuremath{d_{\mathrm{vocab}}}}
\providecommand{\barWE}{\ensuremath{\mathbf{\bar{E}}}}
\providecommand{\qWE}{\ensuremath{\mathbf{E}_q}}
"""
default_OV_colorscale_2024_03_26: Colorscale = px.colors.get_colorscale(
    "RdBu"
)  # px.colors.get_colorscale("Picnic_r")
# default_OV_matplotlib_colorscale_2024_03_26: Colorscale = 'bwr_r'
default_QK_colorscale_2024_03_26: Colorscale = [
    [0, "#ff0000"],
    [0.25, "#ff8247"],
    [0.5, "white"],
    [0.75, "#ffc100"],
    [1, "#ff9c05"],
]
default_OV_colorscale: Colorscale = default_OV_colorscale_2024_03_26
default_QK_colorscale: Colorscale = default_QK_colorscale_2024_03_26
default_QK_SVD_colorscale: Colorscale = default_QK_colorscale
# %%
if cli_args.no_perf:
    PERF_WORKING = False
# %%
latex_values: dict[str, Union[int, float, str]] = {}
latex_all_values_by_value: dict[str, dict[int, Union[int, float, str]]] = defaultdict(
    dict
)
latex_figures: dict[str, Union[go.Figure, matplotlib.figure.Figure]] = {}
latex_externalize_tables: dict[str, bool] = {}
latex_only_externalize_tables: dict[str, bool] = {}


# %%
def maybe_parallel_map(func, *args):
    if N_THREADS is None or N_THREADS <= 1:
        result = list(map(func, *args))
    else:
        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            result = executor.map(func, *args)
    gc.collect()
    return result


# %%


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
def remove_titles(
    fig: Union[go.Figure, matplotlib.figure.Figure],
    plot_with: Literal["plotly", "matplotlib"] = PLOT_WITH,
):
    match plot_with:
        case "matplotlib":
            for ax in fig.axes:
                ax.set_title("")
        case "plotly":
            fig.update_layout(title_text="")
    return fig


# %%
# hack around newlines of black formatting
seeds = (
    sorted(set(map(int, cli_args.seeds.split(","))))
    if compute_expensive_average_across_many_models
    else []
)
if 123 in seeds:
    seeds = [123] + [s for s in seeds if s != 123]
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
    for seed in seeds
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

    maybe_parallel_map(_handle_memo_train_or_load_model, tqdm(cfgs.items()))
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
    *,
    columns: list[str],
    subset: str | list[str] = "seed",
):
    results = None
    if os.path.exists(csv_path):
        results = pd.read_csv(csv_path)

    new_df = pd.DataFrame(new_data, columns=columns)
    if results is None or results.empty:
        results = new_df
    elif not new_df.empty:
        results = pd.concat([results, new_df], ignore_index=True).drop_duplicates(
            subset=subset, keep="last"
        )
    results.to_csv(csv_path, index=False)


def update_csv(
    csv_path: Path,
    data: dict[int, dict[str, Union[float, int, str]]],
    columns: list[str],
    *,
    subset: str | list[str] = "seed",
):
    new_data = [data[seed] for seed in sorted(data.keys())]
    update_csv_with_rows(csv_path, new_data, columns=columns, subset=subset)


# %%
latex_values |= {
    f"{percentile_name}PercentileFloat": percentile_value
    for percentile_name, percentile_value in zip(*data_summary_percentiles())
}
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
    # with PeriodicGarbageCollector(60):
    maybe_parallel_map(
        partial(_handle_train_seed, pbar=pbar), sorted(runtime_models.keys())
    )
# %%
# load csv
train_columns = ["seed", "loss", "accuracy", "model-seed", "dataset-seed"]

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

# %%
num_seeds = len(train_average_loss)
avg_train_average_loss = sum(train_average_loss.values()) / num_seeds
avg_train_average_accuracy = sum(train_average_accuracy.values()) / num_seeds
std_dev_train_average_loss = float(np.std(list(train_average_loss.values())))
std_dev_train_average_accuracy = float(np.std(list(train_average_accuracy.values())))
latex_values["NumSeeds"] = num_seeds
assert all(isinstance(seed, int) for seed in train_average_accuracy.keys())
assert all(isinstance(seed, int) for seed in train_average_loss.keys())
latex_all_values_by_value["TrainAccuracyFloat"] = train_average_accuracy
latex_all_values_by_value["TrainLossFloat"] = train_average_loss
latex_values |= data_summary(train_average_accuracy, prefix="TrainAccuracy")
latex_values |= data_summary(train_average_loss, prefix="TrainLoss")

# %% [markdown]
# # Brute Force Proof
# %%
all_tokens_datasets = {
    seed: SequenceDataset(seq_len=model.cfg.n_ctx, vocab_size=model.cfg.d_vocab)
    for seed, (_runtime, model) in runtime_models.items()
}
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
if os.path.exists(BRUTE_FORCE_CSV_PATH):
    brute_force_results = pd.read_csv(BRUTE_FORCE_CSV_PATH)
else:
    brute_force_results = pd.DataFrame(columns=brute_force_columns)

brute_force_proof_deterministic: bool = True  # @param {type:"boolean"}

batch_size = 4096  # 16_384 # 8182

all_seeds = set(runtime_models.keys())
unknown_seeds = all_seeds - set(brute_force_results["seed"])
known_seeds = all_seeds - unknown_seeds
relevant_seeds = all_seeds if OVERWRITE_CSV_FROM_CACHE else unknown_seeds
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
    all_tokens_dataset = all_tokens_datasets[seed]
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
        cache={},
    )() as run_batch_loss_accuracy_heavy:

        def _run_batch_loss_accuracy_lightweight(*args, **kwargs):
            res = run_batch_loss_accuracy_heavy(*args, **kwargs)
            ((loss, accuracy, size), incorrect_sequences), duration = res
            return (loss, accuracy, size), duration

        with memoshelve(
            _run_batch_loss_accuracy_lightweight,
            filename=cache_dir
            / f"{SHARED_CACHE_STEM}.run_batch_loss_accuracy-lightweight-{cfg_hash_for_filename}-{brute_force_proof_deterministic}",
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        )() as run_batch_loss_accuracy:
            for i in range(0, len(all_tokens_dataset), batch_size):
                (loss, accuracy, size), duration = run_batch_loss_accuracy(i, batch_size)  # type: ignore
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
    for seed in relevant_seeds
]

total_batches = sum(
    length - length % batch_size + (batch_size if length % batch_size != 0 else 0)
    for length in lengths
)

with tqdm(total=total_batches, desc="batches for brute force", position=0) as pbar:
    # with PeriodicGarbageCollector(60):
    maybe_parallel_map(
        partial(_handle_brute_force_for, pbar=pbar), sorted(relevant_seeds)
    )

update_csv(BRUTE_FORCE_CSV_PATH, brute_force_data, columns=brute_force_columns)

# %%
assert len(brute_force_data) == len(
    runtime_models
), f"len(brute_force_data) == {len(brute_force_data)} != {len(runtime_models)} == len(runtime_models)"
all_tokens_datasets_lens = {seed: len(d) for seed, d in all_tokens_datasets.items()}
assert (
    len(set(all_tokens_datasets_lens.values())) == 1
), f"Multiple dataset lengths! {set(all_tokens_datasets_lens.values())}"
latex_values["BruteForceCPU"] = brute_force_proof_deterministic
latex_values["BruteForceBatchSize"] = batch_size
latex_values["BruteForceNumBatches"] = int(
    math.ceil(list(all_tokens_datasets_lens.values())[0] / batch_size)
)

brute_force_data_by_key = defaultdict(dict)
for seed, d in brute_force_data.items():
    for k, v in d.items():
        brute_force_data_by_key[k][seed] = v


for key, latex_key in (
    ("loss", "BruteForceLoss"),
    ("accuracy", "BruteForceAccuracy"),
    ("num_correct", "BruteForceNumCorrect"),
    ("num_incorrect", "BruteForceNumIncorrect"),
    ("duration", "BruteForceTime"),
):
    latex_values |= data_summary(brute_force_data_by_key[key], prefix=latex_key)
    assert all(isinstance(seed, int) for seed in brute_force_data_by_key[key].keys())
    latex_all_values_by_value[f"{latex_key}Float"] = brute_force_data_by_key[key]

# %% [markdown]
# # Cubic proof

# %%
cubic_columns = [
    "seed",
    "accuracy-bound",
    "normalized-accuracy-bound",
    "correct-count-lower-bound",
    "duration-proof-search",
    "duration",
]
if os.path.exists(CUBIC_CSV_PATH):
    cubic_results = pd.read_csv(CUBIC_CSV_PATH)
else:
    cubic_results = pd.DataFrame(columns=cubic_columns)

all_seeds = set(runtime_models.keys())
unknown_seeds = all_seeds - set(cubic_results["seed"])
known_seeds = all_seeds - unknown_seeds
relevant_seeds = all_seeds if OVERWRITE_CSV_FROM_CACHE else unknown_seeds
cubic_data = {
    seed: cubic_results[cubic_results["seed"] == seed].iloc[0].to_dict()
    for seed in known_seeds
}


def get_cubic_row(seed: int, *, pbar: tqdm) -> dict:
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
        partial(
            cubic.verify_proof,
            model,
            pbar=pbar,
            print_complexity=False,
            print_results=False,
        ),
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.cubic_verify_proof-{cfg_hash_for_filename}",
        get_hash_mem=(lambda x: 0),
        get_hash=(lambda x: "0"),
    )() as verify_proof:
        cubic_proof_results = verify_proof(cubic_proof_args)

    # largest_wrong_logit_cubic = cubic_proof_results["largest_wrong_logit"]
    return {
        "seed": seed,
        "accuracy-bound": cubic_proof_results["accuracy_lower_bound"],
        "normalized-accuracy-bound": cubic_proof_results["accuracy_lower_bound"]
        / brute_force_data_by_key["accuracy"][seed],
        "correct-count-lower-bound": cubic_proof_results["correct_count_lower_bound"],
        "duration-proof-search": duration_proof_search,
        "duration": cubic_proof_results["prooftime"],
    }


def _handle_cubic(seed: int, *, pbar: tqdm):
    try:
        cubic_data[seed] = get_cubic_row(seed, pbar=pbar)
    except Exception as e:
        print(f"Error computing cubic proof for seed {seed}: {e}")
        traceback.print_exc()


# \sum_{i=0}^{k} i^2 = k * (k+1) * (k*2+1) // 6
ks = [cfgs[seed].experiment.model_config.d_vocab - 2 for seed in relevant_seeds]
total_batches = sum(k * (k + 1) * (k * 2 + 1) // 6 for k in ks)
with tqdm(total=total_batches, desc="batches for cubic", position=0) as pbar:
    # with PeriodicGarbageCollector(60):
    maybe_parallel_map(partial(_handle_cubic, pbar=pbar), sorted(relevant_seeds))
update_csv(CUBIC_CSV_PATH, cubic_data, columns=cubic_columns)

# %% [markdown]
# Summary satistics cubic
# %%
cubic_data_by_key = defaultdict(dict)
for seed, d in cubic_data.items():
    for k, v in d.items():
        cubic_data_by_key[k][seed] = v

assert len(cubic_data) == len(
    brute_force_data
), f"len(cubic_data) == {len(cubic_data)} != {len(brute_force_data)} == len(brute_force_data)"
for key, latex_key in (
    # ("loss", "CubicLoss"),
    ("accuracy-bound", "CubicAccuracy"),
    ("normalized-accuracy-bound", "CubicNormalizedAccuracy"),
    ("correct-count-lower-bound", "CubicCorrectCount"),
    ("duration", "CubicProofTime"),
):
    latex_values |= data_summary(cubic_data_by_key[key], prefix=latex_key)
    assert all(isinstance(seed, int) for seed in cubic_data_by_key[key].keys())
    latex_all_values_by_value[f"{latex_key}Float"] = cubic_data_by_key[key]


# %% [markdown]
# # Intermediate interp values for export
# %%
max_logit_diffs = {
    seed: EVOU_max_logit_diff(model)
    for seed, (_runtime, model) in runtime_models.items()
}
max_logit_diff_summaries = {
    seed: data_summary(max_logit_diff, prefix="EVOUMaxRowDiff", float_postfix="")
    for seed, max_logit_diff in max_logit_diffs.items()
}
max_logit_diff_summaries_by_keys = defaultdict(dict)
for seed, summary in max_logit_diff_summaries.items():
    for k, v in summary.items():
        max_logit_diff_summaries_by_keys[k][seed] = v
for k, v in max_logit_diff_summaries_by_keys.items():
    latex_values |= data_summary(v, prefix=k)
    assert all(isinstance(seed, int) for seed in v.keys())
    latex_all_values_by_value[f"{k}Float"] = v

# hold some data before summarizing it
latex_values_tmp_data = defaultdict(dict)
for seed, (_runtime, model) in runtime_models.items():
    for duplicate_by_sequence_count in [False, True]:
        key = "EVOU-hist-min-above-diag"
        if duplicate_by_sequence_count:
            key += "-dup-by-seq-count"
        (max_logit_minus_diag, duplication_factors) = EVOU_max_minus_diag_logit_diff(
            model,
            duplicate_by_sequence_count=duplicate_by_sequence_count,
        )
        mean = np.average(
            max_logit_minus_diag.numpy(), weights=duplication_factors.numpy()
        )
        std = np.average(
            (max_logit_minus_diag - mean).numpy() ** 2,
            weights=duplication_factors.numpy(),
        )
        num_std = 1
        most_below_value = int(mean + num_std * std)
        frac_below = (
            duplication_factors[max_logit_minus_diag <= most_below_value].sum()
            / duplication_factors.sum()
        ).item()
        value_key = "".join(
            v.capitalize() if v[0] != v[0].capitalize() else v for v in key.split("-")
        )
        latex_values_tmp_data[value_key + "MostBelowValue"][seed] = most_below_value
        latex_values_tmp_data[value_key + "MostBelowValueNumStd"][seed] = num_std
        latex_values_tmp_data[value_key + "MostBelowValueSequenceFrac"][
            seed
        ] = frac_below
        for k, v in data_summary(
            max_logit_minus_diag,
            sample_weight=duplication_factors,
            prefix=value_key,
            float_postfix="",
        ).items():
            latex_values_tmp_data[k][seed] = v

    for duplicate_by_sequence_count in [False, True]:
        flat_diffs, duplication_factors = attention_difference_over_gap(
            model,
            duplicate_by_sequence_count=duplicate_by_sequence_count,
        )
        key = "EQKE-hist-attention-difference-over-gap" + (
            "-dup-by-seq-count" if duplicate_by_sequence_count else ""
        )
        mean = np.average(flat_diffs.numpy(), weights=duplication_factors.numpy())
        std = np.average(
            (flat_diffs - mean).numpy() ** 2,
            weights=duplication_factors.numpy(),
        )
        value_key = "".join(
            v.capitalize() if v[0] != v[0].capitalize() else v for v in key.split("-")
        )
        for k, v in data_summary(
            flat_diffs,
            sample_weight=duplication_factors,
            prefix=value_key,
            float_postfix="",
        ).items():
            latex_values_tmp_data[k][seed] = v

for k, v in latex_values_tmp_data.items():
    latex_values |= data_summary(v, prefix=k)
    assert all(isinstance(seed, int) for seed in v.keys())
    latex_all_values_by_value[f"{k}Float"] = v


# %%
def analyze_EVOU(model: HookedTransformer):
    EPVOU = all_EVOU(model)
    PVOU = all_PVOU(model)
    PVOU_mean = PVOU.mean(dim=0)
    EPVOU += PVOU_mean
    PVOU -= PVOU_mean
    EPU = EU_PU(model)
    EPVOU_diag = EPVOU.diagonal()
    EPVOU_centered = EPVOU - EPVOU_diag.unsqueeze(-1)
    EPVOU_minf_diag = EPVOU_centered.clone()
    EPVOU_minf_diag[tuple(torch.arange(d) for d in EPVOU.shape)] = -torch.inf
    EPVOU_max_above_diag = EPVOU_minf_diag.amax(dim=-1)
    EPVOU_largest_index_above_diag = torch.arange(EPVOU.shape[0])[
        EPVOU_max_above_diag > 0
    ]
    EPVOU_off_diag = EPVOU.clone()
    EPVOU_off_diag[tuple(torch.arange(d) for d in EPVOU.shape)] = torch.nan
    EPVOU_off_diag = EPVOU_off_diag[~EPVOU_off_diag.isnan()]
    EPVOU_centered_off_diag = EPVOU_centered.clone()
    EPVOU_centered_off_diag[tuple(torch.arange(d) for d in EPVOU_centered.shape)] = (
        torch.nan
    )
    EPVOU_centered_off_diag = EPVOU_centered_off_diag[~EPVOU_centered_off_diag.isnan()]

    result = {}
    result |= data_summary(EPU.flatten(), prefix="EUPU")
    result |= data_summary(EPU.abs().flatten(), prefix="EUPUAbs")
    result |= data_summary(EPU.amax(dim=-1) - EPU.amin(dim=-1), prefix="EUPUMaxRowDiff")

    result |= data_summary(PVOU.flatten(), prefix="PVOU")
    result |= data_summary(PVOU.abs().flatten(), prefix="PVOUAbs")
    result |= data_summary(
        PVOU.amax(dim=-1) - PVOU.amin(dim=-1), prefix="PVOUMaxRowDiff"
    )

    result |= data_summary(EPVOU.flatten(), prefix="EPVOU")
    result |= data_summary(EPVOU.abs().flatten(), prefix="EPVOUAbs")
    result |= data_summary(
        EPVOU.amax(dim=-1) - EPVOU.amin(dim=-1), prefix="EPVOUMaxRowDiff"
    )
    result |= data_summary(EPVOU_diag, prefix="EPVOUDiagonal")
    result |= data_summary(EPVOU_centered.flatten(), prefix="EPVOUCentered")
    result |= data_summary(EPVOU_max_above_diag, prefix="EPVOUMaxAboveDiag")
    result |= data_summary(
        EPVOU_largest_index_above_diag, prefix="EPVOUInputsWithCopyingFailure"
    )
    result |= data_summary(EPVOU_off_diag, prefix="EPVOUOffDiagonal")
    result |= data_summary(EPVOU_off_diag.abs(), prefix="EPVOUOffDiagonalAbs")
    result |= data_summary(EPVOU_centered_off_diag, prefix="EPVOUCenteredOffDiagonal")

    return result


# %%
# with memoshelve(
#     (lambda seed, with_attn_scale: compute_EQKE_SVD_analysis(runtime_models[seed][1], with_attn_scale=with_attn_scale)),
#     filename=cache_dir / f"{SHARED_CACHE_STEM}.compute_EQKE_SVD_analysis",
#     get_hash_mem=(lambda x: x[0]),
#     get_hash=str,
# )() as memo_compute_EQKE_SVD_analysis:
EVOU_analyses = {
    seed: analyze_EVOU(runtime_models[seed][1])
    for seed in tqdm(list(sorted(runtime_models.keys())), desc="EVOU analysis")
}
# %%
EVOU_analyses_by_key = defaultdict(dict)
for seed, d in EVOU_analyses.items():
    for k, v in d.items():
        EVOU_analyses_by_key[k][seed] = v
# %%
for k, v in EVOU_analyses_by_key.items():
    if k.endswith("Float"):
        latex_values |= data_summary(v, prefix=k[: -len("Float")])
        assert all(isinstance(seed, int) for seed in v.keys())
        latex_all_values_by_value[k] = v
    else:
        latex_values |= data_summary(v, prefix=k)
        assert all(isinstance(seed, int) for seed in v.keys())
        latex_all_values_by_value[f"{k}Float"] = v
        # vals = set(v.values())
        # assert len(vals) == 1, f"Too many values for {k}: {vals}"
        # latex_values[k] = list(vals)[0]
# %%


# %% [markdown]
# # SVD analysis
# %%


# %%


# %%
with memoshelve(
    (
        lambda seed: display_EQKE_SVD_analysis(
            runtime_models[seed][1], include_figures=False, show=False, do_print=False
        )[1]
    ),
    filename=cache_dir / f"{SHARED_CACHE_STEM}.compute_EQKE_SVD_analysis",
    get_hash_mem=(lambda x: x[0]),
    get_hash=str,
)() as memo_compute_EQKE_SVD_analysis:
    EQKE_SVD_analyses = {
        seed: memo_compute_EQKE_SVD_analysis(seed)
        for seed in tqdm(list(sorted(runtime_models.keys())), desc="SVD analysis")
    }
# %%
EQKE_SVD_analyses_by_key = defaultdict(dict)
for seed, d in EQKE_SVD_analyses.items():
    for k, v in d.items():
        EQKE_SVD_analyses_by_key[k][seed] = v
# %%
for k, v in EQKE_SVD_analyses_by_key.items():
    if k.endswith("Float"):
        latex_values |= data_summary(v, prefix=k[: -len("Float")])
        assert all(isinstance(seed, int) for seed in v.keys())
        latex_all_values_by_value[k] = v
    else:
        vals = set(v.values())
        assert len(vals) == 1, f"Too many values for {k}: {vals}"
        latex_values[k] = list(vals)[0]
# %%
new_data = []
for seed, d in EQKE_SVD_analyses.items():
    new_data.append(d | {"seed": seed})

for k, v in EQKE_SVD_analyses_by_key.items():
    if k.endswith("Float"):
        latex_values |= data_summary(v, prefix=k[: -len("Float")])
        assert all(isinstance(seed, int) for seed in v.keys())
        latex_all_values_by_value[k] = v
    else:
        vals = set(v.values())
        assert len(vals) == 1, f"Too many values for {k}: {vals}"
        latex_values[k] = list(vals)[0]

update_csv_with_rows(
    SUBCUBIC_ANALYSIS_CSV_PATH,
    new_data,
    columns=["seed"] + list(EQKE_SVD_analyses_by_key.keys()),
    subset=["seed"] + list(EQKE_SVD_analyses_by_key.keys()),
)


# %% [markdown]
# # Plots
# %%
if SAVE_PLOTS or DISPLAY_PLOTS:
    all_axis_limits = defaultdict(dict)
    with tqdm(runtime_models.items(), desc="display_basic_interpretation") as pbar:
        for seed, (_runtime, model) in pbar:
            pbar.set_postfix(dict(seed=seed))
            figs, axis_limits = display_basic_interpretation(
                model,
                include_uncentered=True,
                OV_colorscale=default_OV_colorscale,
                QK_colorscale=default_QK_colorscale,
                QK_SVD_colorscale=default_QK_SVD_colorscale,
                tok_dtick=10,
                plot_with=PLOT_WITH,
                renderer=RENDERER,
                show=DISPLAY_PLOTS,
            )
            for k, v in axis_limits.items():
                all_axis_limits[k][seed] = v
            for attn_scale in ("", "WithAttnScale"):
                for fig in (
                    figs[f"EQKE{attn_scale}"],
                    figs[f"EQKP{attn_scale}"],
                    figs["EVOU"],
                    figs["EVOU-centered"],
                ):
                    remove_titles(fig)
                latex_figures[f"{seed}-EQKE{attn_scale}"] = figs[f"EQKE{attn_scale}"]
                latex_figures[f"{seed}-EQKP{attn_scale}"] = figs[f"EQKP{attn_scale}"]
                latex_figures[f"{seed}-EQKE{attn_scale}-SVD"] = figs[
                    f"EQKE{attn_scale} Attention SVD"
                ]
                del figs[f"EQKE{attn_scale} Attention SVD"]
            latex_figures[f"{seed}-EVOU"] = figs["EVOU"]
            latex_figures[f"{seed}-EVOU-centered"] = figs["EVOU-centered"]
            PVOU_keys = [
                k for k in figs.keys() if k.startswith("irrelevant_") and "V" in k
            ]
            assert len(PVOU_keys) == 1, f"PVOU_keys: {PVOU_keys}"
            latex_figures[f"{seed}-PVOU"] = figs[PVOU_keys[0]]
            del figs[PVOU_keys[0]]
            EUPU_keys = [k for k in figs.keys() if k.startswith("irrelevant_")]
            assert len(EUPU_keys) == 1, f"EUPU_keys: {EUPU_keys}"
            latex_figures[f"{seed}-EUPU"] = figs[EUPU_keys[0]]
            del figs[EUPU_keys[0]]
            latex_figures[f"{seed}-PVOU-scatter"] = figs["irrelevant"]
            del figs["irrelevant"]
            unused_keys = [k for k in figs if k not in latex_figures]
            for fig in (
                latex_figures[f"{seed}-PVOU-scatter"],
                latex_figures[f"{seed}-EUPU"],
                latex_figures[f"{seed}-PVOU"],
            ):
                remove_titles(fig)

        if unused_keys:
            print(f"Unused keys: {unused_keys}")

    axis_limits = {}
    for k, v in all_axis_limits.items():
        if k.endswith("min"):
            axis_limits[k] = np.min(list(v.values()))
        elif k.endswith("max"):
            axis_limits[k] = np.max(list(v.values()))
        else:
            raise ValueError(f"Unknown axis limit key: {k}")

    for k in axis_limits.keys():
        if "centered" not in k.lower():
            k_min = k.replace("max", "min")
            k_max = k.replace("min", "max")
            assert k_min in axis_limits, f"Missing {k_min}"
            assert k_max in axis_limits, f"Missing {k_max}"
            assert k_min == k or k_max == k, f"Unknown key: {k}"
            assert k_min != k_max, f"Same key: {k}"
            v_max = np.max([np.abs(axis_limits[k_min]), np.abs(axis_limits[k_max])])
            axis_limits[k_min] = -v_max
            axis_limits[k_max] = v_max

    for k, v in axis_limits.items():
        k = "".join(
            [
                kpart if kpart[0] == kpart[0].capitalize() else kpart.capitalize()
                for kpart in k.replace("-", "_").split("_")
            ]
        )
        latex_values[f"AxisLimits{k}Float"] = v

    with tqdm(runtime_models.items(), desc="display_basic_interpretation") as pbar:
        for seed, (_runtime, model) in pbar:
            pbar.set_postfix(dict(seed=seed))
            figs, _axis_limits = display_basic_interpretation(
                model,
                include_uncentered=True,
                OV_colorscale=default_OV_colorscale,
                QK_colorscale=default_QK_colorscale,
                QK_SVD_colorscale=default_QK_SVD_colorscale,
                tok_dtick=10,
                **axis_limits,
                plot_with=PLOT_WITH,
                renderer=RENDERER,
                show=DISPLAY_PLOTS,
            )
            for attn_scale in ("", "WithAttnScale"):
                for fig in (
                    figs[f"EQKE{attn_scale}"],
                    figs[f"EQKP{attn_scale}"],
                    figs["EVOU"],
                    figs["EVOU-centered"],
                ):
                    remove_titles(fig)
                latex_figures[f"{seed}-EQKE{attn_scale}UniformLimits"] = figs[
                    f"EQKE{attn_scale}"
                ]
                latex_figures[f"{seed}-EQKP{attn_scale}UniformLimits"] = figs[
                    f"EQKP{attn_scale}"
                ]
                del figs[f"EQKE{attn_scale} Attention SVD"]
            latex_figures[f"{seed}-EVOUUniformLimits"] = figs["EVOU"]
            latex_figures[f"{seed}-EVOU-centeredUniformLimits"] = figs["EVOU-centered"]
            PVOU_keys = [
                k for k in figs.keys() if k.startswith("irrelevant_") and "V" in k
            ]
            assert len(PVOU_keys) == 1, f"PVOU_keys: {PVOU_keys}"
            latex_figures[f"{seed}-PVOUUniformLimits"] = figs[PVOU_keys[0]]
            del figs[PVOU_keys[0]]
            EUPU_keys = [k for k in figs.keys() if k.startswith("irrelevant_")]
            assert len(EUPU_keys) == 1, f"EUPU_keys: {EUPU_keys}"
            latex_figures[f"{seed}-EUPUUniformLimits"] = figs[EUPU_keys[0]]
            del figs[EUPU_keys[0]]
            latex_figures[f"{seed}-PVOU-scatterUniformLimits"] = figs["irrelevant"]
            del figs["irrelevant"]
            unused_keys = [k for k in figs if k not in latex_figures]
            for fig in (
                latex_figures[f"{seed}-PVOU-scatterUniformLimits"],
                latex_figures[f"{seed}-EUPUUniformLimits"],
                latex_figures[f"{seed}-PVOUUniformLimits"],
            ):
                remove_titles(fig)


# %%
## %%
if DISPLAY_PLOTS or SAVE_PLOTS:
    with tqdm(runtime_models.items(), desc="make_better_slides_plots_00") as pbar:
        for seed, (_runtime, model) in pbar:
            pbar.set_postfix(dict(seed=seed))
            figs = make_better_slides_plots_00(
                model,
                OV_colorscale=default_OV_colorscale,
                QK_colorscale=default_QK_colorscale,
                tok_dtick=10,
                plot_with=PLOT_WITH,
                renderer=RENDERER,
                show=DISPLAY_PLOTS,
                do_print=False,
            )
            for k, fig in figs.items():
                latex_figures[f"{seed}-Decomposition-{k}"] = fig
# %%
if DISPLAY_PLOTS or SAVE_PLOTS:
    with tqdm(runtime_models.items(), desc="hist_EVOU_max_logit_diff") as pbar:
        for seed, (_runtime, model) in pbar:
            pbar.set_postfix(dict(seed=seed))
            latex_figures[f"{seed}-EVOU-hist-max-row-diff"], max_logit_diff = (
                hist_EVOU_max_logit_diff(
                    model, plot_with=PLOT_WITH, renderer=RENDERER, show=DISPLAY_PLOTS
                )
            )
            # remove_titles(latex_figures[f"{seed}-EVOU-hist-max-row-diff"])
            for duplicate_by_sequence_count in [False, True]:
                key = "EVOU-hist-min-above-diag"
                if duplicate_by_sequence_count:
                    key += "-dup-by-seq-count"
                latex_figures[f"{seed}-{key}"], (
                    max_logit_minus_diag,
                    duplication_factors,
                ) = hist_EVOU_max_minus_diag_logit_diff(
                    model,
                    duplicate_by_sequence_count=duplicate_by_sequence_count,
                    plot_with=PLOT_WITH,
                    renderer=RENDERER,
                    show=DISPLAY_PLOTS,
                )
                # remove_titles(latex_figures[f"{seed}-{key}"])


# %%
if DISPLAY_PLOTS or SAVE_PLOTS:
    with tqdm(
        runtime_models.items(), desc="scatter_attention_difference_vs_gap"
    ) as pbar:
        for seed, (_runtime, model) in pbar:
            pbar.set_postfix(dict(seed=seed))
            latex_figures[f"{seed}-EQKE-scatter-attention-difference-vs-gap"] = (
                scatter_attention_difference_vs_gap(
                    model,
                    renderer=RENDERER,
                    show=DISPLAY_PLOTS,
                    plot_with=PLOT_WITH,
                    # plot_with="plotly",
                )  # this one is too big to export to TeX
            )
            for duplicate_by_sequence_count in [False, True]:
                fig, (flat_diffs, duplication_factors) = (
                    hist_attention_difference_over_gap(
                        model,
                        duplicate_by_sequence_count=duplicate_by_sequence_count,
                        plot_with=PLOT_WITH,
                        renderer=RENDERER,
                        show=DISPLAY_PLOTS,
                    )
                )
                key = "EQKE-hist-attention-difference-over-gap" + (
                    "-dup-by-seq-count" if duplicate_by_sequence_count else ""
                )
                latex_figures[f"{seed}-{key}"] = fig
# %%
if SAVE_PLOTS or DISPLAY_PLOTS:
    with tqdm(runtime_models.items(), desc="display_EQKE_SVD_analysis") as pbar:
        for seed, (_runtime, model) in pbar:
            pbar.set_postfix(dict(seed=seed))
            figs, values = display_EQKE_SVD_analysis(
                model,
                plot_with=PLOT_WITH,
                QK_colorscale=default_QK_colorscale,
                QK_SVD_colorscale=default_QK_SVD_colorscale,
                tok_dtick=10,
                renderer=RENDERER,
                include_figures=True,
                show=DISPLAY_PLOTS,
                do_print=False,
            )
            key_pairs = {}
            for attn_scale in ("", "WithAttnScale"):
                cur_key_pairs = {
                    f"{k}{attn_scale}": f"{k}{attn_scale}"
                    for k in (
                        "WKkPerp-svd",
                        "WQqPerp-svd",
                        "WEqqPerp-svd",
                        "WEkkPerp-svd",
                    )
                } | {
                    f"EQKE_err{attn_scale}": f"EQKE-err{attn_scale}",
                    f"EQKE_err_svd{attn_scale}": f"EQKE-err-svd{attn_scale}",
                }
                key_pairs |= cur_key_pairs
                for key, latex_key in cur_key_pairs.items():
                    latex_figures[f"{seed}-{latex_key}"] = figs[key]

# %% [markdown]
# # Sub-cubic Proofs
# %%
try_all_configurations: bool = True  # @param {type:"boolean"}
use_tricks: bool = True  # @param {type:"boolean"}
all_configs: list[LargestWrongLogitQuadraticConfig]
if try_all_configurations:
    all_configs = list(enumerate_dataclass_values(LargestWrongLogitQuadraticConfig))
elif use_tricks:
    all_configs = [LargestWrongLogitQuadraticConfig()]
else:
    all_configs = [LargestWrongLogitQuadraticConfig.OFF()]
# %%


def _subcubic_count_verify_proof(
    model: HookedTransformer,
    tricks: LargestWrongLogitQuadraticConfig,
    *,
    sanity_check_instructions: bool = False,
    **kwargs,
) -> Tuple[InstructionCount, dict[str, Any]]:
    # must be outside PatchTorch to avoid triu, tril
    cmodel = CountHookedTransformer(model)
    with PatchTorch():
        with instructions.set_sanity_check(sanity_check_instructions):
            with CountTensorOperations() as subcubic_instruction_count:
                results = subcubic.verify_proof(
                    cmodel,
                    tricks=tricks,
                    **kwargs,
                    print_complexity=False,
                    print_results=False,
                    sanity_check=False,
                    # print_types=True,
                )
    return subcubic_instruction_count, results


# %%
_some_runtime, some_model = runtime_models[123]
d_vocab, n_ctx = some_model.cfg.d_vocab, some_model.cfg.n_ctx
latex_values["BruteForceEffectiveDimensionalityEstimate"] = brute_force_ed = (
    d_vocab ** (n_ctx + 1)
)
EUPU_cost = d_vocab**2
PVOU_cost = n_ctx * d_vocab
EPQKE_cost = d_vocab**2
EPQKP_cost = d_vocab * n_ctx
EVOU_cost = d_vocab**2
latex_values["CubicEffectiveDimensionalityEstimate"] = cubic_ed = (
    EUPU_cost + PVOU_cost + EPQKE_cost + EPQKP_cost + EVOU_cost
)
subcubic_PVOU_cost = d_vocab
subcubic_EPQKP_cost = 0


# %%
subcubic_columns = [
    "seed",
    "accuracy-bound",
    "normalized-accuracy-bound",
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
    "perf-time-enabled-ns",
    "perf-instruction-count",
    "perf-branch-misses",
    "perf-page-faults",
    "proof-flop-estimate",
    "proof-int-op-estimate",
    "proof-branch-estimate",
]
if os.path.exists(SUBCUBIC_CSV_PATH):
    subcubic_results = pd.read_csv(SUBCUBIC_CSV_PATH)
else:
    subcubic_results = pd.DataFrame(columns=subcubic_columns)

all_seeds = set(runtime_models.keys())
unknown_seeds = all_seeds - set(
    seed
    for seed in subcubic_results["seed"]
    if len(subcubic_results[subcubic_results["seed"] == seed].to_dict(orient="records"))
    >= len(all_configs)
)
subcubic_data = {
    seed: subcubic_results[subcubic_results["seed"] == seed].to_dict(orient="records")
    for seed in all_seeds
    if seed not in unknown_seeds
}
known_seeds = all_seeds - unknown_seeds
relevant_seeds = all_seeds if OVERWRITE_CSV_FROM_CACHE else unknown_seeds


@torch.no_grad()
def try_all_proofs_subcubic(
    seed: int,
    *,
    subcfg_pbar: tqdm,
    cfg_pbar: tqdm,
    proof_pbar: tqdm,
    count_proof_pbar: tqdm,
) -> list[dict]:
    cfg = cfgs[seed]
    cfg_hash = cfg_hashes[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    runtime, model = runtime_models[seed]
    training_wrapper = training_wrappers[seed]
    assert cfg.experiment.model_config.seed is not None

    min_gaps_lists = {}

    rows = []

    def _shared_proof_search(seed: int):
        shared_proof_search_duration = 0.0
        start = time.time()
        W_EP_direction_kwargs = analysis_quadratic.W_EP_direction_for_tricks_kwargs(
            model
        )
        find_min_gaps_kwargs = analysis_subcubic.find_min_gaps_with_EQKE_kwargs(model)
        size_and_query_directions_kwargs = (
            analysis_quadratic.find_EKQE_error_directions(model)
        )
        shared_proof_search_duration += time.time() - start
        return (
            W_EP_direction_kwargs,
            find_min_gaps_kwargs,
            size_and_query_directions_kwargs,
            shared_proof_search_duration,
        )

    with memoshelve(
        _shared_proof_search,
        # cache={},
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.shared_proof_search-{cfg_hash_for_filename}",
    )() as shared_proof_search:
        (
            W_EP_direction_kwargs,
            find_min_gaps_kwargs,
            size_and_query_directions_kwargs,
            shared_proof_search_duration,
        ) = shared_proof_search(seed)

    with memoshelve(
        (
            lambda cfg: (
                cfg,
                *analysis_subcubic.find_min_gaps_with_EQKE(
                    model=model,
                    **find_min_gaps_kwargs,  # type: ignore
                    **size_and_query_directions_kwargs,
                    tricks=cfg,
                    sub_pbar=subcfg_pbar,
                    pbar=cfg_pbar,
                    record_time=True,
                ),
            )
        ),
        # cache={},
        filename=cache_dir
        / f"{SHARED_CACHE_STEM}.find_min_gaps-{cfg_hash_for_filename}",
    )() as find_min_gaps_for:
        min_gaps_lists = [find_min_gaps_for(cfg) for cfg in all_configs]

    for tricks, min_gaps, proof_search_duration in min_gaps_lists:
        if N_THREADS is None or N_THREADS <= 1:
            proof_pbar.set_postfix(cfg=tricks.short_description(latex=True))
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

        def _verify_proof(tricks: LargestWrongLogitQuadraticConfig):
            return subcubic.verify_proof(
                model,
                W_EP_direction=W_EP_direction,
                **size_and_query_directions_kwargs,  # type: ignore
                min_gaps=min_gaps,
                tricks=tricks,
                sanity_check=False,
                print_complexity=False,
                print_results=False,
                include_perf=PERF_WORKING,
            )

        with memoshelve(
            _verify_proof,
            filename=cache_dir
            / f"{SHARED_CACHE_STEM}.subcubic_verify_proof{'' if not PERF_WORKING else '-with-perf'}-{cfg_hash_for_filename}",
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        )() as verify_proof:
            proof_results = verify_proof(tricks)

        err_upper_bound = proof_results["err_upper_bound"]
        prooftime = proof_results["prooftime"]
        accuracy_bound = proof_results["accuracy_lower_bound"]
        total_sequences = proof_results["total_sequences"]
        left_behind = proof_results["left_behind"]

        if PERF_WORKING:
            perf_results = {
                "perf-time-enabled-ns": int_or_value(
                    proof_results["proofinstructions"].time_enabled_ns
                ),
                "perf-instruction-count": int_or_value(
                    proof_results["proofinstructions"].instruction_count
                ),
                "perf-branch-misses": int_or_value(
                    proof_results["proofinstructions"].branch_misses
                ),
                "perf-page-faults": int_or_value(
                    proof_results["proofinstructions"].page_faults
                ),
            }
        else:
            perf_results = {}

        with memoshelve(
            partial(
                _subcubic_count_verify_proof,
                model,
                W_EP_direction=(
                    CountTensor.from_numpy(W_EP_direction)
                    if W_EP_direction is not None
                    else W_EP_direction
                ),
                **{k: CountTensor.from_numpy(v) if isinstance(v, torch.Tensor) else v for k, v in size_and_query_directions_kwargs.items()},  # type: ignore
                min_gaps=min_gaps,
                sanity_check_instructions=False,
            ),
            filename=cache_dir
            / f"{SHARED_CACHE_STEM}.subcubic_count_verify_proof-{cfg_hash_for_filename}",
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        )() as count_verify_proof:
            (
                subcubic_instruction_count,
                subcubic_proof_instruction_count_results,
            ) = count_verify_proof(tricks)
        count_proof_pbar.update(1)

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

        def _analyze_gaps(*args, **kwargs):
            d_vocab_q, d_vocab_max, n_ctx_nonmax_copies = min_gaps_lists[0][1].shape
            weights = torch.zeros((d_vocab_q, d_vocab_max, n_ctx_nonmax_copies))
            # weights = ein.array(
            #     (
            #         lambda q_tok, max_tok, n_copies_nonmax: torch.tensor(
            #             (max_tok - 1) ** n_copies_nonmax
            #             * math.comb(model.cfg.n_ctx - 1, n_copies_nonmax)
            #         )
            #     ),
            #     sizes=[d_vocab_q, d_vocab_max, n_ctx_nonmax_copies],
            #     device=torch.tensor(0).device,
            # )
            # weights[:, 0, :] = 1
            # weights[:, 0, 1:] = 0
            # weights = ein.array(
            #     (
            #         lambda q_tok, max_tok, n_copies_nonmax: torch.where(
            #             (
            #                 (q_tok > max_tok)
            #                 | ( # TypeError: unsupported operand type(s) for |: 'Tensor' and 'Tensor'
            #                     (n_copies_nonmax == n_ctx_nonmax_copies - 1)
            #                     & (max_tok != q_tok)
            #                 )
            #                 | ((max_tok == 0) & (n_copies_nonmax > 0))
            #             ),
            #             torch.tensor(0),
            #             torch.where(
            #                 max_tok == 0,
            #                 torch.tensor(1),
            #                 torch.tensor(
            #                     (max_tok - 1) ** n_copies_nonmax
            #                     * math.comb(model.cfg.n_ctx - 1, n_copies_nonmax)
            #                 ),
            #             ),
            #         )
            #     ),
            #     sizes=[d_vocab_q, d_vocab_max, n_ctx_nonmax_copies],
            #     device=torch.tensor(0).device,
            # )
            for max_tok in range(d_vocab_max):
                cur_n_ctx_nonmax_copies = 1 if max_tok == 0 else n_ctx_nonmax_copies
                for n_copies_nonmax in range(cur_n_ctx_nonmax_copies):
                    weights[: max_tok + 1, max_tok, n_copies_nonmax] = (
                        max_tok - 1
                    ) ** n_copies_nonmax * math.comb(
                        model.cfg.n_ctx - 1, n_copies_nonmax
                    )
                weights[:max_tok, max_tok, n_ctx_nonmax_copies - 1] = 0
                # for q_tok in range(max_tok+1):
                #     if (
                #         # (q_tok > max_tok) or
                #          (
                #             n_copies_nonmax == n_ctx_nonmax_copies - 1
                #             and max_tok != q_tok
                #         )
                #         # or (max_tok == 0 and n_copies_nonmax > 0)
                #     ):
                #         weights[q_tok, max_tok, n_copies_nonmax] = 0
                # if max_tok == 0:
                #     assert q_tok == max_tok
                #     assert n_copies_nonmax == 0
            weights[1, 1, 0] = 1

            v = min_gaps.flatten().detach().cpu()
            mean = np.average(v.numpy(), weights=weights.flatten().numpy())
            std = np.average(
                (v - mean).numpy() ** 2,
                weights=weights.flatten().numpy(),
            )
            num_std = 1.5
            most_below_value = int(math.ceil(mean + num_std * std))
            # print(v)
            # print(most_below_value)
            # print(list(sorted(v.tolist())))
            # print(f"max={(min_gaps==min_gaps.max()).nonzero()}")
            # if min_gaps.max() > 100:
            #     print(f"big! {min_gaps.max()}")
            #     args = (tricks,)
            #     kwargs = dict(
            #         filename=cache_dir
            #         / f"{SHARED_CACHE_STEM}.find_min_gaps-{descr}-{cfg_hash_for_filename}"
            #     )
            #     print(f"memoshelve_uncache(*{args}, **{kwargs})")
            #     memoshelve_uncache(*args, **kwargs)
            #     args = (tricks, use_exact_EQKE)
            #     kwargs = dict(
            #         filename=cache_dir
            #         / f"{SHARED_CACHE_STEM}.subcubic_verify_proof-{cfg_hash_for_filename}",
            #         get_hash_mem=(lambda x: x[0]),
            #         get_hash=str,
            #     )
            #     print(f"memoshelve_uncache(*{args}, **{kwargs})")
            #     memoshelve_uncache(*args, **kwargs)
            # print(f"mean={mean}")
            # print(f"std={std}")
            # print(f"max={v.max().item()}")
            # print(f"min={v.min().item()}")
            # print(v <= most_below_value)
            frac_below = (
                weights.flatten()[v <= most_below_value].sum() / weights.sum()
            ).item()

            return (
                frac_below,
                v,
                weights.flatten().detach().cpu(),
                most_below_value,
                mean,
                std,
                num_std,
            )

        with memoshelve(
            _analyze_gaps,
            filename=cache_dir
            / f"{SHARED_CACHE_STEM}.subcubic_analyze_gaps-{cfg_hash_for_filename}",
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        )() as analyze_gaps:
            (frac_below, v, weights, most_below_value, mean, std, num_std) = (
                analyze_gaps(tricks)
            )

        row = {
            "seed": seed,
            "accuracy-bound": accuracy_bound,
            "normalized-accuracy-bound": accuracy_bound
            / brute_force_data_by_key["accuracy"][seed],
            "duration-proof-search": proof_search_duration,
            "duration": prooftime,
            "tricks": tricks.short_description(latex=True),
            "err-upper-bound": err_upper_bound_value,
            "err-upper-bound-is-max": err_upper_bound_is_max,
            "total-sequences": total_sequences,
            "dropped-sequences": left_behind,
            "dropped-sequences-frac": left_behind / total_sequences,
            "most-gap-below-value": most_below_value,
            "most-gap-below-value-frac": frac_below,
            "most-gap-below-value-num-std": num_std,
            "max-gap": v.max().item(),
            "proof-flop-estimate": subcubic_instruction_count.flop,
            "proof-int-op-estimate": subcubic_instruction_count.int_op,
            "proof-branch-estimate": subcubic_instruction_count.branch,
        } | perf_results

        rows.append(row)
        proof_pbar.update(1)
    return rows


def _handle_subcubic(
    seed: int,
    *,
    subcfg_pbar: tqdm,
    cfg_pbar: tqdm,
    proof_pbar: tqdm,
    count_proof_pbar: tqdm,
):
    if N_THREADS is None or N_THREADS <= 1:
        cfg_pbar.set_postfix(seed=seed)
    try:
        subcubic_data[seed] = try_all_proofs_subcubic(
            seed,
            subcfg_pbar=subcfg_pbar,
            cfg_pbar=cfg_pbar,
            proof_pbar=proof_pbar,
            count_proof_pbar=count_proof_pbar,
        )
    except Exception as e:
        print(f"Error computing subcubic proof for seed {seed}: {e}")
        traceback.print_exc()


cfg_counts = {
    seed: sum(
        2 if cfg.attention_error_handling == "max_diff_exact" else 1
        for cfg in all_configs
    )
    for seed in relevant_seeds
}
sub_cfg_counts = {
    seed: runtime_models[seed][1].cfg.d_vocab * num_cfgs
    for seed, num_cfgs in cfg_counts.items()
}

n_cfgs = sum(cfg_counts.values())
n_subcfgs = sum(sub_cfg_counts.values())
with (
    tqdm(total=n_cfgs, desc="configurations for subcubic", position=0) as cfg_pbar,
    tqdm(total=n_subcfgs, desc="subconfig progress", position=1) as subcfg_pbar,
    tqdm(total=n_cfgs, desc="proofs for subcubic", position=2) as proof_pbar,
    tqdm(
        total=n_cfgs, desc="instruction counts for subcubic", position=3
    ) as count_proof_pbar,
):
    # with PeriodicGarbageCollector(60):
    maybe_parallel_map(
        partial(
            _handle_subcubic,
            subcfg_pbar=subcfg_pbar,
            cfg_pbar=cfg_pbar,
            proof_pbar=proof_pbar,
            count_proof_pbar=count_proof_pbar,
        ),
        sorted(relevant_seeds),
    )


def subcubic_approx_effective_dimension(
    model: HookedTransformer, tricks: LargestWrongLogitQuadraticConfig
):
    return (
        int(tricks.effective_dimension_estimate(model.cfg))
        + subcubic_PVOU_cost
        + subcubic_EPQKP_cost
        + EVOU_cost
    )


for seed in subcubic_data:
    for row in subcubic_data[seed]:
        row["effective-dimensionality-estimate"] = subcubic_approx_effective_dimension(
            runtime_models[seed][1],
            LargestWrongLogitQuadraticConfig.parse(row["tricks"], latex=True),
        )

new_data = []
for seed in sorted(subcubic_data.keys()):
    new_data.extend(subcubic_data[seed])

update_csv_with_rows(
    SUBCUBIC_CSV_PATH, new_data, columns=subcubic_columns, subset=["seed", "tricks"]
)

# %%
# %% [markdown]
# Summary satistics subcubic
# %%

assert len(subcubic_data) == len(
    brute_force_data
), f"len(cubic_data) == {len(subcubic_data)} != {len(brute_force_data)} == len(brute_force_data)"


def leading_complexity(tricks: LargestWrongLogitQuadraticConfig):
    # tricks = LargestWrongLogitQuadraticConfig.parse(tricks_str)
    return (
        "AlmostQuadratic"
        if tricks.is_quadratic
        else (
            "SubcubicWithoutVocabSquared"
            if tricks.is_subcubic_no_quadratic_vocab
            else "Subcubic" if tricks.is_subcubic else "FakeSubcubic"
        )
    )


def subcubic_group(tricks: LargestWrongLogitQuadraticConfig):
    # tricks = LargestWrongLogitQuadraticConfig.parse(tricks_str)
    EUPU_str = (
        "DirectQuadratic"
        if tricks.EUPU_handling_quadratic
        else (
            "DirectModelSquaredVocab"
            if tricks.EUPU_handling_subcubic_no_quadratic_vocab
            else None if tricks.EUPU_handling_subcubic else "DirectCubic"
        )
    )
    EPQKE_str = (
        "AttentionQuadratic"
        if tricks.attention_error_handling_quadratic
        and tricks.attention_handling_quadratic
        else (
            "AttentionModelSquaredVocab"
            if tricks.attention_error_handling_subcubic_no_quadratic_vocab
            and tricks.attention_handling_subcubic_no_quadratic_vocab
            else (
                None
                if tricks.attention_error_handling_subcubic
                and tricks.attention_handling_subcubic
                else "AttentionCubic"
            )
        )
    )
    strs = [s for s in (EPQKE_str, EUPU_str) if s is not None]
    return "Subcubic" + (f"{''.join(strs)}" if strs else "Group")


def filter_tricks_str_eq(value: str, tricks_str: str):
    return value == tricks_str


def filter_tricks_by_func(
    value: str, func: Callable[[LargestWrongLogitQuadraticConfig], str], tricks_str: str
):
    return value == func(LargestWrongLogitQuadraticConfig.parse(tricks_str, latex=True))


subcubic_leading_complexities = defaultdict(set)
subcubic_groups = defaultdict(set)

for tricks in all_configs:
    tricks_str = tricks.short_description(latex=True)
    subcubic_leading_complexities[leading_complexity(tricks)].add(tricks_str)
    subcubic_groups[subcubic_group(tricks)].add(tricks_str)

subcubic_key_pairs = [
    ("accuracy-bound", "Accuracy"),
    ("duration-proof-search", "ProofSearchTime"),
    ("duration", "ProofTime"),
    ("normalized-accuracy-bound", "NormalizedAccuracy"),
    ("perf-time-enabled-ns", "PerfTimeEnabledNS"),
    ("perf-instruction-count", "PerfInstructionCount"),
    ("perf-branch-misses", "PerfBranchMisses"),
    ("perf-page-faults", "PerfPageFaults"),
    ("proof-flop-estimate", "InstructionCount"),
    ("proof-int-op-estimate", "InstructionCountInt"),
    ("proof-branch-estimate", "InstructionCountBranch"),
    ("err-upper-bound", "ErrUpperBound"),
    ("dropped-sequences", "DroppedSequences"),
    ("dropped-sequences-frac", "DroppedSequencesFrac"),
    ("most-gap-below-value", "GapMostBelowValue"),
    ("most-gap-below-value-frac", "GapMostBelowValueSequenceFrac"),
    ("most-gap-below-value-num-std", "GapMostBelowValueNumStd"),
    ("max-gap", "MaxGap"),
    ("effective-dimensionality-estimate", "EffectiveDimensionalityEstimate"),
]

for trick_filter_descr, trick_filter in (
    [
        ("AnySubcubic", lambda tricks_str: True),
        (
            "RealSubcubic",
            lambda tricks_str: LargestWrongLogitQuadraticConfig.parse(
                tricks_str, latex=True
            ).is_subcubic,
        ),
        (
            "SubcubicModelSquaredVocab",
            lambda tricks_str: LargestWrongLogitQuadraticConfig.parse(
                tricks_str, latex=True
            ).is_subcubic_no_quadratic_vocab,
        ),
    ]
    + [(k, partial(filter_tricks_by_func, k, subcubic_group)) for k in subcubic_groups]
    + [
        (k, partial(filter_tricks_by_func, k, leading_complexity))
        for k in subcubic_leading_complexities
    ]
    + [
        (
            f"Subcubic{tricks.short_description(latex=True)}",
            partial(filter_tricks_str_eq, tricks.short_description(latex=True)),
        )
        for tricks in all_configs
    ]
):
    filtered_subcubic_data = {
        seed: [row for row in rows if trick_filter(row["tricks"])]
        for seed, rows in subcubic_data.items()
    }
    filtered_subcubic_data_best_by_key = defaultdict(dict)
    for seed, rows in filtered_subcubic_data.items():
        best_row = max(rows, key=lambda row: row["accuracy-bound"])
        for k, v in best_row.items():
            filtered_subcubic_data_best_by_key[k][seed] = v
    for key, latex_key in subcubic_key_pairs:
        if key not in filtered_subcubic_data_best_by_key:
            print(f"Warning! Missing key {key}")
            continue
        latex_values |= data_summary(
            filtered_subcubic_data_best_by_key[key],
            prefix=f"{trick_filter_descr}OnlyBestAccBoundPerSeed{latex_key}",
        )
        assert all(
            isinstance(seed, int)
            for seed in filtered_subcubic_data_best_by_key[key].keys()
        ), list(filtered_subcubic_data_best_by_key[key].keys())
        latex_all_values_by_value[
            f"{trick_filter_descr}OnlyBestAccBoundPerSeed{latex_key}Float"
        ] = filtered_subcubic_data_best_by_key[key]
        if any(len(rows) > 1 for rows in filtered_subcubic_data.values()):
            latex_values |= data_summary(
                [row[key] for rows in filtered_subcubic_data.values() for row in rows],
                prefix=f"{trick_filter_descr}{latex_key}",
            )
        else:
            # print(
            #     f"Skipping key {key} since values have at most one corresponding configuration"
            # )
            pass

for seed, rows in subcubic_data.items():
    for row in rows:
        for key, latex_key in subcubic_key_pairs:
            if key in row:
                assert isinstance(seed, int)
                latex_all_values_by_value[f"{row['tricks']}{latex_key}Float"][seed] = (
                    row[key]
                )

# %%
latex_values["AllModelsHEADSHA"] = git.get_head_sha(short=False)
latex_values["AllModelsHEADSHASHORT"] = git.get_head_sha(short=True)

with open(LATEX_VALUES_PATH, "w") as f:
    f.write(to_latex_defs(latex_values))
# %%
latex_all_values_by_seed: dict[int, dict[str, Union[int, float, str]]] = defaultdict(
    dict
)
for k, d in latex_all_values_by_value.items():
    for seed, v in d.items():
        latex_all_values_by_seed[seed][k] = v

with open(LATEX_VALUES_DATATABLE_PATH, "w", newline="") as f:
    all_keys = sorted(latex_all_values_by_value.keys())
    writer = csv.DictWriter(
        f, fieldnames=["seed"] + all_keys, quoting=csv.QUOTE_MINIMAL
    )

    writer.writeheader()

    for seed in sorted(latex_all_values_by_seed.keys()):
        row = {"seed": seed} | {
            k: format_float_full_precision(v) if isinstance(v, float) else v
            for k, v in latex_all_values_by_seed[seed].items()
        }
        writer.writerow(row)

# %%
# @title export LaTeX code
with open(LATEX_TIKZPLOTLIB_PREAMBLE_PATH, "w") as f:
    f.write(
        re.sub(
            r"\\documentclass{[^}]*}" + "\n*", "", tikzplotlib.Flavors.latex.preamble()
        )
        + r"""
% for line breaks
\pgfplotsset{title/.append style={align=center}}
"""
    )

# %%
# @title export LaTeX figures
title_reps = {
    "W_E": r"\WE ",
    r"W_{\text{pos}}": r"\Wpos ",
    r"W_{\mathrm{pos}}": r"\Wpos ",
    r"W_Q": r"\WQ ",
    r"W_K": r"\WK ",
    r"d_{\text{head}}": r"\dhead ",
    r"d_{\mathrm{head}}": r"\dhead ",
    r"W_V": r"\WV ",
    r"W_O": r"\WO",
    r"W_U": r"\WU ",
    r"\text{EQKE}": r"\EPQKE ",
    r"\mathrm{EQKE}": r"\EPQKE ",
    r"\text{EQKP}": r"\EPQKP ",
    r"\mathrm{EQKP}": r"\EPQKP ",
    r"d_{\mathrm{model}}": r"\dmodel ",
    r"d_{\mathrm{vocab}}": r"\dvocab ",
    r"QK^T": r"\WQ\WK^T",
    r"×": r"\ensuremath{\times}",
}


@contextmanager
def texify_title(
    fig: go.Figure, replace_with_macros: bool = True, show: bool = False, renderer=None
):
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
            # .replace("½", r"\sfrac{1}{2}")
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
            "head",
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
        if replace_with_macros:
            for search, rep in title_reps.items():
                new_title = new_title.replace(search, rep)

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


@contextmanager
def texify_matplotlib_title(
    fig: matplotlib.figure.Figure, show: bool = False, replace_with_macros: bool = True
):
    def texify(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        orig_s = s
        s = s.replace("\n", "\\\\\n")
        if replace_with_macros:
            for search, rep in title_reps.items():
                s = s.replace(search, rep)
        if s != orig_s:
            return s
        return None

    orig_suptitle = fig._suptitle.get_text() if fig._suptitle else None
    orig_titles = [ax.get_title() for ax in fig.axes if fig.axes]
    orig_xlabels = [ax.get_xlabel() for ax in fig.axes if fig.axes]
    orig_ylabels = [ax.get_ylabel() for ax in fig.axes if fig.axes]
    orig_legend_handles_labels = [
        ax.get_legend_handles_labels() if ax.get_legend() else ([], [])
        for ax in fig.axes
    ]
    new_suptitle = texify(orig_suptitle)
    new_titles = [texify(t) for t in orig_titles]
    new_xlabels = [texify(t) for t in orig_xlabels]
    new_ylabels = [texify(t) for t in orig_ylabels]
    new_legend_handles_labels = [
        (handles, [(texify(label) or label) for label in labels])
        for handles, labels in orig_legend_handles_labels
    ]
    try:
        if new_suptitle is not None:
            fig.suptitle(new_suptitle)
        if fig.axes:
            for (
                ax,
                new_title,
                new_xlabel,
                new_ylabel,
                (new_leg_handles, new_leg_labels),
            ) in zip(
                fig.axes,
                new_titles,
                new_xlabels,
                new_ylabels,
                new_legend_handles_labels,
            ):
                if new_title is not None:
                    ax.set_title(new_title)
                if new_xlabel is not None:
                    ax.set_xlabel(new_xlabel)
                if new_ylabel is not None:
                    ax.set_ylabel(new_ylabel)
                if new_leg_labels:
                    ax.legend(new_leg_handles, new_leg_labels)
        yield fig
    finally:
        if new_suptitle is not None:
            fig.suptitle(orig_suptitle)
        if fig.axes:
            for (
                ax,
                orig_title,
                orig_xlabel,
                orig_ylabel,
                (orig_leg_handles, orig_leg_labels),
            ) in zip(
                fig.axes,
                orig_titles,
                orig_xlabels,
                orig_ylabels,
                orig_legend_handles_labels,
            ):
                if orig_title is not None:
                    ax.set_title(orig_title)
                if orig_xlabel is not None:
                    ax.set_xlabel(orig_xlabel)
                if orig_ylabel is not None:
                    ax.set_ylabel(orig_ylabel)
                if orig_leg_labels:
                    ax.legend(orig_leg_handles, orig_leg_labels)


if SAVE_PLOTS:
    errs = []
    for file_path in chain(
        LATEX_FIGURE_PATH.glob("*.png"), LATEX_FIGURE_PATH.glob("*.dat")
    ):
        file_path.unlink()
        print(f"Deleted: {file_path}")
    table_row_sep = r"\\" + "\n"
    for k, fig in latex_figures.items():
        if isinstance(fig, go.Figure):
            fig.update_layout(font_family="Computer Modern")  # Use LaTeX fonts
            unsupported_by_tikzplotly = any(
                isinstance(trace, go.Heatmap) for trace in fig.data
            )
            # if not unsupported_by_tikzplotly:
            #     p = LATEX_FIGURE_PATH / f"{k}.tex"
            #     print(f"Saving {p}...")
            #     p.parent.mkdir(parents=True, exist_ok=True)
            #     tikzplotly.save(p, fig)
            with texify_title(fig, replace_with_macros=False) as fig:
                if True or unsupported_by_tikzplotly:
                    for ext in (".pdf", ".svg"):
                        p = LATEX_FIGURE_PATH / f"{k}{ext}"
                        print(f"Saving {p}...")
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.write_image(p)
                        if ext == ".pdf":
                            try:
                                subprocess.run(["pdfcrop", p, p], check=True)
                            except FileNotFoundError as e:
                                print(f"Warning: {e}")
                                errs.append(e)
        elif isinstance(fig, matplotlib.figure.Figure):
            p = LATEX_FIGURE_PATH / f"{k}.tex"
            p.parent.mkdir(parents=True, exist_ok=True)
            externalize_this_table = latex_externalize_tables.get(k, True)
            if externalize_this_table:
                if not latex_only_externalize_tables.get(k, False):
                    p = LATEX_FIGURE_PATH / f"{k}ExternalTables.tex"
                print(f"Saving {p}...")
                with texify_matplotlib_title(fig) as fig:
                    tikzplotlib.save(
                        p,
                        fig,
                        externalize_tables=externalize_this_table,
                        table_row_sep=table_row_sep,
                    )
            p = LATEX_FIGURE_PATH / f"{k}.tex"
            print(f"Saving {p}...")
            with texify_matplotlib_title(fig, replace_with_macros=True) as fig:
                tikzplotlib.save(
                    p, fig, externalize_tables=False, table_row_sep=table_row_sep
                )
            for ext in (".pdf", ".svg"):
                p = LATEX_FIGURE_PATH / f"{k}{ext}"
                print(f"Saving {p}...")
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p)
                if ext == ".pdf":
                    try:
                        subprocess.run(["pdfcrop", p, p], check=True)
                    except FileNotFoundError as e:
                        print(f"Warning: {e}")
                        errs.append(e)
        else:
            raise TypeError(f"Unsupported figure {fig} of type {type(fig)}")

    for f in LATEX_FIGURE_PATH.glob("*.png"):
        try:
            image_utils.pngcrush(f)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            errs.append(e)
        except subprocess.CalledProcessError as e:
            print(f"Warning: {e}")
            errs.append(e)

        try:
            image_utils.optipng(f)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            errs.append(e)
        except subprocess.CalledProcessError as e:
            print(f"Warning: {e}")
            errs.append(e)

    opt_success = False
    try:
        image_utils.optimize(*LATEX_FIGURE_PATH.glob("*.png"), exhaustive=True)
        opt_success = True
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        errs.append(e)
    except subprocess.CalledProcessError as e:
        print(f"Warning: {e}")
        errs.append(e)

    if not opt_success:
        for f in LATEX_FIGURE_PATH.glob("*.png"):
            try:
                image_utils.optimize(f, exhaustive=True)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                errs.append(e)
            except subprocess.CalledProcessError as e:
                print(f"Warning: {e}")
                errs.append(e)

    if errs:
        print("Errors:")
        for e in errs:
            print(e)
        print(f"Total errors: {len(errs)}")
    for e in errs:
        raise e
