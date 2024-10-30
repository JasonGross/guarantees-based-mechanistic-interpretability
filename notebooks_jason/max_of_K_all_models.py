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
from argparse import ArgumentParser, BooleanOptionalAction

from gbmi.exp_max_of_n import SEEDS, SELECTED_SEED

parser = ArgumentParser()
parser.add_argument(
    "--seeds",
    type=str,
    default=",".join(sorted(map(str, SEEDS))),
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
parser.add_argument(
    "--K",
    type=int,
    default=5,
    help="Sequence length",
)
parser.add_argument(
    "--d_vocab",
    type=int,
    default=64,
    help="Number of tokens",
)
parser.add_argument(
    "--brute-force",
    action=BooleanOptionalAction,
    default=False,
    help="Include brute force and ablations",
)
parser.add_argument(
    "--only-download",
    action=BooleanOptionalAction,
    default=False,
    help="Only download models, then quit",
)
parser.add_argument(
    "--print-cache-glob",
    action=BooleanOptionalAction,
    default=False,
    help="Print glob for cache files for these seeds",
)
parser.add_argument(
    "--print-cache-glob-absolute",
    action=BooleanOptionalAction,
    default=False,
    help="Print glob for cache files for these seeds, with absolute paths",
)
parser.add_argument(
    "--nsamples-per-key",
    type=int,
    default=None,
    help="Number of samples per key for importance sampling estimation of accuracy and loss",
)
parser.add_argument(
    "--huggingface",
    action=BooleanOptionalAction,
    default=False,
    help="Use huggingface",
)
parser.add_argument(
    "--save-to-hf",
    action=BooleanOptionalAction,
    default=True,
    help="Save to huggingface",
)
parser.add_argument(
    "--save-to-hf-from-cache",
    action=BooleanOptionalAction,
    default=True,
    help="Save to huggingface from cache",
)
parser.add_argument(
    "--hf-store-named-data-files",
    action=BooleanOptionalAction,
    default=False,
    help="Store named data files in huggingface in addition to aggregate ones",
)
parser.add_argument(
    "--hf-store-single-named-data-file",
    action=BooleanOptionalAction,
    default=True,
    help="Store a single named data file in huggingface in addition to aggregate ones",
)
parser.add_argument(
    "--compact-image-optimize-output",
    action=BooleanOptionalAction,
    default=False,
    help="Compact stdout/stderr format for image optimization",
)
parser.add_argument(
    "--pdf",
    action=BooleanOptionalAction,
    default=True,
    help="Output pdfs",
)
parser.add_argument(
    "--svg",
    action=BooleanOptionalAction,
    default=True,
    help="Output svgs",
)
parser.add_argument(
    "--only-optimize-images",
    action=BooleanOptionalAction,
    default=False,
    help="Only optimize images, then quit",
)
cli_args = parser.parse_args(None if ipython is None else ["--ignore-csv"])
# %%
#!sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super pdfcrop optipng pngcrush
# %%
import subprocess
import sys
from pathlib import Path

import gbmi.utils.images as image_utils

seq_len: int = cli_args.K
D_VOCAB: int = cli_args.d_vocab
EXTRA_D_VOCAB_FILE_SUFFIX: str = f"_d_vocab_{D_VOCAB}" if D_VOCAB != 64 else ""
COMPACT_IMAGE_OPTIMIZE_OUTPUT: bool = cli_args.compact_image_optimize_output
adjusted_file_path = Path(__file__).parent / Path(__file__).name.replace(
    "_K_", f"_{seq_len}_"
)
adjusted_file_path_dvocab = Path(__file__).parent / Path(__file__).name.replace(
    "_K_", f"_{seq_len}{EXTRA_D_VOCAB_FILE_SUFFIX}_"
)
cache_dir = adjusted_file_path.parent / ".cache"
cache_dir.mkdir(exist_ok=True)

OVERWRITE_CSV_FROM_CACHE: bool = not cli_args.ignore_csv  # @param {type:"boolean"}
compute_expensive_average_across_many_models: bool = True  # @param {type:"boolean"}
TRAIN_CSV_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-train-values.csv"
)
TRAIN_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
INCLUDE_BRUTE_FORCE: bool = cli_args.brute_force  # @param {type:"boolean"}
QUIT_AFTER_MODEL_DOWNLOAD: bool = cli_args.only_download  # @param {type:"boolean"}
BRUTE_FORCE_CSV_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-brute-force-values.csv"
)
BRUTE_FORCE_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
CUBIC_CSV_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-cubic-values.csv"
)
CUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_CSV_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-subcubic-values.csv"
)
SUBCUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_ANALYSIS_CSV_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-subcubic-analysis-values.csv"
)
SUBCUBIC_ANALYSIS_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
PYTHON_VERSION_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values-python-version.txt"
)
PYTHON_VERSION_PATH.parent.mkdir(exist_ok=True, parents=True)
TORCH_VERSION_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values-torch-version.txt"
)
TORCH_VERSION_PATH.parent.mkdir(exist_ok=True, parents=True)


GIT_DIFF_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values-git-diff-info.diff"
)
GIT_DIFF_PATH.parent.mkdir(exist_ok=True, parents=True)
GIT_SHA_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values-git-sha.txt"
)
GIT_SHA_PATH.parent.mkdir(exist_ok=True, parents=True)
GIT_SHA_SHORT_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values-git-sha-short.txt"
)
GIT_SHA_SHORT_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_VALUES_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values.tex"
)
LATEX_VALUES_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_VALUES_DATATABLE_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-all-values.csv"
)
LATEX_VALUES_DATATABLE_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_VALUES_REDUCED_DATATABLE_PATH = (
    adjusted_file_path.with_suffix("")
    / f"all-models{EXTRA_D_VOCAB_FILE_SUFFIX}-values.csv"
)
LATEX_VALUES_REDUCED_DATATABLE_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_FIGURE_PATH = (
    adjusted_file_path.with_suffix("") / f"figures{EXTRA_D_VOCAB_FILE_SUFFIX}"
)
LATEX_FIGURE_PATH.mkdir(exist_ok=True, parents=True)
LATEX_TIKZPLOTLIB_PREAMBLE_PATH = (
    adjusted_file_path.with_suffix("") / "tikzplotlib-preamble.tex"
)
LATEX_TIKZPLOTLIB_PREAMBLE_PATH.parent.mkdir(exist_ok=True, parents=True)
SHARED_CACHE_STEM = adjusted_file_path_dvocab.name.replace("_all_models", "")
(cache_dir / SHARED_CACHE_STEM).mkdir(exist_ok=True, parents=True)
OUTPUT_PDF: bool = cli_args.pdf
OUTPUT_SVG: bool = cli_args.svg
N_SAMPLES_PER_KEY = cli_args.nsamples_per_key
if N_SAMPLES_PER_KEY is None:
    match seq_len:
        case 4:
            N_SAMPLES_PER_KEY = 50
        case 5:
            N_SAMPLES_PER_KEY = 30
        case 10:
            N_SAMPLES_PER_KEY = 10
        case _:
            N_SAMPLES_PER_KEY = max(1, 100 // seq_len)
assert isinstance(N_SAMPLES_PER_KEY, int), (N_SAMPLES_PER_KEY, type(N_SAMPLES_PER_KEY))
N_THREADS: Optional[int] = cli_args.n_threads
DISPLAY_PLOTS: bool = False  # @param {type:"boolean"}
SAVE_PLOTS: bool = cli_args.plots
RENDERER: Optional[str] = "png"  # @param ["png", None]
PLOT_WITH: Literal["plotly", "matplotlib"] = (  # @param ["plotly", "matplotlib"]
    "matplotlib"
)
# %%
errs = []


def wrap_err(f, *args, return_bool: bool = False, **kwargs):
    try:
        result = f(*args, **kwargs)
        return True if return_bool else result
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        errs.append(e)
    except subprocess.CalledProcessError as e:
        print(f"Warning: {e}")
        errs.append(e)
    except OSError as e:
        print(f"Warning: {e}")
        errs.append(e)
    if return_bool:
        return False


def print_and_raise_errs():
    if errs:
        print("Errors:")
        for e in errs:
            print(e)
        print(f"Total errors: {len(errs)}")
    for e in errs:
        raise e


def optimize_pngs(errs: list[Exception] = []):
    # for f in LATEX_FIGURE_PATH.glob("*.png"):
    #     wrap_err(image_utils.ect, f)
    #     wrap_err(image_utils.pngcrush, f)
    #     wrap_err(image_utils.optipng, f)

    opt_success = wrap_err(
        image_utils.optimize,
        *LATEX_FIGURE_PATH.glob("*.png"),
        exhaustive=True,
        return_bool=True,
        trim_printout=COMPACT_IMAGE_OPTIMIZE_OUTPUT,
    )

    if not opt_success:
        for f in LATEX_FIGURE_PATH.glob("*.png"):
            wrap_err(
                image_utils.optimize,
                f,
                exhaustive=True,
                trim_printout=COMPACT_IMAGE_OPTIMIZE_OUTPUT,
            )


# %%
if cli_args.only_optimize_images:
    optimize_pngs()
    print_and_raise_errs()
    sys.exit(0)
# %%
import csv
import gc
import math
import os
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cache, partial
from itertools import chain
from typing import Any, Callable, Collection, Iterator, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tikzplotlib
import torch
from cycler import cycler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

import gbmi.exp_max_of_n.analysis.quadratic as analysis_quadratic
import gbmi.exp_max_of_n.analysis.subcubic as analysis_subcubic
import gbmi.exp_max_of_n.verification.brute_force as brute_force
import gbmi.exp_max_of_n.verification.cubic as cubic
import gbmi.exp_max_of_n.verification.quadratic as quadratic
import gbmi.exp_max_of_n.verification.subcubic as subcubic
import gbmi.utils.ein as ein
import gbmi.utils.git as git
import gbmi.utils.instructions as instructions
from gbmi.analysis_tools.plot import (
    Colorscale,
    EVOU_max_logit_diff,
    colorbar,
    combine_interpolate_color_mapping,
    hist_EVOU_max_logit_diff,
    remove_axis_labels,
    remove_axis_ticklabels,
    remove_colorbars,
    remove_titles,
    scatter,
)
from gbmi.analysis_tools.utils import (
    data_summary,
    data_summary_percentiles,
    pm_mean_std,
    pm_round,
)
from gbmi.exp_max_of_n.analysis import analyze_EVOU
from gbmi.exp_max_of_n.analysis.ablation import (
    compute_ablations,
    latexify_ablation_results,
)
from gbmi.exp_max_of_n.plot import (
    EVOU_max_minus_diag_logit_diff,
    attention_difference_over_gap,
    display_basic_interpretation,
    display_EQKE_SVD_analysis,
    hist_attention_difference_over_gap,
    hist_EVOU_max_minus_diag_logit_diff,
    make_better_slides_plots_00,
    scatter_attention_difference_vs_gap,
)
from gbmi.exp_max_of_n.train import (
    MAX_OF_4_CONFIG,
    MAX_OF_5_CONFIG,
    MAX_OF_10_CONFIG,
    MAX_OF_20_CONFIG,
    MaxOfN,
    MaxOfNDataModule,
    MaxOfNTrainingWrapper,
    train_or_load_model,
)
from gbmi.exp_max_of_n.verification import LargestWrongLogitQuadraticConfig
from gbmi.exp_max_of_n.verification.importance_sample_cubic import importance_sample
from gbmi.utils import deep_getsizeof, default_device, patch, reseed, to_device
from gbmi.utils.c_long import str_list_values_if_any_too_big_for_C_long
from gbmi.utils.dataclass import enumerate_dataclass_values
from gbmi.utils.hashing import get_hash_ascii
from gbmi.utils.instructions import (
    PERF_WORKING,
    CountHookedTransformer,
    CountTensor,
    CountTensorOperations,
    InstructionCount,
    PatchTorch,
    PerfCollector,
    PerfCounter,
    int_or_value,
)
from gbmi.utils.latex_export import (
    format_float_full_precision,
    format_float_full_precision_if_float,
    latex_values_of_counter,
    latex_values_of_instruction_count,
    to_latex_defs,
)
from gbmi.utils.memohf import StorageMethod as MemoHFStorageMethod
from gbmi.utils.memohf import memohf_staged
from gbmi.utils.memoshelve import memoshelve
from gbmi.utils.sequences import SequenceDataset

# %%
hf_repo_id = f"JasonGross/{adjusted_file_path.stem.replace('_','-').replace('-all-models', '')}-proofs"
USE_HF: bool = cli_args.huggingface
SAVE_TO_HF_FROM_CACHE: bool = cli_args.save_to_hf_from_cache
SAVE_TO_HF: bool = cli_args.save_to_hf
HF_STOAGE_METHODS: tuple[MemoHFStorageMethod, ...] = (
    ("named_data_files",) if cli_args.hf_store_named_data_files else ()
)
HF_STORE_SINGLE_NAMED_DATA_FILE: bool = cli_args.hf_store_single_named_data_file

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
default_OV_colorscale_2024_06_15: Colorscale = px.colors.get_colorscale("IceFire_r")
default_QK_colorscale_2024_06_15: Colorscale = px.colors.get_colorscale("IceFire_r")
# alt: Edge_r, Twilight, twilight_shifted, shift_cyclical_colorscale(px.colors.get_colorscale("Edge"), shift=0)
oranges = ["#fefec7", "#f29f05", "#f25c05", "#a62f03", "#400d01"]
blues = ["#e6f3ff", "#5e87f5", "#3d4b91", "#2d2c5e", "#1d0e2c"]
teals = ["#d1e8e8", "#9AD4DE", "#58B8C9", "#10656d", "#0c3547"]
default_colorscale_2024_06_16: Colorscale = combine_interpolate_color_mapping(
    oranges[::-1], blues
)
default_OV_colorscale: Colorscale = default_colorscale_2024_06_16
default_QK_colorscale: Colorscale = default_colorscale_2024_06_16
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
# hack around newlines of black formatting
seeds = (
    sorted(set(map(int, cli_args.seeds.split(","))))
    if compute_expensive_average_across_many_models
    else []
)
if SELECTED_SEED in seeds:
    seeds = [SELECTED_SEED] + [s for s in seeds if s != SELECTED_SEED]
match seq_len:
    case 4:
        make_cfg = MAX_OF_4_CONFIG
    case 5:
        make_cfg = MAX_OF_5_CONFIG
    case 10:
        make_cfg = partial(MAX_OF_10_CONFIG, d_vocab_out=D_VOCAB)
    case 20:
        make_cfg = partial(MAX_OF_20_CONFIG, d_vocab_out=D_VOCAB)
    case _:
        raise ValueError(f"Unsupported seq_len: {seq_len}")
cfgs = {seed: make_cfg(seed) for seed in list(seeds)}
cfg_hashes = {seed: get_hash_ascii(cfg) for seed, cfg in cfgs.items()}
model_cfgs = {
    seed: MaxOfNTrainingWrapper.build_model_config(cfg) for seed, cfg in cfgs.items()
}
datamodules = {seed: MaxOfNDataModule(cfg) for seed, cfg in cfgs.items()}
cfg_hashes_for_filename = {
    seed: f"{seed}_{cfg_hashes[seed].replace('/', '__SLASH__')}"
    for seed, cfg in cfgs.items()
}
# %%
if cli_args.print_cache_glob or cli_args.print_cache_glob_absolute:
    sub_glob = (
        "{" + ",".join(cfg_hash for cfg_hash in cfg_hashes_for_filename.values()) + "}"
    )
    train_or_load_model_glob = "train_or_load_model"
    stem = cache_dir / SHARED_CACHE_STEM
    if not cli_args.print_cache_glob_absolute:
        stem = stem.relative_to(Path.cwd())
    print(f"{stem}/" + "{" + f"{train_or_load_model_glob},*{sub_glob}*" + "}")
    sys.exit(0)


# %%
def hf_sanitize(s: str) -> str:
    return s.replace("-", "_").replace("=", "_").replace("+", "_")


@contextmanager
def memoshelve_hf_staged(
    short_name: Optional[str] = None,
    use_hf: bool = USE_HF,
    save_to_hf: bool = SAVE_TO_HF,
    save_to_hf_from_cache: bool = SAVE_TO_HF_FROM_CACHE,
    include_single_named_data_file: bool = HF_STORE_SINGLE_NAMED_DATA_FILE,
    storage_methods: tuple[MemoHFStorageMethod, ...] = HF_STOAGE_METHODS,
    **kwargs,
):
    if use_hf:
        with memohf_staged(
            hf_repo_id,
            storage_methods=storage_methods
            + (
                ()
                if short_name is None or not include_single_named_data_file
                else (("single_named_data_file", short_name),)
            ),
            save=save_to_hf,
            **kwargs,
        ) as memo_hf:

            @contextmanager
            def inner(
                func: Callable,
                file_suffix: str,
                subfolder: str = "",
                extra_file_suffix: str = "",
                extra_hf_file_suffix: str = "",
                **kwargs,
            ):
                subfolder = str(subfolder)
                extra_file_suffix = str(extra_file_suffix)
                extra_hf_file_suffix = str(extra_hf_file_suffix)
                hf_filename = hf_sanitize(
                    f"{file_suffix}"
                    f"{f'-{subfolder}' if subfolder else ''}"
                    f"{f'-{extra_file_suffix}' if extra_file_suffix else ''}"
                    f"{f'-{extra_hf_file_suffix}' if extra_hf_file_suffix else ''}"
                )
                filename = cache_dir / SHARED_CACHE_STEM
                if subfolder:
                    filename /= subfolder
                filename /= f"{file_suffix}{f'-{extra_file_suffix}' if extra_file_suffix else ''}"
                filename.parent.mkdir(exist_ok=True, parents=True)

                if save_to_hf_from_cache:
                    with memoshelve(
                        func,
                        filename=filename,
                        **kwargs,
                    )() as memo_func:
                        with memo_hf(memo_func, hf_filename, **kwargs) as memo_hf_func:
                            yield memo_hf_func
                else:
                    with memo_hf(func, hf_filename, **kwargs) as memo_hf_func:
                        with memoshelve(
                            memo_hf_func,
                            filename=filename,
                            **kwargs,
                        )() as memo:
                            yield memo

            yield inner
    else:

        @contextmanager
        def inner(
            func: Callable,
            file_suffix: str,
            subfolder: str = "",
            extra_file_suffix: str = "",
            extra_hf_file_suffix: str = "",
            **kwargs,
        ):
            subfolder = str(subfolder)
            extra_file_suffix = str(extra_file_suffix)
            extra_hf_file_suffix = str(extra_hf_file_suffix)
            filename = cache_dir / SHARED_CACHE_STEM
            if subfolder:
                filename /= subfolder
            filename /= (
                f"{file_suffix}{f'-{extra_file_suffix}' if extra_file_suffix else ''}"
            )
            filename.parent.mkdir(exist_ok=True, parents=True)
            with memoshelve(
                func,
                filename=filename,
                **kwargs,
            )() as memo:
                yield memo

        yield inner


# %%
# patch torch.load so that when loading cache from non-CPU devices we can still load
with patch(torch, load=partial(torch.load, map_location=torch.device("cpu"))):
    with memoshelve_hf_staged(
        short_name=f"train_or_load_model{EXTRA_D_VOCAB_FILE_SUFFIX}"
    ) as memoshelve_hf:
        with memoshelve_hf(
            train_or_load_model,
            "train_or_load_model",
            extra_hf_file_suffix=EXTRA_D_VOCAB_FILE_SUFFIX,
            get_hash=get_hash_ascii,
        ) as memo_train_or_load_model:
            runtime_models = {}

            def _handle_memo_train_or_load_model(arg):
                seed, cfg = arg
                try:
                    runtime_models[seed] = memo_train_or_load_model(cfg, force="load")
                except Exception as e:
                    print(f"Error loading model for seed {seed}: {e}")

            maybe_parallel_map(_handle_memo_train_or_load_model, tqdm(cfgs.items()))
# %%
assert all(
    model.cfg.d_vocab == D_VOCAB for _runtime, model in runtime_models.values()
), {seed: model.cfg.d_vocab for seed, (_runtime, model) in runtime_models.items()}
assert all(model.cfg.n_ctx == seq_len for _runtime, model in runtime_models.values()), {
    seed: model.cfg.n_ctx for seed, (_runtime, model) in runtime_models.items()
}
# %%
if __name__ == "__main__" and QUIT_AFTER_MODEL_DOWNLOAD:
    sys.exit(0)

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
        with open(
            adjusted_file_path.with_suffix("") / f"{name}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(result)

with open(GIT_DIFF_PATH, "w", encoding="utf-8") as f:
    f.write(git.get_diff())

with open(GIT_SHA_PATH, "w", encoding="utf-8") as f:
    f.write(git.get_head_sha(short=False))

with open(GIT_SHA_SHORT_PATH, "w", encoding="utf-8") as f:
    f.write(git.get_head_sha(short=True))

with open(PYTHON_VERSION_PATH, "w", encoding="utf-8") as f:
    f.write(sys.version)

with open(TORCH_VERSION_PATH, "w", encoding="utf-8") as f:
    f.write(torch.__version__)


# %%
training_wrappers = {
    seed: MaxOfNTrainingWrapper(cfgs[seed], model)
    for seed, (_runtime, model) in runtime_models.items()
}


# training_wrapper.run_batch = Memoize(training_wrapper.run_batch, name=f"{__file__}.training_wrapper.run_batch", use_pandas=False, use_shelf=True)  # type: ignore
# %%


# %%
def csv_to_hf_path(csv_path: Path, ext: Literal["csv", "parquet"] = "csv") -> str:
    hf_filename_stem = hf_sanitize(csv_path.stem)
    suffix = "" if ext == "csv" else f"_{ext}"
    return f"hf://datasets/{hf_repo_id}/{hf_filename_stem}/alldata{suffix}.{ext}"


def pd_read_csv_or_hf(
    csv_path: Path,
    columns: list[str],
    *,
    use_hf: bool = USE_HF,
    default: Optional[Callable] = pd.DataFrame,
):
    if use_hf:
        hf_path = csv_to_hf_path(csv_path, ext="parquet")
        try:
            return pd.read_parquet(hf_path)
        except FileNotFoundError as e:
            print(f"Could not read from huggingface ({hf_path}): {e}")
        hf_path = csv_to_hf_path(csv_path, ext="csv")
        try:
            return pd.read_csv(hf_path)
        except FileNotFoundError as e:
            print(f"Could not read from huggingface ({hf_path}): {e}")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if default:
        return default(columns=columns)


def update_csv_with_rows(
    csv_path: Path,
    new_data: list[dict[str, Union[float, int, str]]],
    *,
    columns: list[str],
    subset: str | list[str] = "seed",
    use_hf: bool = USE_HF,
    save_to_hf: bool = SAVE_TO_HF,
):
    results = pd_read_csv_or_hf(csv_path, columns=columns, use_hf=use_hf, default=None)

    new_data = str_list_values_if_any_too_big_for_C_long(new_data)

    new_df = pd.DataFrame(new_data, columns=columns)
    if results is None or results.empty:
        results = new_df
    elif not new_df.empty:
        results = pd.concat([results, new_df], ignore_index=True).drop_duplicates(
            subset=subset, keep="last"
        )
    results = results.sort_values(subset)
    results.to_csv(csv_path, index=False)
    if save_to_hf:
        results.to_parquet(csv_to_hf_path(csv_path, ext="parquet"))
        results.to_csv(csv_to_hf_path(csv_path, ext="csv"), index=False)
    return results


def update_csv(
    csv_path: Path,
    data: dict[int, dict[str, Union[float, int, str]]],
    columns: list[str],
    *,
    subset: str | list[str] = "seed",
):
    new_data = [data[seed] for seed in sorted(data.keys())]
    return update_csv_with_rows(csv_path, new_data, columns=columns, subset=subset)


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
    loss, accuracy = training_wrappers[seed].run_batch(
        (xs, ys), log_output=False, device=device
    )
    return loss.item(), accuracy, batch_size


def train_seed(seed: int, *, memoshelve_hf: Callable, pbar: tqdm):
    train_total_loss[seed] = 0.0
    train_total_accuracy[seed] = 0.0
    train_total_samples[seed] = 0

    datamodule = datamodules[seed]
    datamodule.setup("train")
    dataloader = datamodule.train_dataloader()
    dataloader_iter = iter(dataloader)
    with memoshelve_hf(
        partial(_run_train_batch_loss_accuracy, dataloader_iter=dataloader_iter),
        "run_batch_loss_accuracy",
        subfolder=cfg_hashes_for_filename[seed],
        extra_file_suffix=train_measurement_deterministic,
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    ) as run_batch_loss_accuracy:
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


def _handle_train_seed(seed: int, *, memoshelve_hf: Callable, pbar: tqdm):
    try:
        return train_seed(seed, memoshelve_hf=memoshelve_hf, pbar=pbar)
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
    with memoshelve_hf_staged(short_name="run_batch_loss_accuracy") as memoshelve_hf:
        maybe_parallel_map(
            partial(_handle_train_seed, memoshelve_hf=memoshelve_hf, pbar=pbar),
            sorted(runtime_models.keys()),
        )
# %%
# load csv
train_columns = ["seed", "loss", "accuracy", "model-seed", "dataset-seed"]

train_data = {
    seed: {
        "seed": seed,
        "loss": train_average_loss[seed],
        "accuracy": train_average_accuracy[seed],
        "model-seed": model_cfgs[seed].seed,
        "dataset-seed": datamodules[seed].dataset_seed,
    }
    for seed in runtime_models.keys()
}

all_train_data = update_csv(TRAIN_CSV_PATH, train_data, columns=train_columns)

# %%
num_seeds = len(train_average_loss)
avg_train_average_loss = sum(sorted(train_average_loss.values())) / num_seeds
avg_train_average_accuracy = sum(sorted(train_average_accuracy.values())) / num_seeds
std_dev_train_average_loss = float(np.std(list(sorted(train_average_loss.values()))))
std_dev_train_average_accuracy = float(
    np.std(list(sorted(train_average_accuracy.values())))
)
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
brute_force_results = pd_read_csv_or_hf(
    BRUTE_FORCE_CSV_PATH, columns=brute_force_columns
)

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
some_seed = SELECTED_SEED if SELECTED_SEED in all_seeds else list(sorted(all_seeds))[0]
_some_runtime, some_model = runtime_models[some_seed]


# loop for computing overall loss and accuracy
@torch.no_grad()
def _run_batch_loss_accuracy(
    all_tokens_dataset: SequenceDataset,
    training_wrapper: MaxOfNTrainingWrapper,
    i: int,
    batch_size: int,
    return_incorrect_sequences: bool = True,
) -> Tuple[
    Union[Tuple[float, float, int], Tuple[Tuple[float, float, int], Tensor]],
    float,
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


if INCLUDE_BRUTE_FORCE:

    def get_brute_force_for(
        seed: int,
        *,
        memoshelve_hf: Callable,
        memoshelve_hf_lightweight: Callable,
        pbar: tqdm,
    ):
        cfg_hash_for_filename = cfg_hashes_for_filename[seed]
        training_wrapper = training_wrappers[seed]
        all_tokens_dataset = all_tokens_datasets[seed]
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        total_duration = 0.0
        # all_incorrect_sequences = []

        with memoshelve_hf(
            partial(_run_batch_loss_accuracy, all_tokens_dataset, training_wrapper),
            "run_batch_loss_accuracy",
            subfolder=cfg_hash_for_filename,
            extra_file_suffix=brute_force_proof_deterministic,
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        ) as run_batch_loss_accuracy_heavy:

            def _run_batch_loss_accuracy_lightweight(*args, **kwargs):
                res = run_batch_loss_accuracy_heavy(*args, **kwargs)
                ((loss, accuracy, size), incorrect_sequences), duration = res
                return (loss, accuracy, size), duration

            with memoshelve_hf_lightweight(
                _run_batch_loss_accuracy_lightweight,
                "run_batch_loss_accuracy-lightweight",
                subfolder=cfg_hash_for_filename,
                extra_file_suffix=brute_force_proof_deterministic,
                get_hash_mem=(lambda x: x[0]),
                get_hash=str,
            ) as run_batch_loss_accuracy:
                for i in range(0, len(all_tokens_dataset), batch_size):
                    (loss, accuracy, size), duration = run_batch_loss_accuracy(i, batch_size)  # type: ignore
                    total_duration += duration
                    # Accumulate loss and accuracy
                    # start = time.time()
                    total_loss += loss * size
                    total_accuracy += accuracy * size
                    total_samples += size
                    # total_duration += time.time() - start
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

    def _handle_brute_force_for(
        seed: int,
        *,
        memoshelve_hf: Callable,
        memoshelve_hf_lightweight: Callable,
        pbar: tqdm,
    ):
        try:
            brute_force_data[seed] = get_brute_force_for(
                seed,
                memoshelve_hf=memoshelve_hf,
                memoshelve_hf_lightweight=memoshelve_hf_lightweight,
                pbar=pbar,
            )
        except Exception as e:
            print(f"Error computing brute force proof for seed {seed}: {e}")
            traceback.print_exc()

    with (
        tqdm(total=total_batches, desc="batches for brute force", position=0) as pbar,
        memoshelve_hf_staged(short_name="run_batch_loss_accuracy") as memoshelve_hf,
        memoshelve_hf_staged(
            short_name="run_batch_loss_accuracy_lightweight"
        ) as memoshelve_hf_lightweight,
    ):
        maybe_parallel_map(
            partial(
                _handle_brute_force_for,
                memoshelve_hf=memoshelve_hf,
                memoshelve_hf_lightweight=memoshelve_hf_lightweight,
                pbar=pbar,
            ),
            sorted(relevant_seeds),
        )
else:

    def get_brute_force_for(
        seed: int,
        *,
        memoshelve_hf: Callable,
        memoshelve_hf_lightweight: Callable,
        pbar: tqdm,
    ):
        cfg_hash_for_filename = cfg_hashes_for_filename[seed]
        _runtime, model = runtime_models[seed]

        with memoshelve_hf(
            partial(
                importance_sample,
                model,
                batch_size=batch_size,
                pbar=pbar,
                seed=reseed(seed, "importance_sample"),
            ),
            f"importance-sample-{N_SAMPLES_PER_KEY}",
            subfolder=cfg_hash_for_filename,
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        ) as importance_sample_heavy:

            def _importance_sample_lightweight(*args, **kwargs):
                res = importance_sample_heavy(*args, **kwargs).copy()
                del res["incorrect_sequences"]
                return res

            with memoshelve_hf_lightweight(
                _importance_sample_lightweight,
                f"importance-sample-lightweight-{N_SAMPLES_PER_KEY}",
                subfolder=cfg_hash_for_filename,
                get_hash_mem=(lambda x: x[0]),
                get_hash=str,
            ) as importance_sample_lightweight:
                results = importance_sample_lightweight((N_SAMPLES_PER_KEY, "per_key"))

        row = {"seed": seed, **results}
        return row

    def _handle_brute_force_via_importance_sampling_for(
        seed: int,
        *,
        memoshelve_hf: Callable,
        memoshelve_hf_lightweight: Callable,
        pbar: tqdm,
        subpbar: tqdm,
    ):
        pbar.update(1)
        pbar.set_postfix({"seed": seed})
        try:
            brute_force_data[seed] = get_brute_force_for(
                seed,
                memoshelve_hf=memoshelve_hf,
                memoshelve_hf_lightweight=memoshelve_hf_lightweight,
                pbar=subpbar,
            )
        except Exception as e:
            print(f"Error computing brute force proof for seed {seed}: {e}")
            traceback.print_exc()

    with (
        tqdm(total=len(relevant_seeds), desc="brute_force seeds", position=0) as pbar,
        tqdm(desc="importance sampling", position=1) as subpbar,
        # with PeriodicGarbageCollector(60):
        memoshelve_hf_staged(
            short_name=f"importance-sample-{N_SAMPLES_PER_KEY}"
        ) as memoshelve_hf,
        memoshelve_hf_staged(
            short_name=f"importance-sample-lightweight-{N_SAMPLES_PER_KEY}"
        ) as memoshelve_hf_lightweight,
    ):
        maybe_parallel_map(
            partial(
                _handle_brute_force_via_importance_sampling_for,
                memoshelve_hf=memoshelve_hf,
                memoshelve_hf_lightweight=memoshelve_hf_lightweight,
                pbar=pbar,
                subpbar=subpbar,
            ),
            sorted(relevant_seeds),
        )

    total_importance_sampled_sequences = (
        brute_force_data[some_seed]["num_correct"]
        + brute_force_data[some_seed]["num_incorrect"]
    )

all_brute_force_data = update_csv(
    BRUTE_FORCE_CSV_PATH, brute_force_data, columns=brute_force_columns
)

# %%
assert len(brute_force_data) == len(
    runtime_models
), f"len(brute_force_data) == {len(brute_force_data)} != {len(runtime_models)} == len(runtime_models)"
brute_force_key_prefix = "BruteForce" if INCLUDE_BRUTE_FORCE else "ImportanceSampling"
latex_values[f"{brute_force_key_prefix}BatchSize"] = batch_size

if INCLUDE_BRUTE_FORCE:
    all_tokens_datasets_lens = {seed: len(d) for seed, d in all_tokens_datasets.items()}
    assert (
        len(set(all_tokens_datasets_lens.values())) == 1
    ), f"Multiple dataset lengths! {set(all_tokens_datasets_lens.values())}"
    latex_values["BruteForceCPU"] = brute_force_proof_deterministic
    latex_values["BruteForceNumBatches"] = int(
        math.ceil(list(all_tokens_datasets_lens.values())[0] / batch_size)
    )
else:
    latex_values["ImportanceSamplingNSamplesPerKey"] = N_SAMPLES_PER_KEY
    latex_values["ImportanceSamplingTotalSequences"] = (
        total_importance_sampled_sequences
    )


brute_force_data_by_key = defaultdict(dict)
for seed, d in brute_force_data.items():
    for k, v in d.items():
        brute_force_data_by_key[k][seed] = v

for key, latex_key in (
    ("loss", f"{brute_force_key_prefix}Loss"),
    ("accuracy", f"{brute_force_key_prefix}Accuracy"),
    ("num_correct", f"{brute_force_key_prefix}NumCorrect"),
    ("num_incorrect", f"{brute_force_key_prefix}NumIncorrect"),
    ("duration", f"{brute_force_key_prefix}Time"),
):
    latex_values |= data_summary(brute_force_data_by_key[key], prefix=latex_key)
    assert all(isinstance(seed, int) for seed in brute_force_data_by_key[key].keys())
    latex_all_values_by_value[f"{latex_key}Float"] = brute_force_data_by_key[key]


# %%
@torch.no_grad()
def single_batch_instruction_count(
    all_tokens_dataset: SequenceDataset,
    training_wrapper: MaxOfNTrainingWrapper,
    model: HookedTransformer,
    batch_size: int,
) -> Tuple[InstructionCount, PerfCounter]:
    with PerfCollector() as collector:
        if PERF_WORKING:
            _run_batch_loss_accuracy(
                all_tokens_dataset,
                training_wrapper,
                0,
                batch_size,
                return_incorrect_sequences=False,
            )
    perf_instruction_count = collector.counters

    with CountTensorOperations() as result:
        batch = CountTensor.from_numpy(all_tokens_dataset[:batch_size])
        size = batch.shape[0]
        labels: CountTensor = training_wrapper.config.experiment.get_ground_truth(batch)  # type: ignore
        xs, ys = batch, labels
        y_preds: CountTensor = CountHookedTransformer(model)(xs)
        loss: CountTensor = training_wrapper.loss_fn(
            y_preds, ys, log_softmax=CountTensor.log_softmax  # type: ignore
        )  # type: ignore
        full_accuracy: CountTensor = training_wrapper.acc_fn_per_seq(y_preds, ys)  # type: ignore
        accuracy: CountTensor = full_accuracy.float().mean()

    return result, perf_instruction_count


def brute_force_instruction_count(
    all_tokens_dataset: SequenceDataset,
    training_wrapper: MaxOfNTrainingWrapper,
    model: HookedTransformer,
    batch_size: int,
) -> Tuple[InstructionCount, PerfCounter]:
    n_full_batches = (model.cfg.d_vocab_out**model.cfg.n_ctx) // batch_size
    final_batch_size = (model.cfg.d_vocab_out**model.cfg.n_ctx) % batch_size
    single_batch, single_batch_perf = single_batch_instruction_count(
        all_tokens_dataset, training_wrapper, model, batch_size
    )
    result = single_batch * n_full_batches
    result_perf = single_batch_perf * n_full_batches
    if final_batch_size != 0:
        final_batch, final_batch_perf = single_batch_instruction_count(
            all_tokens_dataset, training_wrapper, model, final_batch_size
        )
        result += final_batch
        result_perf += final_batch_perf
    return result, result_perf


brute_force_count, brute_force_perf = brute_force_instruction_count(
    all_tokens_datasets[some_seed],
    training_wrappers[some_seed],
    some_model,
    batch_size,
)
latex_values |= latex_values_of_instruction_count("BruteForce", brute_force_count)
latex_values |= latex_values_of_counter("BruteForce", brute_force_perf)

# %%


@torch.no_grad()
def importance_sample_instruction_count(
    model: HookedTransformer,
    batch_size: int,
    nsamples: int,
    *,
    pbar: Optional[tqdm] = None,
) -> Tuple[InstructionCount, PerfCounter]:
    cache = {}
    with PerfCollector() as collector:
        if PERF_WORKING:
            importance_sample(model, 0, batch_size=batch_size, pbar=pbar, cache=cache)
    perf_instruction_count_shared = collector.counters
    with PerfCollector() as collector:
        if PERF_WORKING:
            importance_sample(
                model, batch_size, batch_size=batch_size, pbar=pbar, cache=cache
            )
    perf_instruction_count_per_batch = collector.counters

    cache = {}

    with CountTensorOperations() as result_shared:
        importance_sample(model, 0, batch_size=batch_size, pbar=pbar, cache=cache)
    print(cache)
    with CountTensorOperations() as result_per_batch:
        importance_sample(
            model, batch_size, batch_size=batch_size, pbar=pbar, cache=cache
        )

    num_batches = nsamples // batch_size

    return (
        result_shared + result_per_batch * num_batches,
        perf_instruction_count_shared + perf_instruction_count_per_batch * num_batches,
    )


if not INCLUDE_BRUTE_FORCE:
    with (
        tqdm(desc="importance sampling instruction counts") as pbar,
        memoshelve_hf_staged(storage_methods=("named_data_files",)) as memoshelve_hf,
    ):
        with memoshelve_hf(
            partial(importance_sample_instruction_count, some_model, pbar=pbar),
            f"importance-sample-instruction-count{'' if not PERF_WORKING else '-with-perf'}-{N_SAMPLES_PER_KEY}",
            extra_hf_file_suffix=f"{EXTRA_D_VOCAB_FILE_SUFFIX}-n_ctx_{seq_len}",
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        ) as memo_importance_sample_instruction_count:
            importance_sample_count, importance_sample_perf = (
                memo_importance_sample_instruction_count(
                    batch_size, total_importance_sampled_sequences
                )
            )
    latex_values |= latex_values_of_instruction_count(
        brute_force_key_prefix, importance_sample_count
    )
    latex_values |= latex_values_of_counter(
        brute_force_key_prefix, importance_sample_perf
    )

# %% [markdown]
# # Ablations
# %%
if INCLUDE_BRUTE_FORCE:
    ablation_data = {}

    def get_ablation_for(seed: int, *, memoshelve_hf: Callable, pbar: tqdm):
        cfg = cfgs[seed]
        cfg_hash = cfg_hashes[seed]
        cfg_hash_for_filename = cfg_hashes_for_filename[seed]
        runtime, model = runtime_models[seed]

        with memoshelve_hf(
            partial(compute_ablations, model, pbar=pbar),
            "compute_ablations",
            subfolder=cfg_hash_for_filename,
            get_hash=get_hash_ascii,
            get_hash_mem=str,
        ) as memo_compute_ablations:
            ablation_results, ablation_time = memo_compute_ablations()
        return latexify_ablation_results(
            ablation_results, float_postfix="", int_postfix=""
        )

    def _handle_ablation_for(
        seed: int, *, memoshelve_hf: Callable, seed_pbar: tqdm, pbar: tqdm
    ):
        try:
            seed_pbar.set_postfix(dict(seed=seed))
            seed_pbar.update(1)
            ablation_data[seed] = get_ablation_for(
                seed, memoshelve_hf=memoshelve_hf, pbar=pbar
            )
        except Exception as e:
            print(f"Error computing ablation for seed {seed}: {e}")
            traceback.print_exc()

    all_d_vocabs = [model.cfg.d_vocab for _runtime, model in runtime_models.values()]
    assert len(set(all_d_vocabs)) == 1, f"Multiple d_vocabs: {all_d_vocabs}"

    with (
        tqdm(
            total=len(all_d_vocabs),
            desc="seeds for ablations",
            position=0,
        ) as seed_pbar,
        tqdm(
            total=sum(all_d_vocabs),
            desc="batches for ablations",
            position=1,
        ) as pbar,
        # with PeriodicGarbageCollector(60):
        memoshelve_hf_staged(short_name="compute_ablations") as memoshelve_hf,
    ):
        maybe_parallel_map(
            partial(
                _handle_ablation_for,
                memoshelve_hf=memoshelve_hf,
                seed_pbar=seed_pbar,
                pbar=pbar,
            ),
            sorted(all_seeds),
        )
# %%
if INCLUDE_BRUTE_FORCE:
    ablation_data_by_key = defaultdict(dict)
    for seed, d in ablation_data.items():
        for k, v in d.items():
            ablation_data_by_key[k][seed] = v

    for key, values in ablation_data_by_key.items():
        latex_values |= data_summary(values, prefix=key)
        assert all(isinstance(seed, int) for seed in values.keys())
        latex_all_values_by_value[f"{key}Float"] = values


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
cubic_results = pd_read_csv_or_hf(CUBIC_CSV_PATH, columns=cubic_columns)

all_seeds = set(runtime_models.keys())
unknown_seeds = all_seeds - set(cubic_results["seed"])
known_seeds = all_seeds - unknown_seeds
relevant_seeds = all_seeds if OVERWRITE_CSV_FROM_CACHE else unknown_seeds
cubic_data = {
    seed: cubic_results[cubic_results["seed"] == seed].iloc[0].to_dict()
    for seed in known_seeds
}


def get_cubic_row(
    seed: int,
    *,
    memoshelve_hf_find_proof: Callable,
    memoshelve_hf_verify_proof: Callable,
    pbar: tqdm,
) -> dict:
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    runtime, model = runtime_models[seed]

    # loop for computing overall loss and accuracy
    @torch.no_grad()
    def _find_proof() -> Tuple[dict, float]:
        start = time.time()
        cubic_proof_args = cubic.find_proof(model)
        duration = time.time() - start
        return cubic_proof_args, duration

    with memoshelve_hf_find_proof(
        _find_proof,
        "cubic_find_proof",
        subfolder=cfg_hash_for_filename,
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    ) as find_proof:
        cubic_proof_args, duration_proof_search = find_proof()

    with memoshelve_hf_verify_proof(
        partial(
            cubic.verify_proof,
            model,
            pbar=pbar,
            print_complexity=False,
            print_results=False,
            include_perf=PERF_WORKING,
        ),
        f"cubic_verify_proof{'' if not PERF_WORKING else '-with-perf'}",
        subfolder=cfg_hash_for_filename,
        get_hash_mem=(lambda x: 0),
        get_hash=(lambda x: "0"),
    ) as verify_proof:
        cubic_proof_results = verify_proof(cubic_proof_args)

    # largest_wrong_logit_cubic = cubic_proof_results["largest_wrong_logit"]
    return (
        {
            "seed": seed,
            "accuracy-bound": cubic_proof_results["accuracy_lower_bound"],
            "correct-count-lower-bound": cubic_proof_results[
                "correct_count_lower_bound"
            ],
            "duration-proof-search": duration_proof_search,
            "duration": cubic_proof_results["prooftime"],
        }
        | (
            {
                "normalized-accuracy-bound": cubic_proof_results["accuracy_lower_bound"]
                / brute_force_data_by_key["accuracy"][seed],
            }
            if INCLUDE_BRUTE_FORCE
            else {}
        )
        | (
            latex_values_of_counter("Cubic", cubic_proof_results["proofinstructions"])
            if PERF_WORKING
            else {}
        )
    )


def _handle_cubic(
    seed: int,
    *,
    memoshelve_hf_find_proof: Callable,
    memoshelve_hf_verify_proof: Callable,
    pbar: tqdm,
):
    try:
        cubic_data[seed] = get_cubic_row(
            seed,
            memoshelve_hf_find_proof=memoshelve_hf_find_proof,
            memoshelve_hf_verify_proof=memoshelve_hf_verify_proof,
            pbar=pbar,
        )
    except Exception as e:
        print(f"Error computing cubic proof for seed {seed}: {e}")
        traceback.print_exc()


# \sum_{i=0}^{k} i^2 = k * (k+1) * (k*2+1) // 6
ks = [model_cfgs[seed].d_vocab - 2 for seed in relevant_seeds]
total_batches = sum(k * (k + 1) * (k * 2 + 1) // 6 for k in ks)
with (
    tqdm(total=total_batches, desc="batches for cubic", position=0) as pbar,
    # with PeriodicGarbageCollector(60):
    memoshelve_hf_staged(short_name="cubic_find_proof") as memoshelve_hf_find_proof,
    memoshelve_hf_staged(
        short_name=f"cubic_verify_proof{'' if not PERF_WORKING else '-with-perf'}"
    ) as memoshelve_hf_verify_proof,
):
    maybe_parallel_map(
        partial(
            _handle_cubic,
            pbar=pbar,
            memoshelve_hf_find_proof=memoshelve_hf_find_proof,
            memoshelve_hf_verify_proof=memoshelve_hf_verify_proof,
        ),
        sorted(relevant_seeds),
    )

# do this externally, because importance sampling is subject to change
for seed, row in cubic_data.items():
    row["normalized-accuracy-bound"] = (
        row["accuracy-bound"] / brute_force_data_by_key["accuracy"][seed]
    )

all_cubic_data = update_csv(CUBIC_CSV_PATH, cubic_data, columns=cubic_columns)

# %% [markdown]
# Summary satistics cubic
# %%
cubic_data_by_key = defaultdict(dict)
for seed, d in cubic_data.items():
    for k, v in d.items():
        cubic_data_by_key[k][seed] = v

assert len(cubic_data) == len(
    runtime_models
), f"len(cubic_data) == {len(cubic_data)} != {len(runtime_models)} == len(runtime_models)"
for key in ("accuracy-bound", "duration", "normalized-accuracy-bound"):
    print(
        f"Cubic {key}: {pm_mean_std(np.array(list(cubic_data_by_key[key].values()), dtype=np.float64))}"
    )

for key, latex_key in (
    # ("loss", "CubicLoss"),
    ("accuracy-bound", "CubicAccuracy"),
    ("correct-count-lower-bound", "CubicCorrectCount"),
    ("duration", "CubicProofTime"),
    # ) + (
    ("normalized-accuracy-bound", "CubicNormalizedAccuracy"),
    # if INCLUDE_BRUTE_FORCE
    # else ()
) + (
    tuple(
        (f"CubicPerf{latex_attr}", f"CubicPerf{latex_attr}")
        for latex_attr in (
            "TimeEnabledNS",
            "InstructionCount",
            "BranchMisses",
            "PageFaults",
        )
    )
    if PERF_WORKING
    else ()
):
    latex_values |= data_summary(cubic_data_by_key[key], prefix=latex_key)
    assert all(isinstance(seed, int) for seed in cubic_data_by_key[key].keys())
    latex_all_values_by_value[f"{latex_key}Float"] = cubic_data_by_key[key]


# %%
# %%
def _cubic_count_verify_proof(
    model: HookedTransformer,
    proof_args: dict,
    *,
    sanity_check_instructions: bool = False,
) -> Tuple[InstructionCount, dict[str, Any]]:
    # must be outside PatchTorch to avoid triu, tril
    cmodel = CountHookedTransformer(model)
    with PatchTorch():
        with instructions.set_sanity_check(sanity_check_instructions):
            with CountTensorOperations() as cubic_instruction_count:
                cubic_proof_instruction_count_results = cubic.verify_proof(
                    cmodel,
                    proof_args,
                    print_complexity=False,
                    print_results=False,
                    sanity_check=False,
                    # print_types=True,
                )
    return cubic_instruction_count, cubic_proof_instruction_count_results


with memoshelve_hf_staged(storage_methods=("named_data_files",)) as memoshelve_hf:
    with memoshelve_hf(
        partial(_cubic_count_verify_proof, some_model, sanity_check_instructions=False),
        f"cubic_count_verify_proof{'' if not PERF_WORKING else '-with-perf'}",
        extra_hf_file_suffix=f"{EXTRA_D_VOCAB_FILE_SUFFIX}-n_ctx_{seq_len}",
        get_hash_mem=(lambda x: 0),
        get_hash=(lambda x: "0"),
    ) as count_verify_proof:
        cubic_proof_args = cubic.find_proof(some_model)
        cubic_instruction_count, cubic_proof_instruction_count_results = (
            count_verify_proof(cubic_proof_args)
        )

latex_values |= latex_values_of_instruction_count("Cubic", cubic_instruction_count)


# %% [markdown]
# # Intermediate interp values for export
# %%
def max_logit_diffs_analysis(
    model: HookedTransformer, *, warning: Callable[[str], None] = print
) -> dict[str, Any]:
    result = {}
    max_logit_diff = EVOU_max_logit_diff(model).cpu()
    max_logit_diff_summary = data_summary(
        max_logit_diff, prefix="EVOUMaxRowDiff", float_postfix=""
    )
    result |= max_logit_diff_summary

    for duplicate_by_sequence_count in [False, True]:
        key = "EVOU-hist-min-above-diag"
        if duplicate_by_sequence_count:
            key += "-dup-by-seq-count"
        (max_logit_minus_diag, duplication_factors) = EVOU_max_minus_diag_logit_diff(
            model,
            duplicate_by_sequence_count=duplicate_by_sequence_count,
            warning=warning,
        )
        max_logit_minus_diag, duplication_factors = (
            max_logit_minus_diag.cpu(),
            duplication_factors.cpu(),
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
        result[value_key + "MostBelowValue"] = most_below_value
        result[value_key + "MostBelowValueNumStd"] = num_std
        result[value_key + "MostBelowValueSequenceFrac"] = frac_below
        for k, v in data_summary(
            max_logit_minus_diag,
            sample_weight=duplication_factors,
            prefix=value_key,
            float_postfix="",
        ).items():
            result[k] = v

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
        result |= data_summary(
            flat_diffs,
            sample_weight=duplication_factors,
            prefix=value_key,
            float_postfix="",
        )

    return result


def handle_compute_max_logit_diffs_analysis(
    seed: int,
    *,
    memoshelve_hf: Callable,
    warning: Callable[[str], None] = print,
):
    runtime, model = runtime_models[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    with memoshelve_hf(
        (lambda seed: max_logit_diffs_analysis(model, warning=warning)),
        "compute_max_logit_diffs_analysis",
        subfolder=cfg_hash_for_filename,
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    ) as memo_compute_max_logit_diffs_analysis:
        return memo_compute_max_logit_diffs_analysis(seed)


with memoshelve_hf_staged(
    short_name="compute_max_logit_diffs_analysis"
) as memoshelve_hf:
    max_logit_diffs_analyses = {
        seed: handle_compute_max_logit_diffs_analysis(
            seed, memoshelve_hf=memoshelve_hf, warning=tqdm.write
        )
        for seed in tqdm(
            list(sorted(runtime_models.keys())), desc="max logit diff analysis"
        )
    }

# %%
max_logit_diffs_analyses_by_key = defaultdict(dict)
for seed, d in max_logit_diffs_analyses.items():
    for k, v in d.items():
        max_logit_diffs_analyses_by_key[k][seed] = v
# %%
for k, v in max_logit_diffs_analyses_by_key.items():
    latex_values |= data_summary(v, prefix=k)
    assert all(isinstance(seed, int) for seed in v.keys())
    latex_all_values_by_value[f"{k}Float"] = v


# %%
# with memoshelve_hf(
#     (lambda seed, with_attn_scale: compute_EQKE_SVD_analysis(runtime_models[seed][1], with_attn_scale=with_attn_scale)),
#     filename=f"compute_EQKE_SVD_analysis",
#     get_hash_mem=(lambda x: x[0]),
#     get_hash=str,
# ) as memo_compute_EQKE_SVD_analysis:
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
def handle_compute_EQKE_SVD_analysis(
    seed: int,
    *,
    memoshelve_hf: Callable,
):
    runtime, model = runtime_models[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    with memoshelve_hf(
        (
            lambda seed: display_EQKE_SVD_analysis(
                model, include_figures=False, show=False, do_print=False
            )[1]
        ),
        "compute_EQKE_SVD_analysis",
        subfolder=cfg_hash_for_filename,
        get_hash_mem=(lambda x: x[0]),
        get_hash=str,
    ) as memo_compute_EQKE_SVD_analysis:
        return memo_compute_EQKE_SVD_analysis(seed)


with memoshelve_hf_staged(short_name="compute_EQKE_SVD_analysis") as memoshelve_hf:
    EQKE_SVD_analyses = {
        seed: handle_compute_EQKE_SVD_analysis(seed, memoshelve_hf=memoshelve_hf)
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

all_subcubic_analysis_data = update_csv_with_rows(
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

    seen = set()
    for k in axis_limits.keys():
        k_no_min_max = (
            k.replace("zmin", "")
            .replace("zmax", "")
            .replace("min", "")
            .replace("max", "")
        )
        latex_key = "".join(
            [
                kpart if kpart[:1] == kpart[:1].capitalize() else kpart.capitalize()
                for kpart in k_no_min_max.replace("-", "_").split("_")
            ]
        )
        k_min = k.replace("max", "min")
        k_max = k.replace("min", "max")
        assert k_min in axis_limits, f"Missing {k_min}"
        assert k_max in axis_limits, f"Missing {k_max}"
        assert k_min == k or k_max == k, f"Unknown key: {k}"
        assert k_min != k_max, f"Same key: {k}"
        if "centered" not in k.lower():
            v_max = np.max([np.abs(axis_limits[k_min]), np.abs(axis_limits[k_max])])
            axis_limits[k_min] = -v_max
            axis_limits[k_max] = v_max

            assert "OV" in k or "QK" in k, f"Unknown key: {k}"
            if k_no_min_max in seen:
                continue
            kwargs = dict(zmin=-v_max, zmax=v_max)
        else:
            if k_no_min_max in seen:
                continue
            kwargs = dict(zmin=axis_limits[k_min], zmax=axis_limits[k_max])
        kwargs |= dict(
            colorscale=(default_OV_colorscale if "OV" in k else default_QK_colorscale),
            show=False,
            plot_with=PLOT_WITH,
            renderer=RENDERER,
        )
        figV = colorbar(**kwargs, orientation="vertical")
        figH = colorbar(**kwargs, orientation="horizontal")
        seen.add(k_no_min_max)
        latex_figures[f"Colorbar-{latex_key}-Vertical"] = figV
        latex_figures[f"Colorbar-{latex_key}-Horizontal"] = figH

    for k, v in axis_limits.items():
        k = "".join(
            [
                kpart if kpart[0] == kpart[0].capitalize() else kpart.capitalize()
                for kpart in k.replace("-", "_").split("_")
            ]
        )
        latex_values[f"AxisLimits{k}Float"] = v

    with tqdm(
        runtime_models.items(), desc="display_basic_interpretation (uniform limits)"
    ) as pbar:
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
                    remove_axis_labels(fig)
                    remove_colorbars(fig)
                    remove_axis_ticklabels(fig, remove_tickmarks=True)
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
                remove_axis_labels(fig)
                remove_colorbars(fig)
                remove_axis_ticklabels(fig, remove_tickmarks=True)


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
                    warning=tqdm.write,
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
                        "WEqqPerp",
                        "WQqPerp",
                        "WKkPerp",
                        "WEkkPerp",
                    )
                } | {
                    f"EQKE_err{attn_scale}": f"EQKE-err{attn_scale}",
                    f"EQKE_err_noticks{attn_scale}": f"EQKE-err-noticks{attn_scale}",
                    f"EQKE_err_simple{attn_scale}": f"EQKE-err-simple{attn_scale}",
                    f"EQKE_err_simple_noticks{attn_scale}": f"EQKE-err-simple-noticks{attn_scale}",
                    f"EQKE_err_svd{attn_scale}": f"EQKE-err-svd{attn_scale}",
                    f"EQKE{attn_scale}1": f"EQKE{attn_scale}1",
                    f"EQKE{attn_scale}2": f"EQKE{attn_scale}2",
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
subcubic_results = pd_read_csv_or_hf(SUBCUBIC_CSV_PATH, columns=subcubic_columns)

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
    memoshelve_hf_shared_proof_search: Callable,
    memoshelve_hf_find_min_gaps: Callable,
    memoshelve_hf_verify_proof: Callable,
    memoshelve_hf_count_verify_proof: Callable,
    memoshelve_hf_analyze_gaps: Callable,
    subcfg_pbar: tqdm,
    cfg_pbar: tqdm,
    proof_pbar: tqdm,
    count_proof_pbar: tqdm,
) -> list[dict]:
    cfg = cfgs[seed]
    cfg_hash_for_filename = cfg_hashes_for_filename[seed]
    runtime, model = runtime_models[seed]

    min_gaps_lists = {}

    rows = []

    with memoshelve_hf_shared_proof_search(
        (lambda seed: analysis_subcubic.find_proof_shared(model)),
        # cache={},
        "shared_proof_search",
        subfolder=cfg_hash_for_filename,
    ) as shared_proof_search:
        (
            W_EP_direction_kwargs,
            find_min_gaps_kwargs,
            size_and_query_directions_kwargs,
            shared_proof_search_duration,
        ) = shared_proof_search(seed)

    with memoshelve_hf_find_min_gaps(
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
        "find_min_gaps",
        subfolder=cfg_hash_for_filename,
    ) as find_min_gaps_for:
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

        with memoshelve_hf_verify_proof(
            _verify_proof,
            f"subcubic_verify_proof{'' if not PERF_WORKING else '-with-perf'}",
            subfolder=cfg_hash_for_filename,
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        ) as verify_proof:
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

        with memoshelve_hf_count_verify_proof(
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
            "subcubic_count_verify_proof",
            subfolder=cfg_hash_for_filename,
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        ) as count_verify_proof:
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
                    # manually convert to float to avoid overflow when unpacking long
                    weights[: max_tok + 1, max_tok, n_copies_nonmax] = float(
                        (max_tok - 1) ** n_copies_nonmax
                        * math.comb(model.cfg.n_ctx - 1, n_copies_nonmax)
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
            mean = np.average(v.numpy(), weights=weights.flatten().cpu().numpy())
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

        with memoshelve_hf_analyze_gaps(
            _analyze_gaps,
            "subcubic_analyze_gaps",
            subfolder=cfg_hash_for_filename,
            get_hash_mem=(lambda x: x[0]),
            get_hash=str,
        ) as analyze_gaps:
            (frac_below, v, weights, most_below_value, mean, std, num_std) = (
                analyze_gaps(tricks)
            )

        row = (
            {
                "seed": seed,
                "accuracy-bound": accuracy_bound,
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
            }
            | perf_results
            | (
                {
                    "normalized-accuracy-bound": accuracy_bound
                    / brute_force_data_by_key["accuracy"][seed],
                }
                if INCLUDE_BRUTE_FORCE
                else {}
            )
        )

        rows.append(row)
        proof_pbar.update(1)
    return rows


def _handle_subcubic(
    seed: int,
    *,
    memoshelve_hf_shared_proof_search: Callable,
    memoshelve_hf_find_min_gaps: Callable,
    memoshelve_hf_verify_proof: Callable,
    memoshelve_hf_count_verify_proof: Callable,
    memoshelve_hf_analyze_gaps: Callable,
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
            memoshelve_hf_shared_proof_search=memoshelve_hf_shared_proof_search,
            memoshelve_hf_find_min_gaps=memoshelve_hf_find_min_gaps,
            memoshelve_hf_verify_proof=memoshelve_hf_verify_proof,
            memoshelve_hf_count_verify_proof=memoshelve_hf_count_verify_proof,
            memoshelve_hf_analyze_gaps=memoshelve_hf_analyze_gaps,
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
    memoshelve_hf_staged(
        short_name="shared_proof_search"
    ) as memoshelve_hf_shared_proof_search,
    memoshelve_hf_staged(short_name="find_min_gaps") as memoshelve_hf_find_min_gaps,
    memoshelve_hf_staged(
        short_name="subcubic_verify_proof"
    ) as memoshelve_hf_verify_proof,
    memoshelve_hf_staged(
        short_name="subcubic_count_verify_proof"
    ) as memoshelve_hf_count_verify_proof,
    memoshelve_hf_staged(
        short_name="subcubic_analyze_gaps"
    ) as memoshelve_hf_analyze_gaps,
):
    # with PeriodicGarbageCollector(60):
    maybe_parallel_map(
        partial(
            _handle_subcubic,
            memoshelve_hf_shared_proof_search=memoshelve_hf_shared_proof_search,
            memoshelve_hf_find_min_gaps=memoshelve_hf_find_min_gaps,
            memoshelve_hf_verify_proof=memoshelve_hf_verify_proof,
            memoshelve_hf_count_verify_proof=memoshelve_hf_count_verify_proof,
            memoshelve_hf_analyze_gaps=memoshelve_hf_analyze_gaps,
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

# do this externally, because importance sampling is subject to change
for seed in subcubic_data:
    for row in subcubic_data[seed]:
        row["normalized-accuracy-bound"] = (
            row["accuracy-bound"] / brute_force_data_by_key["accuracy"][seed]
        )

new_data = []
for seed in sorted(subcubic_data.keys()):
    new_data.extend(subcubic_data[seed])

all_subcubic_data = update_csv_with_rows(
    SUBCUBIC_CSV_PATH, new_data, columns=subcubic_columns, subset=["seed", "tricks"]
)

# %%
# %% [markdown]
# Summary satistics subcubic
# %%

assert len(subcubic_data) == len(
    runtime_models
), f"len(cubic_data) == {len(subcubic_data)} != {len(runtime_models)} == len(runtime_models)"


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
            "DirectVocabModelSquared"
            if tricks.EUPU_handling_subcubic_no_quadratic_vocab
            else None if tricks.EUPU_handling_subcubic else "DirectCubic"
        )
    )
    EPQKE_str = (
        "AttentionQuadratic"
        if tricks.attention_error_handling_quadratic
        and tricks.attention_handling_quadratic
        else (
            "AttentionVocabModelSquared"
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
            "SubcubicVocabModelSquared",
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
            if all(
                key in row for rows in filtered_subcubic_data.values() for row in rows
            ):
                latex_values |= data_summary(
                    [
                        row[key]
                        for rows in filtered_subcubic_data.values()
                        for row in rows
                    ],
                    prefix=f"{trick_filter_descr}{latex_key}",
                )
            else:
                print(
                    f"Skipping key {key} since some rows are missing the key: {tuple(row['seed'] if 'seed' in row else row for rows in filtered_subcubic_data.values() for row in rows if key not in row)}"
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
# Approximating effective dimensionality
# %%
brute_force_df = all_brute_force_data
brute_force_ext_df = brute_force_df.copy()
assert "BruteForceInstructionCount" in latex_values
# print("Warning: falling back on old value for BruteForceInstructionCount")
# brute_force_ext_df["proof-flop-estimate"] = latex_values.get(
#     "BruteForceInstructionCount", 876123000832
# )
brute_force_ext_df["proof-flop-estimate"] = latex_values["BruteForceInstructionCount"]
brute_force_ext_df["normalized-accuracy-bound"] = (
    brute_force_ext_df["accuracy"] / brute_force_ext_df["accuracy"]
)
brute_force_ext_df["accuracy-bound"] = brute_force_ext_df["accuracy"]
brute_force_ext_df["group"] = (
    "brute-force" if INCLUDE_BRUTE_FORCE else "importance-sampling"
)
brute_force_ext_df["effective-dimension-estimate"] = brute_force_ed
brute_force_ext_df["leading-complexity"] = (
    "brute-force" if INCLUDE_BRUTE_FORCE else "importance-sampling"
)
brute_force_ext_df["tricks"] = ""

cubic_df = all_cubic_data
subcubic_df = all_subcubic_data
subcubic_analysis_df = all_subcubic_analysis_data

cubic_ext_df = cubic_df.merge(brute_force_df[["seed", "accuracy"]], on="seed")
assert "CubicInstructionCount" in latex_values
# print("Warning: falling back on old value for CubicInstructionCount")
# cubic_ext_df["proof-flop-estimate"] = latex_values.get(
#     "CubicInstructionCount", 35181664
# )
cubic_ext_df["proof-flop-estimate"] = latex_values["CubicInstructionCount"]
cubic_ext_df["normalized-accuracy-bound"] = (
    cubic_ext_df["accuracy-bound"] / cubic_ext_df["accuracy"]
)
cubic_ext_df["group"] = "cubic"
cubic_ext_df["leading-complexity"] = "cubic"
cubic_ext_df["effective-dimension-estimate"] = cubic_ed
cubic_ext_df["tricks"] = ""

subcubic_PVOU_cost = d_vocab
subcubic_EPQKP_cost = 0

subcubic_ext_df = subcubic_df.merge(brute_force_df[["seed", "accuracy"]], on="seed")
subcubic_ext_df["normalized-accuracy-bound"] = (
    subcubic_ext_df["accuracy-bound"] / subcubic_ext_df["accuracy"]
)
warned = False


@cache
def parse_tricks_legacy(tricks):
    if tricks.startswith("ExactEQKE"):
        global warned
        if not warned:
            print(f"Warning: legacy {tricks}")
        warned = True
        tricks = tricks[len("ExactEQKE") :]
        assert tricks.endswith("AttnErrMaxDiffExact"), tricks
        tricks = tricks[: -len("AttnErrMaxDiffExact")] + "AttnErrExactEqkeMaxDiffExact"
    return LargestWrongLogitQuadraticConfig.parse(tricks, latex=True)


# TODO: merge this with previous version
def subcubic_approx_effective_dimension_df(row, df):
    cfg = (
        model_cfgs[row["seed"]]
        if row["seed"] in model_cfgs
        else MaxOfNTrainingWrapper.build_model_config(make_cfg(row["seed"]))
    )
    tricks = parse_tricks_legacy(row["tricks"])
    return (
        int(tricks.effective_dimension_estimate(cfg))
        + subcubic_PVOU_cost
        + subcubic_EPQKP_cost
        + EVOU_cost
    )


def leading_complexity_df(row, df):
    tricks = parse_tricks_legacy(row["tricks"])
    return (
        "almost-quadratic"
        if tricks.is_quadratic
        else (
            "vocab-model-squared"
            if tricks.is_subcubic_no_quadratic_vocab
            else "subcubic" if tricks.is_subcubic else "fake-cubic"
        )
    )


def subcubic_group_df(row, df):
    tricks = parse_tricks_legacy(row["tricks"])
    EUPU_str = (
        "direct-quadratic"
        if tricks.EUPU_handling_quadratic
        else (
            "direct-vocab-model-squared"
            if tricks.EUPU_handling_subcubic_no_quadratic_vocab
            else None if tricks.EUPU_handling_subcubic else "direct-cubic"
        )
    )
    EPQKE_str = (
        "attention-quadratic"
        if tricks.attention_error_handling_quadratic
        and tricks.attention_handling_quadratic
        else (
            "attention-vocab-model-squared"
            if tricks.attention_error_handling_subcubic_no_quadratic_vocab
            and tricks.attention_handling_subcubic_no_quadratic_vocab
            else (
                None
                if tricks.attention_error_handling_subcubic
                and tricks.attention_handling_subcubic
                else "attention-cubic-reference"
            )
        )
    )
    strs = [s for s in (EPQKE_str, EUPU_str) if s is not None]
    return "subcubic" + (f" ({', '.join(strs)})" if strs else "")


subcubic_ext_df["group"] = subcubic_ext_df.apply(
    subcubic_group_df, args=(subcubic_ext_df,), axis=1
)
subcubic_ext_df["effective-dimension-estimate"] = subcubic_ext_df.apply(
    subcubic_approx_effective_dimension_df, args=(subcubic_ext_df,), axis=1
)
subcubic_ext_df["leading-complexity"] = subcubic_ext_df.apply(
    leading_complexity_df, args=(subcubic_ext_df,), axis=1
)

# Combine all data into a single DataFrame
combined_df = pd.concat(
    [
        subcubic_ext_df[
            [
                "proof-flop-estimate",
                "normalized-accuracy-bound",
                "accuracy-bound",
                "seed",
                "effective-dimension-estimate",
                "leading-complexity",
                "group",
                "tricks",
            ]
        ],
        brute_force_ext_df[
            [
                "proof-flop-estimate",
                "normalized-accuracy-bound",
                "accuracy-bound",
                "seed",
                "effective-dimension-estimate",
                "leading-complexity",
                "group",
                "tricks",
            ]
        ],
        cubic_ext_df[
            [
                "proof-flop-estimate",
                "normalized-accuracy-bound",
                "accuracy-bound",
                "seed",
                "effective-dimension-estimate",
                "leading-complexity",
                "group",
                "tricks",
            ]
        ],
    ],
    ignore_index=True,
)


def is_frontier(row, df):
    seed_group = df[df["seed"] == row["seed"]]
    for _, other in seed_group.iterrows():
        if (
            other["normalized-accuracy-bound"] > row["normalized-accuracy-bound"]
            and other["proof-flop-estimate"] < row["proof-flop-estimate"]
        ):
            return False
    return True


combined_df["frontier"] = combined_df.apply(is_frontier, args=(combined_df,), axis=1)


# %%
def double_singleton_groups(data: pd.DataFrame, column: str) -> pd.DataFrame:
    # hack around https://github.com/nschloe/tikzplotlib/issues/594
    group_counts = data[column].value_counts()
    single_row_groups = group_counts[group_counts == 1].index
    for group in single_row_groups:
        single_row = data[data[column] == group]
        data = pd.concat([data, single_row], ignore_index=True)
    return data


# %%

# %%
subcubic_sing_df = subcubic_ext_df.merge(
    subcubic_analysis_df[["seed", "EQKERatioFirstTwoSingularFloat"]], on="seed"
)[["seed", "normalized-accuracy-bound", "tricks", "EQKERatioFirstTwoSingularFloat"]]
tricks = set(subcubic_sing_df["tricks"])
for trick in tricks:
    if "AttnErrExactEqke" in trick:
        subcubic_sing_df = subcubic_sing_df[subcubic_sing_df["tricks"] != trick]
subcubic_sing_df["attention_error_handling"] = subcubic_sing_df.apply(
    (
        lambda row: LargestWrongLogitQuadraticConfig.parse(
            row["tricks"], latex=True
        ).attention_error_handling.replace("_", "-")
    ),
    axis=1,
)
subcubic_sing_df["attention_handling"] = subcubic_sing_df.apply(
    (
        lambda row: LargestWrongLogitQuadraticConfig.parse(
            row["tricks"], latex=True
        ).attention_handling.replace("_", "-")
    ),
    axis=1,
)
subcubic_sing_df["EUPU_handling"] = subcubic_sing_df.apply(
    (
        lambda row: LargestWrongLogitQuadraticConfig.parse(
            row["tricks"], latex=True
        ).EUPU_handling.replace("_", "-")
    ),
    axis=1,
)
subcubic_sing_df = subcubic_sing_df.loc[
    subcubic_sing_df.groupby(["seed", "attention_error_handling"])[
        "normalized-accuracy-bound"
    ].idxmax()
]
data = subcubic_sing_df[
    [
        "normalized-accuracy-bound",
        "EQKERatioFirstTwoSingularFloat",
        "attention_error_handling",
    ]
]
data = double_singleton_groups(
    data.drop_duplicates(), column="attention_error_handling"
)
if DISPLAY_PLOTS:
    fig = px.scatter(
        data,
        y="normalized-accuracy-bound",
        x="EQKERatioFirstTwoSingularFloat",
        color="attention_error_handling",
        title="Normalized Accuracy Bound vs EQKE Ratio First Two Singular",
        labels={
            "normalized-accuracy-bound": "Normalized Accuracy Bound",
            "EQKERatioFirstTwoSingularFloat": "EQKE Ratio First Two Singular",
        },
    )
    # fig.update_layout(showlegend=False)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5)
    )
    # Show the plot
    fig.show("png")
# %%
# plt.set_prop_cycle(color=['red', 'green', 'blue'])
# default_colors
# cycler(color=plt.cm.Paired.colors)
# cycler(color=plt.cm.tab20c.colors)
# %%
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.cm.Paired.colors[::-1])

subcubic_sing_df = subcubic_ext_df.merge(
    subcubic_analysis_df[["seed", "EQKERatioFirstTwoSingularFloat"]], on="seed"
)[["seed", "normalized-accuracy-bound", "tricks", "EQKERatioFirstTwoSingularFloat"]]
subcubic_sing_df["attention_error_handling"] = subcubic_sing_df.apply(
    (
        lambda row: LargestWrongLogitQuadraticConfig.parse(
            row["tricks"], latex=True
        ).attention_error_handling.replace("_", "-")
    ),
    axis=1,
)
subcubic_sing_df["attention_handling"] = subcubic_sing_df.apply(
    (
        lambda row: LargestWrongLogitQuadraticConfig.parse(
            row["tricks"], latex=True
        ).attention_handling.replace("_", "-")
    ),
    axis=1,
)
subcubic_sing_df["EUPU_handling"] = subcubic_sing_df.apply(
    (
        lambda row: LargestWrongLogitQuadraticConfig.parse(
            row["tricks"], latex=True
        ).EUPU_handling.replace("_", "-")
    ),
    axis=1,
)
tricks = set(subcubic_sing_df["tricks"])
for trick in tricks:
    if (
        "exact_EQKE"
        in LargestWrongLogitQuadraticConfig.parse(
            trick, latex=True
        ).attention_error_handling
    ):
        subcubic_sing_df = subcubic_sing_df[subcubic_sing_df["tricks"] != trick]

# subcubic_sing_df = subcubic_sing_df.loc[
#     subcubic_sing_df.groupby(["seed", "attention_error_handling"])[
#         "normalized-accuracy-bound"
#     ].idxmax()
# ]

for best_bound_only in (True, False):
    df = subcubic_sing_df.copy()
    if best_bound_only:
        print(f"len before: {len(df)}")
        df = df.loc[
            df.groupby(["seed", "attention_error_handling"])[
                "normalized-accuracy-bound"
            ].idxmax()
        ]
        print(f"len after: {len(df)}")

    # Group by 'attention_error_handling' and calculate the max 'normalized-accuracy-bound' for sorting groups
    df = df[
        [
            "normalized-accuracy-bound",
            "EQKERatioFirstTwoSingularFloat",
            "attention_error_handling",
        ]
    ].sort_values(
        by=[
            "attention_error_handling",
            "normalized-accuracy-bound",
            "EQKERatioFirstTwoSingularFloat",
        ]
    )
    # Group by 'attention_error_handling' and calculate the max 'normalized-accuracy-bound' for each group
    max_bound_by_group = df.groupby("attention_error_handling")[
        "normalized-accuracy-bound"
    ].max()

    # Sort the groups by max 'normalized-accuracy-bound'
    sorted_groups = max_bound_by_group.sort_values(ascending=False)

    # Extract the sorted list of 'attention_error_handling' categories
    sorted_attn_err_handling = sorted_groups.index.tolist()

    key = f"NormalizedAccuracyBound{'AllSecondaryTricks' if not best_bound_only else ''}VsEPQKESingularRatio"

    for sing_upper_bound, sing_upper_bound_descr in ((375, "OnlySmallest"), (None, "")):
        for attn_err_handling_key, group in df.groupby("attention_error_handling"):
            subgroup = group
            if sing_upper_bound is not None:
                subgroup = subgroup[
                    subgroup["EQKERatioFirstTwoSingularFloat"] < sing_upper_bound
                ]
            # if there are not enough data points, skip the group
            if len(subgroup) < 2:
                continue
            X = subgroup["EQKERatioFirstTwoSingularFloat"].values.reshape(-1, 1)
            y = subgroup["normalized-accuracy-bound"].values

            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = r2_score(y, model.predict(X))
            attn_err_handling_key_latex = (
                LargestWrongLogitQuadraticConfig.transform_description(
                    attn_err_handling_key, latex=True
                )
            )
            if best_bound_only:
                print(
                    f"{attn_err_handling_key}{sing_upper_bound_descr}:\tbound ≈ {intercept} + {slope} (σ₁/σ₂),\tr^2: {r_squared}"
                )

            latex_values[
                f"{key}{sing_upper_bound_descr}{attn_err_handling_key_latex}LinearFitSlope"
            ] = slope
            latex_values[
                f"{key}{sing_upper_bound_descr}{attn_err_handling_key_latex}LinearFitIntercept"
            ] = intercept
            latex_values[f"{key}{attn_err_handling_key_latex}LinearFitRSquared"] = (
                r_squared
            )

    latex_externalize_tables[key] = True
    df = double_singleton_groups(
        df.drop_duplicates(), column="attention_error_handling"
    )

    if DISPLAY_PLOTS or SAVE_PLOTS:
        latex_figures[key] = fig = scatter(
            df,
            yrange=(0, 1),
            y="normalized-accuracy-bound",
            x="EQKERatioFirstTwoSingularFloat",
            color="attention_error_handling",
            # title='Normalized Accuracy Bound vs EQKE Ratio First Two Singular',
            yaxis="Normalized Accuracy Bound",
            xaxis=r"EPQKE Singular Ratio: $\sigma_1 / \sigma_2$",
            # labels={
            #     'normalized-accuracy-bound': 'Normalized Accuracy Bound',
            #     'EQKERatioFirstTwoSingularFloat': 'EQKE Ratio First Two Singular'
            # }
            color_order=sorted_attn_err_handling,
            renderer=RENDERER,
            plot_with=PLOT_WITH,
            show=DISPLAY_PLOTS,
            # plot_with="plotly"
        )
        # fig.update_layout(showlegend=False)
        # fig.update_layout(
        #     legend=dict(
        #         orientation="h",
        #         yanchor="top",
        #         y=-0.3,
        #         xanchor="center",
        #         x=0.5
        #     )
        # )
        # # Show the plot
        # fig.show()


# %%
for descr, df in (
    (
        "all subcubic",
        subcubic_ext_df[subcubic_ext_df["leading-complexity"] != "fake-cubic"][
            "proof-flop-estimate"
        ],
    ),
    (
        "mainly subcubic",
        subcubic_ext_df[subcubic_ext_df["leading-complexity"] == "subcubic"][
            "proof-flop-estimate"
        ],
    ),
    (
        "almost quadratic",
        subcubic_ext_df[subcubic_ext_df["leading-complexity"] == "almost-quadratic"][
            "proof-flop-estimate"
        ],
    ),
):
    prekey = f"ProofFlopEstimate{''.join([s.capitalize() for s in descr.split(' ')])}"
    latex_values[f"{prekey}Mean"] = df.mean()
    latex_values[f"{prekey}StdDev"] = df.std()
    print(f"{descr}: {pm_mean_std(df)}")
# %%
df_sorted = combined_df.sort_values(by="normalized-accuracy-bound", ascending=False)
category_order = df_sorted["group"].unique().tolist()
category_name_remap = {
    "brute-force": f"brute force (acc: {pm_mean_std(brute_force_df['accuracy'])})",
    "importance-sampling": f"brute force (estimated) (acc: {pm_mean_std(brute_force_df['accuracy'])})",
    "cubic": f"cubic (rel acc: {pm_mean_std(cubic_ext_df['normalized-accuracy-bound'])})",
}
category_name_remap_short = {
    "brute-force": f"brute force",
    "importance-sampling": f"brute force (estimated)",
    "cubic": f"cubic",
}
max_rows = subcubic_ext_df.loc[
    subcubic_ext_df.groupby(["seed", "group"])["normalized-accuracy-bound"].idxmax()
]
result = (
    max_rows.groupby("group")["normalized-accuracy-bound"]
    .agg(["mean", "std"])
    .reset_index()
)
for group_name in category_order:
    if group_name not in category_name_remap:
        avg, std = result[result["group"] == group_name][["mean", "std"]].iloc[0]
        new_group_name = group_name[len("subcubic (") : -1]
        new_group_name = "subcubic" if not new_group_name else new_group_name
        new_group_name = new_group_name.replace(
            "vocab-model-squared", r"$d_{\mathrm{vocab}}d_{\mathrm{model}}^2$"
        )
        category_name_remap[group_name] = (
            f"{new_group_name} (rel acc: {pm_round(avg, std)})"
        )
        category_name_remap_short[group_name] = new_group_name


# PLOT_WITH = "matplotlib"
# %%
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.cm.Paired.colors)
latex_externalize_tables["EffectiveDimensionVsFLOP"] = True
data = combined_df[
    ["proof-flop-estimate", "effective-dimension-estimate", "group"]
].copy()
data = double_singleton_groups(data.drop_duplicates(), column="group")
data = data.sort_values(
    by=["group", "proof-flop-estimate", "effective-dimension-estimate"]
)
data["group"] = data["group"].map(category_name_remap_short)
if DISPLAY_PLOTS or SAVE_PLOTS:
    latex_externalize_tables["EffectiveDimensionVsFLOP"] = True
    latex_figures["EffectiveDimensionVsFLOP"] = fig = scatter(
        data,
        x="proof-flop-estimate",
        y="effective-dimension-estimate",
        color="group",
        title="",
        log_x=2,
        log_y=2,
        reverse_xaxis=False,
        color_order=[category_name_remap_short[c] for c in category_order],
        xaxis="FLOPs to Verify Proof (approximate)",
        yaxis="Unexplained Dimension (Estimated)",
        plot_with=PLOT_WITH,
        renderer=RENDERER,
        show=DISPLAY_PLOTS,
    )
    latex_externalize_tables["EffectiveDimensionVsFLOPDiscontinuousXY"] = True
    latex_figures["EffectiveDimensionVsFLOPDiscontinuousXY"] = fig = scatter(
        data,
        x="proof-flop-estimate",
        y="effective-dimension-estimate",
        color="group",
        title="",
        log_x=2,
        log_y=2,
        reverse_xaxis=False,
        color_order=[category_name_remap_short[c] for c in category_order],
        xaxis="FLOPs to Verify Proof (approximate)",
        yaxis="Unexplained Dimension (Estimated)",
        discontinuous_x=(
            data[(data["group"] == "brute force") | (data["group"] == "cubic")][
                "proof-flop-estimate"
            ].mean(),
        ),
        discontinuous_y=(
            data[(data["group"] == "brute force") | (data["group"] == "cubic")][
                "effective-dimension-estimate"
            ].mean(),
        ),
        plot_with=PLOT_WITH,
        renderer=RENDERER,
        show=DISPLAY_PLOTS,
    )


# %%
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.cm.Paired.colors)
for frontier_only in (True, False):
    for norm, normt in (("", ""), ("normalized-", "Normalized ")):
        key = f"{normt.strip()}AccuracyBoundVsFLOPs{'FrontierOnly' if frontier_only else ''}"
        data = (
            combined_df[combined_df["frontier"] == True]
            if frontier_only
            else combined_df
        )
        data = data[["proof-flop-estimate", f"{norm}accuracy-bound", "group"]].copy()
        data = double_singleton_groups(data.drop_duplicates(), column="group")
        data = data.sort_values(
            by=["group", f"{norm}accuracy-bound", "proof-flop-estimate"]
        )
        discontinuous_x = (
            data[(data["group"] == "brute force") | (data["group"] == "cubic")][
                "proof-flop-estimate"
            ].mean(),
        )
        compress_data = lambda values: (
            f"{values.item() / 2 ** int(math.log2(values.item()))} \\cdot 2^{{{int(math.log2(values.item()))}}}"
            if len(values) == 1
            else f"({pm_mean_std(values / 2 ** int(math.log2(values.mean())))}) \\cdot 2^{{{int(math.log2(values.mean()))}}}"
        )
        print(
            [
                (
                    compress_data(
                        data[data["group"] == c]["proof-flop-estimate"].unique()
                    ),
                    category_name_remap[c],
                )
                for c in category_order
                if len(data[data["group"] == c]["proof-flop-estimate"]) > 0
            ]
        )
        data["group"] = data["group"].map(category_name_remap)
        if DISPLAY_PLOTS or SAVE_PLOTS:
            markersize = (
                plt.rcParams["lines.markersize"] / 8 if not frontier_only else None
            )
            latex_externalize_tables[key] = True
            latex_figures[key] = fig = scatter(
                data,
                x="proof-flop-estimate",
                y=f"{norm}accuracy-bound",
                color="group",
                title="",  # "Pareto Frontier" if frontier_only else "",
                log_x=2,
                reverse_xaxis=False,
                xaxis="FLOPs to Verify Proof (approximate)",
                yaxis=f"{normt}Accuracy Bound",
                color_order=[category_name_remap[c] for c in category_order],
                markersize=markersize,
                plot_with=PLOT_WITH,
                renderer=RENDERER,
                show=DISPLAY_PLOTS,
            )
            latex_externalize_tables[f"{key}DiscontinuousX"] = True
            latex_figures[f"{key}DiscontinuousX"] = fig = scatter(
                data,
                x="proof-flop-estimate",
                y=f"{norm}accuracy-bound",
                color="group",
                title="",  # "Pareto Frontier" if frontier_only else "",
                log_x=2,
                reverse_xaxis=False,
                xaxis="FLOPs to Verify Proof (approximate)",
                yaxis=f"{normt}Accuracy Bound",
                color_order=[category_name_remap[c] for c in category_order],
                markersize=markersize,
                discontinuous_x=discontinuous_x,
                plot_with=PLOT_WITH,
                renderer=RENDERER,
                show=DISPLAY_PLOTS,
            )


# %%
for norm, normt in (("", ""), ("normalized-", "Normalized ")):

    data = combined_df[
        [
            "proof-flop-estimate",
            f"{norm}accuracy-bound",
            "group",
            "frontier",
            "tricks",
        ]
    ].copy()
    data = double_singleton_groups(data.drop_duplicates(), column="group")
    # data["group"] = data["group"].map({k:k[:7] for k in set(data["group"])})
    if DISPLAY_PLOTS:
        fig = px.scatter(
            data,
            x="proof-flop-estimate",
            y=f"{norm}accuracy-bound",
            symbol="group",
            title=f"Scatter Plot of Proof Flop Estimate vs {normt}Accuracy Bound (Logarithmic X-Axis)",
            log_x=True,
            color="tricks",
            # symbol_map={True: "diamond", False: "circle"},
            # legend=False,
        )
        fig.update_layout(showlegend=False)
        # Flip the x-axis
        fig.update_layout(xaxis=dict(autorange="reversed"))

        fig.show()

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

all_keys = sorted(latex_all_values_by_value.keys())
# for saving to HF
update_csv_with_rows(
    LATEX_VALUES_DATATABLE_PATH,
    new_data=[
        {"seed": seed}
        | {
            k: format_float_full_precision_if_float(
                latex_all_values_by_seed[seed].get(k, "")
            )
            for k in all_keys
        }
        for seed in sorted(latex_all_values_by_seed.keys())
    ],
    columns=["seed"] + all_keys,
)
update_csv_with_rows(
    LATEX_VALUES_REDUCED_DATATABLE_PATH,
    new_data=[{"index": 0} | latex_values],
    columns=["index"] + list(sorted(latex_values.keys())),
    subset="index",
)
with open(LATEX_VALUES_DATATABLE_PATH, "w", newline="") as f:
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
# @title Clean up data to free memory while exporting figures
for var in (
    "latex_all_values_by_seed",
    "latex_all_values_by_value",
    "latex_values",
    "subcubic_data",
    "new_data",
    "subcubic_sing_df",
    "subcubic_ext_df",
    "all_subcubic_data",
    "subcubic_df",
    "subcubic_results",
    "df_sorted",
    "combined_df",
    "EQKE_SVD_analyses",
    "EVOU_analyses",
    "data",
    "EQKE_SVD_analyses_by_key",
    "EVOU_analyses_by_key",
    "max_logit_diffs_analyses",
    "row",
    "all_subcubic_analysis_data",
    "subcubic_analysis_df",
    "all_keys",
    "max_logit_diffs_analyses_by_key",
    "max_rows",
    "subgroup",
    "group",
    "filtered_subcubic_data",
    "filtered_subcubic_data_best_by_key",
    "rows",
    "cubic_data",
    "all_axis_limits",
    "brute_force_data",
    "cubic_data_by_key",
    "brute_force_data_by_key",
    "brute_force_ext_df",
    "train_data",
    "cubic_ext_df",
    "df",
    "values",
    "runtime_models",
    "all_cubic_data",
    "cubic_df",
    "cfg_hashes_for_filename",
    "tricks",
    "subcubic_groups",
    "cubic_results",
    "cfg_hashes",
    "subcubic_leading_complexities",
    "model_cfgs",
    "training_wrappers",
    "cfgs",
    "datamodules",
    "all_tokens_datasets",
    "kwargs",
    "train_total_samples",
    "known_seeds",
    "all_seeds",
    "train_average_loss",
    "train_total_loss",
    "train_average_accuracy",
    "train_total_accuracy",
    "default_QK_SVD_colorscale",
    "default_colorscale_2024_06_16",
    "default_OV_colorscale",
    "default_QK_colorscale",
    "all_brute_force_data",
    "brute_force_df",
    "brute_force_results",
    "all_train_data",
    "all_configs",
    "figs",
    "seeds",
    "SEEDS",
    "key_pairs",
    "subcubic_key_pairs",
    "best_row",
    "default_OV_colorscale_2024_06_15",
    "default_QK_colorscale_2024_06_15",
    "cur_key_pairs",
    "category_name_remap",
    "subcubic_columns",
    "ablation_data",
    "ablation_data_by_key",
    "d",
    "all_tokens_datasets_lens",
    "sub_cfg_counts",
):
    if var in locals():
        del locals()[var]
gc.collect()
# %%
# @title Print out names and sizes of locals
sizes = [
    {"name": name, "size": deep_getsizeof(var), "type": str(type(var))}
    for name, var in locals().items()
    if not name.startswith("_")
    and (
        isinstance(var, (pd.DataFrame, dict, list, tuple, torch.Tensor, np.ndarray))
        or sys.getsizeof(var) > 10000
    )
    and deep_getsizeof(var) > 10000
]
if sizes:
    df_sizes = pd.DataFrame(
        sizes,
        columns=["name", "size", "type"],
    )
    df_sizes = df_sizes.sort_values(by="size", ascending=False)
    print(df_sizes.to_string(index=False))
    df_sizes = df_sizes.sort_values(by="name", ascending=True)
    print(df_sizes.to_string(index=False))
    del df_sizes
del sizes
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
    orig_legends = [ax.get_legend() for ax in fig.axes if fig.axes]
    orig_legends_handles = [
        (
            lgnd.legendHandles
            if hasattr(lgnd, "legendHandles")
            else lgnd.legend_handles if lgnd is not None else []
        )
        for lgnd in orig_legends
    ]
    orig_legends_sizes = [
        [h._sizes for h in handles] for handles in orig_legends_handles
    ]
    orig_legend_handles_labels_sizes = [
        (
            (ax.get_legend_handles_labels(), sizes)
            if ax.get_legend()
            else (([], []), sizes)
        )
        for ax, sizes in zip(fig.axes, orig_legends_sizes)
    ]
    new_suptitle = texify(orig_suptitle)
    new_titles = [texify(t) for t in orig_titles]
    new_xlabels = [texify(t) for t in orig_xlabels]
    new_ylabels = [texify(t) for t in orig_ylabels]
    new_legend_handles_labels_sizes = [
        ((handles, [(texify(label) or label) for label in labels]), sizes)
        for (handles, labels), sizes in orig_legend_handles_labels_sizes
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
                ((new_leg_handles, new_leg_labels), new_leg_sizes),
            ) in zip(
                fig.axes,
                new_titles,
                new_xlabels,
                new_ylabels,
                new_legend_handles_labels_sizes,
            ):
                if new_title is not None:
                    ax.set_title(new_title)
                if new_xlabel is not None:
                    ax.set_xlabel(new_xlabel)
                if new_ylabel is not None:
                    ax.set_ylabel(new_ylabel)
                if new_leg_labels:
                    lgnd = ax.legend(new_leg_handles, new_leg_labels)
                    lgnd_handles = (
                        lgnd.legendHandles
                        if hasattr(lgnd, "legendHandles")
                        else lgnd.legend_handles
                    )
                    for h, s in zip(lgnd_handles, new_leg_sizes):
                        h.set_sizes(s)
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
                ((orig_leg_handles, orig_leg_labels), orig_leg_sizes),
            ) in zip(
                fig.axes,
                orig_titles,
                orig_xlabels,
                orig_ylabels,
                orig_legend_handles_labels_sizes,
            ):
                if orig_title is not None:
                    ax.set_title(orig_title)
                if orig_xlabel is not None:
                    ax.set_xlabel(orig_xlabel)
                if orig_ylabel is not None:
                    ax.set_ylabel(orig_ylabel)
                if orig_leg_labels:
                    lgnd = ax.legend(orig_leg_handles, orig_leg_labels)
                    lgnd_handles = (
                        lgnd.legendHandles
                        if hasattr(lgnd, "legendHandles")
                        else lgnd.legend_handles
                    )
                    for h, s in zip(lgnd_handles, orig_leg_sizes):
                        h.set_sizes(s)


if SAVE_PLOTS:
    exts = ((".pdf",) if OUTPUT_PDF else ()) + ((".svg",) if OUTPUT_SVG else ())

    for file_path in chain(
        LATEX_FIGURE_PATH.glob("*.png"), LATEX_FIGURE_PATH.glob("*.dat")
    ):
        file_path.unlink()
        print(f"Deleted: {file_path}")
    table_row_sep = r"\\" + "\n"
    for k, fig in list(latex_figures.items()):
        if isinstance(fig, go.Figure):
            fig.update_layout(font_family="Computer Modern")  # Use LaTeX fonts
            unsupported_by_tikzplotly = any(
                isinstance(trace, go.Heatmap) for trace in fig.data
            )
            # if not unsupported_by_tikzplotly:
            #     p = LATEX_FIGURE_PATH / f"{k}.tex"
            #     print(f"Saving {p} ...")
            #     p.parent.mkdir(parents=True, exist_ok=True)
            #     tikzplotly.save(p, fig)
            with texify_title(fig, replace_with_macros=False) as fig:
                if True or unsupported_by_tikzplotly:
                    for ext in exts:
                        p = LATEX_FIGURE_PATH / f"{k}{ext}"
                        print(f"Saving {p} ...")
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.write_image(p)
                        if ext == ".pdf":
                            wrap_err(subprocess.run, ["pdfcrop", p, p], check=True)
        elif isinstance(fig, matplotlib.figure.Figure):
            p = LATEX_FIGURE_PATH / f"{k}.tex"
            p.parent.mkdir(parents=True, exist_ok=True)
            externalize_this_table = latex_externalize_tables.get(k, True)
            if externalize_this_table:
                if not latex_only_externalize_tables.get(k, False):
                    p = LATEX_FIGURE_PATH / f"{k}ExternalTables.tex"
                print(f"Saving {p} ...")
                with texify_matplotlib_title(fig) as fig:
                    tikzplotlib.save(
                        p,
                        fig,
                        externalize_tables=externalize_this_table,
                        table_row_sep=table_row_sep,
                    )
            p = LATEX_FIGURE_PATH / f"{k}.tex"
            print(f"Saving {p} ...")
            with texify_matplotlib_title(fig, replace_with_macros=True) as fig:
                tikzplotlib.save(
                    p, fig, externalize_tables=False, table_row_sep=table_row_sep
                )
            for ext in exts:
                p = LATEX_FIGURE_PATH / f"{k}{ext}"
                print(f"Saving {p} ...")
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p)
                if ext == ".pdf":
                    wrap_err(subprocess.run, ["pdfcrop", p, p], check=True)
        else:
            raise TypeError(f"Unsupported figure {fig} of type {type(fig)}")
        del latex_figures[k]
        del fig

# %%
# free up some memory
if SAVE_PLOTS:
    del latex_figures
    gc.collect()
# %%
if SAVE_PLOTS:
    optimize_pngs()

    print_and_raise_errs()
# %%
