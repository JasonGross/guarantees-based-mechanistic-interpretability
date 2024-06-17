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
import sys
import re
import time
import subprocess
from itertools import chain
from functools import reduce, partial, cache
from concurrent.futures import ThreadPoolExecutor
import math
from scipy import stats
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.figure
import tikzplotlib
import matplotlib
from typing import (
    Literal,
    Optional,
    Tuple,
    Union,
    Any,
    Iterator,
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from gbmi.exp_max_of_n.plot import (
    scatter_attention_difference_vs_gap,
    hist_attention_difference_over_gap,
    hist_EVOU_max_minus_diag_logit_diff,
    make_better_slides_plots_00,
    display_EQKE_SVD_analysis,
)
from gbmi.analysis_tools.plot import (
    hist_EVOU_max_logit_diff,
    weighted_histogram,
    Colorscale,
    colorscale_to_cmap,
    cmap_to_list,
    imshow,
    line,
    remove_titles,
)
from gbmi.analysis_tools.decomp import analyze_svd, split_svd_contributions
from gbmi.analysis_tools.utils import pm_round, pm_mean_std, data_summary
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
from gbmi.utils.instructions import (
    InstructionCount,
    CountTensor,
    PatchTorch,
    CountHookedTransformer,
    PerfCounter,
    PerfCollector,
    CountTensorOperations,
    PERF_WORKING,
)


try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False


# %%
DISPLAY_PLOTS: bool = True  # @param {type:"boolean"}
RENDERER: Optional[str] = "png"  # @param ["png", None]
PLOT_WITH: Literal["plotly", "matplotlib"] = (  # @param ["plotly", "matplotlib"]
    "matplotlib"
)
cache_dir = Path(__file__.replace("_colorschemes", "")).parent / ".cache"
cache_dir.mkdir(exist_ok=True)
compute_expensive_average_across_many_models: bool = False  # @param {type:"boolean"}
LATEX_FIGURE_PATH = (
    Path(__file__.replace("_colorschemes", "")).with_suffix("") / "figures"
)
LATEX_FIGURE_PATH.mkdir(exist_ok=True, parents=True)
LATEX_VALUES_PATH = (
    Path(__file__.replace("_colorschemes", "")).with_suffix("") / "values.tex"
)
LATEX_VALUES_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_TIKZPLOTLIB_PREAMBLE_PATH = (
    Path(__file__.replace("_colorschemes", "")).with_suffix("")
    / "tikzplotlib-preamble.tex"
)
LATEX_TIKZPLOTLIB_PREAMBLE_PATH.parent.mkdir(exist_ok=True, parents=True)
LATEX_GIT_DIFF_PATH = (
    Path(__file__.replace("_colorschemes", "")).with_suffix("") / "git-diff-info.diff"
)
LATEX_GIT_DIFF_PATH.parent.mkdir(exist_ok=True, parents=True)
ALL_MODELS_PATH = (
    Path(__file__.replace("_colorschemes", ""))
    .with_suffix("")
    .with_name(
        f"{Path(__file__.replace('_colorschemes', '')).with_suffix('').name}_all_models"
    )
)
TRAIN_CSV_PATH = ALL_MODELS_PATH / "all-models-train-values.csv"
TRAIN_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
BRUTE_FORCE_CSV_PATH = ALL_MODELS_PATH / "all-models-brute-force-values.csv"
BRUTE_FORCE_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
CUBIC_CSV_PATH = ALL_MODELS_PATH / "all-models-cubic-values.csv"
CUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_CSV_PATH = ALL_MODELS_PATH / "all-models-subcubic-values.csv"
SUBCUBIC_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
SUBCUBIC_ANALYSIS_CSV_PATH = ALL_MODELS_PATH / "all-models-subcubic-analysis-values.csv"
SUBCUBIC_ANALYSIS_CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
N_THREADS: Optional[int] = 2
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
latex_values: dict[str, Union[int, float, str]] = {}
latex_figures: dict[str, Union[go.Figure, matplotlib.figure.Figure]] = {}
latex_externalize_tables: dict[str, bool] = {}
latex_only_externalize_tables: dict[str, bool] = {}


# %%
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
    seed: f"{seed}_{cfg_hashes[seed].replace('/', '__SLASH__')}"
    for seed, cfg in cfgs.items()
}
datamodules = {seed: MaxOfNDataModule(cfg) for seed, cfg in cfgs.items()}
# %%
with memoshelve(
    train_or_load_model,
    filename=cache_dir
    / f"{Path(__file__.replace('_colorschemes', '')).name}.train_or_load_model",
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
# %%
seed = 123
cfg = cfgs[seed]
cfg_hash = cfg_hashes[seed]
cfg_hash_for_filename = cfg_hashes_for_filename[seed]
runtime, model = runtime_models[seed]
training_wrapper = training_wrappers[seed]
latex_values["seed"] = seed
assert cfg.experiment.model_config.seed is not None

# %% [markdown]


from typing import TypeVar

T = TypeVar("T")


def shift_cyclical_colorscale(
    colors: list[Tuple[float, T]], shift: int = 0
) -> list[Tuple[float, T]]:
    pos = [c[0] for c in colors]
    colors = [c[1] for c in colors]
    mid = len(colors) // 2
    return list(zip(pos, colors[mid + shift :] + colors[: mid + shift]))


# %%

from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import to_hex


def get_gradient(hex_start, hex_end):
    n_steps = 10

    rbg_start = tuple(int(hex_start[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    rgb_end = tuple(int(hex_end[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

    hsv_start = rgb_to_hsv(np.array([rbg_start]))[0]
    hsv_end = rgb_to_hsv(np.array([rgb_end]))[0]

    # Interpolate HSV values
    hues = np.linspace(hsv_start[0], hsv_end[0], n_steps)
    saturations = np.linspace(hsv_start[1], hsv_end[1], n_steps)
    values = np.linspace(hsv_start[2], hsv_end[2], n_steps)

    # Combine interpolated HSV values
    hsv_gradient = np.column_stack((hues, saturations, values))

    # Convert HSV to RGB
    rgb_gradient = hsv_to_rgb(hsv_gradient)

    return rgb_gradient


def get_color_mapping(colors_1, colors_2):

    array_list_1 = []
    array_list_2 = []

    for i in range(4):
        new_array = get_gradient(colors_1[i], colors_1[i + 1])
        array_list_1.append(new_array)

    for i in range(4):
        new_array = get_gradient(colors_2[i], colors_2[i + 1])
        array_list_2.append(new_array)

    array_1 = np.concatenate(array_list_1, axis=0)
    array_2 = np.concatenate(array_list_2, axis=0)

    hex_1 = [to_hex(color) for color in array_1][::-1]
    hex_2 = [to_hex(color) for color in array_2]

    mid = 0.485
    all_hex = (
        list(zip(np.linspace(0, mid, len(hex_1)), hex_1))
        + [(0.5, "#ffffff")]
        + list(zip(np.linspace(1 - mid, 1, len(hex_2)), hex_2))
    )

    return all_hex
    # values = np.linspace(0, 1, len(all_hex))
    # color_mapping = list(zip(values, all_hex))

    # return color_mapping


# %%


oranges = ["#fefec7", "#f29f05", "#f25c05", "#a62f03", "#400d01"]
blues = ["#e6f3ff", "#5e87f5", "#3d4b91", "#2d2c5e", "#1d0e2c"]
teals = ["#d1e8e8", "#9AD4DE", "#58B8C9", "#10656d", "#0c3547"]

color_mapping = get_color_mapping(oranges, blues)
# %%


default_OV_colorscale: Colorscale = color_mapping
default_QK_colorscale: Colorscale = color_mapping
default_QK_SVD_colorscale: Colorscale = default_QK_colorscale

# %%

figs, axis_limits = display_basic_interpretation(
    model,
    include_uncentered=True,
    OV_colorscale=default_OV_colorscale,
    QK_colorscale=default_QK_colorscale,
    QK_SVD_colorscale=default_QK_SVD_colorscale,
    tok_dtick=10,
    plot_with=PLOT_WITH,
    renderer=RENDERER,
    show=False,
    # **axis_limits,
)
# figs, axis_limits = display_basic_interpretation(
#     model,
#     include_uncentered=True,
#     OV_colorscale=default_OV_colorscale,
#     QK_colorscale=default_QK_colorscale,
#     QK_SVD_colorscale=default_QK_SVD_colorscale,
#     tok_dtick=10,
#     plot_with=PLOT_WITH,
#     renderer=RENDERER,
#     show=False,
#     **axis_limits,
# )
PVOU_keys = [k for k in figs.keys() if k.startswith("irrelevant_") and "V" in k]
EUPU_keys = [
    k for k in figs.keys() if k.startswith("irrelevant_") and k != PVOU_keys[0]
]
# %%
for key in ("EQKE", "EQKP", "EVOU", PVOU_keys[0], EUPU_keys[0], "EQKE Attention SVD"):
    plt.figure(figs[key])
    plt.show()
