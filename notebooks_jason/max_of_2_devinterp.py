# -*- coding: utf-8 -*-
# %% [markdown]
## Exploring Grokking in a Very Simple Max of 2 Model
#
### Introduction
#
### Setup
# %%
from functools import partial
from pathlib import Path
from tqdm import tqdm
import devinterp
from devinterp.slt import estimate_learning_coeff, estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD
from devinterp.utils import plot_trace
from devinterp.slt import sample
from devinterp.slt import OnlineLLCEstimator
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
import math
import os
import imageio
from gbmi.exp_max_of_n.plot import (
    compute_l2_norm,
    compute_QK,
    display_basic_interpretation,
)
from gbmi.utils.memoshelve import memoshelve
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    MaxOfN,
    MaxOfNDataModule,
    MaxOfNTrainingWrapper,
    train_or_load_model,
)
import gbmi.utils as utils
from gbmi.utils.hashing import get_hash_ascii
from gbmi.model import Config, RunData
from transformer_lens import HookedTransformerConfig, HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image, display
import torch
import numpy as np
import wandb
from jaxtyping import Float
from torch import Tensor
from typing import (
    Tuple,
    Dict,
    Optional,
    Any,
    List,
    Collection,
)

api = wandb.Api()
# %% [markdown]
### Introduction
#
# Consider a 1 layer attention-only transformer with no normalization trained on inputs of the form $(a, b, =)$ (for $0 \le a, b < 64$) to predict $\max(a, b)$.  Inputs are one-hot encoded.
#
# The training dataset is all sequences of the form $(a, a\pm i, =)$ for $i\in \{0, 1, 2, 17\}$.  17 is chosen to be a medium-size number coprime to 2 and 17.
#
# `=` is encoded as token $-1$
#
#### Model configuration
# %%
seq_len = 2  # training data setup code only works for sequence length 2
vocab = 64  # @param {type:"number"}
d_head = 32  # @param {type:"number"}
d_model = 32  # @param {type:"number"}
model_seed = 613947648  # @param {type:"number"}
seed = 123  # @param {type:"number"}
force_adjacent = (0, 1, 2, 17)  # @param
lr = 0.001  # @param {type:"number"}
betas = (0.9, 0.98)  # @param
momentum = None  # @param {type:"number"}
weight_decay = 1.0  # @param {type:"number"}
optimizer = "AdamW"  # @param ["AdamW", "Adam", "SGD"]
deterministic = True  # @param {type:"boolean"}
# list out the number here explicitly so that it matches with what is saved in wandb
training_ratio = 0.099609375  # @param {type:"number"}
expected_training_ratio = (
    (vocab if 0 in force_adjacent else 0)
    + 2 * sum(vocab - i for i in force_adjacent if i)
) / vocab**seq_len
if abs(training_ratio - expected_training_ratio) > 1e-5:
    f"training_ratio should probably be float.from_hex('{expected_training_ratio.hex()}') ({expected_training_ratio})"
batch_size = int(round(training_ratio * vocab**seq_len))
epochs_to_train_for = 3000  # @param {type:"number"}
include_biases = False  # @param {type:"boolean"}
cfg = Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            act_fn=None,
            attn_only=True,
            d_head=d_head,
            d_mlp=None,
            d_model=d_model,
            d_vocab=vocab + 1,
            d_vocab_out=vocab,
            default_prepend_bos=True,
            device="cpu" if deterministic else None,
            dtype=torch.float32,
            n_ctx=seq_len + 1,
            n_heads=1,
            n_layers=1,
            normalization_type=None,
            seed=model_seed,
        ),
        zero_biases=not include_biases,
        use_log1p=True,
        use_end_of_sequence=True,
        seq_len=2,
        train_dataset_cfg=FullDatasetCfg(
            force_adjacent=force_adjacent,
            training_ratio=training_ratio,
        ),
        test_dataset_cfg=FullDatasetCfg(
            force_adjacent=force_adjacent,
            training_ratio=training_ratio,
        ),
        optimizer_kwargs={
            k: v
            for k, v in dict(
                lr=lr, betas=betas, weight_decay=weight_decay, momentum=momentum
            ).items()
            if v is not None
        },
        optimizer=optimizer,
    ),
    deterministic=deterministic,
    seed=seed,
    batch_size=batch_size,
    train_for=(epochs_to_train_for, "epochs"),
    log_every_n_steps=10,
    validate_every=(10, "epochs"),
    checkpoint_every=(10, "epochs"),
)
# %% [markdown]
#### Model Training / Loading
# %%
# Load (or train) the model
force = "load"  # @param ["load", "train", "allow either"]
if force == "allow either":
    force = None
runtime, model = train_or_load_model(cfg, force=force)
# %%
datamodule = MaxOfNDataModule(cfg)
datamodule.setup("train")
training_wrapper = MaxOfNTrainingWrapper(cfg, model)
# %%
cache_dir = Path(__file__).parent / ".cache"
cache_dir.mkdir(exist_ok=True)

# %%
# load all model versions
# models = runtime.model_versions(cfg, max_count=3000, step=1)
# assert models is not None
# models = list(models)

# %% [markdown]
### Basic Interpretation
#
# The model works by attending to the largest element and copying that elment.  Let's validate this with some basic plots.
# %%
# @title display basic interpretation

# display_basic_interpretation(model)
# %%

# %%
RLCT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CHAINS = 20
NUM_DRAWS = 2000


# PRIMARY, SECONDARY, TERTIARY, QUATERNARY = sns.color_palette("muted")[:4]
# PRIMARY_LIGHT, SECONDARY_LIGHT, TERTIARY_LIGHT, QUATERNARY_LIGHT = sns.color_palette(
#     "pastel"
# )[:4]
def estimate_rlcts(
    models, train_loader, criterion, data_length, device, callbacks=None
):
    if callbacks is None:
        llc_estimator = OnlineLLCEstimator(
            1, 10, len(train_loader.dataset), device=device
        )
        callbacks = [llc_estimator]

    with memoshelve(
        estimate_learning_coeff,
        filename=cache_dir / f"{Path(__file__).name}.estimate_learning_coeff",
        get_hash=partial(get_hash_ascii, dictify_by_default=True),
    )() as memo_estimate_learning_coeff:
        estimates = {"sgld": []}
        for model in tqdm(models):
            for method, kwargs in [
                ("sgld", {"lr": 1e-5, "elasticity": 100.0}),  # 100.0
            ]:
                estimate = memo_estimate_learning_coeff(
                    model,
                    train_loader,
                    criterion=criterion,
                    optimizer_kwargs=dict(num_samples=data_length, **kwargs),
                    sampling_method=SGNHT if method == "sgnht" else SGLD,
                    num_chains=1,
                    num_draws=400,
                    num_burnin_steps=0,
                    num_steps_bw_draws=1,
                    device=device,
                    # verbose=False,
                )
                # sample(
                #     model=model,
                #     loader=train_loader,
                #     criterion=criterion,
                #     optimizer_kwargs=dict(num_samples=data_length, **kwargs),
                #     sampling_method=SGLD,
                #     num_chains=1,
                #     num_draws=10,
                #     callbacks=callbacks,
                #     device=device,
                # )

                # results = {}

                # for callback in callbacks:
                #     if hasattr(callback, "sample"):
                #         results.update(callback.sample())

                # estimates[method].append(results)
                estimates[method].append(estimate)
    return estimates


train_loader = datamodule.train_dataloader()
data_length = len(train_loader.dataset)
from devinterp.slt.norms import GradientNorm, NoiseNorm, WeightNorm

gradient_norm = GradientNorm(num_chains=1, num_draws=10, device=RLCT_DEVICE)
noise_norm = NoiseNorm(num_chains=1, num_draws=10, device=RLCT_DEVICE)
weight_norm = WeightNorm(num_chains=1, num_draws=10, device=RLCT_DEVICE)

norm_callbacks = [gradient_norm, noise_norm, weight_norm]

# for name, param in model.named_parameters():
#     if "b_" in name:
#         param.requires_grad = True
rlct = estimate_rlcts(
    [model],
    train_loader,
    training_wrapper.loss_fn,
    data_length,
    RLCT_DEVICE,
    callbacks=norm_callbacks,
)
print(rlct)
# %%
EPSILONS = [1e-5, 2e-5, 3e-5, 4e-5]  # , 1e-4, 1e-3]
GAMMAS = [100]  # [1, 10, 100]
train_data = train_loader.dataset
criterion = training_wrapper.loss_fn
DEVICE = RLCT_DEVICE
NUM_CHAINS = 10
NUM_DRAWS = 1500
import matplotlib.pyplot as plt


def estimate_llcs_sweeper(model, epsilons, gammas):
    results = {}
    with memoshelve(
        estimate_learning_coeff_with_summary,
        filename=cache_dir / f"{Path(__file__).name}.estimate_learning_coeff",
    )() as memo_estimate_learning_coeff_with_summary:
        for epsilon in epsilons:
            for gamma in gammas:
                optim_kwargs = dict(
                    lr=epsilon,
                    noise_level=1.0,
                    elasticity=gamma,
                    num_samples=len(train_data),
                    temperature="adaptive",
                )
                pair = (epsilon, gamma)
                results[pair] = estimate_learning_coeff_with_summary(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    sampling_method=SGLD,
                    optimizer_kwargs=optim_kwargs,
                    num_chains=NUM_CHAINS,
                    num_draws=NUM_DRAWS,
                    device=DEVICE,
                    online=True,
                )
    return results


def plot_single_graph(result, title=""):
    llc_color = "teal"
    fig, axs = plt.subplots(1, 1)
    # plot loss traces
    loss_traces = result["loss/trace"]
    for trace in loss_traces:
        init_loss = trace[0]
        zeroed_trace = trace - init_loss
        sgld_steps = list(range(len(trace)))
        axs.plot(sgld_steps, zeroed_trace)

    # plot llcs
    means = result["llc/means"]
    stds = result["llc/stds"]
    sgld_steps = list(range(len(means)))
    axs2 = axs.twinx()
    axs2.plot(
        sgld_steps,
        means,
        color=llc_color,
        linestyle="--",
        linewidth=2,
        label=f"llc",
        zorder=3,
    )
    axs2.fill_between(
        sgld_steps, means - stds, means + stds, color=llc_color, alpha=0.3, zorder=2
    )

    # center zero, assume zero is in the range of both y axes already
    y1_min, y1_max = axs.get_ylim()
    y2_min, y2_max = axs2.get_ylim()
    y1_zero_ratio = abs(y1_min) / (abs(y1_min) + abs(y1_max))
    y2_zero_ratio = abs(y2_min) / (abs(y2_min) + abs(y2_max))
    percent_to_add = abs(y1_zero_ratio - y2_zero_ratio)
    y1_amt_to_add = (y1_max - y1_min) * percent_to_add
    y2_amt_to_add = (y2_max - y2_min) * percent_to_add
    if y1_zero_ratio < y2_zero_ratio:
        # add to bottom of y1 and top of y2
        y1_min -= y1_amt_to_add
        y2_max += y2_amt_to_add
    elif y2_zero_ratio < y1_zero_ratio:
        # add to bottom of y2 and top of y1
        y2_min -= y2_amt_to_add
        y1_max += y1_amt_to_add
    axs.set_ylim(y1_min, y1_max)
    axs2.set_ylim(y2_min, y2_max)
    axs.set_xlabel("SGLD time step")
    axs.set_ylabel("loss")
    axs2.set_ylabel("llc", color=llc_color)
    axs2.tick_params(axis="y", labelcolor=llc_color)
    axs.axhline(color="black", linestyle=":")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_sweep_single_model(results, epsilons, gammas, **kwargs):
    llc_color = "teal"
    fig, axs = plt.subplots(len(epsilons), len(gammas))

    for i, epsilon in enumerate(epsilons):
        for j, gamma in enumerate(gammas):
            result = results[(epsilon, gamma)]
            # plot loss traces
            loss_traces = result["loss/trace"]
            for trace in loss_traces:
                init_loss = trace[0]
                zeroed_trace = trace - init_loss
                sgld_steps = list(range(len(trace)))
                axs[i, j].plot(sgld_steps, zeroed_trace)

            # plot llcs
            means = result["llc/means"]
            stds = result["llc/stds"]
            sgld_steps = list(range(len(means)))
            axs2 = axs[i, j].twinx()
            axs2.plot(
                sgld_steps,
                means,
                color=llc_color,
                linestyle="--",
                linewidth=2,
                label=f"llc",
                zorder=3,
            )
            axs2.fill_between(
                sgld_steps,
                means - stds,
                means + stds,
                color=llc_color,
                alpha=0.3,
                zorder=2,
            )

            # center zero, assume zero is in the range of both y axes already
            y1_min, y1_max = axs[i, j].get_ylim()
            y2_min, y2_max = axs2.get_ylim()
            y1_zero_ratio = abs(y1_min) / (abs(y1_min) + abs(y1_max))
            y2_zero_ratio = abs(y2_min) / (abs(y2_min) + abs(y2_max))
            percent_to_add = abs(y1_zero_ratio - y2_zero_ratio)
            y1_amt_to_add = (y1_max - y1_min) * percent_to_add
            y2_amt_to_add = (y2_max - y2_min) * percent_to_add
            if y1_zero_ratio < y2_zero_ratio:
                # add to bottom of y1 and top of y2
                y1_min -= y1_amt_to_add
                y2_max += y2_amt_to_add
            elif y2_zero_ratio < y1_zero_ratio:
                # add to bottom of y2 and top of y1
                y2_min -= y2_amt_to_add
                y1_max += y1_amt_to_add
            axs[i, j].set_ylim(y1_min, y1_max)
            axs2.set_ylim(y2_min, y2_max)

            axs[i, j].set_title(f"$\epsilon$ = {epsilon} : $\gamma$ = {gamma}")
            # only show x axis label on last row
            if i == len(epsilons) - 1:
                axs[i, j].set_xlabel("SGLD time step")
            axs[i, j].set_ylabel("loss")
            axs2.set_ylabel("llc", color=llc_color)
            axs2.tick_params(axis="y", labelcolor=llc_color)
    if kwargs["title"]:
        fig.suptitle(kwargs["title"], fontsize=16)
    plt.tight_layout()
    plt.show()


# %%
results = estimate_llcs_sweeper(model, EPSILONS, GAMMAS)
# %%
plot_sweep_single_model(
    results,
    EPSILONS,
    GAMMAS,
    title="Calibration sweep of MNIST model for lr ($\epsilon$) and elasticity ($\gamma$)",
)
# %%
for v in results.values():
    plot_single_graph(v)
# %%
import sys

sys.exit(0)
# rlct_sgd = estimate_rlcts(
#     models_saved["sgd"], train_loader, criterion, data_length, DEVICE
# )

# %%
# load all model versions
models = runtime.model_versions(cfg, max_count=3000, step=1)
assert models is not None
models = list(models)


# %% [markdown]
### Plotting the Training
#
# Let's plot this analysis, along with the loss and accuracy, across training.
# %%
# @title precompute loss and accuracy lists
def group_metrics_by_epoch(runtime: RunData) -> Dict[str, Dict[int, Any]]:
    result = {}
    max_epoch = 0
    for metric in runtime.train_metrics or []:
        epoch = metric["epoch"]
        max_epoch = max(max_epoch, epoch)
        for k, v in metric.items():
            if k not in ("epoch", "step"):
                result.setdefault(k, {})[epoch] = (
                    v.item() if isinstance(v, torch.Tensor) else v
                )
    return result


metrics = group_metrics_by_epoch(runtime)


def get_epochs_and_metric(
    metric_name: str,
    epoch: Optional[int],
    include_only_epochs: Optional[Collection[int]] = None,
    metrics: Dict[str, Dict[int, float]] = metrics,
    add_to_values: Optional[Dict[int, float]] = None,
) -> Tuple[List[int], List[Any]]:
    add_to_values = add_to_values or {}
    values = metrics[metric_name]
    epochs = [
        i
        for i in sorted(values.keys())
        if (epoch is None or i <= epoch)
        and (include_only_epochs is None or i in include_only_epochs)
    ]
    values = [
        values[i] if add_to_values.get(i) is None else values[i] + add_to_values[i]
        for i in epochs
    ]
    return epochs, values


# %%
# @title compute frames for plotting
@torch.no_grad()
def compute_traces_and_frames(
    models: List[Tuple[Any, Optional[Tuple[RunData, HookedTransformer]], Any]],
    weight_decay: float = cfg.experiment.optimizer_kwargs["weight_decay"],
    include_l2_regularization: bool = True,
):
    # Lists to hold frames and slider steps
    frames = []
    slider_steps = []
    traces = []

    # Variable to track the maximum values for each plot
    max_value_attention = 0
    max_value_losses = 0
    max_value_accuracies = 0
    all_min_value_attention = []
    all_max_value_attention = []
    all_max_value_losses = []
    all_max_value_accuracies = []
    regularizations = {}
    epochs_so_far = set()

    for i, (_version, old_data, _artifact) in enumerate(tqdm(models)):
        assert old_data is not None
        old_runtime, old_model = old_data
        epoch = old_runtime.epoch
        epochs_so_far.add(epoch)
        overlap = compute_QK(old_model)["data"]
        regularizations[epoch] = weight_decay * compute_l2_norm(old_model)  # ** 2

        # kludge with None
        regularization_epochs = list(
            sorted(k for k in regularizations.keys() if k is not None)
        )

        # Update the max_abs_value for the attention plot
        cur_min_attention = min(0, np.min(overlap))
        current_max_attention = np.max(overlap)
        max_value_attention = max(max_value_attention, current_max_attention)

        (
            training_losses_with_regularization_epochs,
            training_losses_with_regularization,
        ) = get_epochs_and_metric(
            "loss",
            epoch,
            include_only_epochs=epochs_so_far,
            add_to_values=regularizations,
        )
        (
            training_losses_without_regularization_epochs,
            training_losses_without_regularization,
        ) = get_epochs_and_metric(
            "loss",
            epoch,
            include_only_epochs=epochs_so_far,
        )
        training_accuracies_epochs, training_accuracies = get_epochs_and_metric(
            "acc",
            epoch,
            include_only_epochs=epochs_so_far,
        )
        (
            test_losses_with_regularization_epochs,
            test_losses_with_regularization,
        ) = get_epochs_and_metric(
            "periodic_test_loss",
            epoch,
            include_only_epochs=epochs_so_far,
            add_to_values=regularizations,
        )
        (
            test_losses_without_regularization_epochs,
            test_losses_without_regularization,
        ) = get_epochs_and_metric(
            "periodic_test_loss",
            epoch,
            include_only_epochs=epochs_so_far,
        )
        test_accuracies_epochs, test_accuracies = get_epochs_and_metric(
            "periodic_test_acc",
            epoch,
            include_only_epochs=epochs_so_far,
        )

        # Update the max_value for the loss and accuracy plots
        max_value_losses = max(
            max(
                training_losses_with_regularization
                if include_l2_regularization
                else training_losses_without_regularization
            ),
            max(
                test_losses_with_regularization
                if include_l2_regularization
                else test_losses_without_regularization
            ),
        )
        max_value_accuracies = max(max(training_accuracies), max(test_accuracies))

        # Update the max values for all plots
        all_min_value_attention.append(cur_min_attention)
        all_max_value_attention.append(max_value_attention)
        all_max_value_losses.append(max_value_losses)
        all_max_value_accuracies.append(max_value_accuracies)

        cur_traces = [
            # Attention plot trace
            (
                (
                    go.Scatter(
                        x=list(range(len(overlap))),
                        y=overlap,
                        mode="lines",
                        name="(E+P)<sub>-1</sub>QK<sup>T</sup>(E+P)<sup>T</sup>",
                    ),
                ),
                dict(
                    row=1,
                    col=1,
                ),
            ),
        ]
        if include_l2_regularization:
            cur_traces += [
                # Loss plot traces
                (
                    (
                        go.Scatter(
                            x=training_losses_with_regularization_epochs,
                            y=training_losses_with_regularization,
                            mode="lines",
                            name="Training Loss + L2",
                        ),
                    ),
                    dict(
                        row=2,
                        col=1,
                    ),
                ),
                (
                    (
                        go.Scatter(
                            x=test_losses_with_regularization_epochs,
                            y=test_losses_with_regularization,
                            mode="lines",
                            name="Test Loss + L2",
                        ),
                    ),
                    dict(
                        row=2,
                        col=1,
                    ),
                ),
            ]
        cur_traces += [
            (
                (
                    go.Scatter(
                        x=training_losses_without_regularization_epochs,
                        y=training_losses_without_regularization,
                        mode="lines",
                        name="Training Loss",
                    ),
                ),
                dict(
                    row=2,
                    col=1,
                ),
            ),
            (
                (
                    go.Scatter(
                        x=test_losses_without_regularization_epochs,
                        y=test_losses_without_regularization,
                        mode="lines",
                        name="Test Loss",
                    ),
                ),
                dict(
                    row=2,
                    col=1,
                ),
            ),
        ]
        if include_l2_regularization:
            cur_traces += [
                (
                    (
                        go.Scatter(
                            x=regularization_epochs,
                            y=[regularizations[e] for e in regularization_epochs],
                            mode="lines",
                            name="Regularization",
                        ),
                    ),
                    dict(
                        row=2,
                        col=1,
                    ),
                ),
            ]
        cur_traces += [
            # Accuracy plot traces
            (
                (
                    go.Scatter(
                        x=training_accuracies_epochs,
                        y=training_accuracies,
                        mode="lines",
                        name="Training Accuracy",
                    ),
                ),
                dict(
                    row=3,
                    col=1,
                ),
            ),
            (
                (
                    go.Scatter(
                        x=test_accuracies_epochs,
                        y=test_accuracies,
                        mode="lines",
                        name="Test Accuracy",
                    ),
                ),
                dict(
                    row=3,
                    col=1,
                ),
            ),
        ]

        # Add a trace for the initial plot (first data point) in all subplots
        if i == 0:
            traces = cur_traces

        # Create a frame combining all plots
        frame = go.Frame(
            data=[g for gs, _ in cur_traces for g in gs],
            name=str(epoch),
            traces=list(
                range(len(cur_traces) + 3)
            ),  # Indices of the traces in this frame
            layout=go.Layout(
                yaxis={
                    "range": [cur_min_attention, max_value_attention]
                },  # Attention plot
                yaxis2={"range": [0, max_value_losses]},  # Loss plot
                yaxis3={"range": [0, max_value_accuracies]},  # Accuracy plot
            ),
        )
        frames.append(frame)

        # Add a step to the slider
        slider_step = dict(
            method="animate",
            args=[
                [str(epoch)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            label=str(epoch),
        )
        slider_steps.append(slider_step)

    layout = dict(
        xaxis_title="Input Token",
        xaxis2_title="Epoch",
        xaxis3_title="Epoch",
        title="Model Analysis: Attention, Losses, and Accuracies Over Epochs",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 60, "easing": "linear"},
                                "mode": "immediate",
                                "repeat": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[{"steps": slider_steps, "active": len(slider_steps) - 1}],
    )

    return dict(
        frames=frames,
        traces=traces,
        layout=layout,
        all_max_value_accuracies=all_max_value_accuracies,
        all_max_value_attention=all_max_value_attention,
        all_max_value_losses=all_max_value_losses,
        all_min_value_attention=all_min_value_attention,
    )


include_l2_regularization = True  # @param {type:"boolean"}
traces_and_frames = compute_traces_and_frames(
    models, include_l2_regularization=include_l2_regularization
)
# %%
# @title plot
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        "Attention Plot",
        f"Loss{'+L2 Regularization' if include_l2_regularization else ''} Plot",
        "Accuracy Plot",
    ),
    # vertical_spacing=0.15,
)

for trace_args, trace_kwargs in traces_and_frames["traces"]:
    fig.add_trace(*trace_args, **trace_kwargs)  # type: ignore

# Add frames to the figure
fig.frames = traces_and_frames["frames"]
# Update layout for the figure
fig.update_layout(**traces_and_frames["layout"])  # type: ignore
# Adjust the height of the figure (e.g., if the original height was 600, now set it to 1200)
fig.update_layout(width=600)
fig.update_layout(height=600)
# Show the figure
fig.show()
# %%
# @title make frames for a gif
# Assuming 'fig', 'models', and the lists 'max_abs_value_attention', 'max_value_losses', 'max_value_accuracies' are defined

# Prepare a directory to save frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

all_min_value_attention = traces_and_frames["all_min_value_attention"]
all_max_value_attention = traces_and_frames["all_max_value_attention"]
all_max_value_losses = traces_and_frames["all_max_value_losses"]
all_max_value_accuracies = traces_and_frames["all_max_value_accuracies"]

filenames = []
for i, frame in enumerate(tqdm(fig.frames)):
    # Update data for each trace
    for j, data in enumerate(frame.data):
        fig.data[j].x = data.x
        fig.data[j].y = data.y

    # Update layout (axis bounds)
    fig.update_layout(
        yaxis={"range": [all_min_value_attention[i], all_max_value_attention[i]]},
        yaxis2={"range": [0, all_max_value_losses[i]]},
        yaxis3={"range": [0, all_max_value_accuracies[i]]},
    )

    # Save as image
    filename = f"{frames_dir}/frame_{i}.png"
    # if os.path.exists(filename):
    #     os.remove(filename)
    fig.write_image(filename, height=fig.layout.height, width=fig.layout.width)
    filenames.append(filename)
# %%
# @title make gif
grokking_gif = (
    f"max_of_2_grokking{'_regularized' if include_l2_regularization else ''}.gif"
)
with imageio.get_writer(grokking_gif, mode="I", duration=0.5, loop=0) as writer:
    for filename in tqdm(filenames):
        image = imageio.imread(filename)
        writer.append_data(image)  # type: ignore

with open(grokking_gif, mode="rb") as f:
    display(Image(f.read()))
# %%
# log artifact to wandb
if True:
    runtime_run = runtime.run()
    assert runtime_run is not None
    run = wandb.init(
        entity=runtime_run.entity,
        project=runtime_run.project,
        name=runtime_run.name,
        id=runtime_run.id,
        resume="must",
    )
    assert run is not None
    run.log(
        {
            f"grokking{'_regularized' if include_l2_regularization else ''}_gif": wandb.Video(
                grokking_gif, fps=2, format="gif"
            )
        }
    )
    wandb.finish()
# %% [markdown]
### Explanation
#
# In this model, the spike in test loss is so big because it's hitting three grokking phase transitions simultaneously.
#
# Since I trained this one on $(n,n)$, $(n,n±1)$, $(n,n±2)$, $(n,n±17)$, it should be simple enough that it's possible to write down some theory about how it trains and match it up to what actually happens.
#
# Prior to the grokking point, the model is slowly smoothing out the query-key overlap vector, sliding the ends of each monotonicity violation closer together.
#
# Because the test set is so small, the monotonicity violations only affect points that are within one or two numbers of them.
#
# Because SGD only captures first-order effects, a single gradient step can only move close-together numbers in tandem; it can't realize "if I just moved this large collection of points, the transitive effects would push the loss lower".  It just sees that moving the monotonicity violations to have less distance helps just a bit more than it hurts, and the loss at adjacent points prevents the gaps from shrinking too quickly.  (TODO: how do I phrase this explanation better?)
#
# Looking at it from the lens of SLT (TODO: how do I do this formally?), there's a lot of symmetry around the monotonicity violations: you can shrink the gap between out-of-order points while adjusting the points around it just a bit to avoid changing the loss much.
#
# However, because the test set includes so many more sequences, and the loss-reduction is carefully tuned to the train set, we see a sharp increase in test loss.
#
# But once the monotonicity violation is resolved, there's a sharp reversal in the generalization from the train set to the test set: now every adjustment helps the test set even more than the train set, so we see a sharp drop in loss.
#
#
# Added note: [2210.01117: Omnigrok: Grokking Beyond Algorithmic Data](https://arxiv.org/abs/2210.01117) claims that "Grokking is caused by the mismatch between training and test loss landscapes."  This demo shows that this explanation isn't a complete picture, though, because there's still a phase transition in *training accuracy* even apart from the train-test loss landscape mismatch.
# %%