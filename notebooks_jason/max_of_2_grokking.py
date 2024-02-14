# -*- coding: utf-8 -*-
# %% [markdown]
## Exploring Grokking in a Very Simple Max of 2 Model
#
### Introduction
#
### Setup
# %%
from tqdm import tqdm
import math
import os
import imageio
from gbmi.exp_max_of_n.plot import (
    compute_l2_norm,
    compute_QK,
    display_basic_interpretation,
)
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    MaxOfN,
    MaxOfNDataModule,
    train_or_load_model,
)
import gbmi.utils as utils
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
UPLOAD_TO_WANDB = False  # @param {type:"boolean"}
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
# load all model versions
models = runtime.model_versions(cfg, max_count=3000, step=1)
assert models is not None
models = list(models)

# %% [markdown]
### Basic Interpretation
#
# The model works by attending to the largest element and copying that elment.  Let's validate this with some basic plots.
# %%
# @title display basic interpretation

display_basic_interpretation(model)


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


# %%
fig_html = {}
traces_and_frames = {}
grokking_fig = {}
# %%
for include_l2_regularization in [True, False]:
    traces_and_frames[include_l2_regularization] = compute_traces_and_frames(
        models, include_l2_regularization=include_l2_regularization
    )
# %%
# @title plot
for include_l2_regularization in [True, False]:
    grokking_fig[include_l2_regularization] = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Attention Plot",
            f"Loss{'+L2 Regularization' if include_l2_regularization else ''} Plot",
            "Accuracy Plot",
        ),
        # vertical_spacing=0.15,
    )

    for trace_args, trace_kwargs in traces_and_frames[include_l2_regularization][
        "traces"
    ]:
        grokking_fig[include_l2_regularization].add_trace(*trace_args, **trace_kwargs)  # type: ignore

    # Add frames to the figure
    grokking_fig[include_l2_regularization].frames = traces_and_frames[
        include_l2_regularization
    ]["frames"]
    # Update layout for the figure
    grokking_fig[include_l2_regularization].update_layout(**traces_and_frames[include_l2_regularization]["layout"])  # type: ignore
    # Adjust the height of the figure (e.g., if the original height was 600, now set it to 1200)
    grokking_fig[include_l2_regularization].update_layout(width=600)
    grokking_fig[include_l2_regularization].update_layout(height=600)

# Show the figure
for include_l2_regularization in [True, False]:
    grokking_fig[include_l2_regularization].show()
# %%
# Save to html
for include_l2_regularization in [True, False]:
    fig_html[include_l2_regularization] = (
        f"max_of_2_grokking{'_regularized' if include_l2_regularization else ''}.html"
    )
    grokking_fig[include_l2_regularization].write_html(
        fig_html[include_l2_regularization], auto_play=False
    )
# %%
# log artifact to wandb
if UPLOAD_TO_WANDB:
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
    for k, v in fig_html.items():
        run.log({f"grokking{'_regularized' if k else ''}": wandb.Html(v)})
    wandb.finish()

# %%
# @title make frames for a gif
include_l2_regularization: bool = False  # @param {type:"boolean"}
# Assuming 'fig', 'models', and the lists 'max_abs_value_attention', 'max_value_losses', 'max_value_accuracies' are defined

# Prepare a directory to save frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

all_min_value_attention = traces_and_frames[include_l2_regularization][
    "all_min_value_attention"
]
all_max_value_attention = traces_and_frames[include_l2_regularization][
    "all_max_value_attention"
]
all_max_value_losses = traces_and_frames[include_l2_regularization][
    "all_max_value_losses"
]
all_max_value_accuracies = traces_and_frames[include_l2_regularization][
    "all_max_value_accuracies"
]

filenames = []
for i, frame in enumerate(tqdm(grokking_fig[include_l2_regularization].frames)):
    # Update data for each trace
    for j, data in enumerate(frame.data):
        grokking_fig[include_l2_regularization].data[j].x = data.x
        grokking_fig[include_l2_regularization].data[j].y = data.y

    # Update layout (axis bounds)
    grokking_fig[include_l2_regularization].update_layout(
        yaxis={"range": [all_min_value_attention[i], all_max_value_attention[i]]},
        yaxis2={"range": [0, all_max_value_losses[i]]},
        yaxis3={"range": [0, all_max_value_accuracies[i]]},
    )

    # Save as image
    filename = f"{frames_dir}/frame_{i}.png"
    # if os.path.exists(filename):
    #     os.remove(filename)
    grokking_fig[include_l2_regularization].write_image(
        filename,
        height=grokking_fig[include_l2_regularization].layout.height,
        width=grokking_fig[include_l2_regularization].layout.width,
    )
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
if UPLOAD_TO_WANDB:
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
# %% [markdown]
# # Essential Dynamics
# %%
# @title make essential dynamics code
@torch.no_grad()
def make_essential_dynamics(
    cfg: Config[MaxOfN],
    models: List[Tuple[Any, Optional[Tuple[RunData, HookedTransformer]], Any]],
):
    datamodule = MaxOfNDataModule(cfg)
    datamodule.setup("train")
    essential_dynamics = [
        torch.cat(
            tuple(
                old_model(
                    torch.cat([d for d, _ in datamodule.test_dataloader()], dim=0)
                )[:, -1, :]
            ),
            dim=0,
        )
        for _, (_, old_model), _ in tqdm(models)
    ]

    essential_dynamics_matrix = torch.stack(essential_dynamics, dim=0)

    essential_dynamics_matrix_centered = essential_dynamics_matrix[:, : 64 * 64 * 10]
    essential_dynamics_matrix_centered -= essential_dynamics_matrix_centered.mean(
        dim=-1, keepdim=True
    )
    U, S, Vh = torch.linalg.svd(essential_dynamics_matrix_centered)

    return essential_dynamics_matrix, essential_dynamics_matrix_centered, (U, S, Vh)


def plot3d_essential_dynamics(
    U: Tensor,
    x_label: str = "pc 0",
    y_label: str = "pc 1",
    z_label: str = "pc 2",
    index_label: str = "epoch",
    indices=None,
    epoch_scale: float = 1,
    title: str = "3D Essential Dynamics",
):
    indices = np.arange(U.shape[0]) * epoch_scale if indices is None else indices
    hover_template = f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{z_label}: %{{z}}<br>{index_label}: %{{marker.color}}"
    fig = px.scatter_3d(
        x=U[:, 0],
        y=U[:, 1],
        z=U[:, 2],
        color=indices,
        labels={"x": x_label, "y": y_label, "z": z_label, "color": index_label},
        title=title,
    )
    fig.update_traces(hovertemplate=hover_template)

    fig.show()
    return fig


def plot_essential_dynamics(
    U: Tensor,
    n: int = 6,
    height: int = 2000,
    width: int = 2000,
    indices=None,
    epoch_scale: float = 1,
):
    indices = np.arange(U.shape[0]) * epoch_scale if indices is None else indices
    fig = make_subplots(
        rows=n,
        cols=n,
        subplot_titles=[f"{i} vs {j}" for i in range(n) for j in range(n)],
    )
    for row in range(n):
        for col in range(n):
            j, i = row, col
            fig.add_trace(
                go.Scatter(
                    x=U[:, i],
                    y=U[:, j],
                    mode="markers",
                    marker=dict(color=indices),
                    name=f"{j} vs {i}",
                    hovertemplate=f"pc{j}: %{{y}}<br>pc{i}: %{{y}}<br>epoch: %{{marker.color}}",
                ),
                row=row + 1,
                col=col + 1,
            )
    fig.update_layout(
        height=height,
        width=width,
        title_text="Essential Dynamics Scatter Plots Grid, Principle Components",
        showlegend=False,
    )
    fig.show()
    return fig


# %%
essential_dynamics_matrix, essential_dynamics_matrix_centered, (U, S, Vh) = (
    make_essential_dynamics(cfg, models)
)
# %%
fig3d012 = plot3d_essential_dynamics(
    U, epoch_scale=(cfg.checkpoint_every[0] if cfg.checkpoint_every is not None else 1)
)
# %%
# log artifact to wandb
if UPLOAD_TO_WANDB:
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
    run.log({f"3d_essential_dynamics_0_1_2": fig3d012})
    wandb.finish()

# %%
fig6 = plot_essential_dynamics(
    U, epoch_scale=(cfg.checkpoint_every[0] if cfg.checkpoint_every is not None else 1)
)
# %%
# log artifact to wandb
if UPLOAD_TO_WANDB:
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
    run.log({f"essential_dynamics": fig6})
    wandb.finish()


# %%
def osculating_circle(curve, t_index):
    # Handle edge cases
    if t_index == 0:
        t_index = 1
    if t_index == len(curve) - 1:
        t_index = len(curve) - 2

    # Central differences for first and second derivatives
    r_prime = (curve[t_index + 1] - curve[t_index - 1]) / 2
    r_double_prime = curve[t_index + 1] - 2 * curve[t_index] + curve[t_index - 1]

    # Append a zero for 3D cross product
    r_prime_3d = np.append(r_prime, [0])
    r_double_prime_3d = np.append(r_double_prime, [0])

    # Curvature calculation and normal vector direction
    cross_product = np.cross(r_prime_3d, r_double_prime_3d)
    curvature = np.linalg.norm(cross_product) / np.linalg.norm(r_prime) ** 3
    signed_curvature = np.sign(cross_product[2])  # Sign of z-component of cross product
    radius_of_curvature = 1 / (curvature + 1e-12)

    # Unit tangent vector
    tangent = r_prime / np.linalg.norm(r_prime)

    # Unit normal vector, direction depends on the sign of the curvature
    if signed_curvature >= 0:
        norm_perp = np.array(
            [-tangent[1], tangent[0]]
        )  # Rotate tangent by 90 degrees counter-clockwise
    else:
        norm_perp = np.array(
            [tangent[1], -tangent[0]]
        )  # Rotate tangent by 90 degrees clockwise

    # Center of the osculating circle
    center = curve[t_index] + radius_of_curvature * norm_perp

    return center, radius_of_curvature


# %%
def get_nearest_step(steps, step):
    idx = np.argmin(np.abs(np.array(steps) - step))
    return steps[idx]


# %%
def plot_essential_dynamics_grid(
    samples,
    steps,
    transitions=None,
    colors=None,
    num_pca_components=3,
    plot_caustic=True,
    figsize=(20, 6),
    marked_cusp_data=None,
    use_cache=False,
    num_sharp_points=20,
    num_vertices=35,
    osculate_start=1,
    osculate_end_offset=0,
    osculate_skip=8,
    smoothing_early=10,
    smoothing_late=60,
    smoothing_boundary=200,
    show_vertex_influence=False,
    use_3D=False,
):

    CUTOFF_START = osculate_start
    CUTOFF_END = osculate_end_offset
    OSCULATE_SKIP = osculate_skip
    sigma_early = smoothing_early
    sigma_late = smoothing_late
    LATE_BOUNDARY = smoothing_boundary
    START_LERP = 0.2 * LATE_BOUNDARY
    END_LERP = LATE_BOUNDARY

    palette = "tab10"
    colors = colors or sns.color_palette(palette)

    num_pca_combos = (num_pca_components * (num_pca_components - 1)) // 2

    if not use_3D:
        fig, axes = plt.subplots(1, num_pca_combos, figsize=figsize)
        if num_pca_combos == 1:
            axes = [axes]
    else:
        # In 3D we show only one PC pair on the left hand side
        figsize = (16, 7)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        axes = [ax1, ax2]
        ax_3D = fig.add_subplot(gs[0, 2], projection="3d")
        elevation = 45  # Angle above the x-y plane
        azimuth = 45  # Angle around the z-axis
        ax_3D.view_init(elev=elevation, azim=azimuth)

    labels = [""]

    print("Number of samples: " + str(len(samples)))

    #
    # 2D plots
    #

    I = 0

    # Make sure we have the smoothed data for each PC
    smoothed_pcs = []

    for i in range(0, num_pca_components):
        print(f"Processing smoothing for PC{i+1}")
        file_path_smoothings = f"smoothed_pc_i{i}.pkl"
        if use_cache and os.path.exists(file_path_smoothings):
            print("    Using cached smoothing data")
            with open(file_path_smoothings, "rb") as file:
                smoothed_pc = pickle.load(file)
        else:
            smoothed_pc = np.copy(samples[:, :1])
            for z in range(len(samples)):
                if z < START_LERP:
                    sigma = sigma_early
                elif z > END_LERP:
                    sigma = sigma_late
                else:
                    sigma = (sigma_late - sigma_early) / (END_LERP - START_LERP) * (
                        z - START_LERP
                    ) + sigma_early

                smoothed_pc[z, 0] = scipy.ndimage.gaussian_filter1d(
                    samples[:, i], sigma
                )[z]

            with open(file_path_smoothings, "wb") as file:
                pickle.dump(smoothed_pc, file)

        smoothed_pcs.append(smoothed_pc)

    # Make sure we have the osculating data for each pair of PCs

    for i in range(1, num_pca_components):
        for j in range(i):
            # For each PC pair we first do some data processing, then plotting
            print(f"Processing PC{j+1} vs PC{i+1}")

            smoothed_pc_i = smoothed_pcs[i]
            smoothed_pc_j = smoothed_pcs[j]
            smoothed_samples = np.column_stack((smoothed_pc_i, smoothed_pc_j))

            file_path_osculating = f"osculating_data_i{i}_j{j}.pkl"
            if use_cache and os.path.exists(file_path_osculating):
                print("    Using cached osculate data")
                with open(file_path_osculating, "rb") as file:
                    osculating_data = pickle.load(file)
            else:
                osculating_data = {}

                for z in range(CUTOFF_START, len(samples) - CUTOFF_END, 1):
                    osculating_data[z] = osculating_circle(smoothed_samples, z)

                with open(file_path_osculating, "wb") as file:
                    pickle.dump(osculating_data, file)

            # The 3D plotting needs osculating data for the marked cusps
            if use_3D:
                for t in range(len(marked_cusp_data)):
                    cusp_index = marked_cusp_data[t]["step"]
                    if not "osculating_data" in marked_cusp_data[t].keys():
                        marked_cusp_data[t]["osculating_data"] = []
                    marked_cusp_data[t]["osculating_data"].append(
                        osculating_data[cusp_index]
                    )

            radii_list = []
            dcenter_list = []
            dcenter = {}
            radii = {}
            prev_center = None

            for z in range(CUTOFF_START, len(samples) - CUTOFF_END, OSCULATE_SKIP):
                center, radius = osculating_data[z]
                radii[z] = radius
                radii_list.append(radius)
                dcenter[z] = 1000

                if prev_center is not None:
                    d = np.linalg.norm(center - prev_center)
                    dcenter[z] = d
                    dcenter_list.append(d)

                color = "lightgray"
                circle = plt.Circle(
                    (center[0].item(), center[1].item()),
                    radius.item(),
                    alpha=0.5,
                    color=color,
                    lw=0.5,
                    fill=False,
                )

                if I < len(axes):
                    axes[I].add_artist(circle)

                prev_center = center

            # Find high curvature points
            radii_list.sort()
            upper_bound = 0
            if len(radii_list) > num_sharp_points:
                upper_bound = radii_list[num_sharp_points]

            # Find caustic cusps
            dcenter_list.sort()
            dcenter_bound = 0
            if len(dcenter_list) > num_vertices:
                dcenter_bound = dcenter_list[num_vertices]

            # Plot un-smoothed points in the background
            # if I < len(axes):
            #    axes[I].scatter(x=samples[:, i], y=samples[:, j], alpha=0.8, color="lightgray", s=10)

            if I < len(axes):
                if transitions:
                    for k, (start, end, stage) in enumerate(transitions):
                        start_idx = steps.index(get_nearest_step(steps, start))
                        end_idx = steps.index(get_nearest_step(steps, end)) + 1

                        axes[I].plot(
                            smoothed_samples[start_idx:end_idx, 0],
                            smoothed_samples[start_idx:end_idx, 1],
                            color=colors[k],
                            lw=2,
                        )
                else:
                    axes[I].plot(samples[:, i], samples[:, j])

            for z in range(CUTOFF_START, len(samples) - CUTOFF_END, OSCULATE_SKIP):
                if (
                    z > 2 * OSCULATE_SKIP
                    and dcenter[z] < dcenter_bound
                    and dcenter[z - OSCULATE_SKIP] < dcenter_bound
                ):
                    print(
                        "    Vertex ["
                        + str(z)
                        + "] rate of change of osculate center "
                        + str(dcenter[z])
                    )
                    if I < len(axes):
                        axes[I].scatter(
                            smoothed_samples[z, 0], smoothed_samples[z, 1], color="gold"
                        )

            # Mark in red high curvature points
            for z in range(CUTOFF_START, len(samples) - CUTOFF_END, OSCULATE_SKIP):
                if radii[z] < upper_bound:
                    print("    Sharp point [" + str(z) + "] curvature " + str(radii[z]))
                    if I < len(axes):
                        axes[I].scatter(
                            smoothed_samples[z, 0], smoothed_samples[z, 1], color="red"
                        )

            if I >= len(axes):
                continue

            current_x_limits = axes[I].get_xlim()
            current_y_limits = axes[I].get_ylim()

            # Draw the evolute
            if plot_caustic:
                for z in range(CUTOFF_START, len(samples) - CUTOFF_END, 1):
                    center, radius = osculating_data[z]

                    if (
                        center[0].item() > current_x_limits[0]
                        and center[0].item() < current_x_limits[1]
                    ):
                        if (
                            center[1].item() > current_y_limits[0]
                            and center[1].item() < current_y_limits[1]
                        ):
                            axes[I].scatter(
                                [center[0].item()],
                                [center[1].item()],
                                color="black",
                                s=0.2,
                            )

            if marked_cusp_data:
                for marked_cusp_id in range(len(marked_cusp_data)):
                    marked_cusp = marked_cusp_data[marked_cusp_id]["step"]
                    axes[I].scatter(
                        smoothed_samples[marked_cusp, 0],
                        smoothed_samples[marked_cusp, 1],
                        color="green",
                        marker="x",
                        s=40,
                    )
                    center, radius = osculating_data[marked_cusp]
                    axes[I].scatter(
                        [center[0].item()],
                        [center[1].item()],
                        color="green",
                        marker="x",
                        s=40,
                    )

                    if show_vertex_influence:
                        vertex_influence_start = marked_cusp_data[marked_cusp_id][
                            "influence_start"
                        ]
                        vertex_influence_end = marked_cusp_data[marked_cusp_id][
                            "influence_end"
                        ]
                        axes[I].scatter(
                            smoothed_samples[vertex_influence_start, 0],
                            smoothed_samples[vertex_influence_start, 1],
                            color="blue",
                            marker="x",
                            s=40,
                        )
                        axes[I].scatter(
                            smoothed_samples[vertex_influence_end, 0],
                            smoothed_samples[vertex_influence_end, 1],
                            color="blue",
                            marker="x",
                            s=40,
                        )

            axes[I].set_xlabel(f"PC {i+1}")
            axes[I].set_ylabel(f"PC {j+1}")

            I += 1

    #
    # 3D plots
    #

    if use_3D:
        # ax_3D.scatter(samples[:, 0], samples[:, 1], samples[:,2], color="lightgray", s=10)

        smoothed_samples = np.copy(samples[:, :3])
        for z in range(len(samples)):
            if z < START_LERP:
                sigma = sigma_early
            elif z > END_LERP:
                sigma = sigma_late
            else:
                sigma = (sigma_late - sigma_early) / (END_LERP - START_LERP) * (
                    z - START_LERP
                ) + sigma_early

            smoothed_samples = np.column_stack(
                (smoothed_pcs[0], smoothed_pcs[1], smoothed_pcs[2])
            )

        if transitions:
            for k, (start, end, stage) in enumerate(transitions):
                start_idx = steps.index(get_nearest_step(steps, start))
                end_idx = steps.index(get_nearest_step(steps, end)) + 1

                ax_3D.plot(
                    smoothed_samples[start_idx:end_idx, 0],
                    smoothed_samples[start_idx:end_idx, 1],
                    smoothed_samples[start_idx:end_idx, 2],
                    color=colors[k],
                )

        for t in range(len(marked_cusp_data)):
            marked_cusp = marked_cusp_data[t]
            cusp_index = marked_cusp["step"]
            osculating_data = marked_cusp["osculating_data"]

            center_pc2_pc1, radius_pc2_pc1 = osculating_data[0]
            center_pc3_pc1, radius_pc3_pc1 = osculating_data[1]

            coeff_pc1 = center_pc2_pc1[1]
            coeff_pc2 = center_pc2_pc1[0]
            coeff_pc3 = center_pc3_pc1[0]

            print(f"Placing sphere at ({coeff_pc1, coeff_pc2, coeff_pc3})")

            print(f"Comparison of radii {radius_pc2_pc1} vs {radius_pc3_pc1}")
            radius = np.max([radius_pc2_pc1, radius_pc3_pc1])

            # Generate points for a sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = (
                radius * np.outer(np.cos(u), np.sin(v)) + coeff_pc1
            )  # x coordinates of the sphere
            y = (
                radius * np.outer(np.sin(u), np.sin(v)) + coeff_pc2
            )  # y coordinates of the sphere
            z = (
                radius * np.outer(np.ones(np.size(u)), np.cos(v)) + coeff_pc3
            )  # z coordinates of the sphere

            # Plot the sphere
            ax_3D.plot_surface(
                x, y, z, color="r", alpha=0.3, linewidth=0
            )  # 'alpha' controls the transparency
            ax_3D.set_xlabel("PC 1")
            ax_3D.set_ylabel("PC 2")
            ax_3D.set_zlabel("PC 3")

    axes[0].set_ylabel(f"{labels[0]}\n\nPC 1")

    plt.tight_layout(rect=[0, 0, 1, 1])

    if transitions:
        legend_ax = fig.add_axes(
            [0.1, -0.03, 0.95, 0.05]
        )  # Adjust these values as needed
        handles = [
            plt.Line2D([0], [0], color=colors[i], linestyle="-")
            for i in range(len(transitions))
        ]
        labels = [label for _, _, label in transitions]
        legend_ax.legend(handles, labels, loc="center", ncol=len(labels), frameon=False)
        legend_ax.axis("off")

    fig.set_facecolor("white")

    return fig


# %%
import seaborn as sns
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.gridspec as gridspec

import pickle
import scipy.ndimage

# from sklearn.decomposition import PCA

plot_essential_dynamics_grid(
    U,
    torch.arange(len(U)) * 10,
    smoothing_early=4,
    smoothing_late=4,
    smoothing_boundary=4,
    num_vertices=7,
    num_sharp_points=1,
)
