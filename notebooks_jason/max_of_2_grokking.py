# %%
from tqdm import tqdm
import os
import imageio
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    MaxOfN,
    train_or_load_model,
)
from gbmi.model import Config, RunData
from transformer_lens import HookedTransformerConfig, HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import torch
import wandb
from jaxtyping import Float
from torch import Tensor
from IPython.display import display, Markdown, Image
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)


api = wandb.Api()

# %%
# config, kwargs = config_of_argv("gbmi.exp_max_of_n.train --max-of 2 --deterministic --train-for-epochs 1500 --validate-every-epochs 1 --force-adjacent-gap 0,1,2 --use-log1p --training-ratio 0.095 --weight-decay 1.0 --betas 0.9 0.98 --optimizer AdamW --use-end-of-sequence --checkpoint-every-epochs 1 --batch-size 389 --force train".split(" "))
# config, kwargs = config_of_argv("gbmi.exp_max_of_n.train --max-of 2 --deterministic --train-for-epochs 3000 --validate-every-epochs 10 --force-adjacent-gap 0,1,2,17 --use-log1p --training-ratio 0.099609375 --weight-decay 1.0 --betas 0.9 0.98 --optimizer AdamW --use-end-of-sequence --batch-size 408 --force train --checkpoint-every-epochs 10".split(" "))
# %%
# print(config)
# %%
seq_len = 2
vocab = 64
cfg = Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            act_fn=None,
            attn_only=True,
            d_head=32,
            d_mlp=None,
            d_model=32,
            d_vocab=vocab + 1,
            d_vocab_out=vocab,
            default_prepend_bos=True,
            device="cpu",
            dtype=torch.float32,
            n_ctx=seq_len + 1,
            n_heads=1,
            n_layers=1,
            normalization_type=None,
            seed=613947648,
        ),
        zero_biases=True,
        use_log1p=True,
        use_end_of_sequence=True,
        seq_len=2,
        train_dataset_cfg=FullDatasetCfg(
            force_adjacent=(0, 1, 2), training_ratio=0.095
        ),
        test_dataset_cfg=FullDatasetCfg(force_adjacent=(0, 1, 2), training_ratio=0.095),
        optimizer_kwargs=dict(lr=0.001, betas=(0.9, 0.98), weight_decay=1.0),
        optimizer="AdamW",
    ),
    deterministic=True,
    seed=123,
    batch_size=389,
    train_for=(1500, "epochs"),
    log_every_n_steps=10,
    validate_every=(10, "epochs"),
    checkpoint_every=(10, "epochs"),
)
cfg = Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            act_fn=None,
            attn_only=True,
            d_head=32,
            d_mlp=None,
            d_model=32,
            d_vocab=vocab + 1,
            d_vocab_out=vocab,
            default_prepend_bos=True,
            device="cpu",
            dtype=torch.float32,
            n_ctx=seq_len + 1,
            n_heads=1,
            n_layers=1,
            normalization_type=None,
            seed=613947648,
        ),
        zero_biases=True,
        use_log1p=True,
        use_end_of_sequence=True,
        seq_len=2,
        train_dataset_cfg=FullDatasetCfg(
            force_adjacent=(0, 1, 2, 17),
            training_ratio=0.099609375,
        ),
        test_dataset_cfg=FullDatasetCfg(
            force_adjacent=(0, 1, 2, 17), training_ratio=0.099609375
        ),
        optimizer_kwargs=dict(lr=0.001, betas=(0.9, 0.98), weight_decay=1.0),
        optimizer="AdamW",
    ),
    deterministic=True,
    seed=123,
    batch_size=408,
    train_for=(3000, "epochs"),
    log_every_n_steps=10,
    validate_every=(10, "epochs"),
    checkpoint_every=(10, "epochs"),
)

# %%
runtime, model = train_or_load_model(cfg, force="load")
# %%
# ExpWrapper = cfg.experiment.get_training_wrapper()
# # wrapped_model = ExpWrapper(config, ExpWrapper.build_model(config))
# datamodule = cfg.experiment.get_datamodule()(cfg)
# datamodule.setup("")
# training_batches = torch.cat([batch for batch in datamodule.train_dataloader()], dim=0)
# test_batches = torch.cat([batch for batch in datamodule.test_dataloader()], dim=0)


# %%
# def evaluate_model(
#     model: HookedTransformer,
# ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
#     with torch.no_grad():
#         wrapped_model = ExpWrapper(cfg, model)
#         training_loss, training_acc = wrapped_model.run_batch(
#             training_batches, prefix="", log_output=False, return_accuracy=True
#         )
#         test_loss, test_acc = wrapped_model.run_batch(
#             test_batches, prefix="", log_output=False, return_accuracy=True
#         )
#     return (training_loss.item(), training_acc), (test_loss.item(), test_acc)


# %%
# @title display basic interpretation
# @title interpretation functions
@torch.no_grad()
def compute_QK(model: HookedTransformer = model) -> dict:
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )
    QK = (
        (W_E[-1] + W_pos[-1])
        @ W_Q[0, 0]
        @ W_K[0, 0].T
        @ (W_E[:-1] + W_pos[:-1].mean(dim=0, keepdim=True)).T
    )
    QK_last = (W_E[-1] + W_pos[-1]) @ W_Q[0, 0] @ W_K[0, 0].T @ (W_E[-1] + W_pos[-1])
    return {
        "data": (QK - QK_last).numpy(),
        "title": "Attention Score<br>QK[p] := (W<sub>E</sub>[-1] + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ (W<sub>E</sub> + W<sub>pos</sub>[p])<sup>T</sup><br>QK[:-1,:-1].mean(dim=0) - QK[-1, -1]",
        "xaxis": "input token",
        "yaxis": "attention score pre-softmax",
    }


@torch.no_grad()
def compute_OV(model: HookedTransformer = model, centered: bool = True) -> dict:
    W_E, W_pos, W_V, W_O, W_U = (
        model.W_E,
        model.W_pos,
        model.W_V,
        model.W_O,
        model.W_U,
    )
    OV = (W_E[:-1] + W_pos[:-1].mean(dim=0)) @ W_V[0, 0] @ W_O[0, 0] @ W_U
    result: dict = {"xaxis": "output logit token", "yaxis": "input token"}
    if not centered:
        result.update(
            {
                "data": OV.numpy(),
                "title": "Attention Computation: (W<sub>E</sub>[:-1] + W<sub>pos</sub>[:-1].mean(dim=0)) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub>",
            }
        )
        return result
    result.update(
        {
            "data": (OV - OV.diag()[:, None]).numpy(),
            "title": "Attention Computation (centered)<br>OV := (W<sub>E</sub>[:-1] + W<sub>pos</sub>[:-1].mean(dim=0)) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub><br>OV - OV.diag()[:, None]",
        }
    )
    return result


@torch.no_grad()
def compute_QK_by_position(model: HookedTransformer = model) -> dict:
    W_E, W_pos, W_Q, W_K = (
        model.W_E,
        model.W_pos,
        model.W_Q,
        model.W_K,
    )
    QK = (
        (W_E[-1] + W_pos[-1])
        @ W_Q[0, 0]
        @ W_K[0, 0].T
        @ (W_pos[:-1] - W_pos[:-1].mean(dim=0)).T
    )
    return {
        "data": {"QK": QK.numpy()},
        "title": "Positional Contribution to Attention Score<br>(W<sub>E</sub>[-1] + W<sub>pos</sub>[-1]) @ W<sub>Q</sub> @ W<sub>K</sub><sup>T</sup> @ (W<sub>pos</sub>[:-1] - W<sub>pos</sub>[:-1].mean(dim=0))<sup>T</sup>",
        "xaxis": "position",
        "yaxis": "attention score pre-softmax",
    }


@torch.no_grad()
def compute_irrelevant(
    model: HookedTransformer = model, include_equals_OV: bool = False
) -> dict:
    W_E, W_pos, W_V, W_O, W_U = (
        model.W_E,
        model.W_pos,
        model.W_V,
        model.W_O,
        model.W_U,
    )
    data = {
        "(W<sub>E</sub>[-1]+W<sub>pos</sub>[-1]) @ W<sub>U</sub>": (
            ((W_E[-1] + W_pos[-1]) @ W_U).numpy()
        ),
    }
    if include_equals_OV:
        data.update(
            {
                "(W<sub>E</sub>[-1]+W<sub>pos</sub>[-1]) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub>": (
                    (W_E[-1] + W_pos[-1]) @ W_V[0, 0] @ W_O[0, 0] @ W_U
                ),
            }
        )
    data.update(
        {
            f"(W<sub>pos</sub>[{i}] - W<sub>pos</sub>[:-1].mean(dim=0)) @ W<sub>V</sub> @ W<sub>O</sub> @ W<sub>U</sub>": (
                (
                    (W_pos[i] - W_pos[:-1].mean(dim=0))
                    @ W_V[0, 0, :, :]
                    @ W_O[0, 0, :, :]
                    @ W_U
                ).numpy()
            )
            for i in range(W_pos.shape[0] - 1)
        }
    )

    return {
        "data": data,
        "title": "Irrelevant Contributions to logits",
        "xaxis": "output logit token",
        "yaxis": "logit value",
    }


# %%


@torch.no_grad()
def display_basic_interpretation(
    model: HookedTransformer = model,
    include_uncentered: bool = False,
    legend_at_bottom: bool = False,
    include_equals_OV: bool = False,
):
    QK = compute_QK(model)
    px.line(
        {"QK": QK["data"]},
        title=QK["title"],
        labels={
            "index": QK["xaxis"],
            "variable": "",
            "value": QK["yaxis"],
        },
    ).show()

    if include_uncentered:
        OV = compute_OV(model, centered=False)
        px.imshow(
            OV["data"],
            title=OV["title"],
            color_continuous_scale="Picnic_r",
            color_continuous_midpoint=0,
            labels={"x": OV["xaxis"], "y": OV["yaxis"]},
        ).show()
    OV = compute_OV(model, centered=True)
    px.imshow(
        OV["data"],
        title=OV["title"],
        color_continuous_scale="Picnic_r",
        labels={"x": OV["xaxis"], "y": OV["yaxis"]},
    ).show()

    pos_QK = compute_QK_by_position(model)
    px.scatter(
        pos_QK["data"],
        title=pos_QK["title"],
        labels={"index": pos_QK["xaxis"], "variable": "", "value": pos_QK["yaxis"]},
    ).show()

    irrelevant = compute_irrelevant(model, include_equals_OV=include_equals_OV)
    fig = px.scatter(
        irrelevant["data"],
        title=irrelevant["title"],
        labels={
            "index": irrelevant["xaxis"],
            "variable": "",
            "value": irrelevant["yaxis"],
        },
    )
    if legend_at_bottom:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.5,
                xanchor="center",
                x=0.5,
            )
        )
    fig.show()


display_basic_interpretation()
# %%
# first_nonnegative = int(torch.nonzero(QK >= 0)[0, 0].item())
# assert (
#     QK[:first_nonnegative] < 0
# ).all(), f"The negatives don't form a contiguous block: {QK}"
# assert (
#     QK[first_nonnegative:] >= 0
# ).all(), f"The nonnegatives don't form a contiguous block: {QK}"
# wrong_sequences = int(
#     1 + sum(1 + seq_len * (i - 1) for i in range(1, first_nonnegative))
# )
# display(
#     Markdown(
#         f"The model pays more attention to '=' than to numbers less than or equal to {first_nonnegative - 1}, but "
#         f"that only occurs in {wrong_sequences} sequences ({100 * wrong_sequences / vocab ** seq_len:.1f}% of the sequences)."
#     )
# )
# print(model(torch.tensor([[4, 5, vocab]]))[:, -1, :].argmax(dim=-1))

# %%
# with torch.no_grad():
#     W_E, W_pos, W_Q, W_K, W_U, W_V, W_O = (
#         model.W_E,
#         model.W_pos,
#         model.W_Q,
#         model.W_K,
#         model.W_U,
#         model.W_V,
#         model.W_O,
#     )
#     OV = W_E[:-1] @ W_V[0, 0] @ W_O[0, 0] @ W_U
#     QK = W_E[-1] @ W_Q[0, 0] @ W_K[0, 0].T @ W_E.T
#     EUPU = (W_E[-1] + W_pos[-1]) @ W_U

#     toks = torch.tensor([[0, 1, vocab]])
#     print(QK_orig[toks].softmax(dim=-1))
#     print(OV[toks])


# %%
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


# %%

metrics = group_metrics_by_epoch(runtime)


def get_epochs_and_metric(
    metric_name: str,
    epoch: Optional[int],
    include_only_epochs: Optional[Collection[int]] = None,
    metrics: Dict[str, Dict[int, float]] = metrics,
) -> Tuple[List[int], List[Any]]:
    values = metrics[metric_name]
    epochs = [
        i
        for i in sorted(values.keys())
        if (epoch is None or i <= epoch)
        and (include_only_epochs is None or i in include_only_epochs)
    ]
    return epochs, [values[i] for i in epochs]


# %%
models = runtime.model_versions(cfg, max_count=3000, step=1)
assert models is not None
models = list(models)


# %%
@torch.no_grad()
def compute_traces_and_frames(
    models: List[Tuple[Any, Optional[Tuple[RunData, HookedTransformer]], Any]]
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
    epochs_so_far = set()

    for i, (_version, old_data, _artifact) in enumerate(tqdm(models)):
        assert old_data is not None
        old_runtime, old_model = old_data
        epoch = old_runtime.epoch
        epochs_so_far.add(epoch)
        overlap = compute_QK(old_model)["data"]

        # Update the max_abs_value for the attention plot
        cur_min_attention = min(0, np.min(overlap))
        current_max_attention = np.max(overlap)
        max_value_attention = max(max_value_attention, current_max_attention)

        training_losses_epochs, training_losses = get_epochs_and_metric(
            "loss", epoch, include_only_epochs=epochs_so_far
        )
        training_accuracies_epochs, training_accuracies = get_epochs_and_metric(
            "acc", epoch, include_only_epochs=epochs_so_far
        )
        test_losses_epochs, test_losses = get_epochs_and_metric(
            "periodic_test_loss", epoch, include_only_epochs=epochs_so_far
        )
        test_accuracies_epochs, test_accuracies = get_epochs_and_metric(
            "periodic_test_acc", epoch, include_only_epochs=epochs_so_far
        )

        # Update the max_value for the loss and accuracy plots
        max_value_losses = max(max(training_losses), max(test_losses))
        max_value_accuracies = max(max(training_accuracies), max(test_accuracies))

        # Update the max values for all plots
        all_min_value_attention.append(cur_min_attention)
        all_max_value_attention.append(max_value_attention)
        all_max_value_losses.append(max_value_losses)
        all_max_value_accuracies.append(max_value_accuracies)

        # Add a trace for the initial plot (first data point) in all subplots
        if i == 0:
            traces = [
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
                # Loss plot traces
                (
                    (
                        go.Scatter(
                            x=training_losses_epochs,
                            y=training_losses,
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
                            x=test_losses_epochs,
                            y=test_losses,
                            mode="lines",
                            name="Test Loss",
                        ),
                    ),
                    dict(
                        row=2,
                        col=1,
                    ),
                ),
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

        # Frame data for the attention plot
        frame_data_attention = go.Scatter(
            x=list(range(len(overlap))), y=overlap, mode="lines"
        )

        # Frame data for the loss and accuracy plots
        frame_data_losses = [
            go.Scatter(
                x=training_losses_epochs,
                y=training_losses,
                mode="lines",
                name="Training Loss",
            ),
            go.Scatter(
                x=test_losses_epochs,
                y=test_losses,
                mode="lines",
                name="Test Loss",
            ),
        ]
        frame_data_accuracies = [
            go.Scatter(
                x=training_accuracies_epochs,
                y=training_accuracies,
                mode="lines",
                name="Training Accuracy",
            ),
            go.Scatter(
                x=test_accuracies_epochs,
                y=test_accuracies,
                mode="lines",
                name="Test Accuracy",
            ),
        ]

        # Create a frame combining all plots
        frame = go.Frame(
            data=[frame_data_attention] + frame_data_losses + frame_data_accuracies,
            name=str(epoch),
            traces=[0, 1, 2, 3, 4, 5, 6],  # Indices of the traces in this frame
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
traces_and_frames = compute_traces_and_frames(models)

# %%

# # Add frames to the figure
# fig.frames = frames

# # Update layout for the figure
# fig.update_layout(


# # Adjust the height of the figure (e.g., if the original height was 600, now set it to 1200)
# fig.update_layout(width=500)
# fig.update_layout(height=600)  # Double the original height

# # Show the figure
# fig.show()


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Assuming 'epochs' list is available
# Create a subplot with 3 rows (1 for attention, 2 for losses, 3 for accuracies)
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=("Attention Plot", "Loss Plot", "Accuracy Plot"),
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
fig.update_layout(height=600)  # Double the original height
# Show the figure
fig.show()

# %%
import plotly.graph_objects as go
import imageio
import numpy as np

# Assuming 'fig', 'models', and the lists 'max_abs_value_attention', 'max_value_losses', 'max_value_accuracies' are defined

# Prepare a directory to save frames
import os

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
# Create the GIF
grokking_gif = "max_of_2_grokking_17.gif"
with imageio.get_writer(grokking_gif, mode="I", duration=0.5, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)  # type: ignore

# Optionally, cleanup the frames
# for filename in filenames:
#     os.remove(filename)

# %%
grokking_gif = "max_of_2_grokking_17.gif"
with open(grokking_gif, mode="rb") as f:
    display(Image(f.read()))
# %%
