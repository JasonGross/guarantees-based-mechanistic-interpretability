# %%
from tqdm import tqdm
import os
import imageio
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    MaxOfN,
    train_or_load_model,
)
from gbmi.model import Config
from transformer_lens import HookedTransformerConfig, HookedTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import wandb
from typing import (
    Tuple,
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
ExpWrapper = cfg.experiment.get_training_wrapper()
# wrapped_model = ExpWrapper(config, ExpWrapper.build_model(config))
datamodule = cfg.experiment.get_datamodule()(cfg)
datamodule.setup("")
training_batches = torch.cat([batch for batch in datamodule.train_dataloader()], dim=0)
test_batches = torch.cat([batch for batch in datamodule.test_dataloader()], dim=0)


# %%
def evaluate_model(
    model: HookedTransformer,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    with torch.no_grad():
        wrapped_model = ExpWrapper(cfg, model)
        training_loss, training_acc = wrapped_model.run_batch(
            training_batches, prefix="", log_output=False, return_accuracy=True
        )
        test_loss, test_acc = wrapped_model.run_batch(
            test_batches, prefix="", log_output=False, return_accuracy=True
        )
    return (training_loss.item(), training_acc), (test_loss.item(), test_acc)


# %%
models = runtime.model_versions(cfg, max_count=3000, step=1)
assert models is not None
models = list(models)
# %%
training_losses, test_losses = [], []
training_accuracies, test_accuracies = [], []
epochs = []
for _version, old_data, _artifact in tqdm(models):
    assert old_data is not None
    old_runtime, old_model = old_data
    epochs.append(old_runtime.epoch)
    (training_loss, training_acc), (test_loss, test_acc) = evaluate_model(old_model)
    training_losses.append(training_loss)
    test_losses.append(test_loss)
    training_accuracies.append(training_acc)
    test_accuracies.append(test_acc)
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

# Lists to hold frames and slider steps
frames = []
slider_steps = []

# Variable to track the maximum values for each plot
max_abs_value_attention = 0
max_value_losses = 0
max_value_accuracies = 0
all_max_abs_value_attention = []
all_max_value_losses = []
all_max_value_accuracies = []

with torch.no_grad():
    for i, (_version, old_data, _artifact) in enumerate(models):
        assert old_data is not None
        old_runtime, old_model = old_data
        epoch = old_runtime.epoch
        W_E, W_pos, W_Q, W_K = (
            old_model.W_E,
            old_model.W_pos,
            old_model.W_Q[0, 0, :, :],
            old_model.W_K[0, 0, :, :],
        )
        overlap = (
            (W_E[-1] + W_pos[-1])
            @ W_Q
            @ W_K.T
            @ (W_E[:-1, :] + W_pos[:-1, 0].mean(dim=0)).T
        )

        # Update the max_abs_value for the attention plot
        current_max_attention = torch.max(torch.abs(overlap)).item()
        max_abs_value_attention = max(max_abs_value_attention, current_max_attention)

        # Update the max_value for the loss and accuracy plots
        current_max_loss = max(training_losses[i], test_losses[i])
        max_value_losses = max(max_value_losses, current_max_loss)
        current_max_accuracy = max(training_accuracies[i], test_accuracies[i])
        max_value_accuracies = max(max_value_accuracies, current_max_accuracy)

        # Update the max values for all plots
        all_max_abs_value_attention.append(max_abs_value_attention)
        all_max_value_losses.append(max_value_losses)
        all_max_value_accuracies.append(max_value_accuracies)

        # Add a trace for the initial plot (first data point) in all subplots
        if i == 0:
            # Attention plot trace
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(overlap))),
                    y=overlap,
                    mode="lines",
                    name="(E+P)<sub>-1</sub>QK<sup>T</sup>(E+P)<sup>T</sup>",
                ),
                row=1,
                col=1,
            )
            # Loss plot traces
            fig.add_trace(
                go.Scatter(
                    x=[epochs[0]],
                    y=[training_losses[0]],
                    mode="lines",
                    name="Training Loss",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[epochs[0]], y=[test_losses[0]], mode="lines", name="Test Loss"
                ),
                row=2,
                col=1,
            )
            # Accuracy plot traces
            fig.add_trace(
                go.Scatter(
                    x=[epochs[0]],
                    y=[training_accuracies[0]],
                    mode="lines",
                    name="Training Accuracy",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[epochs[0]],
                    y=[test_accuracies[0]],
                    mode="lines",
                    name="Test Accuracy",
                ),
                row=3,
                col=1,
            )

        # Frame data for the attention plot
        frame_data_attention = go.Scatter(
            x=list(range(len(overlap))), y=overlap, mode="lines"
        )

        # Frame data for the loss and accuracy plots
        frame_data_losses = [
            go.Scatter(
                x=epochs[: i + 1],
                y=training_losses[: i + 1],
                mode="lines",
                name="Training Loss",
            ),
            go.Scatter(
                x=epochs[: i + 1],
                y=test_losses[: i + 1],
                mode="lines",
                name="Test Loss",
            ),
        ]
        frame_data_accuracies = [
            go.Scatter(
                x=epochs[: i + 1],
                y=training_accuracies[: i + 1],
                mode="lines",
                name="Training Accuracy",
            ),
            go.Scatter(
                x=epochs[: i + 1],
                y=test_accuracies[: i + 1],
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
                    "range": [-max_abs_value_attention, max_abs_value_attention]
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

# Add frames to the figure
fig.frames = frames

# Update layout for the figure
fig.update_layout(
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
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "linear"},
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
    sliders=[{"steps": slider_steps, "active": 0}],
)

# Adjust the height of the figure (e.g., if the original height was 600, now set it to 1200)
fig.update_layout(width=500)
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

filenames = []
for i, frame in enumerate(tqdm(fig.frames)):
    # Update data for each trace
    for j, data in enumerate(frame.data):
        fig.data[j].x = data.x
        fig.data[j].y = data.y

    # Update layout (axis bounds)
    fig.update_layout(
        yaxis={
            "range": [-all_max_abs_value_attention[i], all_max_abs_value_attention[i]]
        },
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
with imageio.get_writer(
    "max_of_2_grokking_17.gif", mode="I", duration=0.5, loop=0
) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optionally, cleanup the frames
# for filename in filenames:
#     os.remove(filename)

# %%
