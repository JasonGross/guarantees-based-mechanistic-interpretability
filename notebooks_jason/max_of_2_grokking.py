# %%
from dataclasses import replace
from tqdm import tqdm
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    MaxOfN,
    train_or_load_model,
    config_of_argv,
)
from gbmi.model import Config, try_load_model_from_wandb_download
from transformer_lens import HookedTransformerConfig, HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import wandb
from typing import (
    Tuple,
)

from gbmi.utils import set_params

api = wandb.Api()

# %%
# config, kwargs = config_of_argv("gbmi.exp_max_of_n.train --max-of 2 --deterministic --train-for-epochs 1500 --validate-every-epochs 1 --force-adjacent-gap 0,1,2 --use-log1p --training-ratio 0.095 --weight-decay 1.0 --betas 0.9 0.98 --optimizer AdamW --use-end-of-sequence --checkpoint-every-epochs 1 --batch-size 389 --force train".split(" "))
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
    validate_every=(1, "epochs"),
    checkpoint_every=(1, "epochs"),
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
models = runtime.model_versions(cfg, max_count=1200, step=20)
assert models is not None
models = list(models)
# %%
training_losses, test_losses = [], []
training_accuracies, test_accuracies = [], []
for _version, old_data, _artifact in tqdm(models):
    assert old_data is not None
    old_runtime, old_model = old_data
    (training_loss, training_acc), (test_loss, test_acc) = evaluate_model(old_model)
    training_losses.append(training_loss)
    test_losses.append(test_loss)
    training_accuracies.append(training_acc)
    test_accuracies.append(test_acc)
# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Assuming 'models', 'training_losses', 'test_losses', 'training_accuracies', 'test_accuracies' are defined
# Create a subplot with 2 rows
fig = make_subplots(rows=2, cols=1)

# Lists to hold frames and slider steps
frames = []
slider_steps = []

# Variable to track the maximum absolute value across all epochs for the attention plot
max_abs_value = 0

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
        current_max = torch.max(torch.abs(overlap)).item()
        max_abs_value = max(max_abs_value, current_max)

        # Add a trace for the initial plot (first epoch) in both subplots
        if epoch == 0:
            # Attention plot trace
            fig.add_trace(
                go.Scatter(x=list(range(len(overlap))), y=overlap, mode="lines"),
                row=1,
                col=1,
            )
            # Losses and accuracies plot traces
            fig.add_trace(
                go.Scatter(
                    x=list(range(i + 1)),
                    y=training_losses[: i + 1],
                    mode="lines",
                    name="Training Loss",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(i + 1)),
                    y=test_losses[: i + 1],
                    mode="lines",
                    name="Test Loss",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(i + 1)),
                    y=training_accuracies[: i + 1],
                    mode="lines",
                    name="Training Accuracy",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(i + 1)),
                    y=test_accuracies[: i + 1],
                    mode="lines",
                    name="Test Accuracy",
                ),
                row=2,
                col=1,
            )

        # Frame data for the attention plot
        frame_data_attention = go.Scatter(
            x=list(range(len(overlap))), y=overlap, mode="lines"
        )

        # Frame data for the losses and accuracies plot
        frame_data_losses = [
            go.Scatter(
                x=list(range(i + 1)),
                y=training_losses[: i + 1],
                mode="lines",
                name="Training Loss",
            ),
            go.Scatter(
                x=list(range(i + 1)),
                y=test_losses[: i + 1],
                mode="lines",
                name="Test Loss",
            ),
            go.Scatter(
                x=list(range(i + 1)),
                y=training_accuracies[: i + 1],
                mode="lines",
                name="Training Accuracy",
            ),
            go.Scatter(
                x=list(range(i + 1)),
                y=test_accuracies[: i + 1],
                mode="lines",
                name="Test Accuracy",
            ),
        ]

        # Create a frame combining both plots
        frame = go.Frame(
            data=[frame_data_attention] + frame_data_losses,
            name=str(epoch),
            traces=[0, 1, 2, 3, 4, 5, 6, 7],  # Indices of the traces in this frame
            layout=go.Layout(
                yaxis={
                    "range": [-max_abs_value, max_abs_value]
                }  # Setting y-axis range for the first subplot
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

# Update layout for the subplot (the attention plot)
fig.update_layout(
    xaxis_title="input token",
    yaxis_title="Attention",
    title="Pre-softmax Attention by Input Token / Training and Test Losses/Accuracies",
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

# Update layout for the second subplot (losses and accuracies)
fig.update_xaxes(title_text="Epoch", row=2, col=1)
fig.update_yaxes(title_text="", row=2, col=1)

# Show the figure
fig.show()

# %%
