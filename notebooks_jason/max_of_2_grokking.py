# %%
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    MaxOfN,
    train_or_load_model,
    config_of_argv,
)
from gbmi.model import Config, try_load_model_from_wandb_download
from transformer_lens import HookedTransformerConfig
import torch
import wandb

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
artifact = api.artifact(runtime.wandb_id)
# %%
model_artifacts = list(artifact.logged_by().logged_artifacts())
# %%
models = [
    (a.version, try_load_model_from_wandb_download(cfg, a.download()), a)
    for a in model_artifacts
    if a.type == "model"
]
# %%
model_artifacts_by_version = {a.version: a for a in models}
# %%
# Walk up and down the graph from an artifact:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
# %%
project, entity, latest_model = runtime.wandb_id.split("/")

# %%
run = wandb.init(project=project, entity=entity)
# %%
