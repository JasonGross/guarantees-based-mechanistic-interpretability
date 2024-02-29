# %%
import torch
from gbmi.exp_bigram_stats.train import DEFAULT_BIGRAM, BigramTrainingWrapper
from gbmi.model import (
    TrainingWrapper,
    Config,
    ExperimentConfig,
    add_HookedTransformerConfig_arguments,
    train_or_load_model,
    DataModule,
    add_force_argument,
    add_no_save_argument,
    update_HookedTransformerConfig_from_args,
)

# %%
print(DEFAULT_BIGRAM)
print(BigramTrainingWrapper.build_model(DEFAULT_BIGRAM).cfg)
_, model = train_or_load_model(DEFAULT_BIGRAM, force="train")

# %%
import plotly.express as px

with torch.no_grad():
    px.imshow(
        (
            (model.W_pos @ model.W_Q[0, 0] + model.b_Q[0, 0])
            @ (model.W_pos @ model.W_K[0, 0] + model.b_K[0, 0]).T
        ).cpu()
    ).show()

# %%
