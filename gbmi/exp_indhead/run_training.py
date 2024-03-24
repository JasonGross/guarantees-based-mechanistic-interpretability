# %%
import torch
from gbmi.exp_indhead.train import DEFAULT_INDHEAD, IndHeadTrainingWrapper
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
torch.set_default_device("cuda")

print(DEFAULT_INDHEAD)
print(IndHeadTrainingWrapper.build_model(DEFAULT_INDHEAD).cfg)
_, model = train_or_load_model(DEFAULT_INDHEAD, force="train")

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
