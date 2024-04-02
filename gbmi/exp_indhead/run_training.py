# %%
import torch
from gbmi.exp_indhead.train import DEFAULT_INDHEAD, main

# %%
torch.set_default_device("cuda")

_, model = main(default=DEFAULT_INDHEAD, default_force="train")

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
