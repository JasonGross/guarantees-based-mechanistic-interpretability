# %%
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    IterableDatasetCfg,
    MaxOfN,
    MaxOfNTrainingWrapper,
    train_or_load_model,
)
from gbmi.model import Config, RunData
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformerConfig, HookedTransformer
from gbmi.utils import default_device

from gbmi.utils.sequences import SequenceDataset, generate_all_sequences_for_model


# %%
cfg = Config(
    experiment=MaxOfN(
        model_config=HookedTransformerConfig(
            act_fn=None,
            attn_only=True,
            d_head=32,
            d_mlp=None,
            d_model=32,
            d_vocab=64,
            device="cpu",
            n_ctx=2,
            n_heads=1,
            n_layers=1,
            normalization_type=None,
            seed=613947648,
        ),
        zero_biases=True,
        use_log1p=True,
        use_end_of_sequence=False,
        seq_len=4,
        train_dataset_cfg=IterableDatasetCfg(),
        test_dataset_cfg=IterableDatasetCfg(n_samples=1024),
    ),
    deterministic=True,
    seed=123,
    batch_size=128,
    train_for=(3000, "steps"),
)
# %%
runtime, model = train_or_load_model(cfg, force="load")
# %%
training_wrapper = MaxOfNTrainingWrapper(cfg, model)
# %%
all_tokens_dataset = SequenceDataset(
    seq_len=model.cfg.n_ctx, vocab_size=model.cfg.d_vocab
)
# %%
batch_size = 128
# Resetting the DataLoader without shuffle for consistent processing
data_loader = DataLoader(all_tokens_dataset, batch_size=batch_size, shuffle=False)

# Variables to accumulate total loss and accuracy
total_loss = 0.0
total_accuracy = 0.0
total_samples = 0

# Training loop for computing overall loss and accuracy
with torch.no_grad():
    for batch in tqdm(data_loader):
        batch.to(default_device(deterministic=False))
        loss, accuracy = training_wrapper.run_batch(
            batch, return_accuracy=True, log_output=False
        )

        # Accumulate loss and accuracy
        total_loss += loss.item() * batch.size(0)
        total_accuracy += accuracy * batch.size(0)
        total_samples += batch.size(0)

# Calculate average loss and accuracy
average_loss = total_loss / total_samples
average_accuracy = total_accuracy / total_samples
# %%
print(f"Model Accuracy: {average_accuracy * 100}%")
print(f"Number Incorrect Sequences: {average_accuracy * all_tokens_dataset.length}")
print(f"Model Loss: {average_loss}")

# %%
