import torch
from gbmi.exp_bigram_stats.train import DEFAULT_BIGRAM
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

model = train_or_load_model(DEFAULT_BIGRAM, force="train")
