# %%
from gbmi.exp_max_of_n.train import (
    FullDatasetCfg,
    IterableDatasetCfg,
    MaxOfN,
    train_or_load_model,
)
from gbmi.model import Config, RunData
from transformer_lens import HookedTransformerConfig, HookedTransformer

from gbmi.utils import generate_all_sequences_for_model


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
# model_wrapper =
# %%
# TODO: figure out batching
all_tokens = generate_all_sequences_for_model(model)
all_logits = model(all_tokens)
expected_max = all_tokens.max(dim=-1).values
predicted_max = all_logits[..., -1, :].argmax(dim=-1)
# print(f"Model Accuracy: {acc_fn(all_logits, all_tokens, return_per_token=False) * 100}%")
# print(f"Number Incorrect Sequences: {(predicted_max != expected_max).sum()}")
# print(f"Model Loss: {loss_fn(all_logits, all_tokens, return_per_token=False)}")
# print(f"{all_logits.dtype} ULP on log-softmax = ULP at 1.0 = -(exp(0) - eps).log() = {torch.finfo(all_logits.dtype).eps}")

# %%
