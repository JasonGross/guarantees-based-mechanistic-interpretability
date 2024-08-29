# %%
import transformer_lens
import torch

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-large")

# Run the model and get logits and activations


# %%
str_ = "Red + Blue ="
for i in range(1):
    logits, activations = model.run_with_cache(str_)
    tok = model.to_string(
        torch.softmax(logits.squeeze()[-1], dim=0).multinomial(
            num_samples=1, replacement=True
        )[0]
    )
    str_ += tok
print(str_)
# %%
