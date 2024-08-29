# %%
import torch
import transformer_lens

model = transformer_lens.HookedTransformer.from_pretrained("gpt2-xl")

# %%
import torch

# %%
str_ = "What common colour do I get if I mix red and white paint? A:"
for i in range(100):

    logits, activations = model.run_with_cache(str_)
    tok = model.to_string(
        torch.multinomial(torch.softmax(2 * logits.squeeze()[-1], dim=0), num_samples=1)
    )
    str_ += tok
    print(tok, end="")
print(str_)
# %%
