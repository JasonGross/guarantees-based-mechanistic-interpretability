import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
import einops
import torch

MEMORY_STRING = """Neil Armstrong Neil Armstrong Neil Armstrong Neil Armstrong"""
CAND_STRING = "Neil"
VAL_TO_MOD = "attn_out"

torch.manual_seed(50)

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Run the model and get logits and activations
logits, activations = model.run_with_cache(MEMORY_STRING)

final_resid_activation = activations[VAL_TO_MOD, 11].detach()
prediction = logits.argmax(dim=-1).squeeze()

print(model.to_str_tokens(prediction))


def tack_on_memory_hook(orig_val, hook):
    orig_val[:, -1, :] = orig_val[:, -1, :] + final_resid_activation[:, -1, :]
    return orig_val


start_string = CAND_STRING
unmod_start_string = CAND_STRING
for i in range(20):
    mod_logits = model.run_with_hooks(
        start_string,
        return_type="logits",
        fwd_hooks=[(utils.get_act_name(VAL_TO_MOD, 11), tack_on_memory_hook)],
    )
    unmod_logits = model.run_with_hooks(
        unmod_start_string, return_type="logits", fwd_hooks=[]
    )
    prediction = mod_logits.argmax(dim=-1).squeeze()
    null_prediction = unmod_logits.argmax(dim=-1).squeeze()

    print(model.to_str_tokens(prediction))
    print(model.to_str_tokens(null_prediction))

    start_string += model.to_str_tokens(prediction)[-1]
    unmod_start_string += model.to_str_tokens(null_prediction)[-1]
