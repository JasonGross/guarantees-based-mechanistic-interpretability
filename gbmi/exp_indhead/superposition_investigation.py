# %%
import transformer_lens
import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from texts import *

torch.cuda.empty_cache()

model_small = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
# model_first = transformer_lens.HookedTransformer.from_pretrained("stanford-gpt2-small-a", checkpoint_index=10)
model = model_small
attn_scale_0 = model.blocks[0].attn.attn_scale
W_pos = model.W_pos
W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
W_E = W_E * torch.sqrt(torch.tensor(model.cfg.d_model)) / e_normalize

W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
E_factor = e_normalize.squeeze() / (
    torch.sqrt(pos_normalize[-1] ** 2 + (e_normalize**2).squeeze())
)
W_U = model.W_U
W_U = W_U - W_U.mean(dim=1, keepdim=True)


def show(matrix):

    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)

    if matrix.shape[0] > 1500 or matrix.shape[1] > 1500:
        print("too big")
        return

    plt.matshow(matrix.detach().cpu())


def trace(neuron, first_layer, second_layer, threshold):
    for first_neuron in (
        (
            model.blocks[first_layer].mlp.W_out
            @ model.blocks[second_layer].mlp.W_in[:, neuron]
        )
        > threshold
    ).nonzero():
        bible.trace_neuron(first_neuron, layer=first_layer)
        code.trace_neuron(first_neuron, layer=first_layer)
        plt.show()
        plt.clf()


class Text:
    def __init__(self, text, name, moving_window=-1):
        self.name = name
        self.text = text
        self.moving_window = moving_window
        self.tokenized_text = model.to_str_tokens(self.text)
        self.length = len(self.tokenized_text)
        self.mean_pattern = (
            1 - torch.ones(self.length, self.length).triu(diagonal=1)
        ) / (torch.arange(1, self.length + 1).unsqueeze(0).T)
        if moving_window != -1:
            for i in range(self.length):
                if i > moving_window:
                    self.mean_pattern[i][: i - moving_window] = 0.0

                    self.mean_pattern[i][i - moving_window : i + 1] = 1 / moving_window
        self.mean_pattern = self.mean_pattern.to("cuda")

    def trace_neuron(self, neuron, length=-1, layer=0, marker=".", label=""):
        length = self.length if length == -1 else min(length, self.length)
        _, self.activations = model.run_with_cache(self.text)
        plt.xlabel("Position in sequence")
        plt.ylabel(f"Pre-activation of neuron {neuron}")
        if label == "":
            plt.scatter(
                [i for i in range(length)],
                self.activations[f"blocks.{layer}.mlp.hook_pre"]
                .squeeze()[:length, neuron]
                .detach()
                .cpu(),
                label=f"{self.name}",
                marker=marker,
            )
        else:
            plt.scatter(
                [i for i in range(length)],
                self.activations[f"blocks.{layer}.mlp.hook_pre"]
                .squeeze()[:length, neuron]
                .detach()
                .cpu(),
                label=label,
                marker=marker,
            )

    def trace_head_neuron(
        self,
        neuron,
        head,
        ax=plt,
        length=-1,
        full_ov=False,
        mean_diff=False,
        layer=0,
        positional=True,
        marker="x",
        plot=True,
    ):
        _, self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length, self.length)

        attn_pattern = self.activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[
            head
        ]

        if mean_diff:
            attn_pattern = self.mean_pattern.to("cuda")
            marker = "."
        sequence = W_E[model.to_tokens(self.text)].squeeze() * E_factor[
            model.to_tokens(self.text)
        ].squeeze().unsqueeze(1)
        if positional:
            sequence = sequence + W_pos[:length] * (
                pos_normalize[-1]
                / (
                    torch.sqrt(
                        pos_normalize[-1] ** 2
                        + (e_normalize[model.to_tokens(self.text)].squeeze()[:length])
                        ** 2
                    )
                )
            ).unsqueeze(1)
        if full_ov:
            head_neuron = (
                (attn_pattern @ (sequence)).squeeze()
                @ gpt2_vecs
                @ model.blocks[layer].mlp.W_in[:, neuron]
            )
        else:
            head_neuron = (
                (attn_pattern @ (sequence)).squeeze()
                @ model.W_V[layer, head]
                @ model.W_O[layer, head]
                @ model.blocks[layer].mlp.W_in[:, neuron]
            )[:length]
        if plot:
            ax.scatter(
                [i for i in range(length)],
                head_neuron[:length].detach().cpu(),
                marker=marker,
            )  # label=f'Contribution of head {head} to neuron {neuron}, on {self.name}')
        return head_neuron[:length]

    def trace_real_neuron(self, neuron, head, length=-1, ax=plt, layer=0):
        _, self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length, self.length)
        marker = "o"
        ax.scatter(
            [i for i in range(length)],
            (
                self.activations[f"blocks.{layer}.attn.hook_z"].squeeze()[:length, head]
                @ model.W_O[layer, head]
                @ model.blocks[0].mlp.W_in[:, neuron]
            )
            .detach()
            .cpu(),
        )

    def trace_first_attention(self, head, ax=plt, length=-1, layer=0):
        _, self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length, self.length)
        first_attn = self.activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[
            head
        ][:length, 0]

        ax.scatter([i for i in range(length)], first_attn.detach().cpu())

    def get_outliers(self, head, threshold=0.01, multiplier=3, layer=0, index=2):
        toks = []
        _, activations = model.run_with_cache(self.text)
        attn_pattern = activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[
            head, -index
        ]
        for i in range(1, self.length):

            if attn_pattern[i - 1] > threshold:
                if attn_pattern[i] / attn_pattern[i - 1] > multiplier:
                    toks.append(i)

        if len(toks) != 0:

            return model.to_str_tokens(
                model.to_tokens(self.text).squeeze()[torch.tensor(toks)].squeeze()
            )
        else:
            return []


# %%
show(((model.blocks[0].mlp.W_out @ model.blocks[1].mlp.W_in[:, 788]).reshape(48, 64)))
plt.colorbar()
# %%
print((model.blocks[0].mlp.W_out @ model.blocks[1].mlp.W_in[:, 788] > 0.5).nonzero())
# %%

bible = Text(genesis[:1000] + " remarked", "bible")
code = Text(java_text + " joking", "code")
neuron = 190
bible.trace_neuron(neuron)
code.trace_neuron(neuron)
# %%
plt.show()
plt.clf()
for head in range(12):
    bible.trace_head_neuron(neuron, head)
    plt.show()
    plt.clf()

# %%
ov_7 = model.W_V[0, 7] @ model.W_O[0, 7]
range_ = (0, 2000)
length = range_[1] - range_[0]
qk_7 = (W_E[range_[0] : range_[1]] @ (model.W_Q[0, 7] @ model.W_K[0, 7].T)) @ W_E[
    range_[0] : range_[1]
].T
qk_3 = (W_E[range_[0] : range_[1]] @ (model.W_Q[0, 3] @ model.W_K[0, 3].T)) @ W_E[
    range_[0] : range_[1]
].T
qk_4 = (W_E[range_[0] : range_[1]] @ (model.W_Q[0, 4] @ model.W_K[0, 4].T)) @ W_E[
    range_[0] : range_[1]
].T
mat_7 = (qk_7 - qk_7.diag().unsqueeze(0))[:length, :length]
mat_3 = (qk_3 - qk_3.diag().unsqueeze(0))[:length, :length]
mat_4 = (qk_4 - qk_4.diag().unsqueeze(0))[:length, :length]
qk_7 = (W_E[range_[0] : range_[1]] @ (model.W_Q[0, 7] @ model.W_K[0, 7].T)) @ W_E[
    range_[0] : range_[1]
].T
mat_7 = (qk_7 - qk_7.diag().unsqueeze(0))[:length, :length]
# %%

bigrams = (
    torch.exp(mat_7 / 10) * torch.exp(mat_4 / 10) * torch.exp(mat_3 / 10) > 3000.0
).nonzero()
for bigram in bigrams:
    print(model.to_str_tokens(bigram[1])[-1] + model.to_str_tokens(bigram[0])[-1])
# %%
text = genesis[:100]
phrase = " Security Contract"
_, activations = model.run_with_cache(text + phrase)
show(activations["blocks.0.attn.hook_pattern"].squeeze()[3][-10:, -10:])
plt.colorbar()
print(model.to_str_tokens(text + phrase)[-10:])
bigram_size = min(500, len(bigrams))
bigram_mat = torch.zeros(bigram_size, model.cfg.d_mlp)
for i in range(bigram_size):
    bigram = bigrams[i]
    first = W_E[bigram[1]] @ model.W_V[0, 7] @ model.W_O[0, 7]
    second = (
        (W_E * e_normalize)[bigram[0]] / (torch.sqrt(torch.tensor(768.0)))
        + (W_E[bigram[0]]) @ model.W_V[0, 4] @ model.W_O[0, 4]
        + (W_E[bigram[0]] @ model.W_V[0, 5] @ model.W_O[0, 5])
        + (W_E[bigram[0]] @ model.W_V[0, 1] @ model.W_O[0, 1])
        + (W_E[bigram[0]] @ model.W_V[0, 3] @ model.W_O[0, 3])
    )
    total_pre_embed = first + second
    mlp_activations = torch.nn.ReLU()(
        total_pre_embed @ model.blocks[0].mlp.W_in + model.blocks[0].mlp.b_in
    )
    end_activ = total_pre_embed + mlp_activations @ model.blocks[0].mlp.W_out
    bigram_mat[i] = end_activ @ model.blocks[1].mlp.W_in

# %%
