# %%
import transformer_lens
import torch
from texts import *
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

torch.cuda.empty_cache()


model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")


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
            sequence = sequence + (model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1))[
                :length
            ] * (
                pos_normalize[-500]
                / (
                    torch.sqrt(
                        pos_normalize[-500] ** 2
                        + (e_normalize[model.to_tokens(self.text)].squeeze()[:length])
                        ** 2
                    )
                )
            ).unsqueeze(
                1
            )
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
        marker = "x"
        ax.scatter(
            [i for i in range(length)],
            (
                self.activations[f"blocks.{layer}.attn.hook_z"].squeeze()[:length, head]
                @ model.W_O[layer, head]
                @ model.blocks[layer].mlp.W_in[:, neuron]
            )
            .detach()
            .cpu(),
            marker=marker,
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


#   def trace_norm(self,head,layer=0,length=-1):
#      _, self.activations = model.run_with_cache(self.text)
#     length = self.length if length == -1 else min(length, self.length)
#
#    attn_pattern = self.activations[f'blocks.{layer}.attn.'


def get_violating_tokens(text, index, threshold, greater=False):
    if greater:
        return model.to_str_tokens(
            model.to_tokens(text)
            .squeeze()[
                (big[:, index][model.to_tokens(text).squeeze()] > threshold).nonzero()
            ]
            .squeeze()
        )
    else:
        return model.to_str_tokens(
            model.to_tokens(text)
            .squeeze()[
                (big[:, index][model.to_tokens(text).squeeze()] < threshold).nonzero()
            ]
            .squeeze()
        )


def get_toks_interval(index, low, high):

    return model.to_str_tokens((low < big[:, index]).nonzero())


def get_toks(index, threshold, greater=False):
    if greater:

        return model.to_str_tokens((big[:, index] > threshold).nonzero())
    else:
        return model.to_str_tokens((big[:, index] < threshold).nonzero())


def plot_toks(index, text):
    plotcdf(big[:, index][model.to_tokens(text).squeeze()])


def splice(text1, text2, splice_start, splice_end):

    return Text(
        "".join(
            text1.tokenized_text[1 : splice_start + 1]
            + text2.tokenized_text[1 : (2 + splice_end - splice_start)]
            + text1.tokenized_text[2 + splice_end :]
        ),
        name=f"{text1.name} with {text2.name} spliced in between tokens {splice_start} and {splice_end}",
        moving_window=text1.moving_window,
    )


moving_window = 100

dnd = Text(dnd_text, "dnd", moving_window)
bible = Text(genesis, "bible", moving_window)
# fishing = Text(fishing_news, "Fishing news story", moving_window)
# league = Text(league_of_legends,'LoL forum post')
code = Text(code_text, "code", moving_window)
tutorial = Text(java_tutorial, "Java tutorial", moving_window)
comments = Text(comment_text, "comment spam")
brackets = Text(
    "((())) )(  )      j j     j  j     (   j j j j j j j j j j j  j j j j ())",
    "brackets",
)
bible_code = splice(bible, code, 100, 600)
code_bible = splice(code, bible, 100, 600)
java = Text(java_text, "java", moving_window)
bbc_news = Text(bbc_text, "bbc", moving_window)


def get_diffs(prompt):
    _, activations = model.run_with_cache(prompt)
    activations["blocks.0.attn.hook_pattern"]


# %%
import plotly.express as px

import plotly.graph_objects as go


# %%
frames = []
model_small = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
for checkpoint in range(0, 100, 1):
    model_first = transformer_lens.HookedTransformer.from_pretrained(
        "stanford-gpt2-small-a", checkpoint_index=checkpoint
    )
    model = model_first
    attn_scale_0 = model.blocks[0].attn.attn_scale

    W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
    W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)

    pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
    e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
    W_E = W_E * torch.sqrt(torch.tensor(model.cfg.d_model)) / e_normalize

    W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
    E_factor = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[-100] ** 2 + (e_normalize**2).squeeze())
    )
    pos_factor = pos_normalize.squeeze() / (
        torch.sqrt(
            (pos_normalize**2).squeeze()
            + (e_normalize[model.to_tokens(" of").squeeze()[-1]] ** 2).squeeze()
        )
    )
    W_E = W_E * (E_factor.unsqueeze(1))
    W_pos = W_pos * (pos_factor.unsqueeze(1))
    W_U = model.W_U
    W_U = W_U - W_U.mean(dim=1, keepdim=True)
    index = 15
    W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
    _, activations = model.run_with_cache(bible.text)
    W_pos = W_pos / (activations["blocks.0.ln1.hook_scale"].squeeze().mean())

    bias = torch.zeros(model.cfg.d_model).to("cuda")

    for head in range(2, 3):
        pos_pattern_presoft = torch.tensor(
            (
                torch.softmax(
                    (
                        (W_pos[-index] + bias)
                        @ model.W_Q[0, head]
                        @ model.W_K[0, head].T
                        @ (1.0 * W_pos[: (1 - index)].T + bias.unsqueeze(1))
                        + (W_pos[-index] + bias)
                        @ model.W_Q[0, head]
                        @ model.b_K[0, head].T
                        + model.b_Q[0, head]
                        @ (
                            model.W_K[0, head].T
                            @ (1.0 * W_pos[: (1 - index)].T + bias.unsqueeze(1))
                            + model.b_K[0, head].unsqueeze(1)
                        )
                        + (W_E[model.to_tokens(bible.text).squeeze()[-index]])
                        @ model.W_Q[0, head]
                        @ (
                            model.b_K[0, head].T.unsqueeze(1)
                            + model.W_K[0, head].T
                            @ (1.0 * W_pos[: (1 - index)].T + bias.unsqueeze(1))
                        )
                    )
                    / 8.0,
                    dim=0,
                )
            ).tolist()
            + [0 for index in range(index - 1)]
        )
        print(f"checkpoint:{checkpoint},head:{head}")
        fig = px.imshow(pos_pattern_presoft.reshape(32, 32))
        frames += [go.Frame(data=fig.data[0], layout=fig.layout)]
        if checkpoint == 0:
            first_fig = fig

# %%
fig = go.Figure(frames=frames)
fig.add_trace(
    first_fig.data[0],
)
fig.layout = first_fig.layout


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]

fig.update_layout(
    title="Slices in volumetric data",
    width=1200,
    height=600,
    scene=dict(
        zaxis=dict(range=[-0.1, 6.8], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders,
)
fig.show()
# %%
