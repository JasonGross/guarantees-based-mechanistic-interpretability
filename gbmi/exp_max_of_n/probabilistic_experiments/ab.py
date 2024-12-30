# %%
import transformer_lens
import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

#from texts import *

torch.cuda.empty_cache()

#model_small = transformer_lens.HookedTransformer.from_pretrained("gpt2-large")
model_first = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
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
   torch.sqrt((pos_normalize ** 2 ).squeeze()+ (e_normalize[model.to_tokens(' of').squeeze()[-1]]**2).squeeze())
)
W_E = W_E*(E_factor.unsqueeze(1))
W_pos = W_pos*(pos_factor.unsqueeze(1))
W_U = model.W_U
W_U = W_U - W_U.mean(dim=1, keepdim=True)


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

    def trace_neuron(self, neuron, length=-1, layer=0,marker='.',label=""):
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
                label=f"{self.name}",marker=marker)
        else:
            plt.scatter(
                [i for i in range(length)],
                self.activations[f"blocks.{layer}.mlp.hook_pre"]
                .squeeze()[:length, neuron]
                .detach()
                .cpu(),
                label=label,
        marker=marker)

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
        plot=True
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
            sequence = sequence + (model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1))[:length] * (
                pos_normalize[-500]
                / (
                    torch.sqrt(
                        pos_normalize[-500] ** 2
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
        marker = "x"
        ax.scatter(
            [i for i in range(length)],
            (
                self.activations[f"blocks.{layer}.attn.hook_z"].squeeze()[:length, head]
                @ model.W_O[layer, head]
                @ model.blocks[layer].mlp.W_in[:, neuron]
            )
            .detach()
            .cpu(),marker=marker
        )

    def trace_first_attention(self, head, ax=plt, length=-1, layer=0):
        _, self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length, self.length)
        first_attn = self.activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[
            head
        ][:length, 0]

        ax.scatter([i for i in range(length)], first_attn.detach().cpu())

    def get_outliers(self,head,threshold=0.01,multiplier=3,layer=0,index=2):
        toks = []
        _,activations = model.run_with_cache(self.text)
        attn_pattern = activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head,-index]
        for i in range(1,self.length):

            if attn_pattern[i-1] > threshold:
                if attn_pattern[i]/attn_pattern[i-1] > multiplier:
                    toks.append(i)

        if len(toks)!=0:

            return model.to_str_tokens(model.to_tokens(self.text).squeeze()[torch.tensor(toks)].squeeze())
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
#fishing = Text(fishing_news, "Fishing news story", moving_window)
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
    _,activations = model.run_with_cache(prompt)
    activations['blocks.0.attn.hook_pattern']

#%%

head=9
full_pattern = torch.zeros(model.W_E.shape[0],768).to('cuda')
pos_pattern = torch.tensor([0.05 for i in range(20)])
texts = [bible,dnd,bbc_news,Text(java.text+java.text,'Java')]
for text in texts[1:2]:
    pattern_ = []
    full_text = text
    text = full_text.text
    _,activations = model.run_with_cache(text)
    post_ln1 = activations['blocks.0.ln1.hook_normalized'].squeeze()
    for index in range(500,501):
        full_pattern = torch.zeros(model.W_E.shape[0],768).to('cuda')
        W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
        W_E_t = W_E*torch.sqrt(torch.tensor(768.0))/ e_normalize
        E_factor = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[index] ** 2 + (e_normalize**2).squeeze())
        )
        E_factor_b = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[index] ** 2 + (e_normalize[11]**2))
        )
        W_E = W_E_t*E_factor.unsqueeze(1)
        W_E_b = W_E_t*E_factor_b.unsqueeze(1)
        W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
        pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
        W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
        pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (e_normalize[11]**2).squeeze())
        )

        W_pos = W_pos*(pos_factor.unsqueeze(1))
        for head in [0,2,6,8,9,10,11]:
            weighting =torch.min(torch.exp(((W_pos[index]+W_E_b[11])@model.W_Q[0,head]+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T)/8.0),torch.tensor(3.0))

            current_weight = ((weighting[model.to_tokens(text).squeeze()].detach().clone()[index-100:index].mean().item()))
            pattern_.append(current_weight)
            full_pattern = full_pattern+((weighting.unsqueeze(1)*W_E)@model.W_V[0,head]@model.W_O[0,head]/(100*current_weight)).detach().clone()
   # plt.scatter([i for i in range(924)],pattern_,label=f'{full_text.name}')
   # plt.ylim(0.0,2.0)
   # plt.xlabel('Position in sequence')
   # plt.ylabel(r'$\sum_{j=1}^{pos} \text{pos}_{j}f_{x_{j}}$')
   # plt.title(f'Head {head}')
#plt.legend()
#plt.show()
#%%
W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)

pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
W_E = W_E * torch.sqrt(torch.tensor(model.cfg.d_model)) / e_normalize

W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
E_factor = e_normalize.squeeze() / (
    torch.sqrt(pos_normalize[500] ** 2 + (e_normalize**2).squeeze())
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
index = 160
#%%
index = 2
pos_patterns = torch.zeros(1024,12,1024)
for index in range(2,1024):
    for head in range(model.cfg.n_heads):
        W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
        e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
        W_E = W_E * torch.sqrt(torch.tensor(model.cfg.d_model)) / e_normalize

        E_factor = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[-index] ** 2 + (e_normalize**2).squeeze())
    )
        W_E = W_E * (E_factor.unsqueeze(1))
        W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
        tokenized_text = model.to_tokens(dnd.text).squeeze()
        W_pos = W_pos*torch.sqrt(torch.tensor(model.cfg.d_model)) / torch.sqrt(torch.norm(W_pos[-index])**2+(torch.mean(e_normalize[11])**2))

        bias = torch.zeros(model.cfg.d_model).to("cuda")

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
            + [0.0 for j in range(index-1)]
        )
        print(f"head:{head}")
        pos_patterns[index,head] =pos_pattern_presoft.detach().clone()

#%%
pos_patterns = pos_patterns.to('cuda')
head=9
full_pattern = torch.zeros(model.W_E.shape[0],768).to('cuda')
texts = [bible,dnd,bbc_news,Text(java.text+java.text,'Java')]
for text in texts[:2]:
    pattern_ = []
    full_text = text
    tokenized_text = model.to_tokens(text.text).squeeze()
    text = full_text.text
    _,activations = model.run_with_cache(text)
    post_ln1 = activations['blocks.0.ln1.hook_normalized'].squeeze()
    for index in range(1024):
        full_pattern = torch.zeros(model.W_E.shape[0],model.cfg.d_model).to('cuda')
        W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
        W_E_t = W_E*torch.sqrt(torch.tensor(768.0))/ e_normalize

        E_factor = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[(1023-index)//2] ** 2 + (e_normalize**2).squeeze())
        )
        E_factor_b = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[1023-index] ** 2 + (torch.mean(e_normalize[tokenized_text])**2))
        )
        W_E = W_E_t*E_factor.unsqueeze(1)
        W_E_b = W_E_t*E_factor_b.unsqueeze(1)
        W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
        pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
        W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
        pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (torch.mean(e_normalize[tokenized_text])**2).squeeze())
        )

        W_pos = W_pos*(pos_factor.unsqueeze(1))
        for head in [0,2,6,8,9,10,11]:

            weighting =torch.exp(((W_pos[index]+torch.mean(W_E_b[tokenized_text]))@(model.W_Q[0,head]+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)
            pattern_.append(((weighting[tokenized_text][:index]@pos_patterns[1023-index,head,:index])).item())
            current_weight = ((weighting[tokenized_text][:index]@pos_patterns[(1023-index),head,:index]))

        #    full_pattern = full_pattern+((weighting.unsqueeze(1)*W_E)@model.W_V[0,head]@model.W_O[0,head]/(current_weight)).detach().clone()
    plt.scatter([i for i in range(1024)],pattern_,label=f'{full_text.name}')
    plt.ylim(0.0,3.0)
    plt.xlabel('Position in sequence')
    plt.ylabel(r'$\sum_{j=1}^{pos} \text{pos}_{j}f_{x_{j}}$')
    plt.title(f'Head {head}')
plt.legend()
plt.show()

#%%

#%%
pos_patterns = pos_patterns.to('cuda')
head=9
full_pattern = torch.zeros(model.W_E.shape[0],768).to('cuda')
texts = [bible,dnd,bbc_news,Text(java.text+java.text,'Java')]
for text in texts[:2]:
    pattern_ = []
    full_text = text
    tokenized_text = model.to_tokens(text.text).squeeze()
    text = full_text.text
    _,activations = model.run_with_cache(text)
    for index in range(2,1024):
        print(index)
        full_pattern = torch.zeros(model.W_E.shape[0],model.cfg.d_model).to('cuda')
        W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
        W_E_t = W_E*torch.sqrt(torch.tensor(768.0))/ e_normalize

        E_factor = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[(index)//2] ** 2 + (e_normalize**2).squeeze())
        )
        E_factor_b = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[index] ** 2 + (torch.mean(e_normalize[11])**2))
        )
        W_E = W_E_t*E_factor.unsqueeze(1)
        W_E_b = W_E_t*E_factor_b.unsqueeze(1)
        W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
        pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
        W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
        pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (torch.mean(e_normalize[11])**2).squeeze())
        )

        W_pos = W_pos*(pos_factor.unsqueeze(1))
        for head in [head]:

            weighting =torch.exp(((W_pos[index]+torch.mean(W_E_b[11]))@(model.W_Q[0,head]+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)
            pattern_.append(((weighting[tokenized_text][:(1+index)]@pos_patterns[index,head,:(1+index)])).item())
           # current_weight = ((weighting[tokenized_text][:index]@pos_patterns[index,head,:(1+index)]))

        #    full_pattern = full_pattern+((weighting.unsqueeze(1)*W_E)@model.W_V[0,head]@model.W_O[0,head]/(current_weight)).detach().clone()
    plt.scatter([i for i in range(2,1024)],pattern_,label=f'{full_text.name}')
    plt.ylim(0.0,3.0)
    plt.xlabel('Position in sequence')
    plt.ylabel(r'$\sum_{j=1}^{pos} \text{pos}_{j}f_{x_{j}}$')
    plt.title(f'Head {head}')
plt.legend()
plt.show()


#%%
pos_patterns = pos_patterns.to('cuda')
def approx_neuron(text,neuron):

    W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
    e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
    W_E_t = W_E*torch.sqrt(torch.tensor(768.0))/ e_normalize

    tokenized_text = model.to_tokens(dnd.text).squeeze().to('cuda')
    W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
    pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
    pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
    W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
    pos_factor = pos_normalize.squeeze() / (
    torch.mean(torch.sqrt((pos_normalize ** 2 ).squeeze()+ (e_normalize[tokenized_text]**2).squeeze()))
    )

    W_pos = W_pos*(pos_factor.unsqueeze(1))




    real_text=  model.to_tokens(text.text).squeeze().to('cuda')
    pat = []
    for index in range(1,1024):
       # print(pos_patterns[1023-index,head,:index])
        print(index)
        E_factor = e_normalize.squeeze() / (torch.sqrt(pos_normalize[index] ** 2 + (e_normalize**2).squeeze()))
        W_E = (W_E_t*E_factor.unsqueeze(1)).detach().clone()


        W_pos = (model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)).detach().clone()
        W_pos = (W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize).detach().clone()
        pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (e_normalize[tokenized_text[index]]**2).squeeze())
        )

        W_pos = (W_pos*(pos_factor.unsqueeze(1))).detach().clone()


        z = torch.zeros(index,model.cfg.d_model).to('cuda')
        for head in [0,2,6,8,9,10,11]:
            if head == 6:
                weight_factor = torch.tensor(1.0)
            else:
                weight_factor = torch.tensor(1.0)

            weighting =(torch.exp((((W_pos[index]+W_E[tokenized_text[index]])@(model.W_Q[0,head])+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)).detach().clone().to('cuda')
            current_weight = ((weighting[tokenized_text][:index]*pos_patterns[1023-index,head,:index])).detach().clone().to('cuda').sum().detach().clone()
            z+= weight_factor*(((((weighting.unsqueeze(1)*W_E)[real_text]*pos_patterns[1023-index,head,:].unsqueeze(1))[:index]).sum(dim=0)/(current_weight))@model.W_V[0,head]@model.W_O[0,head]).detach().clone()
            z+= weight_factor*((pos_patterns[1024-index,head,:(index)].unsqueeze(1)*W_pos[:(index)]).sum(dim=0))@model.W_V[0,head]@(model.W_O[0,head]).detach().clone()
            z+=weight_factor*model.b_O[0,head]

        z = z+(model.W_pos)[index] + (model.W_E)[tokenized_text[index]]
        print(z)
        del current_weight
        del weighting
        pat.append((z[-1]@model.blocks[0].mlp.W_in[:,neuron]).item()/1.2+model.blocks[0].mlp.b_in[neuron].item()) #+ model.blocks[0].mlp.b_in[neuron]).item())
        del z
    return pat
#%%
pos_patterns = pos_patterns.to('cuda')
def approx_neuron_head(text,neuron,head):

    W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
    e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
    W_E_t = W_E*torch.sqrt(torch.tensor(768.0))/ e_normalize


    tokenized_text = model.to_tokens(bible.text).squeeze().to('cuda')
    W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
    pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
    pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
    W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
    pos_factor = pos_normalize.squeeze() / (
    torch.sqrt((pos_normalize ** 2 ).squeeze()+ (torch.mean(e_normalize[tokenized_text]**2)).squeeze())
    )

    W_pos = W_pos*(pos_factor.unsqueeze(1))


    real_text=  model.to_tokens(text.text).squeeze().to('cuda')
    pat = []
    for index in range(1,1024):
       # print(pos_patterns[1023-index,head,:index])

        E_factor = e_normalize.squeeze() / (torch.sqrt(pos_normalize[index] ** 2 + (e_normalize**2).squeeze()))
        W_E = (W_E_t*E_factor.unsqueeze(1)).detach().clone()


        W_pos = (model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)).detach().clone()
        W_pos = (W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize).detach().clone()
        pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (e_normalize[tokenized_text[index]]**2).squeeze())
        )

        W_pos = (W_pos*(pos_factor.unsqueeze(1))).detach().clone()
        if index == 100:

            weighting =(torch.exp((((W_pos[index]+W_E[tokenized_text[index]])@(model.W_Q[0,head])+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)).detach().clone().to('cuda')
            current_weight = ((weighting[tokenized_text]*pos_patterns[1024-index,head,:])).detach().clone().to('cuda').sum().detach().clone()
            show(((weighting)[real_text]*pos_patterns[1024-index,head,:]).reshape(32,32)/current_weight)

        z = torch.zeros(index,model.cfg.d_model).to('cuda')
        for head in [head]:
            if head == 6:
                weight_factor = torch.tensor(1.0)
            else:
                weight_factor = torch.tensor(1.0)
            weighting =(torch.exp((((W_pos[index]+W_E[tokenized_text[index]])@(model.W_Q[0,head])+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)).detach().clone().to('cuda')
            current_weight = ((weighting[tokenized_text]*pos_patterns[1024-index,head,:])).detach().clone().to('cuda').sum().detach().clone()
            z+= weight_factor*(((((weighting.unsqueeze(1)*W_E)[real_text]*pos_patterns[1024-index,head,:].unsqueeze(1))).sum(dim=0)/(current_weight))@model.W_V[0,head]@model.W_O[0,head]).detach().clone()
            z+= weight_factor*((pos_patterns[1024-index,head,:(index)].unsqueeze(1)*W_pos[:(index)]).sum(dim=0))@model.W_V[0,head]@(model.W_O[0,head]).detach().clone()
            z+=weight_factor*model.b_O[0,head]

       # z = z+(model.W_pos)[index] + (model.W_E)[11]

        del current_weight
        del weighting
        pat.append((z[-1]@model.blocks[0].mlp.W_in[:,neuron]).item()) #+ model.blocks[0].mlp.b_in[neuron]).item())
        del z
    return pat
#%%
head = 8
pos_patterns = pos_patterns.to('cuda')
texts = [bible,dnd,bbc_news,Text(java.text+java.text,'java')]
for text in texts:
    print(text.name)
    pat_ = []
    tokenized_text = model.to_tokens(text.text).squeeze().to('cuda')
    for index in range(1,1024):
        weighting =(torch.exp(((W_pos[index]+W_E)@(model.W_Q[0,head]+model.b_Q[0,head])@model.W_K[0,head].T@(W_E[tokenized_text[:index]].T))/8.0)).detach().clone().to('cuda')
        current_weight = ((weighting*pos_patterns[1023-index,head,:index])).detach().clone().to('cuda').sum(dim=-1).detach().clone()
        pat_.append(torch.mean(current_weight[tokenized_text]).item())
    plt.scatter([i for i in range(1,1024)],pat_,label=f'{text.name}')
plt.legend()

plt.show()

#%%



tokenized_text = model.to_tokens(dnd.text).squeeze().to('cuda')
current_weight = ((weighting[tokenized_text][:index]*pos_patterns[1023-index,head,:index])).detach().clone().to('cuda').sum().detach().clone()



#%%
bbc_news.trace_head_neuron(neuron=704,head=6)
#%%
#print(current_weight)
preds = []
for l in range(100,1024):
    preds.append((full_pattern[model.to_tokens(bbc_news.text).squeeze()]@model.blocks[0].mlp.W_in[:,300])[l-100:l].mean().item()*100)
print(preds)
glitch_token_list =   [' dstg',' Flavoring', ' attm','dayName','\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x7f', '.[', 'ÃÂÃÂ', 'ÃÂÃÂÃÂÃÂ', 'wcsstore', '\\.', ' practition', ' Dragonbound', ' guiActive', ' \u200b', '\\\\\\\\\\\\\\\\', 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', ' davidjl', '覚醒', '"]=>', ' --------', ' \u200e', 'ュ', 'ForgeModLoader', '天', ' 裏覚醒', 'PsyNetMessage', ' guiActiveUn', ' guiName', ' externalTo', ' unfocusedRange', ' guiActiveUnfocused', ' guiIcon', ' externalToEVA', ' externalToEVAOnly', 'reportprint', 'embedreportprint', 'cloneembedreportprint', 'rawdownload', 'rawdownloadcloneembedreportprint', 'SpaceEngineers', 'externalActionCode', 'к', '?????-?????-', 'ーン', 'cffff', 'MpServer', ' gmaxwell', 'cffffcc', ' "$:/', ' Smartstocks', '":[{"', '龍喚士', '":"","', ' attRot', "''.", ' Mechdragon', ' PsyNet', ' RandomRedditor', ' RandomRedditorWithNo', 'ertodd', ' sqor', ' istg', ' "\\', ' petertodd', 'StreamerBot', 'TPPStreamerBot', 'FactoryReloaded', ' partName', 'ヤ', '\\">', ' Skydragon', 'iHUD', 'catentry', 'ItemThumbnailImage', ' UCHIJ', ' SetFontSize', 'DeliveryDate', 'quickShip', 'quickShipAvailable', 'isSpecialOrderable', 'inventoryQuantity', 'channelAvailability', 'soType', 'soDeliveryDate', '龍契士', 'oreAndOnline', 'InstoreAndOnline', 'BuyableInstoreAndOnline', 'natureconservancy', 'assetsadobe', '\\-', 'Downloadha', 'Nitrome', ' TheNitrome', ' TheNitromeFan', 'GoldMagikarp', 'DragonMagazine', 'TextColor', ' srfN', ' largeDownload', ' srfAttach', 'EStreamFrame', 'ゼウス', ' SolidGoldMagikarp', 'ーティ', ' サーティ', ' サーティワン', ' Adinida', '":""},{"', 'ItemTracker', ' DevOnline', '@#&', 'EngineDebug', ' strutConnector', ' Leilan', 'uyomi', 'aterasu', 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', 'ÃÂ', 'ÛÛ', ' TAMADRA', 'EStream']
glitch_tokens = []
for token in glitch_token_list:
    glitch_tokens.append(model.to_tokens(token).squeeze()[-1])

 #%%
big = full_pattern@model.blocks[0].mlp.W_in
big[glitch_tokens]=0.0
indices  =[]
for index in range(3072):
    if torch.max(abs(big[:,index]))>0.1:
        indices.append(index)

print(indices)
for index in indices:
    print(index)
    toks = model.to_str_tokens(((big[:,index]>(0.02)).nonzero().squeeze()))
    full_toks = []
    for tok in toks:
        if len(tok)>4 and ' ' in tok:
            full_toks.append(tok)
    print(full_toks)


# %%
def approximate_neuron(text,neuron):
    tokenized = model.to_str_tokens(text)
    W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)

    W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
    full_approx = []
    pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (e_normalize[11]**2).squeeze())
    )
    W_pos = W_pos*(pos_factor.unsqueeze(1))
    for index in range(len(tokenized)):
        approximation = W_pos[index] +W_pos[]@model.W_V[0]

    return full_approx
#%%
pos_patterns = torch.zeros(model.cfg.n_ctx,model.cfg.n_heads,model.cfg.n_ctx)
for index in range(2,model.cfg.n_ctx):
    for head in range(model.cfg.n_heads):
        W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
        W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
        pos_normalize = torch.norm(W_pos,dim=1).unsqueeze(1)
        e_normalize = torch.norm(W_E, dim=1).unsqueeze(1)
        W_E = W_E * torch.sqrt(torch.tensor(model.cfg.d_model)) / (
        torch.sqrt((pos_normalize[index] ** 2).unsqueeze(0) + (e_normalize**2).squeeze())
    ).T
        W_pos = W_pos*torch.sqrt(torch.tensor(model.cfg.d_model)) / torch.sqrt(pos_normalize[index]**2+(e_normalize[model.to_tokens('the').squeeze()[-1]])**2) # Use ',' as a generic starting token

        queries = ((W_pos[index]+W_E[model.to_tokens('the').squeeze()[-1]]) @ model.W_Q[0, head] + model.b_Q[0,head]).unsqueeze(0)
        keys =  model.W_K[0, head].T @ W_pos[:(1+index)].T  + model.b_K[0, head].unsqueeze(1)
        pos_pattern_presoft = torch.tensor((queries @ keys).squeeze().tolist() + [-torch.inf for i in range(1023-index)])
        pos_patterns[index,head] = torch.softmax(pos_pattern_presoft/model.blocks[0].attn.attn_scale,dim=0)
#%%
import random
random_text = ''
stopword_p = 0.3
for i in range(1024):
    if random.random()< stopword_p:
        random_text += model.to_str_tokens(torch.randint(low=300,high=1000,size=(1,)))[-1]
    else:
        random_text += model.to_str_tokens(torch.randint(low=300,high=50000,size=(1,)))[-1]
random_text = Text(random_text,'random')
# %%
pos_patterns = pos_patterns.to('cuda')
head=0
full_pattern = torch.zeros(model.W_E.shape[0],768).to('cuda')
texts =[]# [bbc_news,bible,Text(fishing_news,'Fishing news'),Text(harry_1+harry_1+harry_1,'harry potter')]#Text(java.text+java.text,'java')]
i=-1
pattern_ = [[[] for head in range(12)] for j in range(len(texts))]
for text in texts:
    i+=1

    full_text = text
    tokenized_text = model.to_tokens(text.text).squeeze()
    text = full_text.text
    _,activations = model.run_with_cache(text)

    for index in range(1024):
        print(index)
        full_pattern = torch.zeros(model.W_E.shape[0],model.cfg.d_model).to('cuda')
        W_E = model.W_E - model.W_E.mean(dim=1).unsqueeze(1)
        W_E_t = W_E*torch.sqrt(torch.tensor(768.0))/ e_normalize

        E_factor = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[(index)] ** 2 + (e_normalize**2).squeeze())
        )
        E_factor_b = e_normalize.squeeze() / (
        torch.sqrt(pos_normalize[index] ** 2 + (torch.mean(e_normalize[model.to_tokens(' the').squeeze()[-1]])**2))
        )
        W_E = W_E_t*E_factor.unsqueeze(1)
        W_E_b = W_E_t*E_factor_b.unsqueeze(1)
        W_pos = model.W_pos - model.W_pos.mean(dim=1).unsqueeze(1)
        pos_normalize = torch.norm(W_pos, dim=1).unsqueeze(1)
        W_pos = W_pos * torch.sqrt(torch.tensor(model.cfg.d_model)) / pos_normalize
        pos_factor = pos_normalize.squeeze() / (
        torch.sqrt((pos_normalize ** 2 ).squeeze()+ (torch.mean(e_normalize[model.to_tokens(' the').squeeze()[-1]])**2).squeeze())
        )

        W_pos = W_pos*(pos_factor.unsqueeze(1))
        for head in [9]:
            weighting =torch.exp((((W_pos[index]+torch.mean(W_E_b[model.to_tokens(' the').squeeze()[-1]]))@(model.W_Q[0,head])+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)
            print(weighting)
            pattern_[i][head].append(((weighting[tokenized_text][:(1+index)]@pos_patterns[index,head,:(1+index)])).item())
            current_weight = ((weighting[tokenized_text][:(1+index)]@pos_patterns[index,head,:(1+index)]))

            full_pattern = full_pattern+((weighting.unsqueeze(1)*W_E)@model.W_V[0,head]@model.W_O[0,head]/(current_weight)).detach().clone()

   # plt.scatter([k for k in range(1024)],pattern_,label=f'{full_text.name}')

   # plt.xlabel('Position in sequence')
   # plt.ylabel(r'$\sum_{j=1}^{pos} \text{pos}_{j}f_{x_{j}}$')
   ## plt.title(f'Head {head}')
#plt.legend()
#plt.show()
#%%
for head in range(12):
    for i in range(5):
        plt.scatter([k for k in range(1024)],pattern_[i][head],label=f'{texts[i].name}')

        plt.xlabel('Position in sequence')
        plt.ylabel(r'$\sum_{j=1}^{pos} \text{pos}_{j}f_{x_{j}}$')
        plt.title(f'Head {head}')

    plt.legend()
    plt.show()
    plt.clf()
# %%
big = full_pattern@model.blocks[0].mlp.W_in
indices = []
for index in range(3072):
    if torch.max(big[:,index])>10.0:
        indices.append(index)

# %%
for index in indices:
    print(index)
    print(model.to_str_tokens((big[:,index]>torch.max(big[:,index])/1.7).nonzero().squeeze()))
# %%
for head in range(12):
    fig = px.imshow(pos_patterns[399,head][:400].reshape(20,20),title="Head 11")
    fig.update_layout(
        autosize=False,
    margin=dict(l=0, r=0, t=0, b=0, pad=1),
)

    fig.show()
