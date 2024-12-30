# %%
import transformer_lens
import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from texts import *

torch.cuda.empty_cache()

model_small = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
#model_first = transformer_lens.HookedTransformer.from_pretrained("stanford-gpt2-small-a", checkpoint_index=10)
model = model_small
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
def trace(neuron,first_layer,second_layer,threshold):
    for first_neuron in ((model.blocks[first_layer].mlp.W_out@model.blocks[second_layer].mlp.W_in[:,neuron])>threshold).nonzero():
        bible.trace_neuron(first_neuron,layer=first_layer)
        code.trace_neuron(first_neuron,layer=first_layer)
        plt.show()
        plt.clf()
#%%
the_tok = model.to_tokens('this').squeeze()[-1]
head = 7
pos_pattern = plt.matshow(((W_pos[1000]@model.W_Q[0,head]@model.W_K[0,head].T@W_pos.T
+W_E[the_tok]@model.W_Q[0,head]@model.W_K[0,head].T@W_pos.T).detach().cpu()/8
)[:1000].reshape(20,50))

plt.colorbar()




#%%




_,activations = model.run_with_cache(genesis)
model.to_str_tokens(model.to_tokens(genesis).squeeze()[(activations['blocks.0.mlp.hook_post'].squeeze()[:,2644]>1.0).nonzero().squeeze()])






#%%
head =10
sume = torch.zeros(model.W_E.shape[0],model.W_O[0,0].shape[1]).to('cuda')
for head in [0,2,6,8,9,10,11]:
    weighting = torch.exp(((W_pos[-100]+torch.mean(W_E[tokenized_text]))@model.W_Q[0,head]+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T)/8.0)
    print(weighting)
    head_weight = ((W_E*weighting.unsqueeze(1))@model.W_V[0,head]@model.W_O[0,head]).detach().clone()
    sume = sume+ (head_weight)/(weighting[model.to_tokens(bible.text).squeeze()].mean())


#big = W_E@model.W_V[0,head]@model.W_O[0,head]@model.blocks[0].mlp.W_in
big=sume@model.blocks[0].mlp.W_in
neurons = []
for index in range(3072):
    if torch.max(big[:,index])>10.0:
        neurons.append(index)
print(neurons)

#%%

neuron = 704
bible.trace_real_neuron(neuron=neuron,head=head)
bible.trace_head_neuron
t= big[:,neuron][model.to_tokens(bible.text).squeeze()]

t =t.unsqueeze(1)+W_pos@model.W_V[0,head]@model.W_O[0,head]@model.blocks[0].mlp.W_in[:,neuron]


prediction = []
for i in range(100,1024):


    prediction.append((t[i-100:i].mean().item()))
    plt.scatter([i for i in range(100,100+len(prediction))],prediction)
    plt.ylim(0,0.7)


#%%
tokenized_text = model.to_tokens(dnd.text).squeeze()
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
        full_pattern = torch.zeros(model.W_E.shape[0],model.cfg.d_model).to('cuda')
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

            weighting =torch.exp((((W_pos[index]+W_E[tokenized_text].mean())@(model.W_Q[0,head])+model.b_Q[0,head])@model.W_K[0,head].T@(W_E.T))/8.0)

            current_weight = ((weighting[model.to_tokens(text).squeeze()].detach().clone()[index-100:index].mean().item()))
            pattern_.append(current_weight)
            full_pattern = full_pattern+((weighting.unsqueeze(1)*W_E)@model.W_V[0,head]@model.W_O[0,head]/(current_weight)).detach().clone()
   # plt.scatter([i for i in range(924)],pattern_,label=f'{full_text.name}')
    #plt.ylim(0.0,2.0)
    #plt.xlabel('Position in sequence')
    #plt.ylabel(r'$\sum_{j=1}^{pos} \text{pos}_{j}f_{x_{j}}$')
    #plt.title(f'Head {head}')
#plt.legend()
#plt.show()


#%%





#%%
def neuron_approx(text,neuron):
    ln2_scale = torch.tensor(1.2)

    model.blocks[0].mlp.W_in

    return



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
indices  =[]
for index in range(3072):

    if torch.max(big[:,index])>10.0:
        indices.append(index)
#%%
print(indices)
for index in indices:

    print(index)
    toks = model.to_str_tokens(torch.topk(big[:,index],k=200).indices)[:200]
    alt_toks = model.to_str_tokens(((big[:,index]>(torch.max(big[:,index]/1.5))).nonzero().squeeze()))[:100]
    less_toks = model.to_str_tokens(((big[:,index]<(torch.min(big[:,index]/1.7))).nonzero().squeeze()))[:100]
    print(alt_toks)
    print('less',less_toks)
#%%
#%%
print(indices)
for index in indices:

    print(index)
    toks = model.to_str_tokens(((big[:,index]>(torch.max(big[:,index]/2.5))).nonzero().squeeze()))
    less_toks = model.to_str_tokens(((big[:,index]<(torch.min(big[:,index]/1.5))).nonzero().squeeze()))
    print(toks)
    print('less',less_toks)
#%%

z_scores = (big[:,1797]-torch.mean(big[:,1797]))/(torch.sqrt(torch.var(big[:,index])))
bad_toks = (abs(z_scores)>20.0).nonzero().squeeze()
#%%
print((weighting[model.to_tokens(bible.text).squeeze(
)].mean()))



print((weighting[model.to_tokens(dnd.text).squeeze()].mean()))


print((weighting[model.to_tokens(java.text).squeeze()].mean()))

print((weighting[model.to_tokens(bbc_news.text).squeeze()].mean()))



print((weighting[model.to_tokens(fishing_news).squeeze()].mean()))
#%%

texts = [java.text,bible.text,dnd.text,bbc_news.text,fishing_news]
for text in texts:
    _,activations = model.run_with_cache(text)
    plt.scatter([i for i in range(len(model.to_str_tokens(text)))],activations['blocks.0.ln2.hook_scale'].detach().clone().squeeze().detach().cpu())
    plt.show()
    plt.clf()
    plt.scatter([i for i in range(len(model.to_str_tokens(text)))],activations['blocks.0.ln1.hook_scale'].detach().clone().squeeze().detach().cpu())

    plt.show()























# %%
duplicate_heads =model.W_pos[200]@(0.175*model.W_V[0,10]@model.W_O[0,10]+0.5*model.W_V[0,5]@model.W_O[0,10]+0.5*model.W_V[0,1]@model.W_O[0,1])
# %%

for neuron in (duplicate_heads@model.blocks[0].mlp.W_in>0.4).nonzero().squeeze():
    bible.trace_neuron(neuron)
    plt.show()
    plt.clf()
#%%

def show(matrix):

    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)

    if matrix.shape[0] > 1500 or matrix.shape[1] > 1500:
        print("too big")
        return

    plt.matshow(matrix.detach().cpu())
#%%
length = 1024
pos_block = W_pos[350]@model.W_V[0,1]@model.W_O[0,1]@model.blocks[0].mlp.W_in
for neuron in (pos_block>1.0).nonzero().squeeze():
    bible.trace_neuron(neuron)
    plt.show()
    plt.clf()
#%%

prompt = bible.text
layer = 0
fig, axs = plt.subplots(2, 4, figsize=(10, 10),sharey='all')
prompts = [bible,dnd,java]
for prompt in prompts:
    _,activations = model.run_with_cache(prompt.text)

    for i in range(2):
        for j in range(4):
            head = 4 * i + j

            attn_patterns= activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head].to('cuda')

            diffs = (attn_patterns[1:]-attn_patterns[:-1]).to('cuda')
            diff_pattern = (diffs**2).sum(dim=-1).to('cuda')
            variance_ = (attn_patterns[1:]**2).sum(dim=-1).to('cuda')
            axs[i,j].scatter([i for i in range(len(model.to_str_tokens(prompt.text))-1)],(diff_pattern).detach().cpu(),label=prompt.name)
            axs[i,j].legend()

#%%
prompt = bible.text
layer = 6
fig, axs = plt.subplots(3, 4, figsize=(10, 10),sharey='all')
prompts = [bible,dnd,bbc_news,code]
for prompt in prompts:
    _,activations = model.run_with_cache(prompt.text)

    for i in range(3):
        for j in range(4):
            head = 4 * i + j

            attn_patterns= activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head]

            diffs = attn_patterns[1:]-attn_patterns[:-1]
            diff_pattern = (diffs**2).sum(dim=-1)
            axs[i,j].set_ylim(-25,0)
            axs[i,j].scatter([i for i in range(min(1000,len(model.to_str_tokens(prompt.text))-1))],torch.log(attn_patterns[-4][-1000:].detach().cpu()),vmax=0,vmin=-15)
            axs[i,j].legend()
#%%
prompt = bible.text
layer = 0
neuron = 2452
fig, axs = plt.subplots(3, 4, figsize=(10, 10),sharey='all')
prompts = [bible,java]
for prompt in prompts:
    _,activations = model.run_with_cache(prompt.text)

    for i in range(3):
        for j in range(4):
            head = 4 * i + j

            prompt.trace_real_neuron(neuron,head,ax=axs[i,j],layer=layer)

layer =1

#%%
layer=0
fig, axs = plt.subplots(3, 4, figsize=(10, 10))
prompts = [bible,dnd]
for i in range(3):
    for j in range(4):
        head = 4 * i + j
        for prompt in prompts:
            _,activations = model.run_with_cache(prompt.text)
            axs[i,j].scatter([i for i in range(prompt.length)],torch.exp(activations[f'blocks.{layer}.attn.hook_attn_scores'].squeeze()[head].detach().cpu()/8).sum(dim=-1).detach().cpu(),marker='x')
#%%
prompt =java.text
_,activations = model.run_with_cache(prompt)
fig, axs = plt.subplots(2, 4, figsize=(10, 10),sharey='all')
layer = 1
for i in range(2):
    for j in range(4):
        head = 4 * i + j

        attn_patterns= activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head]
        diffs = attn_patterns[1:]-attn_patterns[:-1]
        diff_pattern = (diffs**2).sum(dim=-1)
        axs[i,j].scatter([i for i in range(min(1000,len(model.to_str_tokens(prompt))-1))],diff_pattern.detach().cpu()[-1000:])
#%%
_,activations = model.run_with_cache(genesis)
layer = 3
for head in range(12):
    plt.matshow(activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head,-200].detach().cpu().reshape(32,32))
    plt.colorbar()

# %%
prompts = [
    bible,
    dnd,
   # java
]
entropies = torch.zeros(len(prompts),model.cfg.n_heads)
layer = 7
head=8

for l in range(len(prompts)):

    _, activations = model.run_with_cache(prompts[l].text)

    t_act = activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[: ,-2,1:]
    entropy =1/(activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[: ,-2,1:]**2).sum(dim=-1)

    entropies[l] = entropy
    print(activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[: ,-2,0])
    plt.scatter([i for i in range(1024)],activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[head ,:,0].detach().cpu())
plt.matshow(torch.log(entropies))
plt.colorbar()
#%%



_,activations = model.run_with_cache(bible.text)

_,code_activations = model.run_with_cache(dnd.text)

def entropy(pattern):
  return 1/((pattern**2).sum())
head =9
for head in range(12):

  layer = 2
  print(head)
  tokenized_text = model.to_str_tokens(bible.text)
  dnd_text = model.to_str_tokens(dnd.text)
  for index in range(30,31):
    print(tokenized_text[-index])
    print(dnd_text[-index])
    print(entropy(activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head,-index][0:]))
    print(entropy(code_activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head,-index][0:]))

    show(activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head,-index][:].reshape(32,32).detach().cpu())
    show(code_activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head,-index][:].reshape(32,32).detach().cpu())





























#%%

length = 1024
plt.clf()
fig, axs = plt.subplots(3, 4, figsize=(10, 10))
layer = 0
neuron =1168
reverse_bbc = Text(''.join(reversed(model.to_str_tokens(bbc_news.text))),'bbc_reverse')

prompts = [bible]
for text in prompts:
    for i in range(3):
        for j in range(4):
            head = 4 * i + j

            neuron_head = text.trace_head_neuron(
                neuron,
                head=head,
                length=length,
                ax=axs[i, j],
                mean_diff=False,
                layer=layer,
                positional=True,
                marker="x",
            )

            mean_neuron = text.trace_head_neuron(
                neuron,
                head=head,
                length=length,
                ax=axs[i, j],
                mean_diff=True,
                layer=layer,
                positional=True,
                marker=".",
            )
            if head == 0:
                total_neuron = mean_neuron
            else:
                total_neuron += mean_neuron
            #  java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=0,positional=True)

            #        java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=layer,positional=False,marker='.')

            axs[i, j].set_title(f"Head {head}")

plt.show()
plt.clf()
length = 100
for prompt in prompts:
    prompt.trace_neuron(neuron)
#plt.scatter([i for i in range(length)],total_neuron.detach().cpu())
#%%
for prompt in [bible]:
    for neuron in range(100):
        prompt.trace_neuron(neuron)
        plt.show()
        plt.clf()
#%%
neuron =2210
for prompt in prompts+[bbc_news]:
    prompt.trace_neuron(neuron)




#%%


entr_type = "collision"
prompts = [
    bbc_news.text,
    java.text+java.text,
    bible.text+bible.text,
    code.text+code.text+code.text,
    league_of_legends + league_of_legends + league_of_legends,
]

entropies = torch.zeros(len(prompts), model.cfg.n_heads)
layer = 1
for l in range(len(prompts)):

    _, activations = model.run_with_cache(prompts[l])

    t_act = activations["attn", layer].squeeze()[:, -2]

    #  plt.scatter(
    ##      [i for i in range(len(model.to_str_tokens(prompts[l])))],
    #      torch.log(t_act).detach().cpu(),
    #  )
    entropy = entr(
        t_act,
        entr_type,
    )
    plt.show()
    plt.clf()
    entropies[l] = entropy
plt.matshow(torch.log(entropies[:, :]))
plt.colorbar()

# %%
"""
gpt2_vecs = (
    + model.W_V[0, 6] @ model.W_O[0, 6]

    +model.W_V[0,2]@model.W_O[0,2]

    +model.W_V[0,0]@model.W_O[0,0]
    +model.W_V[0,3]@model.W_O[0,3]

)
gpt2_vecs = (
    + model.W_V[0, 6] @ model.W_O[0, 6]
    +model.W_V[0,0]@model.W_O[0,0]
    +model.W_V[0,1]@model.W_O[0,1]
    +model.W_V[0,2]@model.W_O[0,2]
    +model.W_V[0,4]@model.W_O[0,4]
    +model.W_V[0,8]@model.W_O[0,8]
    +model.W_V[0,9]@model.W_O[0,9]
    +model.W_V[0,11]@model.W_O[0,11]
    +model.W_V[0,0]@model.W_O[0,0]
    +model.W_V[0,3]@model.W_O[0,3]
    +model.W_V[0,5]@model.W_O[0,5]
    +model.W_V[0,7]@model.W_O[0,7]
    +model.W_V[0,10]@model.W_O[0,10]

)
"""
gpt2_vecs = (

    +model.W_V[0,0]@model.W_O[0,0]

    +model.W_V[0,2]@model.W_O[0,2]
    +model.W_V[0,6]@model.W_O[0,6]
    +model.W_V[0,9]@model.W_O[0,9]
    +model.W_V[0,10]@model.W_O[0,10]
    +model.W_V[0,11]@model.W_O[0,11]
    +model.W_V[0,8]@model.W_O[0,8]
)
big = (W_E*E_factor.unsqueeze(1)) @ (gpt2_vecs) @ model.blocks[0].mlp.W_in



#big = ((W_E * E_factor.unsqueeze(1)) @ gpt2_vecs) @ model.blocks[0].mlp.W_in
# %%
gpt2 = True
if gpt2:
    vecs = (
        model.W_V[0,0]@model.W_O[0,0]

        +model.W_V[0,2]@model.W_O[0,2]

            +model.W_V[0,6]@model.W_O[0,6]

        +model.W_V[0,8]@model.W_O[0,8]
        +model.W_V[0,9]@model.W_O[0,9]

        + model.W_V[0,10]@model.W_O[0,10]
        + model.W_V[0,11]@model.W_O[0,11]
    )
else:
    vecs = (
        model.W_V[0,0]@model.W_O[0,0]
        +  model.W_V[0,6]@model.W_O[0,6]
         +  model.W_V[0,1]@model.W_O[0,1]
    )

big = ((W_E ) @ vecs) @ model.blocks[1].mlp.W_in
#%%


def get_error(lambda_,c,distr):
    return (abs((distr*torch.exp(lambda_*distr)).sum()-c))


def compute_max_distribution(distr,c):


    lambda_ = torch.tensor(0.0)
    lambda_.requires_grad = True
    optimizer = torch.optim.AdamW([lambda_],lr=1e-2)

    for i in range(20000):
        print(lambda_,'lambda')
        optimizer.zero_grad()
        loss = get_error(lambda_,c,distr)

        loss.backward()
        optimizer.step()

        print(loss,'loss')
    distrib = torch.exp(distr*lambda_)

    return distrib/(distrib.sum())
distrib = compute_max_distribution(big[:,1498].detach()/1024,3)
#%%


big = full_pattern@model.blocks[0].mlp.W_in

indices = []
for index in range(3072):
    if torch.mean(abs(big[:10000, index])) >1.5:



        indices.append(index)
print(indices)


for index in indices:
    threshold = torch.max(big[:20000,index])/1.5
    print(index)

    print(get_toks(index, threshold=th

    reshold, greater=True), "positive tail")

 #   print(get_toks(index, threshold=-threshold, greater=False), "negative tail")



# %%
indices = []
threshold=2
threshold =0.5
for index in [0]:

    print(get_toks(index, threshold=threshold, greater=True), "positive tail")

    print(get_toks(index, threshold=-threshold, greater=False), "negative tail")

print(indices)

# %%


indices = []
for index in range(2048):
    if torch.max(abs(big[:, index])) >5.0:



        indices.append(index)
print(indices)

# %%
indices = []
threshold=2
threshold =0.5
for index in [0]:

    print(get_toks(index, threshold=threshold, greater=True), "positive tail")

    print(get_toks(index, threshold=-threshold, greater=False), "negative tail")

print(indices)

#%%

for index in indices:
    threshold = torch.max(big[:,index])/1.3
    print(index)

    print(get_toks(index, threshold=threshold, greater=True), "positive tail")

    print(get_toks(index, threshold=-threshold, greater=False), "negative tail")


# %%

# %%
neuron = 1797
length = 400
# length = min(length,bible_code.length,bible.length)
bible.trace_neuron(neuron, length)

# bible_code.trace_neuron(neuron,length)

plt.legend()
plt.show()
plt.clf()
# bible_diff = bible_code.activations['blocks.0.mlp.hook_pre'].squeeze()[:length,neuron]-bible.activations['blocks.0.mlp.hook_pre'].squeeze()[:length,neuron]
plt.scatter(
    [i for i in range(length)],
    bible_diff.detach().cpu(),
    label=f"Pre-activation difference of '{bible_code.name}' and '{bible.name}' ",
)
plt.xlabel("Position in sequence")
plt.ylabel(f"Pre-activation value of neuron {neuron}")
plt.legend()
plt.show()
# %%
neuron = 9
java.trace_neuron(neuron, length=300)
code.trace_neuron(neuron, length=300)
#HHHEEREEEERERERERERERERERERERERERERERERE
#
#
#
#
#
##
#
#

#
#
#
########################################
# %%
neuron = 2644
layer = 0
length = 1300
for head in range(model.cfg.n_heads):
    print(f"head:{head}")
    bible.trace_head_neuron(neuron, head, length=length, full_ov=False, layer=layer)
    java.trace_head_neuron(neuron, head, length=length, full_ov=False, layer=layer)
    plt.legend()
    plt.show()
    plt.clf()
print("trace of neuron:")
bible.trace_neuron(neuron, length, layer=layer)

java.trace_neuron(neuron, length, layer=layer)
# %%
length = 2048

layer = 0
neuron = 2002
#reverse_bbc = text(''.join(reversed(model.to_str_tokens(bbc_news.text))),'bbc_reverse')
#%%

moving_window = -1
fantasy_text = """
Overcoming the Monster.(      ) The protagonist must defeat an antagonist (usually an individual, force, or entity) that threatens them and the wider world.
Rags to Riches. The protagonist achieves something they lack, loses what they’ve gained, and then gets it back again.
The Quest. The protagonist must set out in pursuit of a treasure, place, or other goal, overcoming challenges along the way.
Voyage and Return. The protagonist travels to a strange new place, experiences hardships and makes discoveries, and then returns home with the lessons they have learned.
Comedy. The protagonist experiences a series of lighthearted or confusing events, before the story resolves into a happy ending.
Tragedy. The protagonist has a central trait or flaw or makes a mistake, which results in catastrophe.
Rebirth. The protagonist undergoes a transformation, and often ends up a better person as a result."""
fantasy=  Text(fantasy_text,'fantasy',moving_window)


league = Text(league_of_legends, "league", moving_window)
dnd = Text(dnd_text, "dnd", moving_window)
bible = Text(genesis, "bible", moving_window)
fishing = Text(fishing_news, "Fishing news story", moving_window)
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
java = Text(java_text, "code", moving_window)
bbc_news = Text(bbc_text, "bbc", moving_window)
woods_text = """
Trees lock up carbon as they grow, but carbon exchange also occurs in the soil. Carbon is added to the soil through plant litter and released by fungi and organisms known as decomposers. If trees are planted on soils already rich in organic carbon, it tips the balance so the soils release more carbon than the young trees can lock up over the coming decades.

Lead author of the research, Dr Nina Friggens, explains the implications. “Tree planting can increase carbon stocks in certain areas and ecological contexts,” she says. “But it is important to understand where in the landscape this approach is best deployed to achieve the best results for climate change mitigation.”


Video

How trees capture and store carbon
00:02:29

Photosynthesis is made simple as you take a journey into the leaf of a tree and discover how trees capture and store carbon.

See how it works
Planting in the right places
This is new evidence from the UK for a view long-held by conservationists and the Woodland Trust: there are places where it isn’t appropriate to expand woodland cover. Following lessons learnt in forestry, the UK Forestry Standard now prohibits planting trees on peat deeper than 50cm in the UK. In light of their research, the authors recommend that current policy should be reviewed and tightened to give greater protection to any soils with organic surface layers less than 50cm thick.

The UK has committed to reduce greenhouse gas emissions to net zero by 2050. That means we can’t afford to plant trees in soils that release more carbon than the tree can absorb in a generation. We need to plant them where they will be most effective.

We need landscapes rich in native woods, trees and wildlife to tackle the climate and nature crises.



"""
woods = Text(woods_text,"woods",moving_window)
legal_text = """
(o),or(p)forpurposesofthissubsection,theSecretaryshall--``(i)adjusttheapplicationofsuchprovisiontoensuretheprovisionis


"""
legal = Text(legal_text,"legal",moving_window)
neuron = 2238
layer=0
length=model.cfg.n_ctx
prompts = [java,bible]

fig, axs = plt.subplots(3, 4, figsize=(10, 10))
for prompt in prompts:
    text = prompt
    relevant_heads = [0,2,6,9,8,10,11,7]
    for i in range(3):
        for j in range(4):
            head = 4 * i + j

            mean_neuron = text.trace_head_neuron(
                neuron,
                head=head,
                length=length,

                mean_diff=True,
                layer=layer,
                positional=True,
                marker=".",
                plot=False
            )
            plt.show()
            plt.clf()
            if head in relevant_heads:
                if head == 0:
                    total_neuron = mean_neuron
                else:
                    total_neuron += mean_neuron

            axs[i, j].set_title(f"Head {head}")
    total_neuron = (total_neuron + ((pos_normalize*W_pos) @ model.blocks[0].mlp.W_in[:, neuron])[: text.length]/torch.sqrt(torch.tensor(model.cfg.d_model)))/1.2+ model.blocks[0].mlp.b_in[neuron]
    text.trace_neuron(neuron,marker='x',label=f'Real output of neuron {neuron} for {text.name}')

    plt.scatter([i for i in range(text.length)], total_neuron.detach().cpu(),label=f'Mean approximation of neuron {neuron} for {text.name}')

    for ax in axs.flat:
        ax.set(xlabel="Position", ylabel=f"Pre-activation contribution to neuron {neuron}")


# %%

java.trace_neuron(neuron)


# %%
text = genesis
vecs = torch.zeros(model.cfg.n_heads, model.cfg.d_model)
for head in range(model.cfg.n_heads):
    vecs[head] = (
        W_E[model.to_tokens(text).squeeze()].squeeze().mean(dim=0)
        @ model.W_V[0, head]
        @ model.W_O[0, head]
    )
show(
    ((vecs @ vecs.T) / (torch.norm(vecs, dim=1).unsqueeze(0)))
    / (torch.norm(vecs, dim=1).unsqueeze(0).T)
)
# %%
layer = 0
for head in range(model.cfg.n_heads):

    bible.trace_first_attention(head, layer=layer, length=1000)

    code.trace_first_attention(head, layer=layer, length=1000)
    plt.show()
    plt.clf()
# %%

fig, axs = plt.subplots(3, 4, figsize=(10, 10), sharex="all", sharey="all")
layer = 0
for i in range(3):
    for j in range(4):
        head = 4 * i + j

        bible.trace_first_attention(head, ax=axs[i, j], layer=layer, length=500)

        comments.trace_first_attention(head, ax=axs[i, j], layer=layer, length=500)

        #  java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=0,positional=True)

        #        java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=layer,positional=False,marker='.')

        axs[i, j].set_title(f"Head {head}")

for ax in axs.flat:
    ax.set(xlabel="Position", ylabel=f"Pre-activation contribution to neuron {neuron}")


# %%
for head in range(model.cfg.n_heads):
    show(
        bible_activation["blocks.0.attn.hook_pattern"]
        .squeeze()[head][-2]
        .reshape(32, 32)
    )

# %%
plt.clf()
fig, axs = plt.subplots(3, 4, figsize=(10, 10))
text = fishing
_, activations = model.run_with_cache(text.text)
for i in range(3):
    for j in range(4):
        head = 4 * i + j

        axs[i, j].imshow(
            activations["blocks.0.attn.hook_pattern"]
            .squeeze()[head][-2]
            .reshape(32, 64)
            .detach()
            .cpu()
        )
        #  java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=0,positional=True)

        #        java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=layer,positional=False,marker='.')

        axs[i, j].set_title(f"Head {head}")

for ax in axs.flat:
    ax.set(xlabel="Position", ylabel=f"Pre-activation contribution to neuron {neuron}")


# %%
head1 = 10
head2 = 10
index = 11
text = code
_, activations = model.run_with_cache(text.text)
print(model.to_str_tokens(text.text)[-index])
print(
    model.to_str_tokens(
        (
            activations["blocks.0.attn.hook_pattern"].squeeze()[head1][-index] > 0.02
        ).nonzero()
    )
)

print(
    model.to_str_tokens(
        (
            activations["blocks.0.attn.hook_pattern"].squeeze()[head2][-index] > 0.02
        ).nonzero()
    )
)

show(
    (
        activations["blocks.0.attn.hook_pattern"].squeeze()[head1][-index]
        + activations["blocks.0.attn.hook_pattern"].squeeze()[head2][-index]
    )[:400].reshape(20, 20)
)


# %%
head = 10
show(
    activations["blocks.0.attn.hook_pattern"]
    .squeeze()[head][-2]
    .reshape(32, 32)
    .detach()
    .cpu()
)


#%%
#111111s

layer = 0
head = 1
fig,axs = plt.subplots(3,3,figsize=(10,10))
neurons = (torch.norm(W_pos@model.W_V[layer,head]@model.W_O[layer,head]@model.blocks[layer].mlp.W_in,dim=0)>30.0).nonzero().squeeze()
for i in range(3):
    for j in range(3):
        neuron = neurons[3*i+j]

        axs[i,j].scatter([i for i in range(1024)],(W_pos@model.W_V[layer,head]@model.W_O[layer,head]@model.blocks[0].mlp.W_in[:,neuron]).detach().cpu())
        axs[i,j].set_xlabel("Position in sequence")
        axs[i,j].set_ylabel(f"PVO contribution")
        axs[i,j].set_title(f"Neuron {neuron}")
plt.show()

#%%
fig, axs = plt.subplots(3, 4, figsize=(10, 10),sharey='all')
prompts = [bible]
for neuron in [703]:





    for prompt in prompts:
        _,activations = model.run_with_cache(prompt.text)

        for i in range(3):
            for j in range(4):
                head = 4 * i + j

                prompt.trace_real_neuron(neuron,head,ax=axs[i,j],layer=layer)


# %%

travel = """trek,
journey,
trip,
tour,
voyage,
roam,
wander,
pilgrimage,
sail,
migrate,
fly,
ride,
cruise,
peregrinate,
drive,
road-trip,
bus,
hop,
traipse,
rove,
jaunt,
cab,
knock,
motor,
galavant,
navigate,
coach,
jet,
ramble,
gallivant,
gig,
perambulate,
trund1le,
roll,
barnstorm""".replace(
    "\n", ""
).split(
    ","
)

# %%
for i in travel:
    print(model.to_str_tokens(i)[-1])
# %%


love_phrase = " I love you" + genesis[:500]
hate_phrase = " I hate you" + genesis[:500]


def get_vec(text, layer, index):
    _, activations = model.run_with_cache(text)
    return activations[f"blocks.{layer}.hook_resid_pre"].squeeze()[index]


def get_mid_vec(text, layer, index):
    _, activations = model.run_with_cache(text)
    return activations[f"blocks.{layer}.hook_resid_mid"].squeeze()[index]


# %%
injection_index = 1
#love_vec = (love - hate) * 1500
#love_vec_mid = (love_mid - hate_mid) * 10000
injection_layer = 0
gpt2_vecs = (
     model.W_V[0,5]@model.W_O[0,5]
)






def hook_fn(activation, hook, layer, vec, index,scale):

    for i in range(0,activation.shape[1]):
        activation[:, index] = activation[:, index] + 2*scale*vec/(activation.shape[1])

    return activation


bible_vector = 3*(model.blocks[0].mlp.W_out[2489])
vec = bible_vector
model.reset_hooks()
model.add_hook(
    name=f"blocks.{injection_layer}.hook_resid_pre",
    hook=lambda activation, hook: hook_fn(
        activation=activation, hook=hook, layer=injection_layer, vec=vec, index=injection_index,scale=torch.tensor(1.0)
    ),
)
#%%

text = "I I I I I I I I I I II I I I I I I I I I I II I french "
str_ = text
for i in range(100):
    logits, activations = model.run_with_cache(str_)
    tok = model.to_str_tokens(
        torch.multinomial(
            torch.softmax(logits.squeeze()[-1], dim=-1), num_samples=1
        ).squeeze()
    )[0]
    str_ = str_ + tok
print(str_)# %%


# %%

fantasy = """ Tyre of ClinA half-divided pathway. Hints: None Uskayaw 2 has 3 statues of grab-and-run Tomb:3. Neither version of glass says a - ofstat buffer Hex: casts - block: Call RelentThrut Stlth B - a cursed +3 chain mail of blasphemy {god gift} Jewellery a - a cursed +10 ring of sustain abilities m - the ring of the Ninjas {rPois rF+ Int+5} o - the ring of Rodolfuary f - a +1 ring of slaying b - a +5 crystal plate armour (worn) b - the ring of Brucea {+/*Tele Int+7} (You bought it in a shop on level 1 of the Lair of Beasts) """
texts = [genesis]
indices=  []
str_tok = model.to_str_tokens(genesis)
index = 300

layer = 4

phrase = " Science fiction"
head = 3
for text in texts:
    _,activation = model.run_with_cache(text[:index]+phrase)
    print(model.to_str_tokens(text[:index]+phrase)[-20:])
    plt.matshow(activation[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head][-50:].detach().cpu()[:,-50:])
    plt.colorbar()
 # %%
for text in texts:
    _,activation = model.run_with_cache(text[:index]+phrase)

    for i in range(len(model.to_str_tokens(text[:index]+phrase))):

        similar_words = (activation['blocks.0.attn.hook_pattern'].squeeze()[6][i] > 0.1).nonzero().squeeze()

        if len(similar_words.shape) >=1 and len(similar_words)>=1:

            print(model.to_str_tokens(model.to_tokens(text[:index]+phrase).squeeze()[similar_words]))


# %%
def get_close_bigrams(tok):
    return model.to_str_tokens(torch.topk(-torch.abs((torch.exp(W_E[tok]@(model.W_Q[0,4]@model.W_K[0,4].T@+model.W_Q[0,3]@model.W_K[0,3].T+model.W_Q[0,7]@model.W_K[0,7].T)@(W_E[tok].unsqueeze(0).T-W_E.T))),k=30).indices)
#%%
new_ = []
big_index = (activation['blocks.0.attn.hook_pattern'].squeeze()[0][-1]>0.03).nonzero().squeeze()
big_list = list(big_index)
for i in range(len(activation['blocks.0.attn.hook_pattern'].squeeze()[4][-1])):
    if not i in big_list:
        new_.append(i)
new_ = torch.tensor(new_)
plt.scatter([i for i in range(len(activation['blocks.0.attn.hook_pattern'].squeeze()[4][-1][new_]))],torch.log(activation['blocks.0.attn.hook_pattern'].squeeze()[10][-1][new_]).detach().cpu())
# %%
head = 9
index_range = (3000,3100)
second_range = (1000,1100)
qk = W_E[index_range[0]:index_range[1]]@model.W_Q[0,head]@model.W_K[0,head].T@W_E[index_range[0]:index_range[1]].T
show(W_E[index_range[0]:index_range[1]]@model.W_Q[0,head]@model.W_K[0,head].T@W_E[second_range[0]:second_range[1]].T-qk.diag().unsqueeze(0).T)
# %%
ov_7 = model.W_V[0,7]@model.W_O[0,7]
range_ = (0,20000)
length = range_[1]-range_[0]
qk_7 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,7]@model.W_K[0,7].T))@W_E[range_[0]:range_[1]].T
qk_3 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,3]@model.W_K[0,3].T))@W_E[range_[0]:range_[1]].T
qk_4 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,4]@model.W_K[0,4].T))@W_E[range_[0]:range_[1]].T
mat_7 = (qk_7-qk_7.diag().unsqueeze(0))[:length,:length]
mat_3 = (qk_3-qk_3.diag().unsqueeze(0))[:length,:length]
mat_4 = (qk_4-qk_4.diag().unsqueeze(0))[:length,:length]
qk_7 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,7]@model.W_K[0,7].T))@W_E[range_[0]:range_[1]].T
mat_7 = (qk_7-qk_7.diag().unsqueeze(0))[:length,:length]
#%%
patterns_7 = set([tuple(x) for x in (torch.logical_and((30.0>mat_7) ,(mat_7>20.0))).nonzero().tolist()])
patterns_4 = set([tuple(x) for x in (torch.logical_and((-5.0>mat_4) ,(mat_4>-10.0))).nonzero().tolist()])
patterns_3 = (torch.logical_and((-5.0>mat_3) ,(mat_3>-10.0))).nonzero()
(torch.exp(mat_7)*torch.exp(mat_4)*torch.exp(mat_3)>10.0).nonzero()
#%%
#neuron = 300
#x = {}
for i in range(len(patterns_3)):
    row = patterns_3[i]
    if tuple([x.item() for x in row]) in patterns_4 and tuple([x.item() for x in row]) in patterns_7:
        phrase = model.to_str_tokens(torch.tensor(range_[0]+row[1]))[-1]+model.to_str_tokens(torch.tensor(range_[0]+row[0]))[-1]
        if phrase.startswith(" ") and len(model.to_str_tokens(phrase)) == 3:

                print(phrase)
          #  first_embedding = W_E[range_[0]+row[1]]@(model.W_V[0,3]@model.W_O[0,3]+model.W_V[0,4]@model.W_O[0,4]+model.W_V[0,7]@model.W_O[0,7])
           # second_embedding = W_E[range_[0]+row[0]]@(model.W_V[0,1]@model.W_O[0,1]+model.W_V[0,5]@model.W_O[0,5])
           ## neuron_contrib = (first_embedding+second_embedding)@model.blocks[0].mlp.W_in[:,neuron]
           ## x[phrase] = neuron_contrib.item()

#def
# embed(row):


#%%


row_list = torch.logical_and(mat_7>30.0,torch.logical_and(mat_4>0.0,mat_3>0.0)).nonzero().tolist()
#%%
text = []
neuron = 0
x = {}
for i in range(30000):
    text.append(model.to_str_tokens(torch.tensor(i)))
print(len(row_list))
j = 0
for i in row_list:
    row = i
    j+=1
    if j%100 == 0:
        print(j)
    phrase = text[range_[0]+row[1]][0]+text[range_[0]+row[0]][0]



    first_embedding = W_E[range_[0]+row[1]]@(model.W_V[0,3]@model.W_O[0,3]+model.W_V[0,4]@model.W_O[0,4]+model.W_V[0,7]@model.W_O[0,7])
    second_embedding = W_E[range_[0]+row[0]]@(model.W_V[0,1]@model.W_O[0,1]+model.W_V[0,5]@model.W_O[0,5])
    neuron_contrib = (first_embedding+second_embedding)@model.blocks[1].mlp.W_in[:,neuron]
    x[phrase] = neuron_contrib.item()

#%%
for i in x.keys():
    if x[i] > 2:
        print(i,x[i])

#bigram_embeddings = torch.zeros(len(patterns),768).to('cuda')
#for i in range(len(patterns)):
 #   row = patterns[i]
  #  second = (W_E[row[0]]@model.W_V[0,3])@model.W_O[0,3]/(1+torch.exp(mat[row[0]][row[1]]))
   # first = ((W_E[row[1]]@model.W_V[0,3])@model.W_O[0,3])*torch.exp(mat[row[0]][row[1]])/(1+torch.exp(mat[row[0]][row[1]]))
    #bigram_embeddings[i] = (first+second)
#big = bigram_embeddings@model.blocks[0].mlp.W_in
#%%
indices = []
for index in range(3072):
    if torch.max(big[:, index]) >0.15:

        indices.append(index)
print(indices)
for index in indices:
    big_bigrams = (big[:,index]>0.15).nonzero().squeeze()
    if len(big_bigrams.shape)>0:
        print(model.to_str_tokens(patterns[big_bigrams][:,1]))
        print(model.to_str_tokens(patterns[big_bigrams][:,0]))
#%%
for index in [1990]:
    big_bigrams = (big[:,index]>0.15).nonzero().squeeze()
    if len(big_bigrams.shape)>0:
        print(model.to_str_tokens(patterns[big_bigrams][:,1]))
        print(model.to_str_tokens(patterns[big_bigrams][:,0]))
# %%
for row in patterns:
    if row[0]!=row[1]:
        print(model.to_str_tokens(row[1]),model.to_str_tokens(row[0]))
# %%

def hook_fn(activation,hook):
    for head in range(model.cfg.n_heads):
        if (1/(activation.squeeze()[head,-1]**2).sum()) >40:
            activation[:,head] =  0.0
    return activation
model.reset_hooks()
model.add_hook('blocks.0.attn.hook_pattern',hook_fn)

model.add_hook('blocks.1.attn.hook_pattern',hook_fn)

model.add_hook('blocks.2.attn.hook_pattern',hook_fn)

model.add_hook('blocks.3.attn.hook_pattern',hook_fn)

#str_ = "
for i in range(200):
    logits, activations = model.run_with_cache(str_)
    tok = model.to_str_tokens(
        torch.multinomial(
            torch.softmax(0.5*logits.squeeze()[-1], dim=-1), num_samples=1
        ).squeeze()
    )[0]
    str_ = str_ + tok
print(str_)


# %%









# %%

# %%
injection_index = 1
#love_vec = (love - hate) * 1500
#love_vec_mid = (love_mid - hate_mid) * 10000
injection_layer = 0
gpt2_vecs = (
   model.W_V[0,9]@model.W_O[0,9]
   +model.W_V[0,6]@model.W_O[0,6]
   +model.W_V[0,2]@model.W_O[0,2]
   +model.W_V[0,8]@model.W_O[0,8]
   +model.W_V[0,10]@model.W_O[0,10]
   +model.W_V[0,11]@model.W_O[0,11]
   +model.W_V[0,0]@model.W_O[0,0]
   +model.W_V[0,7]@model.W_O[0,7]
   +torch.eye(768).to('cuda')
)






def hook_fn(activation, hook, layer, vec, index,scale):

    for i in range(0,activation.shape[1]):
        activation[:, i] = activation[:, i] + scale*vec/(activation.shape[1])

    return activation


bible_vector = model.blocks[0].mlp.W_out[2644]*5
vec = bible_vector
model.reset_hooks()
model.add_hook(
    name=f"blocks.{injection_layer}.hook_resid_pre",
    hook=lambda activation, hook: hook_fn(
        activation=activation, hook=hook, layer=injection_layer, vec=vec, index=injection_index,scale=torch.tensor(1.0)
    ),
)
#%%

text = "John and Mary went to the RandomRedditor and"
str_ = text
len_ = 500
for i in range(len_):
    logits, corrupted_activations = model.run_with_cache(str_)

    tok = model.to_str_tokens(
        torch.multinomial(
            torch.softmax(logits.squeeze()[-1], dim=-1), num_samples=1
        ).squeeze()
    )[0]
    if i ==len_-1:
        break
    str_ = str_ + tok
print(str_)# %%
model.reset_hooks()

__,activations = model.run_with_cache(str_)
#%%
def dist(mat):
    return (mat**2).sum(dim=-1)

index = 429
entropies = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
for layer in range(12):
    entropies[layer] = dist(
        (
            activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[:, -index]
            - corrupted_activations[f"blocks.{layer}.attn.hook_pattern"].squeeze()[
                :, -index
            ]
        )
    )
show(entropies)
# %%
# %%
phrase_1 = ' visited'
phrase_2 = ' Monday'
for index in range(499,500):
    _,activations_1= model.run_with_cache(dnd.text[:index]+phrase_1)
    _,activations_2 =  model.run_with_cache(dnd.text[:index]+phrase_2)

    plt.matshow((activations_1['blocks.0.mlp.hook_post'].squeeze()[-1]-activations_2['blocks.0.mlp.hook_post'].squeeze()[-1]).reshape(48,64))

#%%
indices = []
big_second = second@model.blocks[1].mlp.W_in
for index in range(3072):
    if torch.max(first@(model.blocks[1].mlp.W_in[:,index])) > 3.5:
        indices.append(index)
print(indices)
# %%
gpt2_vecs = (
   model.W_V[0,9]@model.W_O[0,9]
   +model.W_V[0,6]@model.W_O[0,6]
   +model.W_V[0,2]@model.W_O[0,2]
   +model.W_V[0,8]@model.W_O[0,8]
   +model.W_V[0,10]@model.W_O[0,10]
   +model.W_V[0,11]@model.W_O[0,11]
   +model.W_V[0,0]@model.W_O[0,0]

)
#%%
layer1_vecs=  (model.W_V[0,3]@model.W_O[0,3]
              + model.W_V[0,7]@model.W_O[0,7]
               +model.W_V[0,8]@model.W_O[0,8]
               +model.W_V[0,9]@model.W_O[0,9]
               +model.W_V[0,10]@model.W_O[0,10]
               +model.W_V[0,2]@model.W_O[0,2]
               +model.W_V[0,1]@model.W_O[0,1]
               +model.W_V[0,4]@model.W_O[0,4]
               +model.W_V[0,5]@model.W_O[0,5]
               +model.W_V[0,6]@model.W_O[0,6]

               )
ov_7 = model.W_V[0,7]@model.W_O[0,7]
range_ = (5000,20000)
length = range_[1]-range_[0]
qk_7 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,7]@model.W_K[0,7].T))@W_E[range_[0]:range_[1]].T
qk_3 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,3]@model.W_K[0,3].T))@W_E[range_[0]:range_[1]].T
qk_4 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,4]@model.W_K[0,4].T))@W_E[range_[0]:range_[1]].T
mat_7 = (qk_7-qk_7.diag().unsqueeze(0))[:length,:length]
mat_3 = (qk_3-qk_3.diag().unsqueeze(0))[:length,:length]
mat_4 = (qk_4-qk_4.diag().unsqueeze(0))[:length,:length]
qk_7 = (W_E[range_[0]:range_[1]]@(model.W_Q[0,7]@model.W_K[0,7].T))@W_E[range_[0]:range_[1]].T
mat_7 = (qk_7-qk_7.diag().unsqueeze(0))[:length,:length]
patterns_7 = (mat_7>50.0).nonzero()
patterns_3 = (mat_3>-20.0).nonzero()
patterns_4 = (mat_4>13.0).nonzero()
# %%
device='cuda'

second = W_E@(gpt2_vecs+model.W_V[0,3]@model.W_O[0,3]+model.W_V[0,4]@model.W_O[0,4]+model.W_V[0,7]@model.W_O[0,7])
first = (W_E*e_normalize/torch.sqrt(torch.tensor(768.0)))@(torch.eye(768).to(device))+W_E@(gpt2_vecs+model.W_V[0,5]@model.W_O[0,5]+model.W_V[0,1]@model.W_O[0,1])
second = second@layer1_vecs
first = first@layer1_vecs
#%%
import random
bigrams = []
threshold = 1
patterns_3 = (mat_4>0.0).nonzero()
list_ = patterns_3.tolist()
#%%
for i in range(len(list_)):
    row =list_[i]
    print(i)
    if mat_7[row[0]][row[1]]>50.0:

        bigrams.append(row)
#%%
indices = []
big_second = second@model.blocks[1].mlp.W_in
for index in range(3072):
    if 4>torch.max((first)@(model.blocks[1].mlp.W_in[:,index])) >2 and  4>torch.max((second)@(model.blocks[1].mlp.W_in[:,index]))>2:
        indices.append(index)
print(indices)

#%%
for index in indices:
    bigram_list = []
    threshold = (torch.max((first)@(model.blocks[1].mlp.W_in[:,index]))+torch.max((second)@(model.blocks[1].mlp.W_in[:,index])))/3
    print(index)
    for bigram in bigrams:
        row = bigram
        if (second[row[1]]+first[row[0]]) @ model.blocks[1].mlp.W_in[:,index] > threshold:
            bigram_list.append(model.to_str_tokens(torch.tensor(1000+row[1]))[-1]+model.to_str_tokens(torch.tensor(1000+row[0]))[-1])
    print(bigram_list)
# %%
import torch

import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px


def show(matrix):

    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)

    if matrix.shape[0] > 1500 or matrix.shape[1] > 1500:
        print("too big")
        return

    px.imshow(matrix.detach().cpu()).show()


def plotcdf(mat, bins=100):
    pdf, edges = np.histogram(mat.detach().cpu(), bins=bins)
    centers = edges[1:] - np.diff(edges) / 2
    cdf = np.cumsum(pdf) / np.sum(pdf)

    plt.figure(1)

    plt.plot(centers, pdf)


def entr(mat, type):
    if type == "shannon":
        zero_padding = (mat[0] > 0).sum()
        mat_ = mat[:, :zero_padding]
        return torch.sum(mat_ * torch.log(1 / mat_), dim=1) / (torch.log(zero_padding))
    elif type == "collision":

        zero_padding = (mat[0] > 0).sum()
        mat_ = mat[:, :zero_padding]

        return 1 / ((torch.sum(mat_**2, dim=1)))
    return


def kl(mat1, mat2):
    zero_padding = (mat1[0] > 0).sum()
    mat1_ = mat1[:, :zero_padding]
    mat2_ = mat2[:, :zero_padding]
    return torch.sum((mat1_ - mat2_) ** 2, dim=1)


java_tutorial = """
Parameters and local variables are allocated on the stack (with reference types, the object lives on the heap and a variable in the stack references that object on the heap). The stack typically lives at the upper end of your address space and as it is used up it heads towards the bottom of the address space (i.e. towards zero).

Your process also has a heap, which lives at the bottom end of your process. As you allocate memory, this heap can grow towards the upper end of your address space. As you can see, there is a potential for the heap to "collide" with the stack (a bit like tectonic plates!!!).

The common cause for a stack overflow is a bad recursive call. Typically, this is caused when your recursive functions doesn't have the correct termination condition, so it ends up calling itself forever. Or when the termination condition is fine, it can be caused by requiring too many recursive calls before fulfilling it.

However, with GUI programming, it's possible to generate indirect recursion. For example, your app may be handling paint messages, and, whilst processing them, it may call a function that causes the system to send another paint message. Here you've not explicitly called yourself, but the OS/VM has done it for you.

To deal with them, you'll need to examine your code. If you've got functions that call themselves then check that you've got a terminating condition. If you have, then check that when calling the function you have at least modified one of the arguments, otherwise there'll be no visible change for the recursively called function and the terminating condition is useless. Also mind that your stack space can run out of memory before reaching a valid terminating condition, thus make sure your method can handle input values requiring more recursive calls.

If you've got no obvious recursive functions then check to see if you're calling any library functions that indirectly will cause your function to be called (like the implicit case above).


"""

code_text = """
torch.set_default_device("cuda")
dims = 500
linear = torch.rand(model.embed.W_E.shape[1], dims).to("cuda")
bias = 100.0 * torch.ones(dims).to("cuda")

optim = torch.optim.AdamW([linear, bias], lr=1e-1)


def entr(intermediate):
    intermediate_comp = intermediate[intermediate > 10 ** (-7)]
    intermediate_comp = intermediate_comp / (intermediate_comp.sum())
    return torch.sum(intermediate_comp * torch.log(1 / intermediate_comp))



model.embed.W_E.requires_grad = False
for param in model.parameters():
    param.requires_grad = False
linear.requires_grad = True
bias.requires_grad = True
epochs = 10000
for epoch in range(epochs):

    optim.zero_grad()
    intermediate = torch.nn.ReLU()((model.embed.W_E) @ linear + bias)
    reconstruction = interme
    diate @ linear.T
    loss__ = torch.norm(reconstruction - (model.embed.W_E)) + 0.1 * abs(
        entr(intermediate)
    )

    print(loss__)
    loss__.backward()
    optim.step()


bias_1 = model.blocks[0].ln1.b
weight_1 = model.blocks[0].ln1.w

W_El = model.embed.W_E
# show the figure; this was slow
"""
political = """For today’s post, I’d like to take a look at California’s voter initiative to legalize pot. If the measure passes, and the sky doesn’t fall, many other states will probably be looking at similar law changes in the near future. Our drug policy of the last century has simply not worked, and it’s heartening to see a state attempting to legalize marijuana.

The statistics on marijuana arrests are really shocking. According to the Drug Policy Alliance, which is in favor of legalization, blacks are arrested for marijuana possession between four and twelve times more than whites in California, even though studies have consistently shown that whites smoke more pot than blacks. In the last ten years, around 500,000 people have been arrested for possession. That’s absurd! Think about how expensive that is for the criminal justice system. California spends $216,000 for each juvenile inmate in its prison system, yet it spends only $8,000 per student in the Oakland school system. It seems to me that if you really want to limit drug use, it’d make more sense to spend more money keeping kids in school, helping them achieve.

The economic benefits of legalizing marijuana are mind blowing. If marijuana was legalized and taxed at the same rate of tobacco, the money we would save on law enforcement and gain in tax revenue equals about $17 billion. As Nicholas Kristof notes, that is enough money to send every three and four year old in a poor neighborhood to pre-school. Or we could spend that money improving public school education. Or we could use the money to shore up border defense. Whatever we do, $17 billion is not exactly a trivial amount.

For me, the biggest reason to legalize marijuana is to hurt the cartels. Immigration has emerged as a hot button issue recently, with Arizona passing a draconian immigration law and many similar propositions being considered by other states. People are worried about violence, and understandably so. No one wants to have foreign drug dealers operating in their back yard. But no matter how many laws we pass, or how much money we spend, marijuana from Mexico and other Latin American countries will always find a way across the border. Drug importers are smart, and the demand is so high that increased patrols by border agents and harsher prison sentences will not act as an effective deterrent. America will always have a demand for marijuana, and that means as long as the drug stays illegal, violent drug cartels will operate in our borders.

But what if the drug that the cartels are pushing is suddenly legal? No one in their right mind would buy pot off the street if they could instead walk into a dispensary and buy high quality marijuana legally, and probably for less money than the cartels are charging. Very few people actually want to have to hide their drug use. If given a choice, marijuana smokers would absolutely buy legal drugs. This would severely weaken the cartels, and decrease deaths related to drug trafficking.

I’m not advocating drug use here. I know people who have ruined their lives from excess drug use. But it’s not true that marijuana is the gateway drug that people have been demonizing for years. Just because someone smokes pot every once in a while doesn’t mean that person will turn around and become a heroin addict. Yes, marijuana intoxicates you, but so do legal drugs like alcohol. As long as sensible restrictions are built into the law, such as making it illegal to drive under the influence, then there is no reason that marijuana should not be legalized."""
tech = """Displays new emails and the sender's contact photo, get notifications or even listen, read or delete them without opening Gmail! Supports multiple accounts plus many options.

• The fastest and easiest way to manage multiple email accounts • One of the highest rated Chrome extensions - featured many times on PCWorld • Trusted developer of many extensions - more than one million satisfied users worldwide • Lots of features, options and updates • Personal tech support from me (the developer) - very fast response times • I'll add your suggestions • Safer - requires less permissions and only access to Google Mail's website Features... • See the people emailing you just like in the Gmail chat notification, with an option to show their contact photos or your assigned photos for them. • Voice notification: If you get an email while you're busy watching a movie or cooking dinner this extension can optionally read it out loud ie. "Jason says, dinner at my place". It's great for the visually impaired. • Option to monitor any Gmail or custom labels • Option to run in background when Google Chrome is closed and still get new email alerts • Popup mail preview window to read, archive, mark as read or delete emails without leaving the current tab (or option to go directly to your Gmail tab) • Desktop sound or voice notifications when new mail arrives (or add your own sounds) • Support for multiple Gmail and Google Apps accounts • Option to open "Mail to" links in your Gmail instead of your regular mail client • This Gmail notifier has more than 10 different icon sets, choose your favorite! • You change the generated voice by adding TTS (text to speech) voice extensions • The fast way to inbox zero. Yes that's a thing."""
temp_text = """
I ask them to take a poem
and hold it up to the light
like a color slide

or press an ear against its hive.

I say drop a mouse into a poem
and watch him probe his way out,
or walk inside the poem's room
and feel the walls for a light switch.

I want them to waterski
across the surface of a poem
waving at the author's name on the shore.

But all they want to do
is tie the poem to a chair with rope
and torture a confession out of it.

They begin beating it with a hose
to find out what it really means.
April. And the air dry
As the shoulders of a water buffalo.

Grasshoppers scratch at the dirt,
rub their wings with thin legs
flaring out in front of the soldiers
in low arcing flights, wings a blur.

The soldiers don’t notice anymore,
seeing only the wreckage of the streets,
bodies draped with sheets, and the sun,
how bright it is, how hard and flat and white.

It will take many nails from the coffinmakers
to shut out this light, which reflects off everything:
the calloused feet of the dead, their bony hands,
their pale foreheads so cold, brilliant in the sun.
"""
french = """
Je m’appelle Jessica. Je suis une fille, je suis française et j’ai treize ans. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux frères. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats.

Aujourd’hui, on est samedi, nous rendons visite à notre grand-mère. Elle a 84 ans et elle habite à Antibes. J’adore ma grand-mère, elle est très gentille. Elle fait des bons gâteaux.

Lundi, je retourne à l’école. Je suis contente, je vais voir Amélie. C’est ma meilleure amie. J’aime beaucoup l’école. Mes matières préférées sont le français et le sport. J’aime beaucoup lire et je nage très bien.

"""
hyphens = (
    """
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
const const const const const const const const

"""
    * 10
)
political_news = """In February 2021, Price announced that the U.S. would review whether the Taliban were sticking to the terms of the Doha Agreement, guaranteeing the exit of American troops.

It would include an 'assessment of whether the Taliban are fulfilling their commitments to cut ties with terrorist groups, to reduce violence, and to engage in meaningful negotiations with the Afghan Government and other stakeholders.'

It was supposed to ensure that the Taliban kept their part of the deal.

But when he sat for an interview with the investigation, at the end of last year, Price admitted that the assessment did not have any impact on the withdrawal.

'I recall having a number of conversations around the fact that in some ways, Taliban adherence was immaterial,' he told investigators.

For its part, The National Security Council dismissed the findings of the report and its criticism of officials, accusing Rep. Michael McCaul, the chairman of the House Foreign Affairs Committee's, of acting in bad faith.

'Everything we have seen and heard of Chairman McCaul's latest partisan report shows that it is based on cherry-picked facts, inaccurate characterizations, and pre-existing biases that have plagued this investigation from the start,' said Sharon Yang, spokesperson for oversight and investigations.

'As we have said many times, ending our longest war was the right thing to do and our nation is stronger today as a result.

'Bringing our troops home after 20 years put us in a stronger position by allowing us to redirect our resources to confront threats to international peace and stability, such as Russia’s war in Ukraine, an ongoing crisis in the Middle East, China's increasingly aggressive actions, and terror threats that exist around the world.'"""
whitehouse_obama = """President Obama is committed to ensuring that every American family can choose to go solar and to cut their energy bills – and that every American community has the tools they need to tackle local air pollution and global climate change.

Since President Obama took office, solar electricity generation has increased 30 fold and solar jobs are growing 12 times faster than the rest of the economy. Last year, we announced a set of actions to increase access to solar and create a more inclusive workforce, but there is still more work to do. That is why, today, the Obama Administration is announcing a new cross government partnership – the Clean Energy Savings For All Initiative – between the Departments of Energy (DOE), Housing and Urban Development (HUD), Agriculture (USDA), Health and Human Services (HHS), Veteran’s Affairs (VA), and the Environmental Protection Agency (EPA) to increase access to solar energy and promote energy efficiency across the United States and, in particular in low- and moderate- income communities."""

harry_1 = """He dashed back across the road, hurried up to his office, snapped at his
secretary not to disturb him, seized his telephone, and had almost
finished dialing his home number when he changed his mind. He put the
receiver back down and stroked his mustache, thinking... no, he was
being stupid. Potter wasn't such an unusual name. He was sure there were
lots of people called Potter who had a son called Harry. Come to think
of it, he wasn't even sure his nephew was called Harry. He'd never even
seen the boy. It might have been Harvey. Or Harold. There was no point
in worrying Mrs. Dursley; she always got so upset at any mention of her
sister. He didn't blame her -- if he'd had a sister like that... but all
the same, those people in cloaks...
He found it a lot harder to concentrate on drills that afternoon and
when he left the building at five o'clock, he was still so worried that
he walked straight into someone just outside the door.
"Sorry," he grunted, as the tiny old man stumbled and almost fell. It
was a few seconds before Mr. Dursley realized that the man was wearing a
violet cloak. He didn't seem at all upset at being almost knocked to the
ground. On the contrary, his face split into a wide smile and he said in
a squeaky voice that made passersby stare, "Don't be sorry, my dear sir,
for nothing could upset me today! Rejoice, for You-Know-Who has gone at
last! Even Muggles like yourself should be celebrating, this happy,
happy day!


"""

harry_2 = """
Filch took them down to Professor McGonagall's study on the first floor,
where they sat and waited without saying a word to each other. Hermione
was trembling. Excuses, alibis, and wild cover- up stories chased each
other around Harry's brain, each more feeble than the last. He couldn't
see how they were going to get out of trouble this time. They were
cornered. How could they have been so stupid as to forget the cloak?
There was no reason on earth that Professor McGonagall would accept for
their being out of bed and creeping around the school in the dead of
night, let alone being up the tallest astronomy tower, which was
out-of-bounds except for classes. Add Norbert and the invisibility
cloak, and they might as well be packing their bags already.
Had Harry thought that things couldn't have been worse? He was wrong.
When Professor McGonagall appeared, she was leading Neville.
194
"Harry!" Neville burst Out, the moment he saw the other two. "I was
trying to find you to warn you, I heard Malfoy saying he was going to
catch you, he said you had a drag --"
Harry shook his head violently to shut Neville up, but Professor
McGonagall had seen. She looked more likely to breathe fire than Norbert
as she towered over the three of them.
"I would never have believed it of any of you. Mr. Filch says you were
up in the astronomy tower. It's one o'clock in the morning. Explain
yourselves."
It was the first time Hermione had ever failed to answer a teacher's
question. She was staring at her slippers, as still as a statue

"""
harry = """He dashed back across the road, hurried up to his office, snapped at his
secretary not to disturb him, seized his telephone, and had almost
finished dialing his home number when he changed his mind. He put the
receiver back down and stroked his mustache, thinking... no, he was
being stupid. Potter wasn't such an unusual name. He was sure there were
lots of people called Potter who had a son called Harry. Come to think
of it, he wasn't even sure his nephew was called Harry. He'd never even
seen the boy. It might have been Harvey. Or Harold. There was no point
in worrying Mrs. Dursley; she always got so upset at any mention of her
sister. He didn't blame her -- if he'd had a sister like that... but all
the same, those people in cloaks...
He found it a lot harder to concentrate on drills that afternoon and
when he left the building at five o'clock, he was still so worried that
he walked straight into someone just outside the door.
"Sorry," he grunted, as the tiny old man stumbled and almost fell. It
was a few seconds before Mr. Dursley realized that the man was wearing a
violet cloak. He didn't seem at all upset at being almost knocked to the
ground. On the contrary, his face split into a wide smile and he said in
a squeaky voice that made passersby stare, "Don't be sorry, my dear sir,
for nothing could upset me today! Rejoice, for You-Know-Who has gone at
last! Even Muggles like yourself should be celebrating, this happy,
happy day!

Filch took them down to Professor McGonagall's study on the first floor,
where they sat and waited without saying a word to each other. Hermione
was trembling. Excuses, alibis, and wild cover- up stories chased each
other around Harry's brain, each more feeble than the last. He couldn't
see how they were going to get out of trouble this time. They were
cornered. How could they have been so stupid as to forget the cloak?
There was no reason on earth that Professor McGonagall would accept for
their being out of bed and creeping around the school in the dead of
night, let alone being up the tallest astronomy tower, which was
out-of-bounds except for classes. Add Norbert and the invisibility
cloak, and they might as well be packing their bags already.
Had Harry thought that things couldn't have been worse? He was wrong.
When Professor McGonagall appeared, she was leading Neville.
194
"Harry!" Neville burst Out, the moment he saw the other two. "I was
trying to find you to warn you, I heard Malfoy saying he was going to
catch you, he said you had a drag --"
Harry shook his head violently to shut Neville up, but Professor
McGonagall had seen. She looked more likely to breathe fire than Norbert
as she towered over the three of them.
"I would never have believed it of any of you. Mr. Filch says you were
up in the astronomy tower. It's one o'clock in the morning. Explain
yourselves."
It was the first time Hermione had ever failed to answer a teacher's
question. She was staring at her slippers, as still as a statue


"""
fishing_news = """The Marvin-1, a fishing boat, sits on the shore May 16, 2015, in Masinloc, Philippines, unused since the Chinese barred it from Scarborough Shoal in the South China Sea. (Will Englund/The Washington Post)

When nations duel over reefs, rocks and islets, people are going to get hurt, and in the South China Sea dispute, that means the fishermen here who once wrested a living from the contested waters.

Gunmen in a Chinese speedboat drove Macario Forones, for instance, away from a favorite spot called Scarborough Shoal, and now his boat, the Marvin-1, sits useless in the grass and weeds above the high-tide line, and he sells someone else’s fish from a stall in the local market. Efrim Forones now dives for clams in the bay, making about one-tenth of what he earned when he fished the sea. Viany Mula says he was set upon with a Chinese water cannon when he ventured out to the shoal in his boat, and now he makes deliveries around town on a motorbike, barely earning enough each day, as he puts it, to buy the rice he needs.

“I really want to fish the shoal,” Mula said one recent day. “It’s a very rich fishing ground. But that’s not possible now.”

For generations, the South China Sea was a regional common. Fishing boats from all of the surrounding countries would roam its waters, pausing now and then to trade cigarettes or potatoes or gossip.

But then Vietnam, followed by the Philippines, began staking claims to some of the islands, and now China is moving in, in a big way. Beijing is building up the outposts it has established, enlarging islands that it controls and claiming exclusive rights to fishing grounds.





The smaller, poorer nations can’t put up a real fight for the access to the sea that they long enjoyed.

“That’s not for us,” Mula said. “We have nothing.”

But the Philippines does have the United States behind it, after a fashion. The Americans are making more visits here, and stepping up naval patrols and overflights — and in the process, the South China Sea dispute becomes something bigger than a contest for fish. It looks more and more like a geostrategic confrontation between the two great powers, China and the United States; that’s certainly how the Chinese characterize it.

The U.S. military has long been a source of anguish, self-doubt and defiance for the Philippines, a former U.S. colony. Many Filipinos are encouraged by recent U.S. attention to the maritime dispute, but they wonder whether the Americans give much thought to the Philippines and the people who are paying a price as the dispute deepens.

A third of the residents of Masinloc have depended over the years on fishing for their livelihoods, said Mayor Desiree Edora. Scarborough Shoal, a half-day’s sail from shore, was a refuge from storms, a gathering place for fishermen from all over and a home to abundant grouper and giant clams. Now, the Chinese have barred foreign boats. It is like being thrown out of your own house, she said.

“We can’t replicate what Scarborough Shoal can provide,” she said.

The Philippines took China to court — an international tribunal in The Hague — two years ago over competing claims in the sea. China refused to participate; a decision is expected next year, but it probably will be unenforceable. The Philippine move may have provoked the Chinese into trying to cement their claims by occupying and building up as many spots in the sea as they can, but officials in the Philippines say they had no choice after efforts to negotiate came to nothing.

Viany Mula, 43, once fished the South China Sea and says he was forced off the water by the Chinese. Now he makes deliveries around Masinloc on a motorbike. (Will Englund/The Washington Post)

The governor of Zambales province, Hermogenes E. Ebdane Jr., said he wonders what China’s ultimate goal is. “No one’s going to war over fish,” he said. His constituents, the fishermen, will have to find something else to do. But if this confrontation is about something bigger, Ebdane said, it’s unclear what role the Philippines might have. There’s a new defense agreement with the United States, but, he said, neither side seems to have thought through the implications for the murky weeks and months ahead.

A legacy of ambivalence

At the Defense College in Quezon City, on the outskirts of Manila, an entire wall in the lobby is given over to a painting that depicts the massacre of four dozen U.S. soldiers by Philippine insurgents at Balangiga in 1901. A diorama up a staircase shows Filipinos battling Spanish conquistadors and fighting against the Japanese in World War II — alongside Americans.

The United States seized the Philippines from Spain in 1898 and held it until 1946. The U.S. military continued to keep permanent bases here until 1991.

The legacy is a deep ambivalence toward the United States. But the U.S. Navy is the one force that is willing to challenge the Chinese and keep up regular patrols in the region. An agreement signed last year would allow the U.S. military a standing presence here, rotating forces onto Philippine bases. The agreement is held up by a lawsuit in the Philippine Supreme Court.

Washington has stepped up visits and patrols, and it has made much of joint training exercises and the donation of used military equipment.

“That is not to protect the Philippines but to protect their own turf,” said Roilo Golez, a member of the country’s House of Representatives. U.S. military aid, worth about $40 million a year, is nothing but a token, he said.

The Philippine armed forces, in this nation of 100 million, remain in woeful shape. It is an article of faith that the government was caught napping when China began making its moves in the South China Sea.

“We remain quite dependent on allied help, and that is not good,” said Rafael Alunan III, former secretary of the interior. “The focus of the Philippine government has been on politics, politics, politics, at the expense of national security. China is taking advantage of our inertia and lack of assertiveness. We are presenting ourselves as unworthy before friend and foe.”

Walden Bello, founding director of a group called Focus on the Global South, said his country “is right back to its role in the Cold War, when it played the part of handmaiden to the United States.”

But military officials here say they are unsure of the U.S. commitment if hostilities should break out. The United States and the Philippines have a mutual defense treaty pledging assistance if either is attacked, but Washington doesn’t recognize any nation’s territorial claims in the South China Sea, including the Philippines’. Naval analysts in Washington say the U.S. response to conflict there would depend entirely on the circumstances.

“We may have overestimated how the United States will come to the rescue,” said Chito Santa Romana, an expert on China. “We may have underestimated Chinese resolve.”

Civil disobedience at sea

The two biggest vessels in the Philippine navy are former U.S. Coast Guard cutters, retrofitted with deck guns, and of little use in standing up to the Chinese. The government, in any case, has no desire to provoke China into a military confrontation.

That leaves the fishing fleet as the country’s best means of maintaining a presence in the parts of the South China Sea that Beijing claims. Philippine — and Vietnamese — boats challenge the Chinese when and where they can, until the Chinese coast guard drives them off. It is waterborne civil disobedience.

“These are small, subsistence fishermen,” said Evan P. Garcia, undersecretary for policy in the Philippines’ Department of Foreign Affairs. “They’re not a threat to anybody. And it’s not as if they just went there yesterday.”

The fish they’re after may be the other big casualty of the dispute. The tensions over the years have kept anyone from getting good data on fish stocks or devising a conservation plan. Hundreds of millions of people live around the South China Sea and eat its fish. The Marine Stewardship Council, with an office in Singapore, says that the humpback wrasse and bluefin tuna populations are close to collapse. Edgardo Gomez, a marine biologist in Manila, said that the Chinese have wiped out the giant clams on Scarborough and that their construction work is destroying reefs that support the bottom levels of the sea’s food chain.

“You have tons and tons of marine life in and around those reefs that are now gone,” he said.

The hatch is being shut on a way of life. The United States and China are either pursuing strategic advantage or practicing destructive gamesmanship, depending on the perspective. Filipinos have to live with that — with the “odd detour,” as Garcia put it, that brought them here.

Viany Mula would trade his motorbike in the blink of an eye for a chance to return to sea. But that is not going to happen.

Englund visited the Philippines on a Jefferson Fellowship, supported by the East-West Center.

Read more

Beijing’s power play in the South China Sea may be killing coral reefs

Here’s why some in the Philippines want the U.S. Navy back



Chinese warnings to U.S. plane hint of rising stakes over disputed islands"""

league = """
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'v'jungle is a twitch streamer going to win with vladimir as a champion'
"""


spanish = """
Hola! Yo empecé aprendo Español hace dos mes en la escuela. Yo voy la universidad. Yo tratar estudioso Español tres hora todos los días para que yo saco mejor rápido. ¿Cosa algún yo debo hacer además construir mí vocabulario? Muchas veces yo estudioso la palabras solo para que yo construir mí voabulario rápido. Yo quiero empiezo leo el periódico Español la próxima semana. Por favor correcto algún la equivocaciónes yo hisciste. Gracias!"""
# %%
comment_text = (
    """
comment
"""
    * 1000
)

league_of_legends = """
Identity of K'Sante (Design/Live server)
K'Sante was created to fill the role of a "high skill tank" who can "take things into his own hand" and outplay opponents.

From a design perspective, this fantasy was executed exceptionally well imo. Initially, you experience the slower, more deliberate feel of a tank, but after using his Ultimate, which shatters his Ntofos, you really feel the effect of "shedding that weight".

His movements and fighting become faster and more fluid, enhancing the gameplay experience, creating a contrast that also manages to compliment the concept.

Obviously, this design doesn't sit well with a lot of people, which... fair. I'm not here to say that those opinions don't matter.

SoloQ WR and Proplay
I see a lot of people joke about how op.gg and other sites say he has a 47% (current patch) winrate and that he is a bad champion, but this isn't really the case.

The champion himself isn't complex and can easily picked up, but the full potential of K'Sante needs a lot of practice and experience. As a high skill champ, he needs dedication to be piloted properly. The wr you see on stat sites just shows the average and surprisingly, K'Sante has a decently high pickrate, but this doesnt entail everyone who plays him have the mastery the champion needs or wants. While he is in the 45-47% range for average player statistics, his winrate for those dedicated mains are closer to the 50-53% range.



"""
mad_god = """
Dark condensation soul, degenerates can be free, awakens, endless charm of deep sleep in my blood.” I have released the Fallen Angel energy, the black fog of big piece revolves me, the black wing arrives at the world once more, my whole body covers in the dark mist, the Lion-Man mask on face was reduced to ashes under the tyrannical strength,
"""
java_text = """

public class BinaryConverter {

    public static void main(String[] args){
        for(int i = -5; i < 33; i++){
            System.out.println(i + ": " + toBinary(i));
            System.out.println(i);
            //always another way
            System.out.println(i + ": " + Integer.toBinaryString(i));
        }
    }

    /*
     * pre: none
     * post: returns a String with base10Num in base 2
     */
    public static String toBinary(int base10Num){
        boolean isNeg = base10Num < 0;
        base10Num = Math.abs(base10Num);
        String result = "";

        while(base10Num > 1){
            result = (base10Num % 2) + result;
            base10Num /= 2;
        }
        assert base10Num == 0 || base10Num == 1 : "value is not <= 1: " + base10Num;

        result = base10Num + result;
        assert all0sAnd1s(result);

        if( isNeg )
            result = "-" + result;
        return result;
    }

    /*
     * pre: cal != null
     * post: return true if val consists only of characters 1 and 0, false otherwise
     */
    public static boolean all0sAnd1s(String val){
        assert val != null : "Failed precondition all0sAnd1s. parameter cannot be null";
        boolean all = true;
        int i = 0;
        char c;

        while(all && i < val.length()){
            c = val.charAt(i);
            all = c == '0' || c == '1';
            i++;
        }
        return all;
    }
}



"""
genesis = """
Book of Genesis
Chapter 1
In the beginning God created heaven, and earth.
2 And the earth was void and empty, and
darkness was upon the face of the deep; and the
spirit of God moved over the waters.
3 And God said: Be light made. And light
was made.
4 And God saw the light that it was good; and
he divided the light from the darkness.
5 And he called the light Day, and the darkness Night; and there was evening and morning
one day.
6 And God said: Let there be a firmament
made amidst the waters: and let it divide the
waters from the waters.
7 And god made a firmament, and divided
the waters that were under the firmament, from
those that were above the firmament, and it was
so.
8 And God called the firmament, Heaven; and
the evening and morning were the second day.
9 God also said; Let the waters that are under
the heaven, be gathered together into one place:
and let the dry land appear. And it was so done.
10 And God called the dry land, Earth; and
the gathering together of the waters, he called
Seas. And God saw that it was good.
11 And he said: let the earth bring forth green
herb, and such as may seed, and the fruit tree
yielding fruit after its kind, which may have seed
in itself upon the earth. And it was so done.
12 And the earth brought forth the green
herb, and such as yieldeth seed according to its
kind, and the tree that beareth fruit, having seed
each one according to its kind. And God saw
that it was good.
13 And the evening and the morning were the
third day.
14 And God said: Let there be lights made
in the firmament of heaven, to divide the day
and the night, and let them be for signs, and for
seasons, and for days and years:
15 To shine in the firmament of heaven, and
to give light upon the earth, and it was so done.
16 And God made two great lights: a greater
light to rule the day; and a lesser light to rule
the night: and The stars.
17 And he set them in the firmament of heaven
to shine upon the earth.
18 And to rule the day and the night, and to
divide the light and the darkness. And God saw
that it was good.
19 And the evening and morning were the
fourth day.
20 God also said: let the waters bring forth
the creeping creature having life, and the fowl
that may fly over the earth under the firmament
of heaven.
21 And God created the great whales, and
every living and moving creature, which the
waaters brought forth, according to their kinds,
and every winged fowl accordi
22 God blessed them, saying, "Be fruitful and multiply and fill the waters in the seas, and let birds multiply on the earth."
23 And there was evening and there was morning, the fifth day.
24 And God said, "Let the earth bring forth living creatures of every kind: cattle and creeping things and wild animals of the earth of every kind." And it was so.
25 God made the wild animals of the earth of every kind, and the cattle of every kind, and everything that creeps upon the ground of every kind. And God saw that it was good.
26 Then God said, "Let us make humankind in our image, according to our likeness; and let them have dominion over the fish of the sea, and over the birds of the air, and over the cattle, and over all the wild animals of the earth, and over every creeping thing that creeps upon the earth."
27 So God created humankind in his image, in the image of God he created them; male and female he created them.
28 God blessed them, and God said to them, "Be fruitful and multiply, and fill the earth and subdue it; and have dominion over the fish of the sea and over the birds of the air and over every living thing that moves upon the earth."
29 God said, "See, I have given you every plant yielding seed that is upon the face of all the earth, and every tree with seed in its fruit; you shall have them for food.
30 And to every beast of the earth, and to every bird of the air, and to everything that creeps on the earth, everything that has the breath of life, I have given every green plant for food." And it was so.
31 God saw everything that he had made, and indeed, it was very good. And there was evening and there was morning, the sixth day.
"""

dnd_text = """

Every character belongs to a race, one of the many intelligent humanoid species in the D&D world. The most common player character races are dwarves, elves, halflings, and humans. Some races also have subraces, such as mountain dwarf or wood elf. The Races section provides more information about these races.

The race you choose contributes to your character’s identity in an important way, by establishing a general appearance and the natural talents gained from culture and ancestry. Your character’s race grants particular racial traits, such as special senses, proficiency with certain weapons or tools, proficiency in one or more skills, or the ability to use minor spells. These traits sometimes dovetail with the capabilities of certain classes (see step 2). For example, the racial traits of lightfoot halflings make them exceptional rogues, and high elves tend to be powerful wizards. Sometimes playing against type can be fun, too. Halfling paladins and mountain dwarf wizards, for example, can be unusual but memorable characters.

Your race also increases one or more of your ability scores, which you determine in step 3. Note these increases and remember to apply them later.

Record the traits granted by your race on your character sheet. Be sure to note your starting languages and your base speed as well.

BUILDING BRUENOR, STEP 1

Bob is sitting down to create his character. He decides that a gruff mountain dwarf fits the character he wants to play. He notes all the racial traits of dwarves on his character sheet, including his speed of 25 feet and the languages he knows: Common and Dwarvish.

2. Choose a Class
Every adventurer is a member of a class. Class broadly describes a character’s vocation, what special talents he or she possesses, and the tactics he or she is most likely to employ when exploring a dungeon, fighting monsters, or engaging in a tense negotiation. The character classes are described in the Classes section.

Your character receives a number of benefits from your choice of class. Many of these benefits are class features — capabilities (including spellcasting) that set your character apart from members of other classes. You also gain a number of proficiencies: armor, weapons, skills, saving throws, and sometimes tools. Your proficiencies define many of the things your character can do particularly well, from using certain weapons to telling a convincing lie.

On your character sheet, record all the features that your class gives you at 1st level.

Level
Typically, a character starts at 1st level and advances in level by adventuring and gaining experience points (XP). A 1st-level character is inexperienced in the adventuring world, although he or she might have been a soldier or a pirate and done dangerous things before.

Starting off at 1st level marks your character’s entry into the adventuring life. If you’re already familiar with the game, or if you are joining an existing D&D campaign, your DM might decide to have you begin at a higher level, on the assumption that your character has already survived a few harrowing adventures.

Record your level on your character sheet. If you’re starting at a higher level, record the additional elements your class gives you for your levels past 1st. Also record your experience points. A 1st-level character has 0 XP. A higher-level character typically begins with the minimum amount of XP required to reach that level (see “Beyond 1st Level” later in this section).

QUICK BUILD

Each class description in the Classes section includes a section offering suggestions to quickly build a character of that class, including how to assign your highest ability scores, a background suitable to the class, and starting spells.

Hit Points and Hit Dice
Your character’s hit points define how tough your character is in combat and other dangerous situations. Your hit points are determined by your Hit Dice (short for Hit Point Dice).

At 1st level, your character has 1 Hit Die, and the die type is determined by your class. You start with hit points equal to the highest roll of that die, as indicated in your class description. (You also add your Constitution modifier, which you’ll determine in step 3.) This is also your hit point maximum.

Record your character’s hit points on your character sheet. Also record the type of Hit Die your character uses and the number of Hit Dice you have. After you rest, you can spend Hit Dice to regain hit points (see “Resting” in the Adventuring section).

Proficiency Bonus
The table that appears in your class description shows your proficiency bonus, which is +2 for a 1st-level character. Your proficiency bonus applies to many of the numbers you’ll be recording on your character sheet:

Attack rolls using weapons you’re proficient with
Attack rolls with spells you cast
Ability checks using skills you’re proficient in
Ability checks using tools you’re proficient with
Saving throws you’re proficient in
Saving throw DCs for spells you cast (explained in each spellcasting class)

"""
bbc_text = """
Flash floods and heavy rain batter England and Wales
A woman wearing a plastic ran poncho cycles through floodwater after the River Thames overtopped its banks in London
Image source,Reuters
Image caption,
A woman cycles through floodwater after the River Thames overtopped its banks in London

Vicky Wong
BBC News
Matt Taylor
Lead Weather Presenter
@MetMattTaylor
Published
23 September 2024, 08:13 BST
Updated 52 minutes ago
Heavy rain and flash flooding has battered parts of England and Wales, causing widespread travel disruption and damage to properties.

Roads and houses have flooded in central and southern England, after some experienced a month's worth of rain in a matter of hours.

In London, a sinkhole has appeared on AFC Wimbledon's football pitch and 999 call handlers have taken 350 flood-related calls, while in Bedford a main road is totally submerged.

A Met Office amber weather warning in parts of central and southern England ended at 21:00 BST, but a yellow warning remains in place across England and south-east Wales.


Media caption,
Watch: A421 submerged by flood water in Bedfordshire

The yellow weather warning for heavy rain - which means some disruption like floods, travel disruption and power cuts are possible - is due to remain until 23:59. Only the far south-west and parts of northern England are not covered.

The Environment Agency has issued more than 20 flood warnings, meaning flooding is expected, and more than 80 flood alerts, meaning flooding is possible.

Areas affected by the flood warnings include Leighton Buzzard and Luton in Bedfordshire and parts of London.

Among the most dramatic floods was on the A421 main road between Bedford and Milton Keynes, which has been shut - along with the rail line from from Bedford to Bletchley.

Sheep in a temporary pen, there are bags of hay outside it
Image source,PA Media
Image caption,
Sheep in Marston Moretaine had to be dragged to safety in chest-high floodwater and placed in temporary pens

In the village of Grendon, in Northamptonshire, several houses were flooded with clean-up efforts ongoing.

"It was unbelievable," Jon Sayle said describing how about "two feet of water seeped in overnight" into his home.

In the village of Marston Moretaine, Bedfordshire, local residents joined forces to save animals stranded by heavy flooding at a local farm.

Joanna Johnson, 54, said 50 neighbours turned up at Moreteyne's Retreat after she sent an emergency message on social media. "The villagers flocked here so fast," she told PA News agency.

She said her miniature ponies had to swim out of the floodwater, while the sheep were dragged through to safety in chest-high floodwater.

Ms Johnson described the water coming off the nearby A421 as being "like a river", which led the entire farm being flooded within 15 minutes.

"The animals are alive at the moment, I'm now desperately trying to find a piece of land I can leave them on over the winter where they will be safe."

Another resident told PA he had never seen anything like in a decade living there, adding floods on the road were "normally gone within a few hours".

Lee Elliott, 36, said: "I was out last night helping push cars out of the floods because we came home quite late last night and saw the cars stuck in there, so we went down there to help them."

The open boot of a car is visible above the water where the vehicle is submerged in flood water on a421 in Marston Moretaine
Image source,PA Media
Image caption,
All that can be seen of on one car on the A421 is its open boot

A car partly submerged in floodwater on Manor Road in Wallington
Image source,London Fire Brigade
Image caption,
Overnight the London Fire Brigade attended to a vehicle stranded in floodwater in Wallington, Sutton

On Monday afternoon the London Fire Brigade said its 999 control officers had taken some 350 flood-related calls, with firefighters rescuing people trapped inside cars, assisting people from their homes and responding to flooding in Underground stations and roads.

In a post on X, using a photo of a car stranded in floodwater overnight in Wallington, Sutton, the fire brigade warned that "a foot of moving water at just 6mph is enough to float a car, external".

Transport for London has warned passengers that the District, Circle, Metropolitan, Piccadilly, Bakerloo and Central lines have been either partly suspended or subject to minor to severe delays because of flooding caused by heavy rain.

National Rail is also reporting widespread disruption and cancellations to some train services throughout the day and has urged passengers to check their journeys.

In south-east England, a night of heavy rain forced the closure of an M25 slip road at Cobham in Surrey and led to delays on train services.

A number of schools in areas including Bedfordshire and Oxfordshire have been forced to close, with some switching to remote learning, and a number of homes and businesses have been flooded.
"""

# %%
import torch

import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px


def show(matrix):

    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)

    if matrix.shape[0] > 1500 or matrix.shape[1] > 1500:
        print("too big")
        return

    px.imshow(matrix.detach().cpu()).show()


def plotcdf(mat, bins=100):
    pdf, edges = np.histogram(mat.detach().cpu(), bins=bins)
    centers = edges[1:] - np.diff(edges) / 2
    cdf = np.cumsum(pdf) / np.sum(pdf)

    plt.figure(1)

    plt.plot(centers, pdf)


def entr(mat, type):
    if type == "shannon":
        zero_padding = (mat[0] > 0).sum()
        mat_ = mat[:, :zero_padding]
        return torch.sum(mat_ * torch.log(1 / mat_), dim=1) / (torch.log(zero_padding))
    elif type == "collision":

        zero_padding = (mat[0] > 0).sum()
        mat_ = mat[:, :zero_padding]

        return 1 / ((torch.sum(mat_**2, dim=1)))
    return


def kl(mat1, mat2):
    zero_padding = (mat1[0] > 0).sum()
    mat1_ = mat1[:, :zero_padding]
    mat2_ = mat2[:, :zero_padding]
    return torch.sum((mat1_ - mat2_) ** 2, dim=1)


java_tutorial = """
Parameters and local variables are allocated on the stack (with reference types, the object lives on the heap and a variable in the stack references that object on the heap). The stack typically lives at the upper end of your address space and as it is used up it heads towards the bottom of the address space (i.e. towards zero).

Your process also has a heap, which lives at the bottom end of your process. As you allocate memory, this heap can grow towards the upper end of your address space. As you can see, there is a potential for the heap to "collide" with the stack (a bit like tectonic plates!!!).

The common cause for a stack overflow is a bad recursive call. Typically, this is caused when your recursive functions doesn't have the correct termination condition, so it ends up calling itself forever. Or when the termination condition is fine, it can be caused by requiring too many recursive calls before fulfilling it.

However, with GUI programming, it's possible to generate indirect recursion. For example, your app may be handling paint messages, and, whilst processing them, it may call a function that causes the system to send another paint message. Here you've not explicitly called yourself, but the OS/VM has done it for you.

To deal with them, you'll need to examine your code. If you've got functions that call themselves then check that you've got a terminating condition. If you have, then check that when calling the function you have at least modified one of the arguments, otherwise there'll be no visible change for the recursively called function and the terminating condition is useless. Also mind that your stack space can run out of memory before reaching a valid terminating condition, thus make sure your method can handle input values requiring more recursive calls.

If you've got no obvious recursive functions then check to see if you're calling any library functions that indirectly will cause your function to be called (like the implicit case above).


"""

code_text = """
torch.set_default_device("cuda")
dims = 500
linear = torch.rand(model.embed.W_E.shape[1], dims).to("cuda")
bias = 100.0 * torch.ones(dims).to("cuda")

optim = torch.optim.AdamW([linear, bias], lr=1e-1)


def entr(intermediate):
    intermediate_comp = intermediate[intermediate > 10 ** (-7)]
    intermediate_comp = intermediate_comp / (intermediate_comp.sum())
    return torch.sum(intermediate_comp * torch.log(1 / intermediate_comp))



model.embed.W_E.requires_grad = False
for param in model.parameters():
    param.requires_grad = False
linear.requires_grad = True
bias.requires_grad = True
epochs = 10000
for epoch in range(epochs):

    optim.zero_grad()
    intermediate = torch.nn.ReLU()((model.embed.W_E) @ linear + bias)
    reconstruction = intermediate @ linear.T
    loss__ = torch.norm(reconstruction - (model.embed.W_E)) + 0.1 * abs(
        entr(intermediate)
    )

    print(loss__)
    loss__.backward()
    optim.step()


bias_1 = model.blocks[0].ln1.b
weight_1 = model.blocks[0].ln1.w

W_El = model.embed.W_E
# show the figure; this was slow
"""
political = """For today’s post, I’d like to take a look at California’s voter initiative to legalize pot. If the measure passes, and the sky doesn’t fall, many other states will probably be looking at similar law changes in the near future. Our drug policy of the last century has simply not worked, and it’s heartening to see a state attempting to legalize marijuana.

The statistics on marijuana arrests are really shocking. According to the Drug Policy Alliance, which is in favor of legalization, blacks are arrested for marijuana possession between four and twelve times more than whites in California, even though studies have consistently shown that whites smoke more pot than blacks. In the last ten years, around 500,000 people have been arrested for possession. That’s absurd! Think about how expensive that is for the criminal justice system. California spends $216,000 for each juvenile inmate in its prison system, yet it spends only $8,000 per student in the Oakland school system. It seems to me that if you really want to limit drug use, it’d make more sense to spend more money keeping kids in school, helping them achieve.

The economic benefits of legalizing marijuana are mind blowing. If marijuana was legalized and taxed at the same rate of tobacco, the money we would save on law enforcement and gain in tax revenue equals about $17 billion. As Nicholas Kristof notes, that is enough money to send every three and four year old in a poor neighborhood to pre-school. Or we could spend that money improving public school education. Or we could use the money to shore up border defense. Whatever we do, $17 billion is not exactly a trivial amount.

For me, the biggest reason to legalize marijuana is to hurt the cartels. Immigration has emerged as a hot button issue recently, with Arizona passing a draconian immigration law and many similar propositions being considered by other states. People are worried about violence, and understandably so. No one wants to have foreign drug dealers operating in their back yard. But no matter how many laws we pass, or how much money we spend, marijuana from Mexico and other Latin American countries will always find a way across the border. Drug importers are smart, and the demand is so high that increased patrols by border agents and harsher prison sentences will not act as an effective deterrent. America will always have a demand for marijuana, and that means as long as the drug stays illegal, violent drug cartels will operate in our borders.

But what if the drug that the cartels are pushing is suddenly legal? No one in their right mind would buy pot off the street if they could instead walk into a dispensary and buy high quality marijuana legally, and probably for less money than the cartels are charging. Very few people actually want to have to hide their drug use. If given a choice, marijuana smokers would absolutely buy legal drugs. This would severely weaken the cartels, and decrease deaths related to drug trafficking.

I’m not advocating drug use here. I know people who have ruined their lives from excess drug use. But it’s not true that marijuana is the gateway drug that people have been demonizing for years. Just because someone smokes pot every once in a while doesn’t mean that person will turn around and become a heroin addict. Yes, marijuana intoxicates you, but so do legal drugs like alcohol. As long as sensible restrictions are built into the law, such as making it illegal to drive under the influence, then there is no reason that marijuana should not be legalized."""
tech = """Displays new emails and the sender's contact photo, get notifications or even listen, read or delete them without opening Gmail! Supports multiple accounts plus many options.

• The fastest and easiest way to manage multiple email accounts • One of the highest rated Chrome extensions - featured many times on PCWorld • Trusted developer of many extensions - more than one million satisfied users worldwide • Lots of features, options and updates • Personal tech support from me (the developer) - very fast response times • I'll add your suggestions • Safer - requires less permissions and only access to Google Mail's website Features... • See the people emailing you just like in the Gmail chat notification, with an option to show their contact photos or your assigned photos for them. • Voice notification: If you get an email while you're busy watching a movie or cooking dinner this extension can optionally read it out loud ie. "Jason says, dinner at my place". It's great for the visually impaired. • Option to monitor any Gmail or custom labels • Option to run in background when Google Chrome is closed and still get new email alerts • Popup mail preview window to read, archive, mark as read or delete emails without leaving the current tab (or option to go directly to your Gmail tab) • Desktop sound or voice notifications when new mail arrives (or add your own sounds) • Support for multiple Gmail and Google Apps accounts • Option to open "Mail to" links in your Gmail instead of your regular mail client • This Gmail notifier has more than 10 different icon sets, choose your favorite! • You change the generated voice by adding TTS (text to speech) voice extensions • The fast way to inbox zero. Yes that's a thing."""
temp_text = """
I ask them to take a poem
and hold it up to the light
like a color slide

or press an ear against its hive.

I say drop a mouse into a poem
and watch him probe his way out,
or walk inside the poem's room
and feel the walls for a light switch.

I want them to waterski
across the surface of a poem
waving at the author's name on the shore.

But all they want to do
is tie the poem to a chair with rope
and torture a confession out of it.

They begin beating it with a hose
to find out what it really means.
April. And the air dry
As the shoulders of a water buffalo.

Grasshoppers scratch at the dirt,
rub their wings with thin legs
flaring out in front of the soldiers
in low arcing flights, wings a blur.

The soldiers don’t notice anymore,
seeing only the wreckage of the streets,
bodies draped with sheets, and the sun,
how bright it is, how hard and flat and white.

It will take many nails from the coffinmakers
to shut out this light, which reflects off everything:
the calloused feet of the dead, their bony hands,
their pale foreheads so cold, brilliant in the sun.
"""
french = """
Je m’appelle Jessica. Je suis une fille, je suis française et j’ai treize ans. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux frères. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats.

Aujourd’hui, on est samedi, nous rendons visite à notre grand-mère. Elle a 84 ans et elle habite à Antibes. J’adore ma grand-mère, elle est très gentille. Elle fait des bons gâteaux.

Lundi, je retourne à l’école. Je suis contente, je vais voir Amélie. C’est ma meilleure amie. J’aime beaucoup l’école. Mes matières préférées sont le français et le sport. J’aime beaucoup lire et je nage très bien.

"""
hyphens = (
    """
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
const const const const const const const const

"""
    * 10
)
political_news = """In February 2021, Price announced that the U.S. would review whether the Taliban were sticking to the terms of the Doha Agreement, guaranteeing the exit of American troops.

It would include an 'assessment of whether the Taliban are fulfilling their commitments to cut ties with terrorist groups, to reduce violence, and to engage in meaningful negotiations with the Afghan Government and other stakeholders.'

It was supposed to ensure that the Taliban kept their part of the deal.

But when he sat for an interview with the investigation, at the end of last year, Price admitted that the assessment did not have any impact on the withdrawal.

'I recall having a number of conversations around the fact that in some ways, Taliban adherence was immaterial,' he told investigators.

For its part, The National Security Council dismissed the findings of the report and its criticism of officials, accusing Rep. Michael McCaul, the chairman of the House Foreign Affairs Committee's, of acting in bad faith.

'Everything we have seen and heard of Chairman McCaul's latest partisan report shows that it is based on cherry-picked facts, inaccurate characterizations, and pre-existing biases that have plagued this investigation from the start,' said Sharon Yang, spokesperson for oversight and investigations.

'As we have said many times, ending our longest war was the right thing to do and our nation is stronger today as a result.

'Bringing our troops home after 20 years put us in a stronger position by allowing us to redirect our resources to confront threats to international peace and stability, such as Russia’s war in Ukraine, an ongoing crisis in the Middle East, China's increasingly aggressive actions, and terror threats that exist around the world.'"""
whitehouse_obama = """President Obama is committed to ensuring that every American family can choose to go solar and to cut their energy bills – and that every American community has the tools they need to tackle local air pollution and global climate change.

Since President Obama took office, solar electricity generation has increased 30 fold and solar jobs are growing 12 times faster than the rest of the economy. Last year, we announced a set of actions to increase access to solar and create a more inclusive workforce, but there is still more work to do. That is why, today, the Obama Administration is announcing a new cross government partnership – the Clean Energy Savings For All Initiative – between the Departments of Energy (DOE), Housing and Urban Development (HUD), Agriculture (USDA), Health and Human Services (HHS), Veteran’s Affairs (VA), and the Environmental Protection Agency (EPA) to increase access to solar energy and promote energy efficiency across the United States and, in particular in low- and moderate- income communities."""

harry_1 = """He dashed back across the road, hurried up to his office, snapped at his
secretary not to disturb him, seized his telephone, and had almost
finished dialing his home number when he changed his mind. He put the
receiver back down and stroked his mustache, thinking... no, he was
being stupid. Potter wasn't such an unusual name. He was sure there were
lots of people called Potter who had a son called Harry. Come to think
of it, he wasn't even sure his nephew was called Harry. He'd never even
seen the boy. It might have been Harvey. Or Harold. There was no point
in worrying Mrs. Dursley; she always got so upset at any mention of her
sister. He didn't blame her -- if he'd had a sister like that... but all
the same, those people in cloaks...
He found it a lot harder to concentrate on drills that afternoon and
when he left the building at five o'clock, he was still so worried that
he walked straight into someone just outside the door.
"Sorry," he grunted, as the tiny old man stumbled and almost fell. It
was a few seconds before Mr. Dursley realized that the man was wearing a
violet cloak. He didn't seem at all upset at being almost knocked to the
ground. On the contrary, his face split into a wide smile and he said in
a squeaky voice that made passersby stare, "Don't be sorry, my dear sir,
for nothing could upset me today! Rejoice, for You-Know-Who has gone at
last! Even Muggles like yourself should be celebrating, this happy,
happy day!


"""

harry_2 = """
Filch took them down to Professor McGonagall's study on the first floor,
where they sat and waited without saying a word to each other. Hermione
was trembling. Excuses, alibis, and wild cover- up stories chased each
other around Harry's brain, each more feeble than the last. He couldn't
see how they were going to get out of trouble this time. They were
cornered. How could they have been so stupid as to forget the cloak?
There was no reason on earth that Professor McGonagall would accept for
their being out of bed and creeping around the school in the dead of
night, let alone being up the tallest astronomy tower, which was
out-of-bounds except for classes. Add Norbert and the invisibility
cloak, and they might as well be packing their bags already.
Had Harry thought that things couldn't have been worse? He was wrong.
When Professor McGonagall appeared, she was leading Neville.
194
"Harry!" Neville burst Out, the moment he saw the other two. "I was
trying to find you to warn you, I heard Malfoy saying he was going to
catch you, he said you had a drag --"
Harry shook his head violently to shut Neville up, but Professor
McGonagall had seen. She looked more likely to breathe fire than Norbert
as she towered over the three of them.
"I would never have believed it of any of you. Mr. Filch says you were
up in the astronomy tower. It's one o'clock in the morning. Explain
yourselves."
It was the first time Hermione had ever failed to answer a teacher's
question. She was staring at her slippers, as still as a statue

"""
harry = """He dashed back across the road, hurried up to his office, snapped at his
secretary not to disturb him, seized his telephone, and had almost
finished dialing his home number when he changed his mind. He put the
receiver back down and stroked his mustache, thinking... no, he was
being stupid. Potter wasn't such an unusual name. He was sure there were
lots of people called Potter who had a son called Harry. Come to think
of it, he wasn't even sure his nephew was called Harry. He'd never even
seen the boy. It might have been Harvey. Or Harold. There was no point
in worrying Mrs. Dursley; she always got so upset at any mention of her
sister. He didn't blame her -- if he'd had a sister like that... but all
the same, those people in cloaks...
He found it a lot harder to concentrate on drills that afternoon and
when he left the building at five o'clock, he was still so worried that
he walked straight into someone just outside the door.
"Sorry," he grunted, as the tiny old man stumbled and almost fell. It
was a few seconds before Mr. Dursley realized that the man was wearing a
violet cloak. He didn't seem at all upset at being almost knocked to the
ground. On the contrary, his face split into a wide smile and he said in
a squeaky voice that made passersby stare, "Don't be sorry, my dear sir,
for nothing could upset me today! Rejoice, for You-Know-Who has gone at
last! Even Muggles like yourself should be celebrating, this happy,
happy day!

Filch took them down to Professor McGonagall's study on the first floor,
where they sat and waited without saying a word to each other. Hermione
was trembling. Excuses, alibis, and wild cover- up stories chased each
other around Harry's brain, each more feeble than the last. He couldn't
see how they were going to get out of trouble this time. They were
cornered. How could they have been so stupid as to forget the cloak?
There was no reason on earth that Professor McGonagall would accept for
their being out of bed and creeping around the school in the dead of
night, let alone being up the tallest astronomy tower, which was
out-of-bounds except for classes. Add Norbert and the invisibility
cloak, and they might as well be packing their bags already.
Had Harry thought that things couldn't have been worse? He was wrong.
When Professor McGonagall appeared, she was leading Neville.
194
"Harry!" Neville burst Out, the moment he saw the other two. "I was
trying to find you to warn you, I heard Malfoy saying he was going to
catch you, he said you had a drag --"
Harry shook his head violently to shut Neville up, but Professor
McGonagall had seen. She looked more likely to breathe fire than Norbert
as she towered over the three of them.
"I would never have believed it of any of you. Mr. Filch says you were
up in the astronomy tower. It's one o'clock in the morning. Explain
yourselves."
It was the first time Hermione had ever failed to answer a teacher's
question. She was staring at her slippers, as still as a statue


"""
fishing_news = """The Marvin-1, a fishing boat, sits on the shore May 16, 2015, in Masinloc, Philippines, unused since the Chinese barred it from Scarborough Shoal in the South China Sea. (Will Englund/The Washington Post)

When nations duel over reefs, rocks and islets, people are going to get hurt, and in the South China Sea dispute, that means the fishermen here who once wrested a living from the contested waters.

Gunmen in a Chinese speedboat drove Macario Forones, for instance, away from a favorite spot called Scarborough Shoal, and now his boat, the Marvin-1, sits useless in the grass and weeds above the high-tide line, and he sells someone else’s fish from a stall in the local market. Efrim Forones now dives for clams in the bay, making about one-tenth of what he earned when he fished the sea. Viany Mula says he was set upon with a Chinese water cannon when he ventured out to the shoal in his boat, and now he makes deliveries around town on a motorbike, barely earning enough each day, as he puts it, to buy the rice he needs.

“I really want to fish the shoal,” Mula said one recent day. “It’s a very rich fishing ground. But that’s not possible now.”

For generations, the South China Sea was a regional common. Fishing boats from all of the surrounding countries would roam its waters, pausing now and then to trade cigarettes or potatoes or gossip.

But then Vietnam, followed by the Philippines, began staking claims to some of the islands, and now China is moving in, in a big way. Beijing is building up the outposts it has established, enlarging islands that it controls and claiming exclusive rights to fishing grounds.





The smaller, poorer nations can’t put up a real fight for the access to the sea that they long enjoyed.

“That’s not for us,” Mula said. “We have nothing.”

But the Philippines does have the United States behind it, after a fashion. The Americans are making more visits here, and stepping up naval patrols and overflights — and in the process, the South China Sea dispute becomes something bigger than a contest for fish. It looks more and more like a geostrategic confrontation between the two great powers, China and the United States; that’s certainly how the Chinese characterize it.

The U.S. military has long been a source of anguish, self-doubt and defiance for the Philippines, a former U.S. colony. Many Filipinos are encouraged by recent U.S. attention to the maritime dispute, but they wonder whether the Americans give much thought to the Philippines and the people who are paying a price as the dispute deepens.

A third of the residents of Masinloc have depended over the years on fishing for their livelihoods, said Mayor Desiree Edora. Scarborough Shoal, a half-day’s sail from shore, was a refuge from storms, a gathering place for fishermen from all over and a home to abundant grouper and giant clams. Now, the Chinese have barred foreign boats. It is like being thrown out of your own house, she said.

“We can’t replicate what Scarborough Shoal can provide,” she said.

The Philippines took China to court — an international tribunal in The Hague — two years ago over competing claims in the sea. China refused to participate; a decision is expected next year, but it probably will be unenforceable. The Philippine move may have provoked the Chinese into trying to cement their claims by occupying and building up as many spots in the sea as they can, but officials in the Philippines say they had no choice after efforts to negotiate came to nothing.

Viany Mula, 43, once fished the South China Sea and says he was forced off the water by the Chinese. Now he makes deliveries around Masinloc on a motorbike. (Will Englund/The Washington Post)

The governor of Zambales province, Hermogenes E. Ebdane Jr., said he wonders what China’s ultimate goal is. “No one’s going to war over fish,” he said. His constituents, the fishermen, will have to find something else to do. But if this confrontation is about something bigger, Ebdane said, it’s unclear what role the Philippines might have. There’s a new defense agreement with the United States, but, he said, neither side seems to have thought through the implications for the murky weeks and months ahead.

A legacy of ambivalence

At the Defense College in Quezon City, on the outskirts of Manila, an entire wall in the lobby is given over to a painting that depicts the massacre of four dozen U.S. soldiers by Philippine insurgents at Balangiga in 1901. A diorama up a staircase shows Filipinos battling Spanish conquistadors and fighting against the Japanese in World War II — alongside Americans.

The United States seized the Philippines from Spain in 1898 and held it until 1946. The U.S. military continued to keep permanent bases here until 1991.

The legacy is a deep ambivalence toward the United States. But the U.S. Navy is the one force that is willing to challenge the Chinese and keep up regular patrols in the region. An agreement signed last year would allow the U.S. military a standing presence here, rotating forces onto Philippine bases. The agreement is held up by a lawsuit in the Philippine Supreme Court.

Washington has stepped up visits and patrols, and it has made much of joint training exercises and the donation of used military equipment.

“That is not to protect the Philippines but to protect their own turf,” said Roilo Golez, a member of the country’s House of Representatives. U.S. military aid, worth about $40 million a year, is nothing but a token, he said.

The Philippine armed forces, in this nation of 100 million, remain in woeful shape. It is an article of faith that the government was caught napping when China began making its moves in the South China Sea.

“We remain quite dependent on allied help, and that is not good,” said Rafael Alunan III, former secretary of the interior. “The focus of the Philippine government has been on politics, politics, politics, at the expense of national security. China is taking advantage of our inertia and lack of assertiveness. We are presenting ourselves as unworthy before friend and foe.”

Walden Bello, founding director of a group called Focus on the Global South, said his country “is right back to its role in the Cold War, when it played the part of handmaiden to the United States.”

But military officials here say they are unsure of the U.S. commitment if hostilities should break out. The United States and the Philippines have a mutual defense treaty pledging assistance if either is attacked, but Washington doesn’t recognize any nation’s territorial claims in the South China Sea, including the Philippines’. Naval analysts in Washington say the U.S. response to conflict there would depend entirely on the circumstances.

“We may have overestimated how the United States will come to the rescue,” said Chito Santa Romana, an expert on China. “We may have underestimated Chinese resolve.”

Civil disobedience at sea

The two biggest vessels in the Philippine navy are former U.S. Coast Guard cutters, retrofitted with deck guns, and of little use in standing up to the Chinese. The government, in any case, has no desire to provoke China into a military confrontation.

That leaves the fishing fleet as the country’s best means of maintaining a presence in the parts of the South China Sea that Beijing claims. Philippine — and Vietnamese — boats challenge the Chinese when and where they can, until the Chinese coast guard drives them off. It is waterborne civil disobedience.

“These are small, subsistence fishermen,” said Evan P. Garcia, undersecretary for policy in the Philippines’ Department of Foreign Affairs. “They’re not a threat to anybody. And it’s not as if they just went there yesterday.”

The fish they’re after may be the other big casualty of the dispute. The tensions over the years have kept anyone from getting good data on fish stocks or devising a conservation plan. Hundreds of millions of people live around the South China Sea and eat its fish. The Marine Stewardship Council, with an office in Singapore, says that the humpback wrasse and bluefin tuna populations are close to collapse. Edgardo Gomez, a marine biologist in Manila, said that the Chinese have wiped out the giant clams on Scarborough and that their construction work is destroying reefs that support the bottom levels of the sea’s food chain.

“You have tons and tons of marine life in and around those reefs that are now gone,” he said.

The hatch is being shut on a way of life. The United States and China are either pursuing strategic advantage or practicing destructive gamesmanship, depending on the perspective. Filipinos have to live with that — with the “odd detour,” as Garcia put it, that brought them here.

Viany Mula would trade his motorbike in the blink of an eye for a chance to return to sea. But that is not going to happen.

Englund visited the Philippines on a Jefferson Fellowship, supported by the East-West Center.

Read more

Beijing’s power play in the South China Sea may be killing coral reefs

Here’s why some in the Philippines want the U.S. Navy back



Chinese warnings to U.S. plane hint of rising stakes over disputed islands"""

league = """
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'
'jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion''jungle is a twitch streamer going to win with vladimir as a champion'v'jungle is a twitch streamer going to win with vladimir as a champion'
"""


spanish = """
Hola! Yo empecé aprendo Español hace dos mes en la escuela. Yo voy la universidad. Yo tratar estudioso Español tres hora todos los días para que yo saco mejor rápido. ¿Cosa algún yo debo hacer además construir mí vocabulario? Muchas veces yo estudioso la palabras solo para que yo construir mí voabulario rápido. Yo quiero empiezo leo el periódico Español la próxima semana. Por favor correcto algún la equivocaciónes yo hisciste. Gracias!"""
# %%
comment_text = (
    """
comment
"""
    * 1000
)

league_of_legends = """
Identity of K'Sante (Design/Live server)
K'Sante was created to fill the role of a "high skill tank" who can "take things into his own hand" and outplay opponents.

From a design perspective, this fantasy was executed exceptionally well imo. Initially, you experience the slower, more deliberate feel of a tank, but after using his Ultimate, which shatters his Ntofos, you really feel the effect of "shedding that weight".

His movements and fighting become faster and more fluid, enhancing the gameplay experience, creating a contrast that also manages to compliment the concept.

Obviously, this design doesn't sit well with a lot of people, which... fair. I'm not here to say that those opinions don't matter.

SoloQ WR and Proplay
I see a lot of people joke about how op.gg and other sites say he has a 47% (current patch) winrate and that he is a bad champion, but this isn't really the case.

The champion himself isn't complex and can easily picked up, but the full potential of K'Sante needs a lot of practice and experience. As a high skill champ, he needs dedication to be piloted properly. The wr you see on stat sites just shows the average and surprisingly, K'Sante has a decently high pickrate, but this doesnt entail everyone who plays him have the mastery the champion needs or wants. While he is in the 45-47% range for average player statistics, his winrate for those dedicated mains are closer to the 50-53% range.



"""
mad_god = """
Dark condensation soul, degenerates can be free, awakens, endless charm of deep sleep in my blood.” I have released the Fallen Angel energy, the black fog of big piece revolves me, the black wing arrives at the world once more, my whole body covers in the dark mist, the Lion-Man mask on face was reduced to ashes under the tyrannical strength,
"""
java_text = """

public class BinaryConverter {

    public static void main(String[] args){
        for(int i = -5; i < 33; i++){
            System.out.println(i + ": " + toBinary(i));
            System.out.println(i);
            //always another way
            System.out.println(i + ": " + Integer.toBinaryString(i));
        }
    }

    /*
     * pre: none
     * post: returns a String with base10Num in base 2
     */
    public static String toBinary(int base10Num){
        boolean isNeg = base10Num < 0;
        base10Num = Math.abs(base10Num);
        String result = "";

        while(base10Num > 1){
            result = (base10Num % 2) + result;
            base10Num /= 2;
        }
        assert base10Num == 0 || base10Num == 1 : "value is not <= 1: " + base10Num;

        result = base10Num + result;
        assert all0sAnd1s(result);

        if( isNeg )
            result = "-" + result;
        return result;
    }

    /*
     * pre: cal != null
     * post: return true if val consists only of characters 1 and 0, false otherwise
     */
    public static boolean all0sAnd1s(String val){
        assert val != null : "Failed precondition all0sAnd1s. parameter cannot be null";
        boolean all = true;
        int i = 0;
        char c;

        while(all && i < val.length()){
            c = val.charAt(i);
            all = c == '0' || c == '1';
            i++;
        }
        return all;
    }
}



"""
genesis = """
Book of Genesis
Chapter 1
In the beginning God created heaven, and earth.
2 And the earth was void and empty, and
darkness was upon the face of the deep; and the
spirit of God moved over the waters.
3 And God said: Be light made. And light
was made.
4 And God saw the light that it was good; and
he divided the light from the darkness.
5 And he called the light Day, and the darkness Night; and there was evening and morning
one day.
6 And God said: Let there be a firmament
made amidst the waters: and let it divide the
waters from the waters.
7 And god made a firmament, and divided
the waters that were under the firmament, from
those that were above the firmament, and it was
so.
8 And God called the firmament, Heaven; and
the evening and morning were the second day.
9 God also said; Let the waters that are under
the heaven, be gathered together into one place:
and let the dry land appear. And it was so done.
10 And God called the dry land, Earth; and
the gathering together of the waters, he called
Seas. And God saw that it was good.
11 And he said: let the earth bring forth green
herb, and such as may seed, and the fruit tree
yielding fruit after its kind, which may have seed
in itself upon the earth. And it was so done.
12 And the earth brought forth the green
herb, and such as yieldeth seed according to its
kind, and the tree that beareth fruit, having seed
each one according to its kind. And God saw
that it was good.
13 And the evening and the morning were the
third day.
14 And God said: Let there be lights made
in the firmament of heaven, to divide the day
and the night, and let them be for signs, and for
seasons, and for days and years:
15 To shine in the firmament of heaven, and
to give light upon the earth, and it was so done.
16 And God made two great lights: a greater
light to rule the day; and a lesser light to rule
the night: and The stars.
17 And he set them in the firmament of heaven
to shine upon the earth.
18 And to rule the day and the night, and to
divide the light and the darkness. And God saw
that it was good.
19 And the evening and morning were the
fourth day.
20 God also said: let the waters bring forth
the creeping creature having life, and the fowl
that may fly over the earth under the firmament
of heaven.
21 And God created the great whales, and
every living and moving creature, which the
waaters brought forth, according to their kinds,
and every winged fowl accordi
22 God blessed them, saying, "Be fruitful and multiply and fill the waters in the seas, and let birds multiply on the earth."
23 And there was evening and there was morning, the fifth day.
24 And God said, "Let the earth bring forth living creatures of every kind: cattle and creeping things and wild animals of the earth of every kind." And it was so.
25 God made the wild animals of the earth of every kind, and the cattle of every kind, and everything that creeps upon the ground of every kind. And God saw that it was good.
26 Then God said, "Let us make humankind in our image, according to our likeness; and let them have dominion over the fish of the sea, and over the birds of the air, and over the cattle, and over all the wild animals of the earth, and over every creeping thing that creeps upon the earth."
27 So God created humankind in his image, in the image of God he created them; male and female he created them.
28 God blessed them, and God said to them, "Be fruitful and multiply, and fill the earth and subdue it; and have dominion over the fish of the sea and over the birds of the air and over every living thing that moves upon the earth."
29 God said, "See, I have given you every plant yielding seed that is upon the face of all the earth, and every tree with seed in its fruit; you shall have them for food.
30 And to every beast of the earth, and to every bird of the air, and to everything that creeps on the earth, everything that has the breath of life, I have given every green plant for food." And it was so.
31 God saw everything that he had made, and indeed, it was very good. And there was evening and there was morning, the sixth day.
"""

dnd_text = """

Every character belongs to a race, one of the many intelligent humanoid species in the D&D world. The most common player character races are dwarves, elves, halflings, and humans. Some races also have subraces, such as mountain dwarf or wood elf. The Races section provides more information about these races.

The race you choose contributes to your character’s identity in an important way, by establishing a general appearance and the natural talents gained from culture and ancestry. Your character’s race grants particular racial traits, such as special senses, proficiency with certain weapons or tools, proficiency in one or more skills, or the ability to use minor spells. These traits sometimes dovetail with the capabilities of certain classes (see step 2). For example, the racial traits of lightfoot halflings make them exceptional rogues, and high elves tend to be powerful wizards. Sometimes playing against type can be fun, too. Halfling paladins and mountain dwarf wizards, for example, can be unusual but memorable characters.

Your race also increases one or more of your ability scores, which you determine in step 3. Note these increases and remember to apply them later.

Record the traits granted by your race on your character sheet. Be sure to note your starting languages and your base speed as well.

BUILDING BRUENOR, STEP 1

Bob is sitting down to create his character. He decides that a gruff mountain dwarf fits the character he wants to play. He notes all the racial traits of dwarves on his character sheet, including his speed of 25 feet and the languages he knows: Common and Dwarvish.

2. Choose a Class
Every adventurer is a member of a class. Class broadly describes a character’s vocation, what special talents he or she possesses, and the tactics he or she is most likely to employ when exploring a dungeon, fighting monsters, or engaging in a tense negotiation. The character classes are described in the Classes section.

Your character receives a number of benefits from your choice of class. Many of these benefits are class features — capabilities (including spellcasting) that set your character apart from members of other classes. You also gain a number of proficiencies: armor, weapons, skills, saving throws, and sometimes tools. Your proficiencies define many of the things your character can do particularly well, from using certain weapons to telling a convincing lie.

On your character sheet, record all the features that your class gives you at 1st level.

Level
Typically, a character starts at 1st level and advances in level by adventuring and gaining experience points (XP). A 1st-level character is inexperienced in the adventuring world, although he or she might have been a soldier or a pirate and done dangerous things before.

Starting off at 1st level marks your character’s entry into the adventuring life. If you’re already familiar with the game, or if you are joining an existing D&D campaign, your DM might decide to have you begin at a higher level, on the assumption that your character has already survived a few harrowing adventures.

Record your level on your character sheet. If you’re starting at a higher level, record the additional elements your class gives you for your levels past 1st. Also record your experience points. A 1st-level character has 0 XP. A higher-level character typically begins with the minimum amount of XP required to reach that level (see “Beyond 1st Level” later in this section).

QUICK BUILD

Each class description in the Classes section includes a section offering suggestions to quickly build a character of that class, including how to assign your highest ability scores, a background suitable to the class, and starting spells.

Hit Points and Hit Dice
Your character’s hit points define how tough your character is in combat and other dangerous situations. Your hit points are determined by your Hit Dice (short for Hit Point Dice).

At 1st level, your character has 1 Hit Die, and the die type is determined by your class. You start with hit points equal to the highest roll of that die, as indicated in your class description. (You also add your Constitution modifier, which you’ll determine in step 3.) This is also your hit point maximum.

Record your character’s hit points on your character sheet. Also record the type of Hit Die your character uses and the number of Hit Dice you have. After you rest, you can spend Hit Dice to regain hit points (see “Resting” in the Adventuring section).

Proficiency Bonus
The table that appears in your class description shows your proficiency bonus, which is +2 for a 1st-level character. Your proficiency bonus applies to many of the numbers you’ll be recording on your character sheet:

Attack rolls using weapons you’re proficient with
Attack rolls with spells you cast
Ability checks using skills you’re proficient in
Ability checks using tools you’re proficient with
Saving throws you’re proficient in
Saving throw DCs for spells you cast (explained in each spellcasting class)

"""
bbc_text = """
Flash floods and heavy rain batter England and Wales
A woman wearing a plastic ran poncho cycles through floodwater after the River Thames overtopped its banks in London
Image source,Reuters
Image caption,
A woman cycles through floodwater after the River Thames overtopped its banks in London

Vicky Wong
BBC News
Matt Taylor
Lead Weather Presenter
@MetMattTaylor
Published
23 September 2024, 08:13 BST
Updated 52 minutes ago
Heavy rain and flash flooding has battered parts of England and Wales, causing widespread travel disruption and damage to properties.

Roads and houses have flooded in central and southern England, after some experienced a month's worth of rain in a matter of hours.

In London, a sinkhole has appeared on AFC Wimbledon's football pitch and 999 call handlers have taken 350 flood-related calls, while in Bedford a main road is totally submerged.

A Met Office amber weather warning in parts of central and southern England ended at 21:00 BST, but a yellow warning remains in place across England and south-east Wales.


Media caption,
Watch: A421 submerged by flood water in Bedfordshire

The yellow weather warning for heavy rain - which means some disruption like floods, travel disruption and power cuts are possible - is due to remain until 23:59. Only the far south-west and parts of northern England are not covered.

The Environment Agency has issued more than 20 flood warnings, meaning flooding is expected, and more than 80 flood alerts, meaning flooding is possible.

Areas affected by the flood warnings include Leighton Buzzard and Luton in Bedfordshire and parts of London.

Among the most dramatic floods was on the A421 main road between Bedford and Milton Keynes, which has been shut - along with the rail line from from Bedford to Bletchley.

Sheep in a temporary pen, there are bags of hay outside it
Image source,PA Media
Image caption,
Sheep in Marston Moretaine had to be dragged to safety in chest-high floodwater and placed in temporary pens

In the village of Grendon, in Northamptonshire, several houses were flooded with clean-up efforts ongoing.

"It was unbelievable," Jon Sayle said describing how about "two feet of water seeped in overnight" into his home.

In the village of Marston Moretaine, Bedfordshire, local residents joined forces to save animals stranded by heavy flooding at a local farm.

Joanna Johnson, 54, said 50 neighbours turned up at Moreteyne's Retreat after she sent an emergency message on social media. "The villagers flocked here so fast," she told PA News agency.

She said her miniature ponies had to swim out of the floodwater, while the sheep were dragged through to safety in chest-high floodwater.

Ms Johnson described the water coming off the nearby A421 as being "like a river", which led the entire farm being flooded within 15 minutes.

"The animals are alive at the moment, I'm now desperately trying to find a piece of land I can leave them on over the winter where they will be safe."

Another resident told PA he had never seen anything like in a decade living there, adding floods on the road were "normally gone within a few hours".

Lee Elliott, 36, said: "I was out last night helping push cars out of the floods because we came home quite late last night and saw the cars stuck in there, so we went down there to help them."

The open boot of a car is visible above the water where the vehicle is submerged in flood water on a421 in Marston Moretaine
Image source,PA Media
Image caption,
All that can be seen of on one car on the A421 is its open boot

A car partly submerged in floodwater on Manor Road in Wallington
Image source,London Fire Brigade
Image caption,
Overnight the London Fire Brigade attended to a vehicle stranded in floodwater in Wallington, Sutton

On Monday afternoon the London Fire Brigade said its 999 control officers had taken some 350 flood-related calls, with firefighters rescuing people trapped inside cars, assisting people from their homes and responding to flooding in Underground stations and roads.

In a post on X, using a photo of a car stranded in floodwater overnight in Wallington, Sutton, the fire brigade warned that "a foot of moving water at just 6mph is enough to float a car, external".

Transport for London has warned passengers that the District, Circle, Metropolitan, Piccadilly, Bakerloo and Central lines have been either partly suspended or subject to minor to severe delays because of flooding caused by heavy rain.

National Rail is also reporting widespread disruption and cancellations to some train services throughout the day and has urged passengers to check their journeys.

In south-east England, a night of heavy rain forced the closure of an M25 slip road at Cobham in Surrey and led to delays on train services.

A number of schools in areas including Bedfordshire and Oxfordshire have been forced to close, with some switching to remote learning, and a number of homes and businesses have been flooded.
"""

# %%

model.to_str_tokens(((((W_E*e_normalize)/(torch.sqrt(torch.tensor(768.0)))+W_E@gpt2_vecs)@model.blocks[0].mlp.W_in)@(model.blocks[0].mlp.W_out@model.blocks[1].mlp.W_in[:,15])>12.0).nonzero())
/# %%

# %%
