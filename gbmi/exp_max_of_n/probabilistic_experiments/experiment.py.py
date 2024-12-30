# %%
import torch
import transformer_lens
import torch

model = transformer_lens.HookedTransformer.from_pretrained("gpt2-xl")

# %%
import torch

# %%
str_ = "What common colour do I get if I mix red and white paint? A:"
for i in range(100):
    logits, activations = model.run_with_cache(new_str)
    tok_ = model.to_str_tokens(
        torch.multinomial(
            torch.softmax(20* logits.squeeze()[-1], dim=0), num_samples=1
        )
    )[0]
    new_str += tok_
print(new_str)
# %%
import json
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-small")
f = open("british_dict.txt","r")
dict_ = f.read()
british_dict = json.loads(dict_)
f.close()
f = open("american_dict.txt","r")
dict_ = f.read()
american_dict = json.loads(dict_)
f.close()
total_tokens = 50257
l =1
british_sum = torch.tensor(sum([british_dict[i] for i in british_dict.keys()]))
american_sum = torch.tensor(sum([american_dict[i] for i in american_dict.keys()]))
def get_cond_prob(word,dict_,sum_):
    word = str(word)
    return (dict_.get(word,0)+l)/(sum_+l*total_tokens)
cond_probs = []
for i in tqdm(range(50257)):
    if not str(i) in british_dict.keys() or not str(i) in american_dict.keys() or british_dict[str(i)] < 20 or american_dict[str(i)]<20:
        cond_probs.append(0)
    else:
        cond_probs.append(torch.log(get_cond_prob(i,british_dict,british_sum))-torch.log(get_cond_prob(i,american_dict,american_sum)).item())
cond_probs = torch.tensor(cond_probs)
#%%
print(model.to_str_tokens(torch.topk(cond_probs,k=300).indices))
def graph_top(mat,k):
    plt.scatter([i for i in torch.topk(mat,k=k).indices.detach().cpu()],torch.topk(mat,k=k).values.detach().cpu())

k = 200
graph_top(big[:,300],k)
graph_top(cond_probs*5.8,k)
plt.show()
plt.clf()
plt.scatter([i for i in range(k)],torch.topk(big[:,300],k=k).values.detach().cpu())
plt.show()
plt.clf()
plt.scatter([i for i in range(k)],torch.topk(cond_probs,k=k).values.detach().cpu())
# %%
k=300
first_p = torch.topk(cond_probs,1000).indices
second_p = torch.topk(big[:,300],300).indices
arr_ = np.intersect1d(first_p.detach().cpu().numpy(),second_p.detach().cpu().numpy())
# %%
class Text:
    def __init__(self,text,name,moving_window=-1):
        self.name = name
        self.text = text
        self.moving_window = moving_window
        self.tokenized_text = model.to_str_tokens(self.text)
        self.length = len(self.tokenized_text)
        self.mean_pattern = (1-torch.ones(self.length,self.length).triu(diagonal=1))/(torch.arange(1,self.length+1).unsqueeze(0).T)
        if moving_window != -1:
            for i in range(self.length):
                if i>moving_window:
                    self.mean_pattern[i][:i-moving_window] = 0.0

                    self.mean_pattern[i][i-moving_window:i+1] = 1/moving_window
        self.mean_pattern =self.mean_pattern.to('cuda')
    def trace_neuron(self,neuron,length=-1,layer=0):
        length = self.length if length == -1 else min(length,self.length)
        _,self.activations = model.run_with_cache(self.text)
        plt.xlabel('Position in sequence')
        plt.ylabel(f'Pre-activation of neuron {neuron}')
        plt.scatter([i for i in range(length)],self.activations[f'blocks.{layer}.mlp.hook_pre'].squeeze()[:length,neuron].detach().cpu(),label=f'{self.name}')

    def trace_head_neuron(self,neuron,head,ax=plt,length=-1,full_ov = False,mean_diff=False,layer=0,positional=True,marker='x'):
        _,self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length,self.length)


        attn_pattern = self.activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head]

        if mean_diff:
            attn_pattern =self.mean_pattern.to('cuda')
            marker = '.'
        sequence = W_E[model.to_tokens(self.text)].squeeze()*E_factor[model.to_tokens(self.text)].squeeze().unsqueeze(1)
        if positional:
            sequence = sequence + W_pos[:length]*(3.35/(torch.sqrt(11.2+(e_normalize[model.to_tokens(self.text)].squeeze()[:length])**2))).unsqueeze(1)
        if full_ov:
             head_neuron = (attn_pattern@(sequence)).squeeze()@gpt2_vecs@model.blocks[layer].mlp.W_in[:,neuron]
        else:
            head_neuron = ((attn_pattern@(sequence)).squeeze()@model.W_V[layer,head]@model.W_O[layer,head]@model.blocks[layer].mlp.W_in[:,neuron])[:length]


        ax.scatter([i for i in range(length)],head_neuron[:length].detach().cpu(),marker=marker)#label=f'Contribution of head {head} to neuron {neuron}, on {self.name}')
    def trace_real_neuron(self,neuron,head,length=-1,ax=plt,layer=0):
        _,self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length,self.length)
        marker='o'
        ax.scatter([i for i in range(length)],(self.activations[f'blocks.{layer}.attn.hook_z'].squeeze()[:length,head]@model.W_O[layer,head]@model.blocks[0].mlp.W_in[:,neuron]).detach().cpu())


    def trace_first_attention(self,head,ax=plt,length=-1,layer=0):
        _,self.activations = model.run_with_cache(self.text)
        length = self.length if length == -1 else min(length,self.length)
        first_attn = self.activations[f'blocks.{layer}.attn.hook_pattern'].squeeze()[head][:length,0]

        ax.xlabel('Position in sequence')
        ax.scatter([i for i in range(length)],first_attn.detach().cpu())




def splice(text1,text2,splice_start,splice_end):

    return Text(''.join(text1.tokenized_text[1:splice_start+1]+text2.tokenized_text[1:(2+splice_end-splice_start)]+text1.tokenized_text[2+splice_end:]),name=f'{text1.name} with {text2.name} spliced in between tokens {splice_start} and {splice_end}',moving_window=text1.moving_window)
# %%
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
The table that appears in your class description shows your proficiency bonus, which is +2 for a 1st-level character. Your proficiency bonus applies to many of the
Record your character’s hit points on your character sheet. Also record the type of Hit Die your character uses and the number of Hit Dice you have. After you rest, you can spend Hit Dice to regain hit points (see “Resting” in the Adventuring section).

Proficiency Bonus
The table that appears in your class description shows your proficiency bonus, which is +2 for a 1st-level character. Your proficiency bonus applies to many of the"""
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

In a post on X, using a photo of a car stranded in floodwater overnight in Wallington, Sutton, the
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

In a post on X, using a photo of a car stranded in floodwater overnight in Wallington, Sutton, the"""
#%%
moving_window = -1
dnd = Text(dnd_text,"dnd",moving_window)
bible = Text(genesis,'bible',moving_window)
fishing = Text(fishing_news,'Fishing news story',moving_window)
#league = Text(league_of_legends,'LoL forum post')
code = Text(code_text,'code',moving_window)
#comments = Text(comment_text,'comment spam')
brackets = Text('((())) )(  )      j j     j  j     (   j j j j j j j j j j j  j j j j ())','brackets')
bible_code = splice(bible,code,100,600)
code_bible = splice(code,bible,100,600)
java = Text(java_text,'java',moving_window)
bbc_news = Text(bbc_text,'bbc',moving_window)
# %%
neuron = 1498
length = 400
#length = min(length,bible_code.length,bible.length)
bible.trace_neuron(neuron,length)
#bible_code.trace_neuron(neuron,length)

plt.legend()
plt.show()
plt.clf()
#bible_diff = bible_code.activations['blocks.0.mlp.hook_pre'].squeeze()[:length,neuron]-bible.activations['blocks.0.mlp.hook_pre'].squeeze()[:length,neuron]
plt.scatter([i for i in range(length)],bible_diff.detach().cpu(),label=f'Pre-activation difference of \'{bible_code.name}\' and \'{bible.name}\' ')
plt.xlabel('Position in sequence')
plt.ylabel(f'Pre-activation value of neuron {neuron}')
plt.legend()
plt.show()
#%%
brackets.trace_neuron(neuron,length)
gpt2_vecs = (


    + model.W_V[0, 0]@ model.W_O[0, 0]
  + model.W_V[0, 2]@ model.W_O[0, 2]

    + model.W_V[0, 9]@ model.W_O[0, 9]
    +model.W_V[0, 8]@ model.W_O[0, 8]
    +model.W_V[0,10]@model.W_O[0,10]
    +model.W_V[0,7]@model.W_O[0,7]
)
# %%
neuron = 1498

length=1300
for head in range(12):
    print(f'head:{head}')
    bible.trace_head_neuron(neuron,head,length=length,full_ov=False,layer=0)
  #  code.trace_first_attention(head,length=-1,layer=5)
    plt.legend()
    plt.show()
    plt.clf()
print('trace of neuron:')
bible.trace_neuron(neuron,length,layer=0)
# %%
length =1024
plt.clf()
fig, axs = plt.subplots(3,4,figsize=(10,10),sharex='all', sharey='all')
layer=1
neuron = 37
for i in range(3):
    for j in range(4):
        head = 4*i+j

        bible_code.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=False,layer=layer,positional=False,marker='.')

      #  java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=0,positional=True)

#        java.trace_head_neuron(neuron,head=head,length=length,ax=axs[i,j],mean_diff=True,layer=layer,positional=False,marker='.')


        axs[i,j].set_title(f'Head {head}')

for ax in axs.flat:
    ax.set(xlabel='Position', ylabel=f'Pre-activation contribution to neuron {neuron}')


# %%

java.trace_neuron(neuron)


# %%
text = genesis
vecs = torch.zeros(12,768)
for head in range(12):
    vecs[head] = W_E[model.to_tokens(text).squeeze()].squeeze().mean(dim=0)@model.W_V[0,head]@model.W_O[0,head]
show(((vecs@vecs.T)/(torch.norm(vecs,dim=1).unsqueeze(0)))/(torch.norm(vecs,dim=1).unsqueeze(0).T))
# %%
