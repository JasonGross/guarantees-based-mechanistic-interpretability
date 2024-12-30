# %%
import transformer_lens
import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


def show(matrix):

    if len(matrix.shape) == 1:
        matrix = matrix.unsqueeze(0)

    if matrix.shape[0] > 1500 or matrix.shape[1] > 1500:
        print("too big")
        return

    px.imshow(matrix.detach().cpu()).show()


# %%

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


# %%
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

ioi = genesis[:1000] + "Molly and Mike and Mike gave to " ""
ioi_corrupted = genesis[:1000] + "Mike and Molly and Mike gave to "

_, activations = model.run_with_cache(ioi)
_, corrupted_activations = model.run_with_cache(ioi_corrupted)


def dist(mat):
    return (mat**2).sum(dim=-1)


# %%
index = 4
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
plt.colorbar()  # %%

# %%
show(W_pos @ model.W_V[0, 1] @ model.W_O[0, 1])
# %%
layer = 0
head = 10
index = 4
show(
    (activations[f"blocks.{layer}.attn.hook_pattern"])
    .squeeze()[head, -index][-200:]
    .reshape(20, 10)
)
plt.colorbar()
show(
    (corrupted_activations[f"blocks.{layer}.attn.hook_pattern"])
    .squeeze()[head, -index][-200:]
    .reshape(20, 10)
)
plt.colorbar()
# %%
duplicate_heads = model.W_pos[40] @ (
    1.0 * model.W_V[0, 10] @ model.W_O[0, 10]
    + 1.0 * model.W_V[0, 5] @ model.W_O[0, 10]
    + 1.0 * model.W_V[0, 1] @ model.W_O[0, 1]
)
# %%
ioi = genesis[:100] + " Molly and Mike and God"
ioi_corrupted = genesis[:100] + " Mike and Molly and God"

_, activations = model.run_with_cache(ioi)
residual_stream = activations["blocks.0.hook_resid_mid"].squeeze()
plt.matshow((duplicate_heads @ residual_stream.T).detach().cpu()[:200].reshape(2, 16))
# %%
head = 7
first_approximation = (
    activations["blocks.0.attn.hook_z"].squeeze()[-10, head]
) @ model.W_O[0, head]
cosines = []
mike_index = len(model.to_str_tokens(ioi))
for pos in range(1024):
    pos_vec = W_pos[pos] @ model.W_V[0, head] @ model.W_O[0, head]
    cosines.append(
        torch.nn.CosineSimilarity(dim=0)(pos_vec.detach().clone(), first_approximation)
    )
# %%


show(torch.tensor(cosines).reshape(32, 32))
plt.colorbar
# %%
