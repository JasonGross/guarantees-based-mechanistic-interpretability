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
