import math

import torch

torch.set_default_device("cuda")


class MLP(torch.nn.Module):
    def __init__(self, d_mlp=6):
        super(MLP, self).__init__()
        self.d_mlp = d_mlp
        self.W_e = torch.nn.Parameter(-1 / 2 + torch.rand(self.d_mlp, 2))
        # self.W_internal = torch.nn.Parameter(
        #      -1 / 2 + torch.rand(self.d_mlp, self.d_mlp)
        #  )
        self.W_u = torch.nn.Parameter(-1 / 2 + torch.rand(1, self.d_mlp))

    def forward(self, x):
        return self.W_u @ torch.relu(self.W_e @ x)


batch_size = 100
batches = 10
# dataset = torch.pow(torch.cos(2 * (math.pi) * torch.rand(batches, 2, batch_size)), 5)
dataset = -1 + 2 * torch.rand((batches, 2, batch_size))
expectedanswers = torch.prod(dataset, 1)
epochs = 1000
loss = 1
while loss > 0.07 or torch.isnan(torch.tensor([loss])):
    if loss > 0.07:
        mlp = MLP()
        # optimizer = torch.optim.AdamW(
        #   mlp.parameters(), lr=1e-3, weight_decay=1, betas=(0.99, 0.98)
        # )
        optimizer = torch.optim.AdamW(
            mlp.parameters(), lr=1e-3, weight_decay=1, betas=(0.99, 0.98)
        )
    for e in range(epochs):
        for batch in range(batches):
            results = mlp.forward(dataset[batch])

            loss = torch.sqrt(torch.square(expectedanswers[batch] - results).mean())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(mlp.W_e, "embedding")

    print(mlp.W_u, "unembedding")
    # print(mlp.W_internal, "hidden")
    print(loss)
while loss > 0.01 or torch.isnan(torch.tensor([loss])):
    for e in range(epochs):
        for batch in range(batches):
            results = mlp.forward(dataset[batch])

            loss = torch.sqrt(torch.square(-expectedanswers[batch] - results).mean())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(mlp.W_e, "embedding")

    print(mlp.W_u, "unembedding")
    # print(mlp.W_internal, "hidden")
    print(loss)
