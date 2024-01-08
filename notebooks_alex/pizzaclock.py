
from gbmi.exp_modular_fine_tuning.train import MODULAR_ADDITION_113_CLOCK_CONFIG
from gbmi.model import train_or_load_model
import torch
from math import *
from torch import tensor
config = MODULAR_ADDITION_113_CLOCK_CONFIG
rundata,model = train_or_load_model(config)
for param in model.parameters():
    param.requires_grad = False
embeddingmatrix = model.embed.W_E.clone()

model.embed = torch.nn.Identity()
device = 'cuda'
p = 113
class DifferentModClock(torch.nn.Module):
    def __init__(self,q):
        super(DifferentModClock,self).__init__()
        self.W_e = torch.nn.Parameter(-0.5/sqrt(q+1) + torch.rand((p+1,q+1))/sqrt(q+1))
        self.W_u = torch.nn.Parameter(-0.5/sqrt(p) + torch.rand((q,p))/sqrt(p))
    def forward(self,x):

        z = torch.nn.functional.one_hot(x).to(device).T
        y = z @ self.W_e @ embeddingmatrix

        y.unsqueeze(-1)

        return self.W_u @ (model(y)[0])
Clock = DifferentModClock(2*113)

print(Clock(torch.tensor([1,2,113])))