import torch


def array(n, f):
    arr = [f(i) for i in range(n)]
    arr = [i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in arr]
    return torch.stack(arr)


def sum(n, f):
    return array(n, f).sum(dim=0)
