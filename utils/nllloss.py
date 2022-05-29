import numpy as np
import torch.nn as nn
from torch import Tensor
import torch


def npnllloss(arr: np.ndarray, target: np.ndarray):
    result = np.array([-a[t] for a, t in zip(arr, target)]).mean()
    return result


def nllloss(x: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.tensor([-_x[t] for _x, t in zip(x, target)]))


if __name__ == "__main__":
    tensor = torch.randn(3, 3)
    target = torch.tensor([2, 0, 1])

    ndarr = np.array(tensor)
    ndtar = np.array(target)

    nn_nllloss = nn.NLLLoss()
    nnloss = nn_nllloss(tensor, target)
    print('{} : {}'.format('nn_nlllos', nnloss))

    my_npnllloss = npnllloss(ndarr, ndtar)
    print('{} : {}'.format('my_npnllloss', my_npnllloss))

    my_nllloss = nllloss(tensor, target)
    print('{} : {}'.format('my_nllloss', my_nllloss))
