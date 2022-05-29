import torch.nn as nn
import numpy as np
import torch
from softmax import softmax, npsoftmax
from nllloss import nllloss, npnllloss
from torch import Tensor


def npCrossEntropyLoss(arr: np.ndarray, target: np.ndarray):
    sout = npsoftmax(arr)
    lout = np.log(sout)
    result = npnllloss(lout, target)
    return result


def CrossEntropyLoss(x: Tensor, target: Tensor) -> Tensor:
    softmax_out = softmax(x)
    log_out = torch.log(softmax_out)
    nllloss_loss = nllloss(log_out, target)
    return nllloss_loss


if __name__ == "__main__":
    tensor = torch.randn(3, 3)
    target = torch.tensor([2, 0, 1])

    ndarr = np.array(tensor)
    ndtar = np.array(target)

    nncel = nn.CrossEntropyLoss()
    nncelloss = nncel(tensor, target)
    print('{} : {}'.format('nncelloss', nncelloss))

    mynpcel = npCrossEntropyLoss(ndarr, ndtar)
    print('{} : {}'.format('mynpcel', mynpcel))

    mycel = CrossEntropyLoss(tensor, target)
    print('{} : {}'.format('mycel', mycel))
