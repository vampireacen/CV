import numpy as np
import torch.nn as nn
import torch
from torch import Tensor


def my_batch_norm(x: Tensor, running_mean: Tensor, running_var: Tensor, weight: Tensor = None, bias: Tensor = None,
                training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """BN(F.batch_norm()). 对NHW做归一化.

    :param x: shape = (N, In) or (N, C, H, W)
    :param running_mean: shape = (In,) 或 (C,) 下同
    :param running_var:
    :param weight: gamma
    :param bias: beta
    :param training:
    :param momentum: 动量实际为 1 - momentum. (同torch)
    :param eps:
    :return: shape = x.shape"""

    if training:
        if x.dim() == 2:
            _dim = (0,)
        elif x.dim() == 4:
            _dim = (0, 2, 3)
        else:
            raise ValueError("x dim error")
        mean = eval_mean = torch.mean(x, _dim)  # 总体 = 估计. shape = (In,) or (C,)
        eval_var = torch.var(x, _dim, unbiased=True)  # 无偏估计, x作为样本
        var = torch.var(x, _dim, unbiased=False)  # 用于标准化, x作为总体
        running_mean.data = (1 - momentum) * running_mean + momentum * eval_mean
        running_var.data = (1 - momentum) * running_var + momentum * eval_var  # 无偏估计
    else:
        mean = running_mean
        var = running_var
    # 2D时, mean.shape = (In,)
    # 4D时, mean.shape = (C, 1, 1)
    if x.dim() == 4:  # 扩维
        mean, var = mean[:, None, None], var[:, None, None]
        weight, bias = weight[:, None, None], bias[:, None, None]
    return (x - mean) * torch.rsqrt(var + eps) * (weight if weight is not None else 1.) + (bias if bias is not None else 0.)


if __name__ == "__main__":
    tensor = torch.randn(4, 3, 2, 2)
    N, C, H, W = tensor.shape
    nnbn = nn.BatchNorm2d(num_features=C, momentum=0.01, eps=1e-7)  # num_features 必须与 channel 相同
    nnr = nnbn(tensor)
    print(nnr)
    mybn = my_batch_norm(tensor, nnbn.running_mean, nnbn.running_var, nnbn.weight, nnbn.bias, momentum=nnbn.momentum,
                eps=nnbn.eps)
    print(mybn)
