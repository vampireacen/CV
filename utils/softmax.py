import numpy as np
import torch
from torch import Tensor


def npsoftmax(arr: np.ndarray):
    row_max = np.max(arr, axis=1)  # 每一行最大值
    arr -= row_max.reshape(-1, 1)  # 防止溢出
    exp_X = np.exp(arr)  # e^xi
    exp_SUM = np.sum(exp_X, axis=1, keepdims=True)  # sum(e^yi)
    result = exp_X / exp_SUM
    return result


def softmax(x: Tensor) -> Tensor:
    row_max = torch.max(x, dim=1, keepdim=True).values
    x -= row_max
    exp_x = torch.exp(x)
    exp_sum = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x / exp_sum


if __name__ == "__main__":
    arr = torch.randn(3, 3)
    ndarr = np.array(arr)
    result = npsoftmax(ndarr)
    tresutl = torch.softmax(arr, dim=1)
    result1 = softmax(arr)
    print(result)
    print(tresutl)
    print(result1)
