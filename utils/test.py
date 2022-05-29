import torch.nn as nn
import torch
from basic import _conv2d, _max_pool2d
import cv2

if __name__ == "__main__":
    tensor = torch.randn((1, 3, 224, 224))
    N, C, W, H = tensor.shape
    np_img = cv2.imread(r"F:\project\helmet_uniform\uniform_helmet\train\images\0000000.jpg")
    cv2.imshow('img', np_img)
    c = cv2.waitKey()
    # conv2d = nn.Conv2d(in_channels=8, out_channels=4, padding=1, stride=(2, 2), kernel_size=(3, 3))
    # result = conv2d(tensor)
    # r = _conv2d(tensor, conv2d.weight, conv2d.bias, stride=2, padding=1)
    r = _max_pool2d(tensor, 3, 1)
    # print(result)
    print(r)
