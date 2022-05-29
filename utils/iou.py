import numpy as np
import torch

from yolov5iou import bbox_iou


def iou(box1: np.ndarray, box2: np.ndarray):
    x11, x12 = box1[0], box2[0]
    y11, y12 = box1[1], box2[1]
    x21, x22 = box1[2], box2[2]
    y21, y22 = box1[3], box2[3]

    area1 = (x21 - x11) * (y21 - y11)
    area2 = (x22 - x12) * (y22 - y12)

    xx1 = np.maximum(x11, x12)
    yy1 = np.maximum(y11, y12)
    xx2 = np.minimum(x21, x22)
    yy2 = np.minimum(y21, y22)

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    union = (area1 + area2 - inter)
    iou = inter / union
    return iou


def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (carea + garea - area)

    return iou


if __name__ == "__main__":
    box1 = np.array([0, 0, 50, 50])
    box2 = np.array([25, 25, 75, 75])
    print(box1)
    print(box2)
    r1 = iou(box1, box2)
    r2 = calculateIoU(box1, box2)
    r3 = bbox_iou(torch.from_numpy(box1), torch.from_numpy(box2))
    print(r1)
    print(r2)
    print(r3)
