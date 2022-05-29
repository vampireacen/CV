import numpy as np
import cv2


def draw(arr):
    bg = cv2.imread('./white.jpg')
    print(arr[0])
    print(type(arr[0]))
    print(arr[0][0])
    print(type(arr[0][0]))
    bg = cv2.resize(bg, (1500, 1500))
    for i in arr:
        cv2.rectangle(bg, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255))

    cv2.imshow('bg', bg)
    cv2.waitKey()
    pass


def NMS(arr: np.ndarray, thresh: float) -> list:
    # arr [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],···[x1,y1,x2,y2,score]]
    x1 = arr[:, 0]  # all x1
    y1 = arr[:, 1]  # all y1
    x2 = arr[:, 2]  # all x2
    y2 = arr[:, 3]  # all y2
    score = arr[:, 4]  # all score
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # all area
    order = score.argsort()[::-1]  # index of score sort high to low
    keep = []
    while order.size > 0:
        i = order[0]  # the first must be the highest score in its related bbox
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ious = inter / (areas[i] + areas[order[1:]] - inter)
        index = np.where(ious <= thresh)[0]
        order = order[index + 1]
        draw(arr[order])
    return keep


if __name__ == '__main__':
    dets = [[218, 322, 385, 491, 0.98], [247, 312, 419, 461, 0.83], [237, 344, 407, 510, 0.92],
            [757, 218, 937, 394, 0.96], [768, 198, 962, 364, 0.85], [740, 240, 906, 414, 0.83],
            [1101, 84, 1302, 303, 0.82], [1110, 67, 1331, 260, 0.97], [1123, 42, 1362, 220, 0.85]]
    arr = np.array(dets)

    keep = NMS(arr, 0.5)
    print(keep)
