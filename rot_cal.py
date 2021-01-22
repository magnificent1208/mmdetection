import numpy as np
import cv2
import math


def iou_rotate_calculate(boxes1, boxes2):
    """Calculate iou for a pair of rot box.(Matrix not support)
    """
    # import pdb; pdb.set_trace()
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = np.zeros([boxes1.shape[0], boxes2.shape[0]])

    for i in range(len(boxes1)):
        r1 = ((boxes1[i, 0], boxes1[i, 1]), (boxes1[i, 2], boxes1[i, 3]), boxes1[i, 4])
        for j in range(len(boxes2)):
            r2 = ((boxes2[j, 0], boxes2[j, 1]), (boxes2[j, 2], boxes2[j, 3]), boxes2[j, 4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                ious[i][j] = int_area * 1.0 / (area1[i] + area2[j] - int_area)
            else:
                ious[i][j] = 0
    return ious


box1 = np.array([[0, 0, 395, 295, 0]])
box2 = np.array([[0, 0, 395, 295, 340]])

if __name__ == "__main__":
    print(iou_rotate_calculate(box1, box2))