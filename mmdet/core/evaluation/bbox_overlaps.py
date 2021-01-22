import numpy as np
import cv2
import math


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def iou_rotate_calculate(boxes1, boxes2, rad=True):
    """Calculate iou for a pair of rot box.(Matrix not support)
    """
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = np.zeros([boxes1.shape[0], boxes2.shape[0]])

    if rad:
        boxes1[:, 4] *= 180 / math.pi
        boxes2[:, 4] *= 180 / math.pi

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