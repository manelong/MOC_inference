import numpy as np


def takeLast(elem):
    return elem[-1]


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

def intersection_over_union(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    bbox1_area = bbox_area(bbox1)
    bbox2_area = bbox_area(bbox2)

    return inter_area / (bbox1_area + bbox2_area - inter_area)

def giou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Union area
    bbox1_area = bbox_area(bbox1)
    bbox2_area = bbox_area(bbox2)
    union_area = bbox1_area + bbox2_area - inter_area

    # IoU
    iou = inter_area / union_area

    # Smallest enclosing box
    enclosing_x_min = min(x1_min, x2_min)
    enclosing_y_min = min(y1_min, y2_min)
    enclosing_x_max = max(x1_max, x2_max)
    enclosing_y_max = max(y1_max, y2_max)

    enclosing_area = (enclosing_x_max - enclosing_x_min) * (enclosing_y_max - enclosing_y_min)

    # GIoU
    giou = iou - (enclosing_area - union_area) / enclosing_area

    return giou

def bbox_edge_distance(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    if x1_max < x2_min:  # bbox1 在 bbox2 左侧
        dx = x2_min - x1_max
    elif x2_max < x1_min:  # bbox2 在 bbox1 左侧
        dx = x1_min - x2_max
    else:  # bbox1 和 bbox2 在水平上有重叠
        dx = 0
    
    if y1_max < y2_min:  # bbox1 在 bbox2 上侧
        dy = y2_min - y1_max
    elif y2_max < y1_min:  # bbox2 在 bbox1 上侧
        dy = y1_min - y2_max
    else:  # bbox1 和 bbox2 在垂直上有重叠
        dy = 0

    distance = np.sqrt(dx**2 + dy**2)

    dis = distance/np.sqrt(bbox_area(bbox1))
    return dis

def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)

# TUBELETS
""" tubelets of length K are represented using numpy array with 4K columns """


def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0:
        dets
    if top_k is None:
        top_k = len(dets)

    K = int((dets.shape[1] - 1) / 4)

    # Coordinates of bounding boxes
    x1 = [dets[:, 4 * k] for k in range(K)]
    y1 = [dets[:, 4 * k + 1] for k in range(K)]
    x2 = [dets[:, 4 * k + 2] for k in range(K)]
    y2 = [dets[:, 4 * k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, -1]
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + K
    counter = 0

    while order.size > 0:
        i = order[0]
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][order[1:]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][order[1:]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][order[1:]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][order[1:]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([inter_area[k] / (area[k][order[1:]] + area[k][i] - inter_area[k]) for k in range(K)])
        index = np.where(ious > overlapThresh*K)[0]
        weight[order[index + 1]] = K - ious[index]

        index2 = np.where(ious <= overlapThresh * K)[0]
        order = order[index2 + 1]

    dets[:, -1] = dets[:, -1] * weight/K

    new_scores = dets[:, -1]
    new_order = np.argsort(new_scores)[::-1]
    dets = dets[new_order, :]

    return dets[:top_k, :]