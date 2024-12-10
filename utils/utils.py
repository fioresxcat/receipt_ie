import numpy as np
import cv2
from pathlib import Path
from bpemb import BPEmb
import unidecode
import pdb
import os
import json
import copy
from shapely.geometry import Polygon


def order_points(self, pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect



def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def max_left(bb):
    return min(bb[0], bb[2], bb[4], bb[6])

def max_right(bb):
    return max(bb[0], bb[2], bb[4], bb[6])

def row_bbs(bbs):
    bbs.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in bbs:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters

def sort_bbs(bbs):
    bb_clusters = row_bbs(bbs)
    bbs = []
    for cl in bb_clusters:
        bbs.extend(cl)
    return bbs, bb_clusters


def sort_json(json_data):
    bbs, labels, texts = [], [], []
    for shape in json_data['shapes']:
        if len(shape['points']) == 2:
            continue
        x1, y1 = shape['points'][0]  # tl
        x2, y2 = shape['points'][1]  # tr
        x3, y3 = shape['points'][2]  # br
        x4, y4 = shape['points'][3]  # bl
        bb = tuple(int(i) for i in (x1,y1,x2,y2,x3,y3,x4,y4))
        bbs.append(bb)
        labels.append(shape['label'])
        if 'text' in shape:
            texts.append(shape['text'])
        else:
            texts.append('')

    bb2label = dict(zip(bbs, labels))   # theo thu tu truyen vao trong data['shapes']
    bb2text = dict(zip(bbs, texts))
    bb2idx_original = {x: idx for idx, x in enumerate(bbs)}   # theo thu tu truyen vao trong data['shapes']
    rbbs = row_bbs(copy.deepcopy(bbs))
    sorted_bbs = [bb for row in rbbs for bb in row]  # theo thu tu tu trai sang phai, tu tren xuong duoi
    bb2idx_sorted = {tuple(x): idx for idx, x in enumerate(sorted_bbs)}   # theo thu tu tu trai sang phai, tu tren xuong duoi
    sorted_indices = [bb2idx_sorted[bb] for bb in bb2idx_original.keys()]

    return bb2label, bb2text, rbbs, bb2idx_original



def iou_poly(poly1, poly2):
    poly1 = np.array(poly1).flatten().tolist()
    poly2 = np.array(poly2).flatten().tolist()

    xmin1, xmax1 = min(poly1[::2]), max(poly1[::2])
    ymin1, ymax1 = min(poly1[1::2]), max(poly1[1::2])
    xmin2, xmax2 = min(poly2[::2]), max(poly2[::2])
    ymin2, ymax2 = min(poly2[1::2]), max(poly2[1::2])

    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return 0, 0, 0

    if len(poly1) == 4:  # if poly1 is a box
        x1, y1, x2, y2 = poly1
        poly1 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    if len(poly2) == 4:  # if poly2 is a box
        x1, y1, x2, y2 = poly2
        poly2 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if len(poly1) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly1
        poly1 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    if len(poly2) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly2
        poly2 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    ratio1 = intersect / poly1.area
    ratio2 = intersect / poly2.area
    iou = intersect / union
    
    return ratio1, ratio2, iou