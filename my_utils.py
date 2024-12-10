import os
import re
import pdb
import json
import math
import numpy as np
import unidecode
from bpemb import BPEmb
from pathlib import Path
from typing import List, Tuple
import copy
from model import *
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import itertools
import string
import random


DATA_DICT = {
    '711': {
        'train_dir': '../ie_data/711/train',
        'val_dir': '../ie_data/711/val',
        'test_dir': '../ie_data/711/test',
    },
    'aeon_citimart': {
        'train_dir': '../ie_data/aeon_citimart/train',
        'val_dir': '../ie_data/aeon_citimart/val',
        'test_dir': '../ie_data/aeon_citimart/test',
    },
    'aeon_combined': {
        'train_dir': '../ie_data/aeon_combined/train',
        'val_dir': '../ie_data/aeon_combined/val',
        'test_dir': '../ie_data/aeon_combined/test',

    },
    'BHD': {
        'train_dir': '../ie_data/BHD/train',
        'val_dir': '../ie_data/BHD/val',
        'test_dir': '../ie_data/BHD/test',
    },
    'bhx': {
        'train_dir': '../ie_data/bhx/train',
        'val_dir': '../ie_data/bhx/val',
        'test_dir': '../ie_data/bhx/test',
    },
    'bigc_old': {
        'train_dir': '../ie_data/bigc_old/train',
        'val_dir': '../ie_data/bigc_old/val',
        'test_dir': '../ie_data/bigc_old/test',

    },
    'bitis': {
        'train_dir': '../ie_data/bitis/train',
        'val_dir': '../ie_data/bitis/val',
        'test_dir': '../ie_data/bitis/test',

    },
    'bonchon': {
        'train_dir': '../ie_data/bonchon/train',
        'val_dir': '../ie_data/bonchon/val',
        'test_dir': '../ie_data/bonchon/test',

    },
    'brg': {
        'train_dir': '../ie_data/brg/train',
        'val_dir': '../ie_data/brg/val',
        'test_dir': '../ie_data/brg/test',
    },
    'bsmart': {
        'train_dir': '../ie_data/bsmart/train',
        'val_dir': '../ie_data/bsmart/val',
        'test_dir': '../ie_data/bsmart/test',
    },
    'cheers': {
        'train_dir': '../ie_data/cheers/train',
        'val_dir': '../ie_data/cheers/val',
        'test_dir': '../ie_data/cheers/test',
    },
    'circlek': {
        'train_dir': '../ie_data/circlek/train',
        'val_dir': '../ie_data/circlek/val',
        'test_dir': '../ie_data/circlek/test',

    },
    'coopfood': {
        'train_dir': '../ie_data/coopfood/train',
        'val_dir': '../ie_data/coopfood/val',
        'test_dir': '../ie_data/coopfood/test',
        
    },
    'coopmart': {
        'train_dir': '../ie_data/coopmart/train',
        'val_dir': '../ie_data/coopmart/val',
        'test_dir': '../ie_data/coopmart/test',

    },
    'dmx': {
        'train_dir': '../ie_data/dmx/train',
        'val_dir': '../ie_data/dmx/val',
        'test_dir': '../ie_data/dmx/test',

    },
    'don_chicken': {
        'train_dir': '../ie_data/don_chicken/train',
        'val_dir': '../ie_data/don_chicken/val',
        'test_dir': '../ie_data/don_chicken/test',

    },
    'emart': {
        'train_dir': '../ie_data/emart/train',
        'val_dir': '../ie_data/emart/val',
        'test_dir': '../ie_data/emart/test',
    },
    'familymart': {
        'train_dir': '../ie_data/familymart/train',
        'val_dir': '../ie_data/familymart/val',
        'test_dir': '../ie_data/familymart/test',
    },
    'fujimart': {
        'train_dir': '../ie_data/fujimart/train',
        'val_dir': '../ie_data/fujimart/val',
        'test_dir': '../ie_data/fujimart/test',

    },
    'galaxy_cinema': {
        'train_dir': '../ie_data/galaxy_cinema/train',
        'val_dir': '../ie_data/galaxy_cinema/val',
        'test_dir': '../ie_data/galaxy_cinema/test',

    },
    'gs25': {
        'train_dir': '../ie_data/gs25/train',
        'val_dir': '../ie_data/gs25/val',
        'test_dir': '../ie_data/gs25/test',

    },
    'guardian': {
        'train_dir': '../ie_data/guardian/train',
        'val_dir': '../ie_data/guardian/val',
        'test_dir': '../ie_data/guardian/test',

    },
    'hc': {
        'train_dir': '../ie_data/hc/train',
        'val_dir': '../ie_data/hc/val',
        'test_dir': '../ie_data/hc/test',
    },
    'heineken': {
        'train_dir': '../ie_data/heineken/train',
        'val_dir': '../ie_data/heineken/val',
        'test_dir': '../ie_data/heineken/test',
    },
    'heineken_2024': {
        'train_dir': '../ie_data/heineken_2024/train',
        'val_dir': '../ie_data/heineken_2024/val',
        'test_dir': '../ie_data/heineken_2024/test',
    },
    'kfc': {
        'train_dir': '../ie_data/kfc/train',
        'val_dir': '../ie_data/kfc/val',
        'test_dir': '../ie_data/kfc/test',

    },
    'lamthao': {
        'train_dir': '../ie_data/lamthao/train',
        'val_dir': '../ie_data/lamthao/val',
        'test_dir': '../ie_data/lamthao/test',
    },
    'lotte': {
        'train_dir': '../ie_data/lotte/train',
        'val_dir': '../ie_data/lotte/val',
        'test_dir': '../ie_data/lotte/test',
    },
    'lotte-drop-0.4': {
        'train_dir': '../ie_data/lotte-drop-0.4/train',
        'val_dir': '../ie_data/lotte-drop-0.4/val',
        'test_dir': '../ie_data/lotte-drop-0.4/test',
    },
    'lotte_cinema': {
        'train_dir': '../ie_data/lotte_cinema/train',
        'val_dir': '../ie_data/lotte_cinema/val',
        'test_dir': '../ie_data/lotte_cinema/test',
    },
    'lotteria': {
        'train_dir': '../ie_data/lotteria/train',
        'val_dir': '../ie_data/lotteria/val',
        'test_dir': '../ie_data/lotteria/test',
    },
    'mega_2022': {
        'train_dir': '../ie_data/mega_2022/train',
        'val_dir': '../ie_data/mega_2022/val',
        'test_dir': '../ie_data/mega_2022/test'
    },
    'ministop': {
        'train_dir': '../ie_data/ministop/train',
        'val_dir': '../ie_data/ministop/val',
        'test_dir': '../ie_data/ministop/test'
    },
    'newbigc_go_top': {
        'train_dir': '../ie_data/newbigc_go_top/train',
        'val_dir': '../ie_data/newbigc_go_top/val',
        'test_dir': '../ie_data/newbigc_go_top/test',
    },
    'new_gs25': {
        'train_dir': '../ie_data/new_gs25/train',
        'val_dir': '../ie_data/new_gs25/val',
        'test_dir': '../ie_data/new_gs25/test',
    },
    'nguyenkim': {
        'train_dir': '../ie_data/nguyenkim/train',
        'val_dir': '../ie_data/nguyenkim/val',
        'test_dir': '../ie_data/nguyenkim/test',
    },
    'nova': {
        'train_dir': '../ie_data/nova/train',
        'val_dir': '../ie_data/nova/val',
        'test_dir': '../ie_data/nova/test'

    },
    'nuty': {
        'train_dir': '../ie_data/nuty/train',
        'val_dir': '../ie_data/nuty/val',
        'test_dir': '../ie_data/nuty/test',
    },
    'okono': {
        'train_dir': '../ie_data/okono/train',
        'val_dir': '../ie_data/okono/val',
        'test_dir': '../ie_data/okono/test',
    },
    'pepper_lunch': {
        'train_dir': '../ie_data/pepper_lunch/train',
        'val_dir': '../ie_data/pepper_lunch/val',
        'test_dir': '../ie_data/pepper_lunch/test',
    },
    'pizza_company': {
        'train_dir': '../ie_data/pizza_company/train',
        'val_dir': '../ie_data/pizza_company/val',
        'test_dir': '../ie_data/pizza_company/test',
    },
    'satra': {
        'train_dir': '../ie_data/satra/train',
        'val_dir': '../ie_data/satra/val',
        'test_dir': '../ie_data/satra/test',

    },
    'tgs': {
        'train_dir': '../ie_data/tgs/train',
        'val_dir': '../ie_data/tgs/val',
        'test_dir': '../ie_data/tgs/test',

    },
    'thegioiskinfood': {
        'train_dir': '../ie_data/thegioiskinfood/train',
        'val_dir': '../ie_data/thegioiskinfood/val',
        'test_dir': '../ie_data/thegioiskinfood/test',
    },
    'winmart_combined': {
        'train_dir': '../ie_data/winmart_combined/train',
        'val_dir': '../ie_data/winmart_combined/val',
        'test_dir': '../ie_data/winmart_combined/test'
    },
}


uc = {
    'a':'a',
    'á':'a',
    'à':'a',
    'ả':'a',
    'ã':'a',
    'ạ':'a',
    'ă':'a',
    'ắ':'a',
    'ằ':'a',
    'ẳ':'a',
    'ẵ':'a',
    'ặ':'a',
    'â':'a',
    'ấ':'a',
    'ầ':'a',
    'ẩ':'a',
    'ẫ':'a',
    'ậ':'a',
    'e':'e',
    'é':'e',
    'è':'e',
    'ẻ':'e',
    'ẽ':'e',
    'ẹ':'e',
    'ê':'e',
    'ế':'e',
    'ề':'e',
    'ể':'e',
    'ễ':'e',
    'ệ':'e',
    'i':'i',
    'í':'i',
    'ì':'i',
    'ỉ':'i',
    'ĩ':'i',
    'ị':'i',
    'o':'o',
    'ó':'o',
    'ò':'o',
    'ỏ':'o',
    'õ':'o',
    'ọ':'o',
    'ô':'o',
    'ố':'o',
    'ồ':'o',
    'ổ':'o',
    'ỗ':'o',
    'ộ':'o',
    'ơ':'o',
    'ớ':'o',
    'ờ':'o',
    'ở':'o',
    'ỡ':'o',
    'ợ':'o',
    'u':'u',
    'ú':'u',
    'ù':'u',
    'ủ':'u',
    'ũ':'u',
    'ụ':'u',
    'ư':'u',
    'ứ':'u',
    'ừ':'u',
    'ử':'u',
    'ữ':'u',
    'ự':'u',
    'y':'y',
    'ý':'y',
    'ỳ':'y',
    'ỷ':'y',
    'ỹ':'y',
    'ỵ':'y',
    'đ':'d'
}


SUPPORTED_MODEL = {
    'rgcn': RGCN_Model,
    'improved_rgcn': ImprovedRGCN_Model,
    'improved_rgcn_visual': ImprovedRGCNVisual_Model,
    'gatv2': GATv2_Model,
    'gnn_film': GNN_FiLM_Model,
    'baseline': BaselineModel,
    'baseline_mlp': BaselineMLPModel,
    'gin': GIN_Model,
    'gcn': ChebGCN_Model,
}


def load_model(general_cfg, model_cfg, n_classes, ckpt_path=None):
    model_type = general_cfg['options']['model_type']
    if model_type not in SUPPORTED_MODEL:
        raise ValueError(f'Model type {model_type} is not supported yet')
    
    if ckpt_path is not None:
        model = SUPPORTED_MODEL[model_type].load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    general_cfg=general_cfg, 
                    model_cfg=model_cfg, 
                    n_classes=n_classes
                )
    else:
        model = SUPPORTED_MODEL[model_type](
            general_cfg=general_cfg, 
            model_cfg=model_cfg, 
            n_classes=n_classes
        )
    
    return model



def remove_accent(text):
    return unidecode.unidecode(text)


def rotate(xy, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )


def translate(xy, offset, img_h):
    return [xy[0] + offset[0], img_h - (xy[1] + offset[1])]

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
        x1, y1 = shape['points'][0]  # tl
        x2, y2 = shape['points'][1]  # tr
        x3, y3 = shape['points'][2]  # br
        x4, y4 = shape['points'][3]  # bl
        bb = tuple(int(i) for i in (x1,y1,x2,y2,x3,y3,x4,y4))
        bbs.append(bb)
        labels.append(shape['label'])
        try:
            texts.append(shape['text'])
        except:
            texts.append('')

    bb2label = dict(zip(bbs, labels))   # theo thu tu truyen vao trong data['shapes']
    bb2text = dict(zip(bbs, texts))
    bb2idx_original = {x: idx for idx, x in enumerate(bbs)}   # theo thu tu truyen vao trong data['shapes']
    rbbs = row_bbs(copy.deepcopy(bbs))
    sorted_bbs = [bb for row in rbbs for bb in row]  # theo thu tu tu trai sang phai, tu tren xuong duoi
    bb2idx_sorted = {tuple(x): idx for idx, x in enumerate(sorted_bbs)}   # theo thu tu tu trai sang phai, tu tren xuong duoi
    sorted_indices = [bb2idx_sorted[bb] for bb in bb2idx_original.keys()]

    return bb2label, bb2text, rbbs, bb2idx_sorted, sorted_indices



def unsign(text):
    unsign_text = ''
    for c in text.lower():
        if c in uc.keys():
            unsign_text += uc[c]
        else:
            unsign_text += c
    return unsign_text


def get_img_fp_from_json_fp(json_fp: Path):
    ls_ext = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in ls_ext:
        img_fp = json_fp.with_suffix(ext)
        if img_fp.exists():
            return img_fp
    return None


def augment_text(text):
    """
        randomly change number in the original text
    """
    augmented_text = ''
    for c in text:
        if c.isdigit():
            augmented_text += str(np.random.randint(0, 10))
        else:
            augmented_text += c
    
    return augmented_text


def get_bb_from_poly(poly: Tuple, img_w, img_h):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly    # tl -> tr -> br -> bl
    xmin = min (x1, x2, x3, x4)
    xmin = max(0, min(xmin, img_w))
    xmax = max (x1, x2, x3, x4)
    xmax = max(0, min(xmax, img_w))
    ymin = min (y1, y2, y3, y4)
    ymin = max(0, min(ymin, img_h))
    ymax = max (y1, y2, y3, y4)
    ymax = max(0, min(ymax, img_h))

    return xmin, ymin, xmax, ymax


def augment_box(xmin, ymin, xmax, ymax, img_w, img_h, percent=5):
    box_w = xmax - xmin
    box_h = ymax - ymin
    xmin += np.random.randint(-int(box_w/100*percent), int(box_w/100*percent)+1)
    xmax += np.random.randint(-int(box_w/100*percent), int(box_w/100*percent)+1)
    ymin += np.random.randint(-int(box_h/100*percent), int(box_h/100*percent)+1)
    ymax += np.random.randint(-int(box_h/100*percent), int(box_h/100*percent)+1)

    xmin = min(img_w, max(0, xmin))
    ymin = min(img_h, max(0, ymin))
    xmax = min(img_w, max(0, xmax))
    ymax = min(img_h, max(0, ymax))

    return xmin, ymin, xmax, ymax



def get_list_json(data_dir: str, max_sample=int(1e4), remove_json_with_no_img=True, shuffle=False):
    ls_json_fp = sorted(list(Path(data_dir).rglob('*.json')))
    if remove_json_with_no_img:
        ls_json_fp = [fp for fp in ls_json_fp if get_img_fp_from_json_fp(fp) is not None]
    if shuffle:
        np.random.shuffle(ls_json_fp)

    return ls_json_fp[:max_sample]


def random_drop_shape(shapes: List, num_general_drop: int, num_field_drop: int, outlier_label='text'):
    ls_idx2drop = []
    if np.random.rand() < 0.5:  # chon random ca text va non-text
        ls_idx2drop = np.random.randint(0, len(shapes), size=num_general_drop)
    else:   # chi drop non-text
        non_text_indices = [i for i, shape in enumerate(shapes) if shape['label'] != outlier_label]
        if len(non_text_indices) > 5:
            ls_idx2drop = np.random.choice(non_text_indices, min(num_field_drop, len(non_text_indices)//8))
    shapes = [shape for i, shape in enumerate(shapes) if i not in ls_idx2drop]

    return shapes

augment = iaa.SomeOf((1, 3), [
    iaa.Affine(scale={"x": (0.98, 1.02), "y": (0.9, 1.1)}),
    iaa.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.1, 0.1)}),
    iaa.Affine(rotate=(-3, 3)),
    iaa.Affine(shear=(-5, 5)),
    # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    iaa.PerspectiveTransform(scale=(0.01, 0.02)),
    # iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0, 0.02)))
], random_order=True)

def random_geometric_transform(image, shapes: List, augment=augment):
    kps = KeypointsOnImage(list(itertools.chain.from_iterable(
        [[Keypoint(x=i['points'][0][0], y=i['points'][0][1]),
        Keypoint(x=i['points'][1][0], y=i['points'][1][1]),
        Keypoint(x=i['points'][2][0], y=i['points'][2][1]),
        Keypoint(x=i['points'][3][0], y=i['points'][3][1])] for i in shapes]))
    , shape=image.shape)
    # Augment keypoints and images.
    image_aug, kps_aug = augment(image=image, keypoints=kps)
            #     json_data['shapes'], img_w, img_h = augment_pad(json_data['shapes'], img_w, img_h, max_offset=15)

    aug_kps_ls = list(itertools.chain.from_iterable([[i.x, i.y] for i in kps_aug]))
    aug_kps_ls = [aug_kps_ls[i*8:i*8+8] for i in range(len(aug_kps_ls)//8)]
    aug_kps_ls = [np.array(i).reshape(4, 2) for i in aug_kps_ls]
    for i in range(len(shapes)):
        shapes[i]['points'] = aug_kps_ls[i]
    new_shapes_list = []
    for i in shapes:
        if min(np.min(i['points'], axis=0)) < 0 or np.max(i['points'], axis=0)[0] > image.shape[1] or np.max(i['points'], axis=0)[1] > image.shape[0]:
            continue
        else:
            new_shapes_list.append(i)
    shapes = new_shapes_list
    new_shapes_list = []
    for i in shapes:
        i['points'] = i['points'].tolist()
        new_shapes_list.append(i)
    return new_shapes_list


def random_string(letter_count, digit_count):  
    str1 = ''.join((random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ- ') for x in range(letter_count)))  
    str1 += ''.join((random.choice(string.digits) for x in range(digit_count)))  
    sam_list = list(str1) # it converts the string to list.  
    random.shuffle(sam_list) # It uses a random.shuffle() function to shuffle the string.  
    final_string = ''.join(sam_list)  
    return final_string

def random_lower_upper_newbigc(text):
    new_text = text.lower()
    new_text = list(new_text)
    new_text[0] = new_text[0].upper()
    new_text = ''.join(new_text)
    return random.choice([text, new_text])

def random_drop_field(shapes, field, drop_rate):
    new_shapes = []
    for i, shape in enumerate(shapes):
        if shape['label'] == field and random.random() < drop_rate:
            continue
        new_shapes.append(shape)
    return new_shapes

def random_fieldtext_generate(shapes, field, letter_count, digit_count, provided_list=None):
    if provided_list is None:
        for i, shape in enumerate(shapes):
            if shape['label'] == field:
                # if fuzz.ratio(shapes[i]['text'].lower(), 'BIG'.lower()) > 60 or fuzz.ratio(shapes[i]['text'].lower(), 'GO!'.lower()) > 60 or fuzz.ratio(shapes[i]['text'].lower(), 'TOPS'.lower()) > 60:
                # if fuzz.ratio(shapes[i]['text'].lower(), 'Co.opXtra'.lower()) > 60 or fuzz.ratio(shapes[i]['text'].lower(), 'Co.opMart'.lower()) > 60:
                    # if fuzz.ratio(shapes[i]['text'].lower(), 'BIG'.lower()) > 60:
                        # shapes[i]['text'] = random_lower_upper_newbigc(shapes[i]['text'])
                # if fuzz.ratio(shapes[i]['text'].lower(), 'Emart'.lower()) > 70:
                    # continue
                # else:
                    shapes[i]['text'] = random_string(letter_count, digit_count)
                    # shapes[i]['text'] = random_string(random.randint(9, 11), random.randint(9, 11))
        return shapes
    else:
        for i, shape in enumerate(shapes):
            if shape['label'] == field:
                shapes[i]['text'] = random.choice(provided_list)
        return shapes

def augment_pad(shapes, img_w, img_h, max_offset=15):
    """
        add offset width and height to image
    """
    offset_x1, offset_x2, offset_y1, offset_y2 = [np.random.randint(-max_offset, max_offset), np.random.randint(-max_offset, max_offset),  \
                                                  np.random.randint(-max_offset, max_offset), np.random.randint(-max_offset, max_offset)]
    img_w += offset_x1 + offset_x2
    img_h += offset_y1 + offset_y2

    for i, shape in enumerate(shapes):
        for pt_idx, pt in enumerate(shape['points']):
            pt[0] = min(img_w, max(0, pt[0]+offset_x1))
            pt[1] = min(img_h, max(0, pt[1]+offset_y1))
            shapes[i]['points'][pt_idx] = [pt[0], pt[1]]

    return shapes, img_w, img_h


def random_translate(shapes, img_w, img_h):
    limit_w = img_w//70
    for i, shape in enumerate(shapes):
        shape_w = max(np.abs(shape['points'][1][0] - shape['points'][0][0])//10, 1)
        shape_h = max(np.abs(shape['points'][3][1] - shape['points'][0][1])//10, 1)
        for pt_idx, pt in enumerate(shape['points']):
            pt[0] = min(img_w, max(0, pt[0]+np.random.randint(-min(shape_w, limit_w), min(shape_w, limit_w))))
            pt[1] = min(img_h, max(0, pt[1]+np.random.randint(-shape_h, shape_h)))
            shapes[i]['points'][pt_idx] = [pt[0], pt[1]]

    return shapes

def get_manual_text_feature(text: str):
    feature = []

    # có phải ngày tháng không
    feature.append(int(re.search('(\d{1,2})\/(\d{1,2})\/(\d{4})', text) != None))

    # co phai gio khong
    feature.append(int(re.search('(\d{1,2}):(\d{1,2})', text) != None))
        
    # có phải tiền dương không
    feature.append(int(re.search('^\d{1,3}(\,\d{3})*(\,00)+$', text.replace('.', ',')) != None or re.search('^\d{1,3}(\,\d{3})+$', text.replace('.', ',')) != None))
    
    # co phai tien am khong
    feature.append(int(text.startswith('-') and re.search('^[\d(\,)]+$', text[1:].replace('.', ',')) != None and len(text) >= 3))

    # có phải uppercase
    feature.append(int(text.isupper()))

    # có phải title
    feature.append(int(text.istitle()))

    # có phải lowercase
    feature.append(int(text.islower()))

    # chỉ có số
    feature.append(int(re.search('^\d+$', text) != None))

    # chỉ có chữ cái
    feature.append(int(re.search('^[a-zA-Z]+$', text) != None))

    # chi co chu hoac so
    feature.append(int(re.search('^[a-zA-Z0-9]+$', text) != None))

    # chỉ có số và dấu
    feature.append(int(re.search('^[\d|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) != None))

    # chỉ có chữ và dấu
    feature.append(int(re.search('^[a-zA-Z|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) != None))

    return feature


def get_word_encoder(encoder_options):
    encoder = BPEmb(
        lang=encoder_options['lang'],
        dim=encoder_options['dim'],
        vs=encoder_options['vs']
    )
    return encoder


def get_experiment_dir(root_dir, description=None):
    os.makedirs(root_dir, exist_ok=True)
    exp_nums = [int(subdir[3:]) if '_' not in subdir else int(subdir.split('_')[0][3:]) for subdir in os.listdir(root_dir)]
    max_exp_num = max(exp_nums) if len(exp_nums) > 0 else 0
    exp_name = f'exp{max_exp_num+1}' if description is None else f'exp{max_exp_num+1}_{description}'
    return os.path.join(root_dir, exp_name)


# def map_label_list(raw_labels):
#     from label_list_hub import COMMON_LABEL_LIST

#     new_labels = []
#     for label in raw_labels:
#         if label in COMMON_LABEL_LIST:
#             new_labels.append(label)
#         else:
#             new_labels.append('text')
#     return new_labels