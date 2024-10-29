import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import argparse
import json
import shutil
import yaml
from pathlib import Path
from my_utils import *
from label_list_hub import all_label_list
from PIL import Image
import onnx
import onnxruntime
import numpy as np


def get_input_from_json(json_fp, word_encoder, use_emb, emb_range):
    with open(json_fp, 'r') as f:
            json_data = json.load(f)
    img_fp = os.path.splitext(json_fp)[0]+'.jpg'
    img_w, img_h = Image.open(img_fp).size

    # get node features
    x_indexes = [] # list of all x_indexes of all nodes in graph (each node has an x_index)
    y_indexes = [] # list of all y_indexes of all nodes in graph (each node has an y_index)
    text_features = [] # list of all features of all nodes in graph (each node has a feature)

    bb2label, bb2text, rbbs, bbs2idx_sorted, sorted_indices = sort_json(json_data)

    edges = []
    for row_idx, rbb in enumerate(rbbs):
        for bb_idx_in_row, bb in enumerate(rbb):  # duyet qua tung bb (tung node)
            # ----------------- process text feature -----------------
            text = bb2text[bb]
            if word_encoder.lang != 'vi':
                text = unidecode.unidecode(text)  # nếu hóa đơn ko dấu thì bpemb để tiếng việt hay tiếng anh ?
            bb_text_feature = get_manual_text_feature(text) + list(np.sum(word_encoder.embed(text), axis=0))
            text_features.append(bb_text_feature)

            # ----------------- process geometry feature -----------------
            xmin, ymin, xmax, ymax = get_bb_from_poly(bb, img_w, img_h)
            if use_emb: 
                # rescale coord at width=emb_range
                x_index = [int(xmin * emb_range / img_w), int(xmax * emb_range / img_w), int((xmax - xmin) * emb_range / img_w)]
                y_index = [int(ymin * emb_range / img_h), int(ymax * emb_range / img_h), int((ymax - ymin) * emb_range / img_h)]
            else:
                # normalize in rnage(0, 1)
                x_index = [float(xmin * 1.0 / img_w), float(xmax * 1.0 / img_w), float((xmax - xmin) * 1.0 / img_w)]
                y_index = [float(ymin * 1.0 / img_h), float(ymax * 1.0 / img_h), float((ymax - ymin) * 1.0 / img_h)]
            x_indexes.append(x_index)
            y_indexes.append(y_index)
            
            # ------------------------ build graph ----------------------
            # find right node
            right_node = rbb[bb_idx_in_row+1] if bb_idx_in_row < len(rbb) - 1 else None
            if right_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[right_node], 1])
                edges.append([bbs2idx_sorted[right_node], bbs2idx_sorted[bb], 2])
            
            # find left node
            left_node = rbb[bb_idx_in_row-1] if bb_idx_in_row > 0 else None
            if left_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[left_node], 2])
                edges.append([bbs2idx_sorted[left_node], bbs2idx_sorted[bb], 1])
            
            # find above node
            max_x_overlap = -1e9
            above_node = None
            if row_idx > 0:
                for prev_bb in rbbs[row_idx-1]:
                    xmax_prev_bb = max(prev_bb[2], prev_bb[4])
                    xmin_prev_bb = min(prev_bb[0], prev_bb[6])
                    x_overlap = (xmax_prev_bb - xmin_prev_bb) + (xmax-xmin) - (max(xmax_prev_bb, xmax) - min(xmin_prev_bb, xmin))
                    if x_overlap > max_x_overlap:
                        max_x_overlap = x_overlap
                        above_node = prev_bb
            if above_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[above_node], 4])
                edges.append([bbs2idx_sorted[above_node], bbs2idx_sorted[bb], 3])
            
            # find below node
            max_x_overlap = -1e9
            below_node = None
            if row_idx < len(rbbs) - 1:
                for next_bb in rbbs[row_idx+1]:
                    xmax_next_bb = max(next_bb[2], next_bb[4])
                    xmin_next_bb = min(next_bb[0], next_bb[2])
                    x_overlap = (xmax_next_bb - xmin_next_bb) + (xmax-xmin) - (max(xmax_next_bb, xmax) - min(xmin_next_bb, xmin))
                    if x_overlap > max_x_overlap:
                        max_x_overlap = x_overlap
                        below_node = next_bb
            if below_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[below_node], 3])
                edges.append([bbs2idx_sorted[below_node], bbs2idx_sorted[bb], 4])

    # 1 - right, 2 - left, 3 - down, 4  - up
    edges = torch.tensor(edges, dtype=torch.int32)
    edges = torch.unique(edges, dim=0, return_inverse=False)   # remove duplicate rows
    edge_index, edge_type = edges[:, :2], edges[:, -1]

    return torch.tensor(x_indexes, dtype=torch.int if use_emb else torch.float),  \
            torch.tensor(y_indexes, dtype=torch.int if use_emb else torch.float), \
            torch.tensor(text_features, dtype=torch.float), \
            edge_index.t().to(torch.int64), \
            edge_type, \


def convert(ckpt_path, out_path):
    print(f'Convert with checkpoint {ckpt_path} ...')
    ckpt_dir = Path(ckpt_path).parent
    with open(os.path.join(ckpt_dir, 'train_cfg.yaml')) as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, 'model_cfg.yaml')) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    label_list = all_label_list[general_cfg['data']['label_list']]
    word_encoder = BPEmb(**general_cfg['options']['word_encoder'])
    use_emb = general_cfg['options']['use_emb'],
    emb_range = general_cfg['model']['emb_range']
    model = load_model(general_cfg, model_cfg, n_classes=len(label_list), ckpt_path=ckpt_path).cpu()
    model.eval()

    json_data = json.load(open('../ie_data/aeon_citimart/test/1eb2d42bfdbd3be362ac97.json'))
    _, _, _, _, sorted_indices = sort_json(json_data)

    # model infer
    x_indexes, y_indexes, text_features, edge_index, edge_type = get_input_from_json(
        '../ie_data/aeon_citimart/test/1eb2d42bfdbd3be362ac97.json',
        word_encoder,
        use_emb,
        emb_range
    )
    print(x_indexes.shape, y_indexes.shape, text_features.shape, edge_index.shape, edge_type.shape)
    out = model(x_indexes, y_indexes, text_features, edge_index, edge_type)
    print(out.shape)
    torch.onnx.export(model,               # model being run
                    (x_indexes, y_indexes, text_features, edge_index, edge_type),   # model input (or a tuple for multiple inputs)
                    out_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['x_indexes', 'y_indexes', 'text_features', 'edge_index', 'edge_type'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'x_indexes' : {0 : 'num_node'},
                                'y_indexes' : {0: 'num_node'},
                                'text_features' : {0: 'num_node'},
                                'edge_index' : {1: 'num_edge'},
                                'edge_type' : {0: 'num_edge'},
                                'output' : {0 : 'num_node'}}
    )
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(out_path, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x_indexes), ort_session.get_inputs()[1].name: to_numpy(y_indexes),
                ort_session.get_inputs()[2].name: to_numpy(text_features), ort_session.get_inputs()[3].name: to_numpy(edge_index),
                ort_session.get_inputs()[4].name: to_numpy(edge_type)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    print(np.mean(np.abs(ort_outs[0]-to_numpy(out))))
    a = ort_outs[0].flatten()
    b = to_numpy(out).flatten()
    for i, j in zip(a, b):
        if abs(i-j) > 0.01:
            print(i, j)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments...')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to lightning checkpoint')
    parser.add_argument('--out_path', type=str, required=True, help='path to output onnx path')

    args = parser.parse_args()
    convert(
        args.ckpt_path,
        args.out_path
    )