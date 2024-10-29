import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import argparse
import json
import pdb
import shutil
import yaml
from pathlib import Path
from my_utils import *
from label_list_hub import all_label_list
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from bpemb import BPEmb
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics import F1Score, Precision, Recall
import timm
import torchvision.transforms as transforms
from torchvision.ops import roi_align, roi_pool


visual_encoder = timm.create_model('resnet34', pretrained=True, num_classes=0, global_pool='', features_only=True).eval().to('cuda')
visual_shape = (720, 480)
im_transforms = transforms.Compose([
    transforms.Resize(visual_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def save_result(out_dir, target_names, all_trues, all_preds, err_dict):
    f1_calc = F1Score(task='multiclass', threshold=0.5, num_classes=len(target_names))
    precision_calc = Precision(task='multiclass', threshold=0.5, num_classes=len(target_names))
    recall_calc = Recall(task='multiclass', threshold=0.5, num_classes=len(target_names))

    f1_score = round(f1_calc(torch.tensor(all_preds), torch.tensor(all_trues)).item(), 3)
    precision = round(precision_calc(torch.tensor(all_preds), torch.tensor(all_trues)).item(), 3)
    recall = round(recall_calc(torch.tensor(all_preds), torch.tensor(all_trues)).item(), 3)
    num_err = len([i for i in range(len(all_trues)) if all_trues[i] != all_preds[i]])
    num_total = len(all_trues)
    report = classification_report(y_true=all_trues, y_pred=all_preds, target_names=target_names, digits=3)
    print(report)
    confuse_matrix = confusion_matrix(y_true=all_trues, y_pred=all_preds)
    
    # visualize confusion matrix and save plot with seaborn
    plt.figure(figsize=(20, 20))
    sns.heatmap(confuse_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.savefig(os.path.join(out_dir, 'confuse_matrix.png'))
    
    with open(os.path.join(out_dir, 'error_dict.json'), 'w') as f:
        json.dump(err_dict, f)
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(report)
        f.write('\n')
        f.write(f'overall f1 score: {f1_score}\n')
        f.write(f'overall precision: {precision}\n')
        f.write(f'overall recall: {recall}\n')
        f.write(f'num_err: {num_err}\n')
        f.write(f'num_total: {num_total}\n')
    # write all_trues to txt files
    with open(os.path.join(out_dir, 'all_trues.txt'), 'w') as f:
        for true in all_trues:
            f.write(f'{true}\n')
    # write all_preds to txt files
    with open(os.path.join(out_dir, 'all_preds.txt'), 'w') as f:
        for pred in all_preds:
            f.write(f'{pred}\n')
    # write target_names to txt files
    with open(os.path.join(out_dir, 'target_names.txt'), 'w') as f:
        for name in target_names:
            f.write(f'{name}\n')


def get_input_from_json(json_fp, word_encoder, use_emb, emb_range):
    with open(json_fp, 'r') as f:
            json_data = json.load(f)
    img_fp = get_img_fp_from_json_fp(json_fp)
    img = Image.open(img_fp)
    img_w, img_h = img.size

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
            if bb_idx_in_row < len(rbb) - 1:
                for temp_idx in range(bb_idx_in_row+1, len(rbb)-1):
                    right_node = rbb[temp_idx]
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[right_node], 0])
                    edges.append([bbs2idx_sorted[right_node], bbs2idx_sorted[bb], 1])
            
            # find left node
            if bb_idx_in_row > 0:
                for temp_idx in range(0, bb_idx_in_row):
                    left_node = rbb[temp_idx]
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[left_node], 1])
                    edges.append([bbs2idx_sorted[left_node], bbs2idx_sorted[bb], 0])
            
            # find above node
            if row_idx > 0:
                for prev_bb in rbbs[row_idx-1]:
                    above_node = prev_bb
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[above_node], 3])
                    edges.append([bbs2idx_sorted[above_node], bbs2idx_sorted[bb], 2])
            
            # find below node
            if row_idx < len(rbbs) - 1:
                for next_bb in rbbs[row_idx+1]:
                    below_node = next_bb
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[below_node], 2])
                    edges.append([bbs2idx_sorted[below_node], bbs2idx_sorted[bb], 3])

    # ----------------- process visual feature -----------------
    visual_features = []
    sorted_bbs = [get_bb_from_poly(bb, img_w, img_h) for row in rbbs for bb in row]
    resized_bbs = []
    for bb in sorted_bbs:
        xmin, ymin, xmax, ymax = bb
        resized_bbs.append([
            int(xmin/img_w*visual_shape[1]),
            int(ymin/img_h*visual_shape[0]),
            int(xmax/img_w*visual_shape[1]),
            int(ymax/img_h*visual_shape[0]),
        ])
    resized_bbs = torch.tensor(resized_bbs, dtype=torch.float)
    im_input = im_transforms(img).unsqueeze(0).to('cuda')
    feature_map = visual_encoder(im_input)[2]  # /8 in size
    aligned_rois = roi_align(input=feature_map, boxes=[resized_bbs.to('cuda')], output_size=(8, 8), spatial_scale=1/8, aligned=True)
    aligned_rois = aligned_rois.to('cpu')
    for roi in aligned_rois:
        pooled_roi = torch.mean(roi, dim=(1,2))  # shape (512,)
        visual_features.append(pooled_roi)
    assert len(visual_features) == len(text_features)

    # 1 - right, 2 - left, 3 - down, 4  - up
    edges = torch.tensor(edges, dtype=torch.int32)
    edges = torch.unique(edges, dim=0, return_inverse=False)   # remove duplicate rows
    edge_index, edge_type = edges[:, :2], edges[:, -1]

    return torch.tensor(x_indexes, dtype=torch.int if use_emb else torch.float),  \
            torch.tensor(y_indexes, dtype=torch.int if use_emb else torch.float), \
            torch.tensor(text_features, dtype=torch.float), \
            torch.stack(visual_features, dim=0), \
            edge_index.t().to(torch.int64), \
            edge_type, \



def inference(ckpt_path, src_dir=None, out_dir=None):
    print(f'Inference with checkpoint {ckpt_path} ...')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = Path(ckpt_path).parent
    with open(os.path.join(ckpt_dir, 'train_cfg.yaml')) as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, 'model_cfg.yaml')) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    label_list = all_label_list[general_cfg['data']['label_list']]
    word_encoder = BPEmb(**general_cfg['options']['word_encoder'])
    use_emb = general_cfg['options']['use_emb'],
    emb_range = general_cfg['model']['emb_range']
    model = load_model(general_cfg, model_cfg, n_classes=len(label_list), ckpt_path=ckpt_path)
    model = model.to('cuda')
    model.eval()

    json_files = get_list_json(src_dir, max_sample=int(1e4), remove_json_with_no_img=True)
    err_dict = {}
    all_trues, all_preds = [], []
    for i, json_fp in enumerate(json_files):
        print('json fp: ', json_fp)
        json_data = json.load(open(json_fp))
        _, _, _, _, sorted_indices = sort_json(json_data)

        # model infer
        x_indexes, y_indexes, text_features, visual_features, edge_index, edge_type = get_input_from_json(
            json_fp,
            word_encoder,
            use_emb,
            emb_range
        )
        x_indexes, y_indexes, text_features, visual_features, edge_index, edge_type = x_indexes.to('cuda'), y_indexes.to('cuda'), text_features.to('cuda'), visual_features.to('cuda'), edge_index.to('cuda'), edge_type.to('cuda')
        # pdb.set_trace()
        out = model(x_indexes, y_indexes, text_features, visual_features, edge_index, edge_type)
        preds = torch.argmax(out, dim=-1)

        
        for i, shape in enumerate(json_data['shapes']):
            try:
                pred_label = label_list[preds[sorted_indices[i]]]
            except:
                pdb.set_trace()
            
            if pred_label != shape['label']:
                err_info = {
                    'box': [int(coord) for pt in shape['points'] for coord in pt],
                    'text': shape['text'],
                    'true_label': shape['label'],
                    'pred_label': pred_label
                }
                if str(json_fp) not in err_dict:
                    err_dict[str(json_fp)] = [err_info]
                else:
                    err_dict[str(json_fp)].append(err_info) 

            try:
                all_trues.append(label_list.index(shape['label']))
            except:
                all_trues.append(label_list.index('text'))

            all_preds.append(label_list.index(pred_label))

            # modify json_label
            json_data['shapes'][i]['label'] = 'text'
            json_data['shapes'][i]['label'] = pred_label

        # # save
        # with open(os.path.join(out_dir, json_fp.name), 'w') as f:
        #     json.dump(json_data, f)
        # shutil.copy(get_img_fp_from_json_fp(json_fp), out_dir)
        print(f'Done {json_fp.name}')
    
    target_names = [label_list[i] for i in sorted(list(set(all_trues+all_preds)))]
    save_result(out_dir, target_names, all_trues, all_preds, err_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments...')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to lightning checkpoint')
    parser.add_argument('--src_dir', type=str, required=True, help='path to inference dir')
    parser.add_argument('--out_dir', type=str, required=True, help='path to write result')

    args = parser.parse_args()
    inference(
        args.ckpt_path,
        args.src_dir,
        args.out_dir
    )
