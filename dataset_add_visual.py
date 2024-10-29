import torch

import os
import re
import json
import math
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from bpemb import BPEmb
from my_utils import *
import unidecode
from pathlib import Path
from typing import Optional, Tuple, List
import torch
import pytorch_lightning as pl
from label_list_hub import all_label_list
from PIL import Image
import cv2
import timm
from torchvision.ops import roi_align, roi_pool
from torchvision import transforms


class VisualGraphDataset(Dataset):
    def __init__(self, general_cfg, mode='train'):
        self.general_cfg = general_cfg
        self.data_dir = general_cfg['data']['train_dir'] if mode == 'train' else general_cfg['data']['val_dir']
        self.label_list = all_label_list[general_cfg['data']['martname']]
        self.n_labels = len(self.label_list)
        self.json_files = get_list_json(self.data_dir, max_sample=general_cfg['data']['max_sample'], remove_json_with_no_img=True)
        self.mode = mode
        self.use_emb = general_cfg['options']['use_emb']
        self.bpemb = get_word_encoder(general_cfg['options']['word_encoder'])
        self.emb_range = general_cfg['model']['emb_range']

        self.visual_encoder = timm.create_model('resnet34', pretrained=True, num_classes=0, global_pool='', features_only=True).eval().to('cuda')
        self.visual_shape = (720, 480)
        self.im_transforms = transforms.Compose([
            transforms.Resize(self.visual_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.print_dataset_info()
        self.invalid_labels = self.check_label_valid()


    def print_dataset_info(self):
        print('----------------------- Dataset Info ---------------------------')
        print('dir: ', self.data_dir)
        print('num json files: ', len(self.json_files))
        print('num labels: ', self.n_labels)
        print('label list: ', self.label_list)

    
    def save_data_info(self, out_path):
        with open(out_path, 'w') as f:
            f.write(f'dir: {self.data_dir}\n')
            f.write(f'num json files: {len(self.json_files)}\n')
            f.write(f'num labels: {self.n_labels}\n')
            f.write(f'label list: {self.label_list}')

    
    def check_label_valid(self):
        ls_json_fp = list(Path(self.data_dir).rglob('*.json'))
        invalid_labels = []
        for jp in ls_json_fp:
            json_data = json.load(open(jp))
            for shape in json_data['shapes']:
                if shape['label'] not in self.label_list:
                    print(f'{jp} has outlier label: ', shape['label'])
                    invalid_labels.append(shape['label'])
        invalid_labels = list(set(invalid_labels))
        if len(invalid_labels) == 0:
            print('All labels are valid!')
        else:
            print('Some labels are invalid!')

        return invalid_labels


    def __len__(self):
        return len(self.json_files)


    def __getitem__(self, index):
        json_fp = self.json_files[index]
        with open(json_fp, 'r') as f:
            json_data = json.load(f)
        img_fp = get_img_fp_from_json_fp(json_fp)
        img = Image.open(img_fp)
        img_w, img_h = img.size
        # get node features
        nodes = []  # list of all node in graph
        x_indexes = [] # list of all x_indexes of all nodes in graph (each node has an x_index)
        y_indexes = [] # list of all y_indexes of all nodes in graph (each node has an y_index)
        text_features = [] # list of all features of all nodes in graph (each node has a feature)
        # filter out all shape with num points # 4
        # json_data['shapes'] = [shape for shape in json_data['shapes'] if len(shape['points']) == 4]

        # if self.mode == 'train':
        #     # if np.random.rand() < 0.5:
        #     #    json_data['shapes'] = random_fieldtext_generate(json_data['shapes'], 'mart_name', random.randint(1, 1), random.randint(13, 13))
        #     # if np.random.rand() < 0.5:
        #         # json_data['shapes'] = random_fieldtext_generate(json_data['shapes'], 'date', 0, 0, provided_list=[f'{random.randint(10, 31)}/{random.randint(10, 12)}/{random.randint(1800, 2500)}'])
        #     json_data['shapes'] = random_drop_field(json_data['shapes'], 'product_id', 0.2)
        #     json_data['shapes'] = random_drop_field(json_data['shapes'], 'product_unit_price', 0.2)
        #     json_data['shapes'] = random_drop_field(json_data['shapes'], 'product_quantity', 0.2)
        #     json_data['shapes'] = random_drop_field(json_data['shapes'], 'product_total_money', 0.2)

        #     # json_data['shapes'] = random_geometric_transform(image, json_data['shapes'])
        #     # random drop out boxes
        #     if len(json_data['shapes']) > 5 and np.random.rand() < 0.6:
        #         json_data['shapes'] = random_drop_shape(
        #             json_data['shapes'], 
        #             num_general_drop=min(10, len(json_data['shapes'])//15),
        #             num_field_drop=np.random.randint(3, 10)
        #         )
        #     # augment pad
        #     if np.random.rand() < 0.6:
        #         json_data['shapes'], img_w, img_h = augment_pad(json_data['shapes'], img_w, img_h, max_offset=15)

        #     # # augment geometry jitter
        #     # if np.random.rand() < 0.6:
        #     #     json_data['shapes'] = random_translate(json_data['shapes'], img_w, img_h)

        bb2label, bb2text, rbbs, bbs2idx_sorted, sorted_indices = sort_json(json_data)
        nodes, edges, labels  = [], [], []
        for row_idx, rbb in enumerate(rbbs):
            for bb_idx_in_row, bb in enumerate(rbb):  # duyet qua tung bb (tung node)
                # ----------------- process text feature -----------------
                text = bb2text[bb]
                if self.bpemb.lang != 'vi':
                    text = unidecode.unidecode(text)  # nếu hóa đơn ko dấu thì bpemb để tiếng việt hay tiếng anh ?
                bb_text_feature = get_manual_text_feature(text) + list(np.sum(self.bpemb.embed(text), axis=0))
                text_features.append(bb_text_feature)

                # ----------------- process geometry feature -----------------
                xmin, ymin, xmax, ymax = get_bb_from_poly(bb, img_w, img_h)
                # augment box coord
                # if self.mode == 'train' and np.random.rand() < 0.3:
                #     xmin, ymin, xmax, ymax = augment_box(xmin, ymin, xmax, ymax, img_w, img_h, percent=5)

                if self.use_emb: 
                    # rescale coord at width=self.emb_range
                    x_index = [int(xmin * 1. / img_w * self.emb_range), int(xmax * 1. / img_w * self.emb_range), int((xmax - xmin) * 1. / img_w * self.emb_range)]
                    y_index = [int(ymin * 1. / img_h * self.emb_range), int(ymax * 1. / img_h * self.emb_range), int((ymax - ymin) * 1. / img_h * self.emb_range)]
                else:
                    # normalize in rnage(0, 1)
                    x_index = [float(xmin * 1.0 / img_w), float(xmax * 1.0 / img_w), float((xmax - xmin) * 1.0 / img_w)]
                    y_index = [float(ymin * 1.0 / img_h), float(ymax * 1.0 / img_h), float((ymax - ymin) * 1.0 / img_h)]
                x_indexes.append(x_index)
                y_indexes.append(y_index)


                # ------------------ encode label -----------------------------
                text_label = bb2label[bb]
                if text_label in self.invalid_labels:
                    text_label = 'text'
                label = self.label_list.index(text_label)
                labels.append(label)

                # add to node   
                nodes.append({
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'feature': [x_index, y_index, bb_text_feature],
                    'label': label
                })
                
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
                int(xmin/img_w*self.visual_shape[1]),
                int(ymin/img_h*self.visual_shape[0]),
                int(xmax/img_w*self.visual_shape[1]),
                int(ymax/img_h*self.visual_shape[0]),
            ])
        resized_bbs = torch.tensor(resized_bbs, dtype=torch.float)
        im_input = self.im_transforms(img).unsqueeze(0).to('cuda')
        feature_map = self.visual_encoder(im_input)[2]  # /8 in size
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

        # print('x index: ', x_indexes)
        # print('y index: ', y_indexes)

        return torch.tensor(x_indexes, dtype=torch.int if self.use_emb else torch.float),  \
               torch.tensor(y_indexes, dtype=torch.int if self.use_emb else torch.float), \
               torch.tensor(text_features, dtype=torch.float), \
               torch.stack(visual_features, dim=0), \
               edge_index.t().to(torch.int64), \
               edge_type, \
               torch.tensor(labels).type(torch.LongTensor)





class VisualGraphDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup(stage=None)

    def setup(self, stage):
        self.train_ds = VisualGraphDataset(general_cfg=self.config, mode='train')
        self.val_ds = VisualGraphDataset(general_cfg=self.config, mode='val')
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=None, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=None, shuffle=False, num_workers=0)
    



if __name__ == '__main__':
    import yaml
    with open("configs/train.yaml") as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)

    data_module = VisualGraphDataModule(config=general_config)
    pdb.set_trace()

    train_ds = VisualGraphDataset(general_cfg=general_config, mode='train')
    for item in train_ds:
        x_indexes, y_indexes, text_features, edge_index, edge_type, labels = item
        for subitem in item:
            print(subitem.shape)
        break












