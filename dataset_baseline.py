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




class BaselineGraphDataset(Dataset):
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
                    # print(f'{jp} has outlier label: ', shape['label'])
                    invalid_labels.append(shape['label'])
        invalid_labels = list(set(invalid_labels))
        if len(invalid_labels) == 0:
            print('All labels are valid!')
        else:
            print(f'Some labels are invalid: {invalid_labels}')
        return invalid_labels


    def __len__(self):
        return len(self.json_files)


    def __getitem__(self, index):
        json_fp = self.json_files[index]
        with open(json_fp, 'r') as f:
            json_data = json.load(f)
        img_fp = get_img_fp_from_json_fp(json_fp)
        img_w, img_h = Image.open(img_fp).size
        image = np.ones((img_h, img_w, 3), dtype='uint8')
        # get node features
        node_features = [] # list of all features of all nodes in graph (each node has a feature)

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
                node_feature = []
                # ----------------- process text feature -----------------
                text = bb2text[bb]
                if self.bpemb.lang != 'vi':
                    text = unidecode.unidecode(text)  # nếu hóa đơn ko dấu thì bpemb để tiếng việt hay tiếng anh ?
                bb_text_feature = get_manual_text_feature(text) + list(np.sum(self.bpemb.embed(text), axis=0))
                node_feature.extend(bb_text_feature)

                # ----------------- encode label -----------------
                # one hot encode label 
                text_label = bb2label[bb]
                if text_label in self.invalid_labels:
                    text_label = 'text'
                # label = [0] * len(self.label_list)
                # label[self.label_list.index(text_label)] = 1
                # labels.append(label)
                label = self.label_list.index(text_label)
                labels.append(label)

                # ------------------------ build graph ----------------------
                xmin, ymin, xmax, ymax = get_bb_from_poly(bb, img_w, img_h)
                bb_layout_feature = []
                # find right node
                right_node = rbb[bb_idx_in_row+1] if bb_idx_in_row < len(rbb) - 1 else None
                if right_node:
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[right_node], 0])
                    edges.append([bbs2idx_sorted[right_node], bbs2idx_sorted[bb], 1])
                    dist = (min(right_node[::2]) - max(bb[::2])) / img_w
                    bb_layout_feature.append(dist)
                else:
                    bb_layout_feature.append(-1)
                
                # find left node
                left_node = rbb[bb_idx_in_row-1] if bb_idx_in_row > 0 else None
                if left_node:
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[left_node], 1])
                    edges.append([bbs2idx_sorted[left_node], bbs2idx_sorted[bb], 0])
                    dist = (min(bb[::2]) - max(left_node[::2])) / img_w
                    bb_layout_feature.append(dist)
                else:
                    bb_layout_feature.append(-1)
                
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
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[above_node], 3])
                    edges.append([bbs2idx_sorted[above_node], bbs2idx_sorted[bb], 2])
                    dist = (min(bb[1::2]) - max(above_node[1::2])) / img_h
                    bb_layout_feature.append(dist)
                else:
                    bb_layout_feature.append(-1)


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
                    edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[below_node], 2])
                    edges.append([bbs2idx_sorted[below_node], bbs2idx_sorted[bb], 3])
                    dist = (min(below_node[1::2]) - max(bb[1::2])) / img_h
                    bb_layout_feature.append(dist)
                else:
                    bb_layout_feature.append(-1)

                node_feature.extend(bb_layout_feature)
                node_features.append(node_feature)

        # 1 - right, 2 - left, 3 - down, 4  - up
        edges = torch.tensor(edges, dtype=torch.int32)
        edges = torch.unique(edges, dim=0, return_inverse=False)   # remove duplicate rows
        edge_index, edge_type = edges[:, :2], edges[:, -1]


        node_features = torch.tensor(node_features, dtype=torch.float)
        return node_features, \
               edge_index.t().to(torch.int64), \
               edge_type, \
               torch.tensor(labels).type(torch.LongTensor)



class BaselineGraphDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup(stage=None)

    def setup(self, stage):
        self.train_ds = BaselineGraphDataset(general_cfg=self.config, mode='train')
        self.val_ds = BaselineGraphDataset(general_cfg=self.config, mode='val')
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=None, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=None, shuffle=False, num_workers=16)
    



if __name__ == '__main__':
    import yaml
    with open("configs/train.yaml") as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)

    data_module = BaselineGraphDataModule(config=general_config)
    pdb.set_trace()

    train_ds = BaselineGraphDataset(general_cfg=general_config, mode='train')
    for item in train_ds:
        x_indexes, y_indexes, text_features, edge_index, edge_type, labels = item
        for subitem in item:
            print(subitem.shape)
        break












