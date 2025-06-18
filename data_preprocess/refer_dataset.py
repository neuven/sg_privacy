import os
import sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json
import random
import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


#示例用
graphs_list = [
    {
        'node_features': [[0.1, 0.2], [0.4, 0.5], [0.6, 0.7]],
        'edge_index': [[0, 1, 2], [1, 2, 0]],
        'relation_features': [[0.5], [0.3], [0.2]],
        'label': 0,
        'subgraph_masks': [[1, 1, 0], [0, 0, 1]]  # 第一个子图0,1节点，第二个子图2节点
    },
    {
        'node_features': [[0.8, 0.9], [0.2, 0.3], [0.4, 0.5]],
        'edge_index': [[0, 1], [1, 2]],
        'relation_features': [[0.1], [0.4]],
        'label': 1,
        'subgraph_masks': [[1, 1, 0], [0, 1, 1]]  # 第一个子图0,1节点，第二个子图2,3节点
    }
]



#加载和批处理数据
def get_dataloader(objects_dir, partition='train'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet values
    # scale_size = 256  # args.scale_size
    # crop_size = 224  # args.crop_size

    #compose(将操作按照指定的顺序组合)
    data_full_transform = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor(), normalize])  # what about horizontal flip

    dataset = SceneGraphDataset(graphs=graphs_list)
    #dataset = SceneGraphDataset(objects_dir, partition=partition)
    #之后把这句转到main去训练
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    #data_set = IPDataset_FromFolder(objects_dir, data_full_transform, partition=partition)
    # returns target (also called labels), full_im (full sized original image without reshape), bboxes_14 (bbox
    # locations for max N objects (N can be 12 or 14). locations are resized to fit the 448x448 dim) , categories (of
    # objects found in bboxes), image_name
    # test_set = IPDataset(data_dir, objects_dir, test_full_transform)

    return dataset


class SceneGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        node_features = torch.tensor(graph['node_features'], dtype=torch.float)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph['relation_features'], dtype=torch.float)
        label = torch.tensor(graph['label'], dtype=torch.long)
        subgraph_masks = torch.tensor(graph['subgraph_masks'], dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
        data.subgraph_masks = subgraph_masks
        return data



full_transform = transforms.Compose([
    transforms.Resize((448, 448)),  #  transforms.Resize((448, 448)),
    transforms.ToTensor()])


class IPDataset_FromFolder(data.Dataset):
    def __init__(self, anno_dir, full_im_transform=None, dpath=''
            '/GIP/data_preprocess_', partition=''):
        super(IPDataset_FromFolder, self).__init__()

        self.anno_dir = anno_dir
        self.full_im_transform = full_im_transform

        f = open(str(dpath) + '/PrivacyAlert_' + str(partition) + '_private_files_path.txt', 'r')
        private_imgs = f.readlines()

        g = open(dpath + '/PrivacyAlert_' + str(partition) + '_public_files_path.txt', 'r')
        public_imgs = g.readlines()

        self.imgs = private_imgs + public_imgs
        # how to make continuous labels [0,1]
        self.labels = [0] * len(private_imgs) + [1] * len(public_imgs)

    def __getitem__(self, index):
        # For normalize
        if self.imgs[index].endswith('\n'):
            img = Image.open(self.imgs[index].split("\n")[0]).convert('RGB')
        else:
            img = Image.open(self.imgs[index]).convert('RGB')  # convert gray to rgb

        target = self.labels[index]

        if self.full_im_transform:
            full_im = self.full_im_transform(img)  # e.g; for index 10 full_im.shape = [3, 448, 448]
        else:
            full_im = img

        path = self.imgs[index].split('/')[-2:]  # e.g; ['train2017', '2017_80112549.jpg']
        path = os.path.join(self.anno_dir, path[1].split('.')[0] + '.json')

        image_name = path.split('/')[-1]  # e.g; '2017_80112549.json'
        (w, h) = img.size  # e.g; (1024, 1019)
        bboxes_objects = json.load(open(path))
        bboxes = torch.Tensor(bboxes_objects["bboxes"])

        max_rois_num = 12  # {detection threshold: max rois num} {0.3: 19, 0.4: 17, 0.5: 14, 0.6: 13, 0.7: 12}
        bboxes_14 = torch.zeros((max_rois_num, 4))

        if bboxes.size()[0] > max_rois_num:
            bboxes = bboxes[0:max_rois_num]

        if bboxes.size()[0] != 0:
            # re-scale, image size is wxh so change bounding boxes dimensions from wxh space to 448x448 range
            bboxes[:, 0::4] = 448. / w * bboxes[:, 0::4]
            bboxes[:, 1::4] = 448. / h * bboxes[:, 1::4]
            bboxes[:, 2::4] = 448. / w * bboxes[:, 2::4]
            bboxes[:, 3::4] = 448. / h * bboxes[:, 3::4]

            bboxes_14[0:bboxes.size(0), :] = bboxes

        categories = torch.IntTensor(max_rois_num + 1).fill_(-1)
        categories[0] = len(bboxes_objects['categories'])  # e.g; 5

        if categories[0] > max_rois_num:
            categories[0] = max_rois_num
        else:
            categories[0] = categories[0]
        end_idx = categories[0] + 1

        categories[1: end_idx] = torch.IntTensor(bboxes_objects['categories'])[
                                 0:categories[0]]  # e.g; [ 5,  0,  0,  0,  7, 72, -1, -1, -1, -1, -1, -1, -1]
        return target, full_im, categories, image_name

    def __len__(self):
        return len(self.imgs)
