import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
import copy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

@DATASETS.register_module()
class ScanObjectNN_Paired(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        self.n_sample = config.n_sample
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()
                
        self.unique_cls = np.unique(self.labels).tolist()
        self.unique_cls_temp = copy.deepcopy(self.unique_cls)
        self.per_class_idx = {}
        for cls in (self.unique_cls):
            self.per_class_idx[cls] = []
        for idx, cls in enumerate(self.labels.reshape((-1))):
            self.per_class_idx[cls].append(idx)
        self.per_class_idx_ori = copy.deepcopy(self.per_class_idx)

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        point_list = []
        label_list = []
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        point_list.append(current_points)
        label_list.append(label)
        label = label.item()
        current_idx = [idx]
        try:
            self.per_class_idx[label].remove(idx)
        except: #the idx0 can be used in the previous pair and deleted
            pass
        for i in range(self.n_sample-1):
            #get next sample's index in the pair
            try:
                idx = np.random.choice(self.per_class_idx[label])
            except: #in case the idx list is already empty, clone from the original list
                self.per_class_idx[label] = copy.deepcopy(self.per_class_idx_ori[label])
                for item in current_idx:
                    self.per_class_idx[label].remove(item)
                idx = np.random.choice(self.per_class_idx[label])
            current_idx.append(idx)
            #get next sample in the pair
            current_points = self.points[idx, pt_idxs].copy()
            current_points = torch.from_numpy(current_points).float()
            label = self.labels[idx]
            point_list.append(current_points)
            label_list.append(label)
            label = label.item()

            try: #exclude the 'idx' for the next pair candidate
                self.per_class_idx[label].remove(idx)
            except:
                pass
        
        return 'ScanObjectNN', 'sample', (point_list, label_list)

    def __len__(self):
        return self.points.shape[0]