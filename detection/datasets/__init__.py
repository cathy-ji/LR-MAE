# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig, SelfScannetDetectionDataset, \
    UpmaeScannetDetectionDataset
from .sunrgbd import SunrgbdDetectionDataset, UPmaeSunrgbdDetectionDataset,SunrgbdDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "selfscannet": [SelfScannetDetectionDataset, ScannetDatasetConfig],
    "upmaescannet": [UpmaeScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "upmaesunrgbd":[UPmaeSunrgbdDetectionDataset, SunrgbdDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    dataset_dict = {
        "train": dataset_builder(
            dataset_config, 
            split_set="train",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=True
        ),
        "test": dataset_builder(
            dataset_config, 
            split_set="val", 
            root_dir=args.dataset_root_dir, 
            use_color=args.use_color,
            augment=False
        ),
    }
    return dataset_dict, dataset_config


def build_self_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()

    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="trainval", #如果使用sunrgbd同时加载训练集与验证集进行预训练，需要设置为trainval，否则设置为all
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=True
        ),
        "test": dataset_builder(
            dataset_config,
            split_set="val",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            augment=False
        ),
    }
    return dataset_dict, dataset_config

