# LR-MAE: Locate while Reconstructing with Masked Autoencoders for Point Cloud Self-supervised Learning

This repository provides the official implementation of **Locate while Reconstructing with Masked Autoencoders for Point Cloud Self-supervised Learning**.

## 1. Introduction

As an efficient self-supervised pre-training approach, Masked autoencoder (MAE) has shown promising improvement across various 3D point cloud understanding tasks. However, the pretext task of existing point-based MAE is to reconstruct the geometry of masked points only, hence it learns features at lower semantic levels which is not appropriate for high-level downstream tasks. To address this challenge, we propose a novel self-supervised approach named Locate while Reconstructing with Masked Autoencoders (LR-MAE). Specifically, a multi-head decoder is designed to simultaneously localize the global position of masked patches while reconstructing masked points, aimed at learning better semantic features that align with downstream tasks. Moreover, we design a random query patch detection strategy for 3D object detection tasks in the pre-training stage, which significantly boosts the model performance with faster convergence speed. Extensive experiments show that our LR-MAE achieves superior performance on various point cloud understanding tasks. By fine-tuning on downstream datasets, LR-MAE outperforms the Point-MAE baseline by 3.65% classification accuracy  on the ScanObjectNN dataset, and significantly exceeds the 3DETR baseline by 6.1\% $AP_{50}$ on the ScanNetV2 dataset.


## 2. Preparation
Our code is tested with PyTorch 1.8.0, CUDA 11.1 and Python 3.7.0. 
### 2.1 Requirement
```
conda create -y -n lrmae python=3.7
conda activate lrmae
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Chamfer Distance & emd
cd /extensions/chamfer_dist
python setup.py install --user
cd /extensions/emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
Optionally, you can install a Cythonized implementation of gIOU for faster training.
```
conda install cython
cd ./detection/utils && python cython_compile.py build_ext --inplace
```

### 2.2 Download dataset
Before running the code, you need to download dataset and **modify the corresponding file path** in the code.
Here we have also collected the download links of required datasets for you:
- ShapeNet55/34: [[link](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)].
- ScanObjectNN: [[link](https://hkust-vgd.github.io/scanobjectnn/)].
- ModelNet40: [[link](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md)].
- ShapeNetPart: [[link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)].
- SUN RGB-D: [[link]](https://github.com/facebookresearch/votenet/tree/main/sunrgbd).
- ScanNet: [[link]](https://github.com/facebookresearch/votenet/tree/main/scannet).

## 3. Pre-training
### 3.1 Unsupervised pre-training on ShapeNet
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain.yaml --exp_name ./pretrain_upmae
```

### 3.2 Unsupervised pre-training on SUN RGB-D
```
cd ./detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_upmae.py --dataset_name upmaesunrgbd --checkpoint_dir ./checkpoint_upmae --model_name up_mae --ngpus 4
```
## 4. Tune pre-trained models on downstream tasks
### 4.1 Object classification
- ModelNet40
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet.yaml --finetune_model --exp_name ./modelnet1k_ft --ckpts ./experiments/pretrain/cfgs/pretrain_upmae/ckpt-epoch-300.pth

# if you want to test the model with vote, please run:
CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet.yaml --test --exp_name ./modelnet1k_ft_vote --ckpts path/to/model
```
- ScanObjectNN (OBJ-BG)
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg.yaml --finetune_model --exp_name ./scan_objbg_upmae_ft --ckpts ./experiments/pretrain/cfgs/pretrain_upmae/ckpt-epoch-300.pth
```
- ScanObjectNN (OBJ-ONLY)
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly.yaml --finetune_model --exp_name ./scan_objonly_upmae_ft --ckpts ./experiments/pretrain/cfgs/pretrain_upmae/ckpt-epoch-300.pth
```
- ScanObjectNN (PB-T50-RS)
```
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest.yaml --finetune_model --exp_name ./scan_hardest_upmae_ft --ckpts ./experiments/pretrain/cfgs/pretrain_upmae/ckpt-epoch-300.pth
```
### 4.2 Part Segmentation 
- ShapeNet-Part
```
cd ./segmentation
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts ../experiments/pretrain/cfgs/pretrain_upmae/ckpt-epoch-300.pth --log_dir ./shapenetpart1 --seed 1 --root data/path --learning_rate 0.0002 --epoch 300
```
### 4.3 3D object detection
- ScanNet
```
cd ./detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--model_name up_mae_3detr --ngpus 4 --nqueries 256 \
--batchsize_per_gpu 12 \
--pretrain_ckpt checkpoint_upmae/ckpt-last.pth \
--dataset_name scannet \
--max_epoch 1080 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--checkpoint_dir ./checkpoint_mae_q256_scannet
```
- SUN RGB-D
```
cd ./detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--model_name up_mae_3detr --ngpus 4 --nqueries 256 \
--batchsize_per_gpu 10 \
--pretrain_ckpt checkpoint_upmae/ckpt-last.pth \
--dataset_name sunrgbd  \
--base_lr 7e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--seed 2 \
--checkpoint_dir ./checkpoint_mae_q256_sunrgbd
```





## 5. Acknowledgements
Our code is based on prior work such as [3DETR](https://github.com/facebookresearch/3detr) and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE). Thanks for their efforts.

