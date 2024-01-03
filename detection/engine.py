# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import datetime
import logging
import math
import time
import sys
from third_party.pointnet2 import pointnet2_utils
from utils import pc_util
from utils.pc_util import scale_points, shift_scale_points
from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.ap_calculator_simple import APCalculator as APCalculator2
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    final_eval,
):

    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)

        loss_reduced = all_reduce_average(loss)
        # loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)
        loss.backward()

        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()


        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            with open(final_eval, "a") as fh:
                fh.write(f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB")
                fh.write("\n")
            #logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            # train_dict = {}
            # train_dict["lr"] = curr_lr
            # train_dict["memory"] = mem_mb
            # train_dict["loss"] = loss_avg.avg
            # train_dict["batch_time"] = time_delta.avg
            #logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return ap_calculator

def mask_center_rand(center, num_mask):
    '''
        center : B G 3
        --------------
        mask : B G (bool)
    '''
    B, G, _ = center.shape

    overall_mask = np.zeros([B, G])
    for i in range(B):
        mask = np.hstack([
            np.zeros(G-num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        overall_mask[i, :] = mask
    overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

    return overall_mask.to(center.device)


def calculate_bounding_boxes(point_cloud, group_indices):
    bounding_boxes_all = []
    nearest_points_all_all = []

    for i in range(point_cloud.size(0)):
        point_cloud_temp = point_cloud[i]
        group_indices_temp = group_indices[i]
        bounding_boxes = []

        nearest_points_all = []
        for indices in group_indices_temp:
            # 步骤3: 获取KNN点
            nearest_points = point_cloud_temp[indices]

            # 步骤4: 计算边界框的中心
            box_center = nearest_points.mean(dim=0)

            # 步骤5: 计算边界框的旋转矩阵
            # 你可以根据你的具体需求添加旋转矩阵的计算

            # 步骤6: 计算边界框的半边长
            half_lengths = 0.5 * (torch.max(nearest_points, dim=0).values - torch.min(nearest_points, dim=0).values)

            bounding_boxes.append(torch.cat((box_center, half_lengths), dim=0))
            nearest_points_all.append(nearest_points)

        bounding_boxes_all.append(torch.stack(bounding_boxes))
        nearest_points_all_all.append(torch.stack(nearest_points_all))
    bounding_boxes_all = torch.stack(bounding_boxes_all)
    nearest_points_all_all = torch.stack(nearest_points_all_all)
    return bounding_boxes_all, nearest_points_all_all

def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]     # 提取输入边界框的中心点坐标和尺寸信息
    new_centers = np.dot(centers, np.transpose(rot_mat))            # 计算旋转后的中心点坐标，通过将原始中心点与旋转矩阵相乘得到

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0               # 计算边界框在新坐标系下的四个角点的坐标
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)                                 # 计算旋转后的边界框的尺寸，即新坐标系下的宽度和高度
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)       # 将旋转后的中心点坐标和尺寸信息拼接在一起，形成新的边界框

def flip_axis_to_camera_np(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = pc.copy()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output

def get_3d_box_batch_np(box_size, angle, center):
    input_shape = angle.shape
    R = roty_batch(angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate(
        (l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1
    )
    corners_3d[..., :, 1] = np.concatenate(
        (h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1
    )
    corners_3d[..., :, 2] = np.concatenate(
        (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
    )
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d

def box_parametrization_to_corners_np(box_center_unnorm, box_size, box_angle):
    box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
    boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
    return boxes

def get_label(pc, ib,center_points,nearest_point):
    box_corners_all = []
    box_centers_all = []
    box_centers_normalized_all = []
    angle_classes_all = []
    angle_residuals_all = []
    target_bboxes_mask_all = []
    raw_sizes_all = []
    box_sizes_normalized_all = []
    raw_angles_all = []
    point_cloud_dims_min_all = []
    point_cloud_dims_max_all = []

    for i in range(center_points.size(0)):
        point_cloud = pc[i].cpu().numpy()
        instance_bboxes = ib[i].cpu().numpy()
        MAX_NUM_OBJ = 1024
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        target_bboxes_mask[0: instance_bboxes.shape[0]] = 1  # 由于instance_bboxes不确定 有多少个就target_bboxes_mask的前多少个为1
        target_bboxes[0: instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]  # target_bboxes前n个放xyzwhl这6个值



        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        rot_mat = pc_util.rotz(rot_angle)  # 根据一个旋转角度得到一个旋转矩阵
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))  # 旋转点云
        target_bboxes = rotate_aligned_boxes(  # 旋转检测框，只改变检测狂的xyzwhl 因为原始点云的xyz改变了
            target_bboxes, rot_mat
        )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(  # 它用于将一组点云的坐标从一个坐标范围映射到另一个坐标范围
            box_centers[None, ...],
            src_range=[  # src_range 源坐标范围，包含每个维度的最小和最大值 原始坐标区间
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=center_normalizing_range,  # dst_range 目标坐标范围，包含每个维度的最小和最大值 [0,1]区间
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)  # (64, 3)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]  # 标准化后的盒子中心点坐标
        mult_factor = point_cloud_dims_max - point_cloud_dims_min  # xyz最大差距
        box_sizes_normalized = scale_points(  # 缩放因子是1/最大差距
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)  # (64, 8, 3)

        box_corners_all.append(torch.from_numpy(box_corners.astype(np.float32)))
        box_centers_all.append(torch.from_numpy(box_centers.astype(np.float32)))
        box_centers_normalized_all.append(torch.from_numpy(box_centers_normalized.astype(np.float32)))
        angle_classes_all.append(torch.from_numpy(angle_classes.astype(np.int64)))
        angle_residuals_all.append(torch.from_numpy(angle_residuals.astype(np.float32)))
        target_bboxes_mask_all.append(torch.from_numpy(target_bboxes_mask.astype(np.float32)))
        raw_sizes_all.append(torch.from_numpy(raw_sizes.astype(np.float32)))
        box_sizes_normalized_all.append(torch.from_numpy(box_sizes_normalized.astype(np.float32)))
        raw_angles_all.append(torch.from_numpy(raw_angles.astype(np.float32)))
        point_cloud_dims_min_all.append(torch.from_numpy(point_cloud_dims_min.astype(np.float32)))
        point_cloud_dims_max_all.append(torch.from_numpy(point_cloud_dims_max.astype(np.float32)))


    ret_dict = {}
    ret_dict["point_clouds"] = pc
    ret_dict["point_clouds_center"] = center_points
    ret_dict["point_clouds_nearest"] = nearest_point
    ret_dict["gt_box_corners"] = torch.stack(box_corners_all).to(pc.device)
    ret_dict["gt_box_centers"] = torch.stack(box_centers_all).to(pc.device)
    ret_dict["gt_box_centers_normalized"] = torch.stack(box_centers_normalized_all).to(pc.device)
    ret_dict["gt_angle_class_label"] = torch.stack(angle_classes_all).to(pc.device)
    ret_dict["gt_angle_residual_label"] = torch.stack(angle_residuals_all).to(pc.device)
    ret_dict["gt_box_present"] = torch.stack(target_bboxes_mask_all).to(pc.device)
    ret_dict["gt_box_sizes"] = torch.stack(raw_sizes_all).to(pc.device)
    ret_dict["gt_box_sizes_normalized"] = torch.stack(box_sizes_normalized_all).to(pc.device)
    ret_dict["gt_box_angles"] = torch.stack(raw_angles_all).to(pc.device)
    ret_dict["point_cloud_dims_min"] = torch.stack(point_cloud_dims_min_all).to(pc.device)
    ret_dict["point_cloud_dims_max"] = torch.stack(point_cloud_dims_max_all).to(pc.device)

    return ret_dict

def train_simple_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):


    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    loss_rec_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        num_mask = 512
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        ###FPS采样2048个点，并保留inds作为后续3detr encoder编码的中心点位置
        xyz = batch_data_label["point_clouds"]
        xyz_flipped = xyz.transpose(1,2).contiguous()
        batch_size, num_points, _ = xyz.shape
        # 最远点采样的索引和中心点
        inds = pointnet2_utils.furthest_point_sample(xyz, args.preenc_npoints)#(B 2048)
        center = pointnet2_utils.gather_operation(
            xyz_flipped, inds.contiguous()
        ).transpose(1, 2).contiguous()

        ###取1024点作为mask point,计算被mask的中心点和未被mask的中心索引
        bool_masked_pos = mask_center_rand(center, num_mask)
        box_center_points = center[bool_masked_pos].reshape(batch_size, -1, 3)
        # index_vis_center = inds[~bool_masked_pos].reshape(batch_size, -1)


        idx_raw = knn_point(args.group_size, xyz, box_center_points)#被mask的中心点的patch索引
        # idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # idx = idx_raw + idx_base
        # idx = idx.view(-1)
        # box_patch = xyz.view(batch_size * num_points, -1)[idx, :]
        # box_patch = box_patch.view(batch_size, num_mask, args.group_size, 3).contiguous()#(B 256 32 3) 绝对坐标
        ###计算patch中扩大后的bbox中心/尺寸信息，并做归一化
        # 得到检测框和重建patch的绝对坐标
        instance_bboxes, nearest_point = calculate_bounding_boxes(xyz,idx_raw)
        # 计算box
        batch_data_label = get_label(pc=xyz,ib=instance_bboxes,center_points=box_center_points,nearest_point=nearest_point)



        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_clouds_nearest": batch_data_label["point_clouds_nearest"],
            "point_clouds_center": batch_data_label["point_clouds_center"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            "center_point_inds": inds,
            "bool_masked_pos": bool_masked_pos,
        }
        outputs, loss_rec = model(inputs)

        # # Forward pass
        # optimizer.zero_grad()
        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_nearest": batch_data_label["point_clouds_nearest"],
        #     "point_clouds_center": batch_data_label["point_clouds_center"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #     "center_point_inds": batch_data_label["center_point_inds"]
        # }
        # outputs, loss_rec = model(inputs)

        # Compute loss
        loss_dete, loss_dict = criterion(outputs, batch_data_label)
        loss_rec = loss_rec * 1000
        loss = loss_dete + loss_rec
        loss_reduced = all_reduce_average(loss_dete)
        loss_reduced_rec = all_reduce_average(loss_rec)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)


        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            # ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())
        loss_rec_avg.update(loss_reduced_rec.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; Loss Rec {loss_rec_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return None
def train_ablation2_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):


    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    #loss_rec_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        num_mask = 512
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        ###FPS采样2048个点，并保留inds作为后续3detr encoder编码的中心点位置
        xyz = batch_data_label["point_clouds"]
        xyz_flipped = xyz.transpose(1,2).contiguous()
        batch_size, num_points, _ = xyz.shape
        # 最远点采样的索引和中心点
        inds = pointnet2_utils.furthest_point_sample(xyz, args.preenc_npoints)#(B 2048)
        center = pointnet2_utils.gather_operation(
            xyz_flipped, inds.contiguous()
        ).transpose(1, 2).contiguous()

        ###取1024点作为mask point,计算被mask的中心点和未被mask的中心索引
        bool_masked_pos = mask_center_rand(center, num_mask)
        box_center_points = center[bool_masked_pos].reshape(batch_size, -1, 3)
        # index_vis_center = inds[~bool_masked_pos].reshape(batch_size, -1)


        idx_raw = knn_point(args.group_size, xyz, box_center_points)#被mask的中心点的patch索引
        # idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # idx = idx_raw + idx_base
        # idx = idx.view(-1)
        # box_patch = xyz.view(batch_size * num_points, -1)[idx, :]
        # box_patch = box_patch.view(batch_size, num_mask, args.group_size, 3).contiguous()#(B 256 32 3) 绝对坐标
        ###计算patch中扩大后的bbox中心/尺寸信息，并做归一化
        # 得到检测框和重建patch的绝对坐标
        instance_bboxes, nearest_point = calculate_bounding_boxes(xyz,idx_raw)
        # 计算box
        batch_data_label = get_label(pc=xyz,ib=instance_bboxes,center_points=box_center_points,nearest_point=nearest_point)



        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_clouds_nearest": batch_data_label["point_clouds_nearest"],
            "point_clouds_center": batch_data_label["point_clouds_center"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            "center_point_inds": inds,
            "bool_masked_pos": bool_masked_pos,
        }
        outputs = model(inputs)

        # # Forward pass
        # optimizer.zero_grad()
        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_nearest": batch_data_label["point_clouds_nearest"],
        #     "point_clouds_center": batch_data_label["point_clouds_center"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #     "center_point_inds": batch_data_label["center_point_inds"]
        # }
        # outputs, loss_rec = model(inputs)

        # Compute loss
        loss_dete, loss_dict = criterion(outputs, batch_data_label)
        #loss_rec = loss_rec * 1000
        loss = loss_dete 
        loss_reduced = all_reduce_average(loss_dete)
        #loss_reduced_rec = all_reduce_average(loss_rec)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)


        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            # ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())
        #loss_rec_avg.update(loss_reduced_rec.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return None

def train_ablation_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):


    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    #loss_rec_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        num_mask = 512
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        ###FPS采样2048个点，并保留inds作为后续3detr encoder编码的中心点位置
        xyz = batch_data_label["point_clouds"]
        xyz_flipped = xyz.transpose(1,2).contiguous()
        batch_size, num_points, _ = xyz.shape
        # 最远点采样的索引和中心点
        inds = pointnet2_utils.furthest_point_sample(xyz, args.preenc_npoints)#(B 2048)
        center = pointnet2_utils.gather_operation(
            xyz_flipped, inds.contiguous()
        ).transpose(1, 2).contiguous()

        ###取1024点作为mask point,计算被mask的中心点和未被mask的中心索引
        bool_masked_pos = mask_center_rand(center, num_mask)
        box_center_points = center[bool_masked_pos].reshape(batch_size, -1, 3)
        # index_vis_center = inds[~bool_masked_pos].reshape(batch_size, -1)


        idx_raw = knn_point(args.group_size, xyz, box_center_points)#被mask的中心点的patch索引
        # idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # idx = idx_raw + idx_base
        # idx = idx.view(-1)
        # box_patch = xyz.view(batch_size * num_points, -1)[idx, :]
        # box_patch = box_patch.view(batch_size, num_mask, args.group_size, 3).contiguous()#(B 256 32 3) 绝对坐标
        ###计算patch中扩大后的bbox中心/尺寸信息，并做归一化
        # 得到检测框和重建patch的绝对坐标
        instance_bboxes, nearest_point = calculate_bounding_boxes(xyz,idx_raw)
        # 计算box
        batch_data_label = get_label(pc=xyz,ib=instance_bboxes,center_points=box_center_points,nearest_point=nearest_point)



        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_clouds_nearest": batch_data_label["point_clouds_nearest"],
            "point_clouds_center": batch_data_label["point_clouds_center"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            "center_point_inds": inds,
            "bool_masked_pos": bool_masked_pos,
        }
        loss = model(inputs)

        # # Forward pass
        # optimizer.zero_grad()
        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_nearest": batch_data_label["point_clouds_nearest"],
        #     "point_clouds_center": batch_data_label["point_clouds_center"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #     "center_point_inds": batch_data_label["center_point_inds"]
        # }
        # outputs, loss_rec = model(inputs)

        # Compute loss
        # loss_dete, loss_dict = criterion(outputs, batch_data_label)
        loss = loss * 1000
        # loss = loss_dete + loss_rec
        # loss_reduced = all_reduce_average(loss_dete)
        loss_reduced = all_reduce_average(loss)
        # loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)


        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            # ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())
        # loss_rec_avg.update(loss_reduced_rec.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            #logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return None

@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            # loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        # outputs["outputs"] = all_gather_dict(outputs["outputs"])
        # batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            # test_dict = {}
            # test_dict["memory"] = mem_mb
            # test_dict["batch_time"] = time_delta.avg
            # if criterion is not None:
            #     test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    # if is_primary():
    #     if criterion is not None:
    #         logger.log_scalars(
    #             loss_dict_reduced, curr_train_iter, prefix="Test_details/"
    #         )
    #     logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return ap_calculator
