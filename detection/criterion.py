# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment


class MatcherSimple(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["center_dist"].shape[0]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        # pred_cls_prob = outputs["sem_cls_prob"]

        # objectness cost: batch x nqueries x 1
        # objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            # self.cost_objectness * objectness_mat
            self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=outputs["center_dist"].device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=outputs["center_dist"].device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=outputs["center_dist"].device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }

class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):      # 计算目标检测中的一个损失项，用于目标数量（cardinality）估计
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}

    def loss_sem_cls(self, outputs, targets, assignments):          # 计算语义分类损失项，用于目标的语义分类。


        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )

        return {"loss_sem_cls": loss}

    def loss_angle(self, outputs, targets, assignments):            # 计算角度损失项，用于目标的朝向角度。
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments):           # 计算中心点损失项，用于目标的中心点位置。
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):             # 计算GIOU损失项，用于边界框的位置和尺寸。
        gious_dist = 1 - outputs["gious"]

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):             # 计算尺寸损失项，用于目标的尺寸。
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()           # 得到每个点云中存在的目标数量
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()     # 计算总目标数量
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


class SetSimpleCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)

        self.loss_functions = {
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            # "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):      # 计算目标检测中的一个损失项，用于目标数量（cardinality）估计
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}

    def loss_angle(self, outputs, targets, assignments):            # 计算角度损失项，用于目标的朝向角度。
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments):           # 计算中心点损失项，用于目标的中心点位置。
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):             # 计算GIOU损失项，用于边界框的位置和尺寸。
        gious_dist = 1 - outputs["gious"]

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):             # 计算尺寸损失项，用于目标的尺寸。
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()           # 得到每个点云中存在的目标数量
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()     # 计算总目标数量
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict
class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights
    
class SetKPSCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)
        self.loss_query_kps_weight = loss_weight_dict["loss_query_kps_weight"]
        del loss_weight_dict["loss_query_kps_weight"]

        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):      # 计算目标检测中的一个损失项，用于目标数量（cardinality）估计
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}
    def loss_query_kps(self, outputs, targets,topk=8):
        box_label_mask = targets["gt_box_present"]
        seed_inds =  outputs["seed_inds"].long()#(B nquery)
        seed_xyz = outputs["seed_xyz"]#(B nquery 3)
        points_obj_cls_logits = outputs["points_obj_cls_logits"]#(B 1 nquery)
        gt_center = targets["gt_box_centers"]  # B, K2, 3
        gt_size = targets["gt_box_sizes"]  # B, K2, 3
        B = gt_center.shape[0]
        K = seed_xyz.shape[1]
        K2 = gt_center.shape[1]
        
        point_instance_label=targets["point_instance_label"]#B npoints
        object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
        object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
        object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
        object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
        delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
        delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
        new_dist = torch.sum(delta_xyz ** 2, dim=-1)
        euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
        euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
        euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
        topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                    (box_label_mask[:, :, None] - 1)  # BxK2xtopk
        topk_inds = topk_inds.long()  # BxK2xtopk
        topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
        batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
        batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

        objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
        objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
        objectness_label = objectness_label[:, :K]
        objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
        objectness_label[objectness_label_mask < 0] = 0


        # Compute objectness loss
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = (objectness_label >= 0).float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        cls_loss_src = criterion(points_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B
        return objectness_loss
        
    def loss_sem_cls(self, outputs, targets, assignments):          # 计算语义分类损失项，用于目标的语义分类。
        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )

        return {"loss_sem_cls": loss}

    def loss_angle(self, outputs, targets, assignments):            # 计算角度损失项，用于目标的朝向角度。
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments):           # 计算中心点损失项，用于目标的中心点位置。
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):             # 计算GIOU损失项，用于边界框的位置和尺寸。
        gious_dist = 1 - outputs["gious"]

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):             # 计算尺寸损失项，用于目标的尺寸。
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments)
                losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()           # 得到每个点云中存在的目标数量
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()     # 计算总目标数量
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        loss += self.loss_query_kps_weight*self.loss_query_kps(outputs["query_prediction"],targets)
        return loss, loss_dict

def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
    }
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion

def build_simple_criterion(args, dataset_config):
    matcher = MatcherSimple(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
    }
    criterion = SetSimpleCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion

def build_kps_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
        "loss_query_kps_weight":args.loss_query_kps_weight,
    }
    criterion = SetKPSCriterion(matcher, dataset_config, loss_weight_dict)
    return criterion