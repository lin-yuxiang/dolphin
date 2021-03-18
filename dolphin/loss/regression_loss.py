import torch
import torch.nn as nn
import torch.nn.functional as F

from dolphin.utils import Registers
from .base import BaseWeightedLoss
from dolphin.utils.postprocess import tranpose_and_gather_feat


def binary_logistic_regression_loss(reg_score,
                                    label,
                                    threshold=0.5,
                                    ratio_range=(1.05, 21),
                                    eps=1e-5):
    """Binary Logistic Regression Loss."""
    label = label.view(-1).to(reg_score.device)
    reg_score = reg_score.contiguous().view(-1)

    pmask = (label > threshold).float().to(reg_score.device)
    num_positive = max(torch.sum(pmask), 1)
    num_entries = len(label)
    ratio = num_entries / num_positive
    # clip ratio value between ratio_range
    ratio = min(max(ratio, ratio_range[0]), ratio_range[1])

    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    loss = coef_1 * pmask * torch.log(reg_score + eps) + coef_0 * (
        1.0 - pmask) * torch.log(1.0 - reg_score + eps)
    loss = -torch.mean(loss)
    return loss


def _reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = F.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss /= (num + 1e-4)
    return regr_loss


@Registers.loss.register
class L1Loss(BaseWeightedLoss):

    def _forward(self, inputs, label):
        loss = F.l1_loss(inputs, label)
        return loss


@Registers.loss.register
class RegLoss(BaseWeightedLoss):

    def _forward(self, inputs, mask, ind, label, **kwargs):
        inputs = tranpose_and_gather_feat(inputs, ind)
        loss = _reg_loss(inputs, label, mask)
        return loss
        

@Registers.loss.register
class RegL1Loss(BaseWeightedLoss):

    def _forward(self, inputs, mask, ind, label, **kwargs):
        inputs = tranpose_and_gather_feat(inputs, ind)
        mask = mask.unsqueeze(2).expand_as(inputs).float()
        loss = F.l1_loss(inputs * mask, label * mask, size_average=False)
        loss /= (mask.sum() + 1e-4)
        return loss


@Registers.loss.register
class NormRegL1Loss(BaseWeightedLoss):

    def _forward(self, inputs, mask, ind, label):
        inputs = tranpose_and_gather_feat(inputs, ind)
        mask = mask.unsqueeze(2).expand_as(inputs).float()
        inputs = inputs / (label + 1e-4)
        label = label * 0 + 1
        loss = F.l1_loss(inputs * mask, label * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


@Registers.loss.register
class RegWeightedL1Loss(BaseWeightedLoss):

    def _forward(self, inputs, mask, ind, label):
        inputs = tranpose_and_gather_feat(inputs, ind)
        mask = mask.float()

        loss = F.l1_loss(inputs * mask, label * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


@Registers.loss.register
class MSELoss(BaseWeightedLoss):

    def _forward(self, inputs, label):
        loss = F.mse_loss(inputs, label)
        return loss


@Registers.loss.register
class TemporalEvaluationModuleLoss(BaseWeightedLoss):

    def tem_loss(self, pred_start, pred_end, gt_start, gt_end, tem_thresh=0.5):
        """Calculate Temporal Evaluation Module Loss.

        This function calculate the binary_logistic_regression_loss for start
        and end respectively and returns the sum of their losses.

        Args:
            pred_start (torch.Tensor): Predicted start score by BMN model.
            pred_end (torch.Tensor): Predicted end score by BMN model.
            gt_start (torch.Tensor): Groundtruth confidence score for start.
            gt_end (torch.Tensor): Groundtruth confidence score for end.

        Returns:
            torch.Tensor: Returned binary logistic loss.
        """
        loss_start = binary_logistic_regression_loss(
            pred_start, gt_start, threshold=tem_thresh)
        loss_end = binary_logistic_regression_loss(
            pred_end, gt_end, threshold=tem_thresh)
        loss = loss_start + loss_end
        return loss
    
    def _forward(self, pred_start, pred_end, gt_start, gt_end, **kwargs):
        loss = self.tem_loss(pred_start, pred_end, gt_start, gt_end, **kwargs)
        return loss


@Registers.loss.register
class ProposalEvaluationRegressionLoss(BaseWeightedLoss):

    def pem_reg_loss(self,
                     pred_score,
                     gt_iou_map,
                     mask,
                     pem_reg_high=0.7,
                     pem_reg_low=0.3):
        """Calculate Proposal Evaluation Module Regression Loss.

        Args:
            pred_score (torch.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (torch.Tensor): Groundtruth temporal_iou score.
            mask (torch.Tensor): Boundary-Matching mask.
            high_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.7.
            low_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.3.

        Returns:
            torch.Tensor: Proposal evalutaion regression loss.
        """
        u_hmask = (gt_iou_map > pem_reg_high).float()
        u_mmask = ((gt_iou_map <= pem_reg_high) &
                   (gt_iou_map > pem_reg_low)).float()
        u_lmask = ((gt_iou_map <= pem_reg_low) &
                   (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.rand_like(gt_iou_map)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(
            loss * torch.ones_like(weights)) / torch.sum(weights)

        return loss

    def _forward(self, pred_score, gt_iou_map, mask, **kwargs):
        loss = self.pem_reg_loss(pred_score, gt_iou_map, mask, **kwargs)
        return loss