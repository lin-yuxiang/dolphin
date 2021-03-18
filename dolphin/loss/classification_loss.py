import torch
import torch.nn as nn
import torch.nn.functional as F

from dolphin.utils import Registers
from .base import BaseWeightedLoss
from .utils import weight_reduce_loss


def classwise_regression_loss(pred, labels, targets):
    indexer = labels.data - 1
    prep = pred[:, indexer, :]
    class_pred = torch.cat((torch.diag(prep[:, :, 0]).view(-1, 1), 
        torch.diag(prep[:, :, 1]).view(-1, 1)), dim=1)
    loss = F.smooth_l1_loss(class_pred.view(-1), targets.view(-1)) * 2
    return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  **kwargs):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


def _neg_loss(pred, gt):
    '''Modified focal loss. Exactly the same as CornerNet.
       Runs faster and costs a little bit more memory.
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


@Registers.loss.register
class FocalLoss(BaseWeightedLoss):

    def _forward(self, inputs, label, **kwargs):
        loss = _neg_loss(inputs, label)
        return loss


@Registers.loss.register
class CrossEntropyLoss(BaseWeightedLoss):

    def _forward(self, inputs, label, **kwargs):
        if kwargs.get('use_sigmoid', False):
            loss = binary_cross_entropy(inputs, label, **kwargs)
        elif kwargs.get('use_mask', False):
            loss = mask_cross_entropy(inputs, label, **kwargs)
        else:
            loss = cross_entropy(inputs, label, **kwargs)
        return loss
    
@Registers.loss.register
class BinaryCrossEntropyLoss(BaseWeightedLoss):

    def _forward(self, inputs, label, **kwargs):
        loss = F.binary_cross_entropy(inputs, label.squeeze().float(), **kwargs)
        return loss


@Registers.loss.register
class ProposalEvaluationModuleClassificationLoss(BaseWeightedLoss):

    def pem_cls_loss(self,
                     pred_score,
                     gt_iou_map,
                     mask,
                     threshold=0.9,
                     ratio_range=(1.05, 21),
                     eps=1e-5):
        """Calculate Proposal Evaluation Module Classification Loss.

        Args:
            pred_score (torch.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (torch.Tensor): Groundtruth temporal_iou score.
            mask (torch.Tensor): Boundary-Matching mask.
            threshold (float): Threshold of temporal_iou for positive
                instances. Default: 0.9.
            ratio_range (tuple): Lower bound and upper bound for ratio.
                Default: (1.05, 21)
            eps (float): Epsilon for small value. Default: 1e-5

        Returns:
            torch.Tensor: Proposal evalutaion classification loss.
        """
        pmask = (gt_iou_map > threshold).float()
        nmask = (gt_iou_map <= threshold).float()
        nmask = nmask * mask

        num_positive = max(torch.sum(pmask), 1)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        ratio = torch.clamp(ratio, ratio_range[0], ratio_range[1])

        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio

        loss_pos = coef_1 * torch.log(pred_score + eps) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + eps) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss

    def _forward(self, pred_score, gt_iou_map, mask, **kwargs):
        loss = self.pem_cls_loss(pred_score, gt_iou_map, mask, **kwargs)
        return loss