import sys

import numpy as np
import torch

from dolphin import ext


# This function is modified from: https://github.com/pytorch/vision/
class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, offset):
        inds = ext.nms(
            bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        return inds

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, offset):
        from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze
        boxes = unsqueeze(g, bboxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op(
            'Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        iou_threshold = g.op(
            'Constant',
            value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores,
                       max_output_per_class, iou_threshold)
        return squeeze(
            g,
            select(
                g, nms_out, 1,
                g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))),
            1)


def nms(boxes, scores, iou_threshold, offset=0):
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    if torch.__version__ == 'parrots':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        _, order = scores.sort(0, descending=True)
        if boxes.device == 'cpu':
            indata_list = [boxes, order, areas]
            indata_dict = {
                'iou_threshold': float(iou_threshold),
                'offset': int(offset)}
            select = ext.nms(*indata_list, **indata_dict).byte()
        else:
            boxes_sorted = boxes.index_select(0, order)
            indata_list = [boxes_sorted, order, areas]
            indata_dict = {
                'iou_threshold': float(iou_threshold),
                'offset': int(offset)}
            select = ext.nms(*indata_list, **indata_dict)
        inds = order.masked_select(select)
    else:
        if torch.onnx.is_in_onnx_export() and offset == 0:
            # ONNX only support offset == 1
            boxes[:, -2:] -= 1
        inds = NMSop.apply(boxes, scores, iou_threshold, offset)
        if torch.onnx.is_in_onnx_export() and offset == 0:
            # ONNX only support offset == 1
            boxes[:, -2:] += 1
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def soft_nms(boxes,
             scores,
             iou_threshold=0.3,
             sigma=0.5,
             min_score=1e-3,
             method='linear',
             offset=0):

    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)
    method_dict = {'naive': 0, 'linear': 1, 'gaussian': 2}
    assert method in method_dict.keys()

    if torch.__version__ == 'parrots':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        indata_list = [boxes.cpu(), scores.cpu(), areas.cpu()]
        indata_dict = {
            'iou_threshold': float(iou_threshold),
            'sigma': float(sigma),
            'min_score': min_score,
            'method': method_dict[method],
            'offset': int(offset)
        }
        dets, inds, num_out = ext.softnms(*indata_list, **indata_dict)
        inds = inds[:num_out]
    else:
        dets = boxes.new_empty((boxes.size(0), 5), device='cpu')
        inds = ext.softnms(
            boxes.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=method_dict[method],
            offset=int(offset))
    dets = dets[:inds.size(0)]
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
        return dets, inds
    else:
        return dets.to(device=boxes.device), inds.to(device=boxes.device)


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
