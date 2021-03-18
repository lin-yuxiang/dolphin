import torch
import torch.nn as nn
import numpy as np
import cv2
import sys

from dolphin.dataset.utils import iou_with_anchors, iou3dt


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat(
        [xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2,
         xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds


def flip_tensor(x):
    return torch.flip(x, [3])


def moc_decode(heat, wh, mov, N=100, K=5):
    batch, cat, height, width = heat.size()

    # perform 'nms' on heatmaps
    heat = _nms(heat)
    scores, index, classes, ys, xs = _topk(heat, K=N)

    mov = tranpose_and_gather_feat(mov, index)
    mov = mov.view(batch, N, 2 * K)

    mov_copy = mov.clone()
    mov_copy = mov_copy.view(batch, N, K, 2)
    index_all = torch.zeros((batch, N, K, 2)).cuda()
    xs_all = xs.clone().unsqueeze(2).expand(batch, N, K)
    ys_all = ys.clone().unsqueeze(2).expand(batch, N, K)
    xs_all = xs_all + mov_copy[:, :, :, 0]
    ys_all = ys_all + mov_copy[:, :, :, 1]
    xs_all[:, :, K // 2] = xs
    ys_all[:, :, K // 2] = ys

    xs_all = xs_all.long()
    ys_all = ys_all.long()

    index_all[:, :, :, 0] = xs_all + ys_all * width
    index_all[:, :, :, 1] = xs_all + ys_all * width
    index_all[index_all < 0] = 0
    index_all[index_all > width * height - 1] = width * height - 1
    index_all = index_all.view(batch, N, K * 2).long()

    # gather wh in each location after movement
    wh = tranpose_and_gather_feat(wh, index, index_all=index_all)
    wh = wh.view(batch, N, 2 * K)

    classes = classes.view(batch, N, 1).float()
    scores = scores.view(batch, N, 1)
    xs = xs.view(batch, N, 1)
    ys = ys.view(batch, N, 1)
    bboxes = []
    for i in range(K):
        bboxes.extend([
            xs + mov[..., 2 * i:2 * i + 1] - wh[..., 2 * i:2 * i + 1] / 2,
            ys + mov[..., 2 * i + 1:2 * i + 2] - wh[
                ..., 2 * i + 1:2 * i + 2] / 2,
            xs + mov[..., 2 * i:2 * i + 1] + wh[..., 2 * i:2 * i + 1] / 2,
            ys + mov[..., 2 * i + 1:2 * i + 2] + wh[
                ..., 2 * i + 1:2 * i + 2] / 2])
    bboxes = torch.cat(bboxes, dim=2)
    detections = torch.cat([bboxes, scores, classes], dim=2)

    return detections


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    
    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dor(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 90, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)

    return ret


def moc_post_process(detections, h, w, o_h, o_w, num_classes, K):
    detections = detections.detach().cpu().numpy()

    results = []
    for i in range(detections.shape[0]):
        top_preds = {}
        for j in range((detections.shape[2] - 2) // 2): 
            detections[i, :, 2 * j] = np.maximum(
                0, np.minimum(w - 1, detections[i, :, 2 * j] / o_w * w))
            detections[i, :, 2 * j + 1] = np.maximum(
                0, np.minimum(h - 1, detections[i, :, 2 * j + 1] / o_h * h))
        classes = detections[:, :, -1]
        for c in range(num_classes):
            inds = (classes == c)
            top_preds[c + 1] = detections[
                i, inds, :4 * K + 1].astype(np.float32)
        results.append(top_preds)
    return results


def bmn_post_process(result, video_info, soft_nms_alpha, soft_nms_low_thres,
                     soft_nms_high_thres, post_process_top_k):
    if len(result) > 1:
        result = bmn_soft_nms(
            result, soft_nms_alpha, soft_nms_low_thres, soft_nms_high_thres, 
            post_process_top_k)
    
    result = result[result[:, -1].argsort()[::-1]]
    video_duration = float(
        video_info['duration_frame'] // 16 * 16) / video_info[
            'duration_frame'] * video_info['duration_second']
    proposal_list = []

    for j in range(min(post_process_top_k, len(result))):
        proposal = {}
        proposal['score'] = float(result[j, -1])
        proposal['segment'] = [
            max(0, result[j, 0]) * video_duration,
            min(1, result[j, 1]) * video_duration ]
        proposal_list.append(proposal)
    return proposal_list


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def nms2d(boxes, overlap=0.6):
    """Compute the soft nms given a set of scored boxes,
    as numpy array with 5 columns <x1> <y1> <x2> <y2> <score>
    return the indices of the tubelets to keep
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1

    while order.size > 0:
        i = order[0]

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        index = np.where(iou > overlap)[0]
        weight[order[index + 1]] = 1 - iou[index]

        index2 = np.where(iou <= overlap)[0]
        order = order[index2 + 1]

    boxes[:, 4] = boxes[:, 4] * weight

    return boxes


def bmn_soft_nms(proposals, alpha, low_threshold, high_threshold, top_k):

    proposals = proposals[proposals[:, -1].argsort()[::-1]]
    tstart = list(proposals[:, 0])
    tend = list(proposals[:, 1])
    tscore = list(proposals[:, -1])
    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 0 and len(rscore) <= top_k:
        max_index = np.argmax(tscore)
        max_width = tend[max_index] - tstart[max_index]
        iou_list = iou_with_anchors(
            tstart[max_index], tend[max_index], 
            np.array(tstart), np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / alpha)
        for idx in range(len(tscore)):
            if idx != max_index:
                current_iou = iou_list[idx]
                if current_iou > low_threshold + (
                    high_threshold - low_threshold) * max_width:
                    tscore[idx] = tscore[idx] * iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    rstart = np.array(rstart).reshape(-1, 1)
    rend = np.array(rend).reshape(-1, 1)
    rscore = np.array(rscore).reshape(-1, 1)
    new_proposals = np.concatenate((rstart, rend, rscore), axis=1)
    return new_proposals


def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0:
        dets
    if top_k is None:
        top_k = len(dets)

    K = int((dets.shape[1] - 1) / 4)

    # Coordinates of bounding boxes
    x1 = [dets[:, 4 * k] for k in range(K)]
    y1 = [dets[:, 4 * k + 1] for k in range(K)]
    x2 = [dets[:, 4 * k + 2] for k in range(K)]
    y2 = [dets[:, 4 * k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, -1]
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1
    counter = 0

    while order.size > 0:
        i = order[0]
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][order[1:]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][order[1:]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][order[1:]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][order[1:]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum(
            [inter_area[k] / (area[k][order[1:]] + area[k][i] - inter_area[k]) 
            for k in range(K)])
        index = np.where(ious > overlapThresh * K)[0]
        weight[order[index + 1]] = 1 - ious[index]

        index2 = np.where(ious <= overlapThresh * K)[0]
        order = order[index2 + 1]

    dets[:, -1] = dets[:, -1] * weight

    new_scores = dets[:, -1]
    new_order = np.argsort(new_scores)[::-1]
    dets = dets[new_order, :]

    return dets[:top_k, :]


def nms3dt(tubes, overlap=0.5):

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t[1] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([iou3dt(tubes[ii][0], tubes[i][0]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]

    return indices[:counter]


#=====================================================================


class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, offset):
        inds = extensions.nms(
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


def bbox2roi(bbox_list):
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4]
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4]
    return flipped


def bbox_mapping(bboxes,
                 img_shape,
                 scale_factor,
                 flip,
                 flip_direction='horizontal'):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes