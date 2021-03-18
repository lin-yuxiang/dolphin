import cv2
import numpy as np
import math
import random
import numbers
import torch
import os
import os.path as osp
from pathlib import Path
from terminaltables import AsciiTable

from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
from dolphin import utils

try:
    from PIL import Image
except ImportError:
    Image = None

imread_backend = 'cv2'

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}


def get_img_bytes(filename):
    filename = str(filename)
    with open(filename, 'rb') as f:
        value_buf = f.read()
    return value_buf


def imread(img_or_path, flag='color', channel_order='bgr'):
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
            Note that the `turbojpeg` backened does not support `unchanged`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, str):
        utils.check_file_exist(
            img_or_path, f'img file does not exist: {img_or_path}')
        if imread_backend == 'turbojpeg':
            pass
        else:
            flag = imread_flags[flag] if isinstance(flag, str) else flag
            img = cv2.imread(img_or_path, flag)
            if flag == IMREAD_COLOR and channel_order == 'rgb':
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')


def imfrombytes(content, flag='color', channel_order='bgr', backend='cv2'):
    if backend == 'cv2':
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if isinstance(flag, str) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    elif backend == 'pillow':
        raise NotImplementedError
    return img


def rescale_size(old_size, scale, return_scale=False, rescale_edge='max'):

    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, (tuple, list)):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        if rescale_edge == 'max':
            scale_factor = min(
                max_long_edge / max(h, w), max_short_edge / min(h, w))
        else:
            scale_factor = max(
                max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def _scale_size(size, scale):

    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):

    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, tuple(size), dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imrescale(img,
              scale,
              return_scale=False,
              interpolation='bilinear',
              rescale_edge='max',
              backend=None):

    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size(
        (w, h), scale, return_scale=True, rescale_edge=rescale_edge)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.
    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
    Returns:
        ndarray: The normalized image.
    """
    obj = img.copy().astype(np.float32)
    assert obj.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    stdinv = 1 / std.reshape(1, -1).astype(np.float64)
    if to_rgb:
        cv2.cvtColor(obj, cv2.COLOR_BGR2RGB, obj)  # inplace
    cv2.subtract(obj, mean, obj)  # inplace
    cv2.multiply(obj, stdinv, obj)  # inplace
    return obj


def random_affine(img=None, 
                  targets=None, 
                  degrees=(-10, 10), 
                  translate=(.1, .1), 
                  scale=(.9, 1.1), 
                  shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):

    if img is not None:
        border = 0  # width of added border (optional)
        height = img.shape[0]
        width = img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
            img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
            img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
            img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + 
            shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + 
            shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(
            img, M, dsize=(width, height), flags=cv2.INTER_LINEAR, 
            borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * \
                (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** \
                0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate(
                (x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

    if img is not None and targets is not None:
        return imw, targets
    elif img is not None and targets is None:
        return imw
    else:
        return targets


def xyxy2xywh(x):
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def draw_msra_gaussian(hm, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = hm.shape[0], hm.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return hm
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2 + 1e-6))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    hm[img_y[0]: img_y[1], img_x[0]: img_x[1]] = np.maximum(
        hm[img_y[0]: img_y[1], img_x[0]: img_x[1]],
        g[g_y[0]: g_y[1], g_x[0]: g_x[1]])
    return hm


def gaussian_radius(det_size, min_overlap=0.7):
    h, w = det_size

    a1 = 1
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        (b1_x1, b1_y1, b1_x2, b1_y2) = (box1[:, 0], box1[:, 1], 
            box1[:, 2], box1[:, 3])
        (b2_x1, b2_y1, b2_x2, b2_y2) = (box2[:, 0], box2[:, 1], 
            box2[:, 2], box2[:, 3])
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def impad(img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant'):

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, (list, tuple)):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a list or tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, (list, tuple)) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img
    

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divided(inter_len, union_len)
    return jaccard


def iou3dt(b1, b2, spatialonly=False):

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(
                                        np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(
                                        np.where(b2[:, 0] == tmax)[0]) + 1, :]
    result = iou3d(tube1, tube2) * (1. 
                        if spatialonly else temporal_inter / temporal_union)
    return result


def iou3d(b1, b2):

    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:, 1:5], b2[:, 1:5])
    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov))


def iou2d(b1, b2):
    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)
    return ov / (area2d(b1) + area2d(b2) - ov)


def overlap2d(b1, b2):
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2], b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3], b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)
    return width * height


def area2d(b):
    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[
        radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    logger.info('\n' + table.table)