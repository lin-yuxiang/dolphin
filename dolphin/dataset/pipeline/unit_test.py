import cv2
import numpy as np
import random
import math
import copy

# from ..utils import (rescale_size, imresize, random_affine, imnormalize, 
                     # imrescale, impad, impad_to_multiple)
import scipy.ndimage as ndimage

import cv2
import numbers
import os
import os.path as osp
from pathlib import Path

from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED

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

img_extensions = [
    '.jpg', '.JPG', 'jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']


#==============================
def is_list_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=list)


def is_seq_of(seq, expected_type, seq_type=None):
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of_list_or_nontype(seq):
    if not isinstance(seq, list):
        return False
    for item in seq:
        if not isinstance(item, list) and item is not None:
            return False
    return True


def is_list_of_list_or_int(seq):
    if not isinstance(seq, list):
        return False
    for item in seq:
        if not isinstance(item, list) and isinstance(item, int):
            return False
    return True


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
    if isinstance(pad_val, list):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(f'pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
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
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
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
              backend=None):

    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(
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
#==============================


class Pad(object):

    def __init__(self,
                 keys=['imgs'],
                 size=None,
                 pad_val=0):

        self.keys = keys

        if is_list_of_list_or_nontype(size):
            if len(size) < len(self.keys) and len(size) == 1:
                self.size = [size for _ in range(len(self.keys))]
            else:
                assert len(size) == len(self.keys)
                self.size = size
        elif isinstance(size, list) or size is None:
            self.size = [size for _ in range(len(self.keys))] 
        else:
            raise TypeError(
                f'"size" must be type of list with list or None inside, or '
                f'type of list or None.')

        if is_list_of(pad_val, list):
            if len(pad_val) < len(self.keys) and len(pad_val) == 1:
                self.pad_val = [
                    pad_val if len(pad_val) == 1 else tuple(pad_val)
                    for _ in range(len(self.keys))]
            else:
                assert len(pad_val) == len(self.keys)
                self.pad_val = [
                    item[0] if len(item) == 1 else tuple(item) 
                    for item in pad_val]
        elif is_list_of(pad_val, int):
            self.pad_val = [
                pad_val if len(pad_val) == 1 else tuple(pad_val)
                for _ in range(len(self.keys))]
        elif isinstance(pad_val, int):
            self.pad_val = [pad_val for _ in range(len(self.keys))]
        else:
            raise TypeError(
            f'"pad_val" must be type of list of list, or list of int or int')

    def pad(self, results, key, size, pad_val):
        assert results.get(key + '_cfg') is not None and \
                results[key + '_cfg'].get('shape') is not None
        obj = copy.deepcopy(results[key])
        if size is None:
            shape = results[key + '_cfg']['shape']
        else:
            assert len(size) == 2 and size[0] > 0 and size[1] > 0
            shape = (size[0], size[1])
        if isinstance(obj, list):
            results[key] = [
                impad(item, shape=shape, pad_val=pad_val) for item in obj]
        else:
            results[key] = impad(obj, shape=shape, pad_val=pad_val)
        results[key + '_cfg']['shape'] = shape

    def __call__(self, results):
        for key, size, pad_val in zip(self.keys, self.size, self.pad_val):
            self.pad(results, key, size, pad_val)
        
        return results