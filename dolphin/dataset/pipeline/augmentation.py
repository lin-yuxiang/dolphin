import cv2
import numpy as np
import random
import math
import copy
import scipy.ndimage as ndimage

from dolphin.utils import Registers, is_list_of, is_list_of_list_or_nontype
from ..utils import (rescale_size, imresize, random_affine, imnormalize, 
                     imrescale, impad, xyxy2xywh, gaussian_radius, 
                     draw_msra_gaussian)


@Registers.pipeline.register
class Resize(object):

    def __init__(self,
                 key='imgs',
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 interpolation='bilinear',
                 keep_ratio=True,
                 rescale_edge='max',
                 with_box=False,
                 output_stride=None,
                 box_key='gt_bbox',
                 begin_idx=0):
        if img_scale is None:
            self.img_scale = None
        else:
            if is_list_of(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, list)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.rescale_edge = rescale_edge
        self.interpolation = interpolation
        self.key = key
        self.with_box = with_box
        self.output_stride = output_stride
        self.box_key = box_key
        self.begin_idx = begin_idx

    @staticmethod
    def random_select(img_scales):

        assert is_list_of(img_scales, list)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):

        assert is_list_of(img_scales, list) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):

        assert isinstance(img_scale, list) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results[self.key + '_cfg']['scale'] = scale
        results[self.key + '_cfg']['scale_idx'] = scale_idx

    def _resize(self, results):
        if results.get(self.key + '_cfg', None) is not None:
            cfg = results[self.key + '_cfg']
            h, w = cfg['shape']
        else:
            h, w = results[self.key].shape[:2]
        if self.keep_ratio:
            results[self.key], scale_factor = imrescale(
                results[self.key], cfg['scale'], 
                interpolation=self.interpolation, return_scale=True, 
                rescale_edge=self.rescale_edge)
            new_h, new_w = results[self.key].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            results[self.key], w_scale, h_scale = imresize(
                results[self.key], cfg['scale'], 
                interpolation=self.interpolation, return_scale=True)
            new_h, new_w = results[self.key].shape[:2]
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        cfg['shape'] = (new_h, new_w)
        cfg['scale_factor'] = scale_factor
        cfg['keep_ratio'] = self.keep_ratio
        results[self.key + '_cfg'] = cfg

    def _resize_box(self, box, scale_factor, results):
        idx = self.begin_idx
        h, w = results['imgs_cfg']['shape']
        if self.output_stride is not None:
            scale_factor = scale_factor / self.output_stride
        new_box = copy.deepcopy(box)
        new_box[:, idx: idx + 4] = box[:, idx: idx + 4] * scale_factor
        new_box[:, idx] = np.clip(new_box[:, idx], 0, w)
        new_box[:, idx + 1] = np.clip(new_box[:, idx + 1], 0, h)
        new_box[:, idx + 2] = np.clip(new_box[:, idx + 2], 0, w)
        new_box[:, idx + 3] = np.clip(new_box[:, idx + 3], 0, h)
        return new_box

    def resize_box(self, results):
        assert results.get(self.box_key, None) is not None
        assert results.get(self.box_key + '_cfg') is not None
        assert results[self.box_key + '_cfg'].get('is_box_resize', False)
        if not results[self.box_key + '_cfg'].get('norm_box', False):
            factor = results['imgs_cfg']['scale_factor']
            obj = copy.deepcopy(results[self.box_key])
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, list):
                        results[self.box_key][k] = [
                            self._resize_box(item, factor, results) 
                            for item in v]
            elif isinstance(obj, list):
                results[self.box_key] = [
                    self._resize_box(item, factor, results) for item in obj]
            else:
                results[self.box_key] = self._resize_box(obj, factor, results)
            results[self.box_key]['is_box_resize'] = True
        else:
            raise RuntimeError(
            f'{self.box_key} do not need to be resize because its sizes '
            f'are normed.')

    def __call__(self, results):
        assert results.get(self.key, None) is not None
        assert results.get(self.key + '_cfg', None) is not None
        if 'scale' not in results[self.key + '_cfg']:
            self._random_scale(results)
        self._resize(results)
        if self.with_box:
            self.resize_box(results)
        return results


@Registers.pipeline.register
class LetterBoxResize(object):

    def __init__(self,
                 keys=['imgs'],
                 img_scale=[],
                 border_color=[127.5, 127.5, 127.5],
                 interpolation='area',
                 with_box=False,
                 box_key=None):
        self.keys = keys
        if len(keys) < 1:
            raise ValueError(f'please assign keys that you want to apply to.')
        elif len(keys) == 1:
            if is_list_of(img_scale, int):
                self.img_scale = [img_scale]
            elif is_list_of(img_scale, list) and len(img_scale) == 1:
                self.img_scale = img_scale
            else:
                raise TypeError(
                    f'paramater "img_scale" must be type of "list of list" or '
                    f'"list of int" when there is only one key.')
            if isinstance(interpolation, str):
                self.interpolation = [interpolation]
            elif is_list_of(interpolation, str) and \
                len(interpolation) == len(keys):
                self.interpolation = interpolation
            else:
                raise TypeError(
                    f'parameter "interpolation" must be type of "list of str" '
                    f'or "str" when there is only one key.')
        else:
            if is_list_of(img_scale, list) and len(img_scale) == len(keys):
                self.img_scale = img_scale
            else:
                raise TypeError(
                    f'parameter "img_scale" must be type of "list of list" '
                    f'with same length as "keys" when there are more '
                    f'than one keys.')
            if is_list_of(interpolation, str) and \
                len(interpolation) == len(keys):
                self.interpolation = interpolation
            else:
                raise TypeError(
                    f'parameter "interpolation" must be type of "list of str" '
                    f'with some length as "keys" when there are more '
                    f'than one keys.')
        self.border_color = border_color
        self.with_box = with_box
        self.box_key = box_key
    
    def _resize(self, results, key, img_scale, interpolation):
        assert results.get(key) is not None
        assert results.get(key + '_cfg') is not None
        assert results[key + '_cfg']['shape'] is not None
        obj = copy.deepcopy(results[key])
        h, w = results[key + '_cfg']['shape']
        h_scale, w_scale = img_scale
        ratio = min(float(h_scale) / h, float(w_scale) / w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        dw, dh = (w_scale - new_w) / 2, (h_scale - new_h) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        if isinstance(obj, list):
            results[key] = [
                imresize(item, (new_w, new_h), interpolation=interpolation)
                for item in obj]
            results[key] = [
                impad(item, padding=(left, top, right, bottom), 
                pad_val=self.border_color) for item in results[key]]
        else:
            results[key] = imresize(
                obj, (new_w, new_h), interpolation=interpolation)
            results[key] = impad(
                results[key], padding=(left, top, right, bottom), 
                pad_val=self.border_color)
        results[key + '_cfg']['shape'] = (h_scale, w_scale)
        results[key + '_cfg']['border'] = (dh, dw)
        results[key + '_cfg']['scale_factor'] = ratio
    
    def _resize_box(self, results):
        ratio = results['imgs_cfg']['scale_factor']
        dh, dw = results['imgs_cfg']['border']
        assert results.get(self.box_key) is not None
        label = copy.deepcopy(results[self.box_key])
        label[:, 2] = ratio * label[:, 2] + dw
        label[:, 3] = ratio * label[:, 3] + dh
        label[:, 4] = ratio * label[:, 4] + dw
        label[:, 5] = ratio * label[:, 5] + dh
        results[self.box_key] = label
    
    def __call__(self, results):
        for key, img_scale, interpolation in zip(
            self.keys, self.img_scale, self.interpolation):
            self._resize(results, key, img_scale, interpolation)
        if self.with_box and self.box_key is not None:
            self._resize_box(results)
            
        return results


@Registers.pipeline.register
class RandomFlip(object):

    def __init__(self, 
                 keys=['imgs'], 
                 flip_ratio=0.5, 
                 direction='horizontal', 
                 with_box=False,
                 box_key=None,
                 restore_size=False,
                 begin_idx=0):
        self.keys = keys
        if len(keys) < 1:
            raise ValueError(f'please assign keys that you want to apply to.')
        elif len(keys) == 1:
            if isinstance(direction, str):
                self.direction = [direction]
            elif is_list_of(direction, str) and len(direction) == 1:
                self.direction = direction
            else:
                raise TypeError(
                    f'parameter "direction" must be type of "list of str" or '
                    f'"str" when there is only one key, please check for it.')
        else:
            if is_list_of(direction, str) and len(direction) == len(keys):
                self.direction = direction
            else:
                raise TypeError(
                    f'parameter "direction" must be type of "list of str" and '
                    f'have the same length as "keys".')

        for item in self.direction:
            assert item in ['horizontal', 'vertical']
        self.with_box = with_box
        self.box_key = box_key
        self.begin_idx = begin_idx
        self.restore_size = restore_size

        if np.random.rand() < flip_ratio:
            self.flip = True
        else:
            self.flip = False

    def flip_bbox(self, bbox, direction, results):
        idx = self.begin_idx
        h, w = results['imgs_cfg']['shape']
        if results.get(self.box_key + '_cfg') is not None and \
            results[self.box_key + '_cfg'].get('norm_box', False):
            if direction == 'horizontal':
                bbox[..., idx::4] = 1 - bbox[..., idx::4]
            else:
                bbox[..., (idx + 1)::4] = 1 - bbox[..., (idx + 1)::4]
        else:
            if direction == 'horizontal':
                bbox[..., idx::4] = w - bbox[..., idx::4]
            else:
                bbox[..., (idx + 1)::4] = h - bbox[..., (idx + 1)::4]
                # only bbox format as xywh
        if self.restore_size:
            bbox[:, idx] *= w
            bbox[:, idx + 1] *= h
            bbox[:, idx + 2] *= w
            bbox[:, idx + 3] *= h
        return bbox
    
    def flip_img(self, img, direction):
        if direction == 'horizontal':
            img = np.flip(img, axis=1)
        else:
            img = np.flip(img, axis=0)
        return img

    def _flip(self, results, key, direction):
        assert results.get(key) is not None
        assert results.get(key + '_cfg') is not None
        obj = copy.deepcopy(results[key])
        if self.flip:
            if isinstance(obj, list):
                results[key] = [self.flip_img(item, direction) for item in obj]
            else:
                results[key] = self.flip_img(obj, direction)
            results[key + '_cfg']['flip_direction'] = direction
            if self.with_box and not results.get('is_box_flip', False) and \
                self.box_key:
                assert results.get(self.box_key) is not None
                gt_bbox = results[self.box_key]
                if isinstance(gt_bbox, dict):
                    for name, box in gt_bbox.items():
                        if isinstance(box, list):
                            results[self.box_key][name] = [
                                self.flip_bbox(item, direction, results) 
                                for item in box]
                        else:
                            results[self.box_key][name] = self.flip_bbox(
                                box, direction, results)
                elif isinstance(gt_bbox, list):
                    results[self.box_key] = [
                        self.flip_bbox(item, direction, results) 
                        for item in gt_bbox]
                else:
                    results[self.box_key] = self.flip_bbox(
                        gt_bbox, direction, results)
                results['is_box_flip'] = True
                results[
                    self.box_key + '_cfg']['norm_box'] = not self.restore_size
        results[key + '_cfg']['flip'] = self.flip

    def __call__(self, results):
        for key, direction in zip(self.keys, self.direction):
            self._flip(results, key, direction)
        return results


@Registers.pipeline.register
class RandomCrop(object):

    def __init__(self, 
                 crop_size, 
                 keys=['imgs'], 
                 cat_max_ratio=1.0):
        self.keys = keys
        if len(keys) < 1:
            raise ValueError(f'please assign keys that you want to apply to.')
        elif len(keys) == 1:
            if is_list_of(crop_size, int):
                self.crop_size = [crop_size]
            elif is_list_of(crop_size, list) and len(crop_size) == 1:
                self.crop_size = crop_size
            else:
                raise TypeError(
            f'parameter "crop_size" must be type of "list of int" or '
            f'"list of one list" when there is only one key, please check it.')
        else:
            if is_list_of(crop_size, list) and len(crop_size) == len(keys):
                self.crop_size = crop_size
            elif is_list_of(crop_size, int) and len(crop_size) == 1:
                self.crop_size = [crop_size for _ in range(len(keys))]
            else:
                raise TypeError(
                f'parameter "crop_size" must be type of "list of list" and '
                f'have the same length as "keys".')

        self.cat_max_ratio = cat_max_ratio

    def crop_shape(self, h, w, results, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        margin_h = max(h - crop_size[0], 0)
        margin_w = max(w - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        crop_box = (crop_y1, crop_y2, crop_x1, crop_x2)
        offset = (offset_h, offset_w)
        return crop_box, offset
    
    def get_crop_box(self, h, w, results, crop_size):
        crop_box, offset = self.crop_shape(h, w, results, crop_size)
        if self.cat_max_ratio < 1.0:
            for _ in range(10):
                seg_tmp = self.crop(results['label'], crop_box)
                labels, cnt = np.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                    cnt) < self.cat_max_ratio:
                    break
                crop_box = self.crop_shape(h, w, results, crop_size)
        return crop_box, offset

    def crop(self, img, crop_box):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_box
        img = img[crop_y1: crop_y2, crop_x1: crop_x2, ...]
        return img

    def _crop(self, results, key, crop_size):
        obj = copy.deepcopy(results[key])
        if isinstance(obj, list):
            # crop_box, offset = self.get_crop_box(h, w, results, crop_size)
            results[key] = [self.crop(item, self.crop_box) for item in obj]
        else:
            # crop_box, offset = self.get_crop_box(h, w, results, crop_size)
            results[key] = self.crop(obj, self.crop_box)
        results[key + '_cfg']['shape'] = crop_size
        results[key + '_cfg']['crop_offset'] = self.offset

    def __call__(self, results):
        sizes = []
        for key, crop_size in zip(self.keys, self.crop_size):
            assert results.get(key) is not None
            assert results.get(key + '_cfg') is not None
            assert results[key + '_cfg'].get('shape') is not None
            sizes.append(results[key + '_cfg']['shape'])
            if len(set(sizes)) == 1:
                h, w = results[key + '_cfg']['shape']
                if not hasattr(self, 'crop_box'):
                    self.crop_box, self.offset = self.get_crop_box(
                        h, w, results, crop_size)
                self._crop(results, key, crop_size)
        return results


@Registers.pipeline.register
class Pad(object):

    def __init__(self,
                 keys=['imgs'],
                 size=None,
                 pad_val=0):

        self.keys = keys
        if len(keys) < 1:
            raise ValueError(f'please assign keys that you want to apply to.')
        elif len(keys) == 1:
            if size is None or is_list_of(size, int):
                self.size = [size]
            elif is_list_of_list_or_nontype(size) and len(size) == 1:
                self.size = size
            else:
                raise TypeError(
                f'parameter "size" must be type of "list of one int", "None", '
                f'"list of one list" or "list of None" when '
                f'there is only one key, please check for it.')

            if isinstance(pad_val, int):
                self.pad_val = [pad_val]
            elif is_list_of(pad_val, int) and len(pad_val) != 1:
                self.pad_val = [pad_val]
            elif is_list_of(pad_val, list) and len(pad_val) == 1:
                self.pad_val = pad_val
            elif is_list_of(pad_val, int) and len(pad_val) == 1:
                self.pad_val = pad_val
            else:
                raise TypeError(
                f'parameter "pad_val" must be type of "list of one int", '
                f'"list of list", "list of one list" or "int" when '
                f'there is only one key, please check for it.')
        else:
            if is_list_of(size, list) and len(size) == len(keys):
                self.size = size
            else:
                raise TypeError(
                f'parameter "size" must be type of "list of list" and '
                f'have the same length as "keys".')
            if is_list_of(pad_val, list) and len(pad_val) == len(keys):
                self.pad_val = [
                    item[0] if len(item) == 1 else item for item in pad_val] 
            else:
                raise TypeError(
                f'parameter "pad_val" must be type of "list of list" and '
                f'have the same length as "keys".')

    def _pad(self, results, key, size, pad_val):
        assert results.get(key) is not None
        assert results.get(key + '_cfg') is not None
        assert results[key + '_cfg'].get('shape') is not None
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
            self._pad(results, key, size, pad_val)
        return results


@Registers.pipeline.register
class Normalize(object):

    def __init__(self, mean, std, keys=['imgs'], to_rgb=True):
        self.keys = keys
        if len(keys) < 1:
            raise ValueError(f'please assign keys that you want to apply to.')
        elif len(keys) == 1:
            if isinstance(mean, (float, int)) or is_list_of(mean, (float, int)):
                self.mean = [mean]
            elif is_list_of(mean, list) and len(mean) == 1:
                self.mean = mean
            else:
                raise TypeError(
                f'parameter "mean" must be type of "list of int(float)", '
                f'"float", "int" or "list of one list" when '
                f'there is only one key, please check for it.')
            if isinstance(std, (float, int)) or is_list_of(std, (float, int)):
                self.std = [std]
            elif is_list_of(std, list) and len(std) == 1:
                self.std = std
            else:
                raise TypeError(
                f'parameter "std" must be type of "list of int(float)", '
                f'"float", "int" or "list of one list" when '
                f'there is only one key, please check for it.')
        else:
            if is_list_of(mean, list) and len(mean) == len(keys):
                self.mean = mean
            else:
                raise TypeError(
                f'parameter "mean" must be type of "list of list" and '
                f'have the same length as "keys".')
            if is_list_of(std, list) and len(std) == len(keys):
                self.std = std
            else:
                raise TypeError(
                f'parameter "std" must be type of "list of list" and '
                f'have the same length as "keys".')

        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
        self.to_rgb = to_rgb

    def _normalize(self, results, key, mean, std):
        assert results.get(key) is not None
        assert results.get(key + '_cfg') is not None
        obj = copy.deepcopy(results[key])
        if isinstance(obj, list):
            results[key] = [
                imnormalize(item, mean, std, self.to_rgb) for item in obj]
        else:
            results[key] = imnormalize(obj, mean, std, self.to_rgb)
        results[key + '_cfg']['norm_cfg'] = dict(
            mean=mean, std=std, to_rgb=self.to_rgb)

    def __call__(self, results):

        for key, mean, std in zip(self.keys, self.mean, self.std):
            self._normalize(results, key, mean, std)
        return results


@Registers.pipeline.register
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 with_saturation=False,
                 value_range=(0.5, 1.5),
                 with_value=False,
                 hue_delta=18):

        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.value_lower, self.value_upper = value_range
        self.hue_delta = hue_delta
        self.with_saturation = with_saturation
        self.with_value = with_value

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(
                    -self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=np.random.uniform(
                    self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2) or self.with_saturation:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(
                    self.saturation_lower, self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img
    
    def value(self, img):
        """Value distortion."""
        if np.random.randint(2) or self.with_value:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 2] = self.convert(
                img[:, :, 2],
                alpha=random.uniform(self.value_lower, self.value_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, results):

        # 要转换为hsvaument的时候，brightness_delta 为 0. 
        # contrast_range 为 (1.0, 1.0).
        # hue_delta 为 0.
        # 只有 saturation 和 value 两个有值. and with_saturation, with_value

        # 如果用这个的原型的话，只需要把 value_range 设为 1.0，并且 with_saturation
        # with_value 都设为 False.

        img = results['imgs']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        img = self.value(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['imgs'] = img
        return results


@Registers.pipeline.register
class RandomAffine(object):

    def __init__(self,
                 keys=['imgs'],
                 degrees=[-10, 10],
                 translate=[0.1, 0.1],
                 scale=[0.9, 1.1],
                 shear=[-2, 2],
                 border_value=[127.5, 127.5, 127.5],
                 with_box=False,
                 box_key=None,
                 to_xywh=False):
        self.keys = keys
        if len(keys) < 1:
            raise TypeError(f'please assign keys that you want to apply to.')
        elif len(keys) == 1:
            if is_list_of(degrees, int) or is_list_of(degrees, float):
                self.degrees = [degrees]
            elif is_list_of(degrees, list) and len(degrees) == len(keys):
                self.degrees = degrees
            else:
                raise TypeError(
                f'parameter "degrees" must be type of "list of int(float)", '
                f'"list of one list" when there is only one key, '
                f'please check for it.')
            if is_list_of(translate, int) or is_list_of(translate, float):
                self.translate = [translate]
            elif is_list_of(translate, list) and len(translate) == len(keys):
                self.translate = translate 
            else:
                raise TypeError(
                f'parameter "translate" must be type of "list of int(float)", '
                f'"list of one list" when there is only one key, '
                f'please check for it.')
            if is_list_of(scale, int) or is_list_of(scale, float):
                self.scale = [scale]
            elif is_list_of(scale, list) and len(scale) == len(keys):
                self.scale = scale 
            else:
                raise TypeError(
                f'parameter "scale" must be type of "list of int(float)", '
                f'"list of one list" when there is only one key, '
                f'please check for it.')
            if is_list_of(shear, int) or is_list_of(shear, float):
                self.shear = [shear]
            elif is_list_of(shear, list) and len(shear) == len(keys):
                self.shear = shear
            else:
                raise TypeError(
                f'parameter "shear" must be type of "list of int(float)", '
                f'"list of one list" when there is only one key, '
                f'please check for it.')
        else:
            if is_list_of(degrees, list) and len(degrees) == len(keys):
                self.degress = degrees
            else:
                raise TypeError(
                f'parameter "degrees" must be type of "list of list" and '
                f'have the same length as "keys".')
            if is_list_of(scale, list) and len(scale) == len(scale):
                self.scale = scale 
            else:
                raise TypeError(
                f'parameter "scale" must be type of "list of list" and '
                f'have the same length as "keys".')
            if is_list_of(translate, list) and len(translate) == len(keys):
                self.translate = translate
            else:
                raise TypeError(
                f'parameter "translate" must be type of "list of list" and '
                f'have the same length as "keys".')
            if is_list_of(shear, list) and len(shear) == len(shear):
                self.shear = shear
            else:
                raise TypeError(
                f'parameter "shear must be type of "list of list" and '
                f'have the same length as "keys".')
        
        self.border_value = border_value
        self.with_box = with_box
        self.box_key = box_key
        self.to_xywh = to_xywh

    def _affine(self, results, key, degrees, translate, scale, shear):
        assert results.get(key) is not None
        assert results.get(key + '_cfg') is not None
        assert results[key + '_cfg'].get('shape') is not None
        assert len(degrees) == 2 and len(scale) == 2 and len(translate) == 2 \
            and len(shear) == 2
        obj = copy.deepcopy(results[key])
        h, w = results[key + '_cfg']['shape']
        border = 0
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(w / 2, h / 2), scale=s)

        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * h + border
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * w + border

        S = np.eye(3)
        S[0, 1] = math.tan((
            random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
        S[1, 0] = math.tan((
            random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
        M = S @ T @ R
        results[key] = cv2.warpPerspective(
            obj, M, dsize=(w, h), flags=cv2.INTER_LINEAR, 
            borderValue=self.border_value)
        results['imgs_cfg']['angle'] = a
        results['imgs_cfg']['R'] = R
        results['imgs_cfg']['T'] = T
        results['imgs_cfg']['S'] = S
        results['imgs_cfg']['M'] = M
    
    def _affine_box(self, results):
        img_h, img_w = results['imgs_cfg']['shape']
        a = results['imgs_cfg']['angle']
        M = results['imgs_cfg']['M']
        assert results.get(self.box_key) is not None
        target = results[self.box_key]
        n = target.shape[0]
        points = target[:, 2: 6].copy()
        area_box = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate(
            (x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        
        np.clip(xy[:, 0], 0, img_w, out=xy[:, 0])
        np.clip(xy[:, 2], 0, img_w, out=xy[:, 2])
        np.clip(xy[:, 1], 0, img_h, out=xy[:, 1])
        np.clip(xy[:, 3], 0, img_h, out=xy[:, 3])
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area_box + 1e-16) > 0.1) & (ar < 10)

        target = target[i]
        target[:, 2:6] = xy[i]
        if self.to_xywh:
            target[:, 2:6] = xyxy2xywh(target[:, 2:6].copy())
            target[:, 2] /= img_w
            target[:, 3] /= img_h
            target[:, 4] /= img_w
            target[:, 5] /= img_h
            if isinstance(results[self.box_key + '_cfg'], dict):
                results[self.box_key + '_cfg']['norm_box'] = True
            else:
                results[self.box_key + '_cfg'] = dict()
                results[self.box_key + '_cfg']['norm_box'] = True
        results[self.box_key] = target

    def __call__(self, results):
        for key, degrees, translate, scale, shear in zip(
            self.keys, self.degrees, self.translate, self.scale, self.shear):
            self._affine(results, key, degrees, translate, scale, shear)
        if self.with_box and self.box_key is not None:
            self._affine_box(results)

        return results


@Registers.pipeline.register
class GenerateLabelMap(object):

    def __init__(self,
                 down_ratio=4,
                 num_classes=1,
                 max_objs=128):
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.num_classes = num_classes

    def __call__(self, results):
        label = results['label']
        new_h, new_w = results['imgs_cfg']['shape']
        out_h, out_w = new_h // self.down_ratio, new_w // self.down_ratio
        num_objs = label.shape[0] 
        hm = np.zeros((self.num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)

        for k in range(num_objs):
            l = label[k]
            bbox = l[2:]
            cls_id = int(l[0])
            bbox[[0, 2]] = bbox[[0, 2]] * out_w
            bbox[[1, 3]] = bbox[[1, 3]] * out_h
            bbox[0] = np.clip(bbox[0], 0, out_w - 1)
            bbox[1] = np.clip(bbox[1], 0, out_h - 1)
            h = bbox[3]
            w = bbox[2]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_msra_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * out_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = l[1]
        
        results['hm'] = hm
        results['reg_mask'] = reg_mask
        results['ind'] = ind
        results['wh'] = wh
        results['reg'] = reg
        results['ids'] = ids
        
        return results


@Registers.pipeline.register
class RandomExpand(object):

    def __init__(self, 
                 expand_prob=0.5,
                 max_expand_ratio=4.0,
                 expand_value=[104.0136177, 114.0342201, 119.91659325]):
        self.expand_prob = expand_prob
        self.max_expand_ratio = max_expand_ratio
        self.expand_value = expand_value

    def __call__(self, results):
        imgs = results['imgs']
        gt_bbox = results['gt_bbox']
        oh, ow = results['imgs_cfg']['original_shape']
        out_imgs = imgs
        out_bbox = gt_bbox

        if random.random() < self.expand_prob:
            expand_ratio = random.uniform(1, self.max_expand_ratio)
            h = int(oh * expand_ratio)
            w = int(ow * expand_ratio)
            out_imgs = [
                np.zeros((h, w, 3), dtype=np.float32) for i in range(len(imgs))] 
            h_off = int(np.floor(h - oh))
            w_off = int(np.floor(w - ow))
            if self.mean_values is not None:
                for i in range(len(imgs)):
                    out_imgs[i] += np.array(self.mean_values).reshape(1, 1, 3)
            for i in range(len(imgs)):
                out_imgs[i][h_off: h_off + oh, w_off: w_off + ow, :] = imgs[i]
            for ilabel in gt_bbox:
                for box in range(len(gt_bbox[ilabel])):
                    out_bbox[ilabel][box] += np.array(
                        [[w_off, h_off, w_off, h_off]], dtype=np.float32)

        results['imgs'] = out_imgs
        results['gt_bbox'] = out_bbox
        return results