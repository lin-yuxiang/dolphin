import torch
import numpy as np
import copy
from collections.abc import Sequence

from dolphin.utils import Registers, is_list_of
from dolphin.dataset.parallel import DataContainer as DC
# from ..parallel import DataContainer as DC


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, dict):
        return {name: to_tensor(value) for name, value in data.items()}
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@Registers.pipeline.register
class ToTensor(object):
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


@Registers.pipeline.register
class ToDataContainer(object):

    def __init__(self, 
                 key='label',
                 stack=False, 
                 padding_value=0, 
                 cpu_only=False, 
                 pad_dims=2):
        self.key = key
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims
    
    def __call__(self, results):
        assert results.get(self.key) is not None
        results[self.key] = DC(
            results[self.key], stack=self.stack, pad_dims=self.pad_dims,
            padding_value=self.padding_value, cpu_only=self.cpu_only)
        return results


@Registers.pipeline.register
class ConvertBoxFormat(object):

    def __init__(self,
                 key='gt_bbox',
                 to_format=None,
                 begin_idx=0):
        self.to_format = to_format
        self.key = key
        self.begin_idx = begin_idx
        assert to_format in ['xyxy', 'xywh']
    
    def _convert(self, box):
        idx = self.begin_idx
        new_box = copy.deepcopy(box)
        if self.to_format == 'xyxy':
            x = box[..., idx::4]
            y = box[..., (idx + 1)::4]
            w = box[..., (idx + 2)::4]
            h = box[..., (idx + 3)::4]
            if isinstance(w, float) or isinstance(h, float):
                dw = w / 2
                dh = h / 2
            elif isinstance(w, int) and isinstance(h, int):
                dw = w // 2
                dh = h // 2
            new_box[..., idx::4] = x - dw
            new_box[..., (idx + 1)::4] = y - dh
            new_box[..., (idx + 2)::4] = x + dw
            new_box[..., (idx + 1)::4] = y + dh
        else:
            x1 = box[..., idx::4]
            y1 = box[..., (idx + 1)::4]
            x2 = box[..., (idx + 2)::4]
            y2 = box[..., (idx + 3)::4]
            if isinstance((x2 - x1), float) or isinstance((y2 - y1), float):
                dw = (x2 - x1) / 2
                dh = (y2 - y1) / 2
            elif isinstance((x2 - x1), int) and isinstance((y2 - y1), int):
                dw = (x2 - x1) // 2
                dh = (y2 - y1) // 2
            new_box[..., idx::4] = x1 + dw
            new_box[..., (idx + 1)::4] = y1 + dh
            new_box[..., (idx + 2)::4] = x2 - x1
            new_box[..., (idx + 3)::4] = y2 - y1
        return new_box

    def __call__(self, results):
        assert results.get(self.key) is not None and self.to_format is not None
        obj = results[self.key]
        if isinstance(obj, dict):
            for n, v in obj.items():
                if isinstance(v, list):
                    results[self.key][n] = [self._convert(item) for item in v]
                else:
                    results[self.key][n] = self._convert(v)
        elif isinstance(obj, list):
            results[self.key] = [self._convert(item) for item in obj]
        else:
            results[self.key] = self._convert(obj)
        if results.get(self.key + '_cfg') is not None:
            results[self.key + '_cfg']['format'] = self.to_format
        return results


@Registers.pipeline.register
class Transpose(object):

    def __init__(self, keys, order):
        if not isinstance(keys, list):
            raise TypeError(
                f'parameter "keys" should be type of list, '
                f'but got {type(keys)}.')
        else:
            self.keys = keys

        if is_list_of(order, list):
            self.order = order
        elif is_list_of(order, int):
            self.order = [order for _ in range(len(keys))]
        else:
            raise TypeError(
                f'parameter "order" should be type of list, '
                f'but got {type(order)}.')

    def __call__(self, results):
        assert len(self.keys) == len(self.order)
        for key, order in zip(self.keys, self.order):
            if isinstance(results[key], list):
                data_list = results[key]
                results[key] = [i.transpose(order) for i in data_list]
            else:
                results[key] = results[key].transpose(order)
        return results


@Registers.pipeline.register
class FormatShape:
    """Format final imgs shape to the given input_format.
    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".
    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format):
        self.input_format = input_format
        if self.input_format not in ['NCTHW', 'NCHW', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x L
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results


@Registers.pipeline.register
class Zip(object):

    def __init__(self,
                 out_keys='imgs',
                 zip_keys=[]):
        self.out_keys = out_keys
        self.zip_keys = zip_keys
    
    def __call__(self, results):
        # assert results.get(self.out_keys) is None
        assert len(self.zip_keys) > 0 and is_list_of(self.zip_keys, str)
        data = {}
        for key in self.zip_keys:
            assert results.get(key) is not None
            data[key] = results[key]
        results[self.out_keys] = data
        return results


@Registers.pipeline.register
class Unzip(object):

    def __init__(self, keys=['imgs']):
        self.keys = keys
    
    def __call__(self, results):
        for key in self.keys:
            assert results.get(key, None) is not None, \
                f'key {key} does not exist in results dict.'
            objs = results[key]
            assert isinstance(objs, dict), \
                f'type of {key} is not dict, unnecessary to unzip.'
            for k, v in objs.items():
                results[k] = v
        return results


@Registers.pipeline.register
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_meta', the results will be a dict with
    keys 'imgs' and 'img_meta', where 'img_meta' is a DataContainer of another
    dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_meta".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file

            - "label": label of the image file

            - "original_shape": original shape of the image as a tuple
            (h, w, c)

            - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the
            bottom/right, if the batch tensor is larger than this shape.

            - "pad_shape": image shape after padding

            - "flip_direction": a str in ("horiziontal", "vertival") to
            indicate if the image is fliped horizontally or vertically.

            - "img_norm_cfg": a dict of normalization information:

                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=[],
                 meta_name=[]):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0 and \
            len(self.meta_keys) == len(self.meta_name):
            for key, meta_name in zip(self.meta_keys, self.meta_name):
                data[meta_name] = DC(results[key], cpu_only=True)
        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')