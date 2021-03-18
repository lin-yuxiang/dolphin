import cv2
import numpy as np
import os.path as osp
import math

from dolphin.utils import Registers
from ..utils import (gaussian_radius, draw_msra_gaussian, imread,
                     imfrombytes, get_img_bytes, draw_umich_gaussian)


@Registers.pipeline.register
class LoadImageFromFile(object):

    def __init__(self,
                 to_float32=False,
                 color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def _load(self, filename):
        img_bytes = get_img_bytes(filename)
        img = imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def __call__(self, results):
        if results.get('filename') is not None:
            filename = results['filename']
        elif results.get('img_path') is not None:
            filename = results['img_path']
        elif results.get('img_file') is not None:
            filename = results['img_file']
        if isinstance(filename, dict):
            for key, value in filename.items():
                results[key] = self._load(value)
                results[key + '_cfg'] = dict()
                results[key + '_cfg']['shape'] = results[key].shape[:2]
                results[key + '_cfg']['original_shape'] = results[key].shape[:2]
        elif isinstance(filename, list):
            results['imgs'] = [self._load(item) for item in filename]
            results['imgs_cfg'] = dict()
            results['imgs_cfg']['shape'] = results['imgs'][0].shape[:2]
            results['imgs_cfg']['original_shape'] = results['imgs'][0].shape[:2]
        else:
            results['imgs'] = self._load(filename)
            results['imgs_cfg'] = dict()
            results['imgs_cfg']['shape'] = results['imgs'].shape[:2]
            results['imgs_cfg']['original_shape'] = results['imgs'].shape[:2]
        return results


@Registers.pipeline.register
class LoadSequentialBoundingBoxFromFile(object):

    def __init__(self, restore_size=False, xywh2xyxy=True, joint_data=False):
        self.restore_size = restore_size
        self.xywh2xyxy = xywh2xyxy
        self.joint_data = joint_data
    
    def _load(self, filename, w, h, results): 
        if osp.isfile(filename): 
            label = np.loadtxt(filename, dtype=np.float32).reshape(-1, 6) 
            res = label.copy()
            if self.xywh2xyxy:
                res[:, 2] = label[:, 2] - label[:, 4] / 2
                res[:, 3] = label[:, 3] - label[:, 5] / 2
                res[:, 4] = label[:, 2] + label[:, 4] / 2
                res[:, 5] = label[:, 3] + label[:, 5] / 2
            if self.restore_size:
                res[:, 2] *= w
                res[:, 3] *= h
                res[:, 4] *= w
                res[:, 5] *= h
        else:
            res = np.array([])
        
        if self.joint_data:
            tid_start_idx = results['tid_start_index']
            for i, _ in enumerate(res):
                if res[i, 1] > -1:
                    res[i, 1] += tid_start_idx[results['data_name']]
        return res
    
    def __call__(self, results):
        if results.get('label_path') is not None:
            filename = results['label_path']
        elif results.get('box_path') is not None:
            filename = results['box_path']
        assert results['imgs_cfg'].get('shape') is not None
        h, w = results['imgs_cfg']['shape']
        if isinstance(filename, list):
            results['label'] = [
                self._load(item, w, h, results) for item in filename]
        else:
            results['label'] = self._load(filename, w, h, results)
        results['label_cfg'] = dict()
        return results


@Registers.pipeline.register
class LoadSegmentationMap(object):

    def __init__(self,
                 reduce_zero_label=False):
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(
                results['seg_prefix'], results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = get_img_bytes(filename)
        gt_semantic_seg = imfrombytes(
            img_bytes, flag='unchanged').squeeze().astype(np.uint8)
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['label'] = gt_semantic_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        return repr_str


@Registers.pipeline.register
class LoadLocalizationFeature(object):
    """Load Video features for localizer with given video_name list.

    Required keys are "video_name" and "data_prefix",
    added or modified keys are "raw_feature".

    Args:
        raw_feature_ext (str): Raw feature file extension.  Default: '.csv'.
    """

    def __init__(self, raw_feature_ext='.npy'):
        valid_raw_feature_ext = ('.csv', '.npy')
        if raw_feature_ext not in valid_raw_feature_ext:
            raise NotImplementedError
        self.raw_feature_ext = raw_feature_ext

    def __call__(self, results):
        """Perform the LoadLocalizationFeature loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        data_prefix = results['data_prefix']

        data_path = osp.join(data_prefix, video_name + self.raw_feature_ext)
        if self.raw_feature_ext == '.csv':
            raw_feature = np.loadtxt(
                data_path, dtype=np.float32, delimiter=',', skiprows=1)
        elif self.raw_feature_ext == '.npy':
            raw_feature = np.load(data_path)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results


@Registers.pipeline.register
class LoadLocalizationLabels(object):
    """Load video label for localizer with given video_name list.

    Required keys are "duration_frame", "duration_second", "feature_frame",
    "annotations", added or modified keys are "gt_bbox".
    """

    def __call__(self, results):
        """Perform the GenerateLocalizationLabels loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@Registers.pipeline.register
class SampleFrames(object):

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 start_index=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.start_index = start_index
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ))

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ))
        return clip_offsets

    def _sample_clips(self, num_frames):
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        if 'total_frames' not in results:
            video_reader = cv2.VideoCapture(results['filename'])
            total_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
            results['total_frames'] = total_frames
        else:
            total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')
        frame_inds = np.concatenate(frame_inds) + self.start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    
@Registers.pipeline.register
class SequentialSampleFrames(object):

    def __call__(self, results):
        assert results.get('frame_idx') is not None
        frame_idx = results['frame_idx']
        assert results.get('clip_len') is not None
        clip_len = results['clip_len']
        assert results.get('num_frames') is not None
        num_frames = results['num_frames']
        frame_inds = [min(frame_idx + i, num_frames) for i in range(clip_len)]
        results['frame_inds'] = frame_inds
        return results


@Registers.pipeline.register
class FrameLoader(object):

    def __call__(self, results):
        """Perform the FrameSelector selecting given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if results.get('frame_dir') is not None: 
            directory = results['frame_dir']
        elif results.get('data_prefix') is not None:
            directory = results['data_prefix']
        filename_tmpl = results['filename_tmpl']
        if results.get('modality') is not None:
            modality = results['modality']
        else:
            modality = 'RGB'

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_idx in results['frame_inds']:
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                with open(filepath, 'rb') as f:
                    img_bytes = f.read()
                # Get frame with channel order RGB directly.

                cur_frame = imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(
                    directory, filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(
                    directory, filename_tmpl.format('y', frame_idx))
                with open(x_filepath, 'rb') as f:
                    x_img_bytes = f.read()
                x_frame = imfrombytes(x_img_bytes, flag='grayscale')
                with open(y_filepath, 'rb') as f:
                    y_img_bytes = f.read()
                y_frame = imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['imgs_cfg'] = dict()
        results['imgs_cfg']['original_shape'] = imgs[0].shape[:2]
        results['imgs_cfg']['shape'] = imgs[0].shape[:2]

        return results


@Registers.pipeline.register
class LoadTubeletLabel(object):

    def __init__(self, norm_box=False):
        self.norm_box = norm_box

    def __call__(self, results):

        video_name = results['video_name']
        gttubes = results['gttubes']
        frame_idx = results['frame_idx']
        clip_len = results['clip_len']

        if results['imgs_cfg'].get('shape') is None and self.norm_box:
            self.norm_box = False

        gt_bbox = {}
        for ilabel, tubes in gttubes[video_name].items():
            for t in tubes:
                if frame_idx not in t[:, 0]:
                    continue
                assert frame_idx + clip_len - 1 in t[:, 0]
                t = t.copy()
                boxes = t[
                    (t[:, 0] >= frame_idx) * (t[:, 0] < frame_idx + clip_len), 
                    1: 5]
                assert boxes.shape[0] == clip_len
                if ilabel not in gt_bbox:
                    gt_bbox[ilabel] = []
                if self.norm_box:
                    h, w = results['imgs_cfg']['shape']
                    boxes[:, 1] /= w
                    boxes[:, 2] /= h
                    boxes[:, 3] /= w
                    boxes[:, 4] /= h
                gt_bbox[ilabel].append(boxes)
        results['gt_bbox_cfg'] = dict()
        results['gt_bbox_cfg']['norm_box'] = self.norm_box
        results['gt_bbox'] = gt_bbox
        
        return results


@Registers.pipeline.register
class GenerateMOCHeatmap(object):

    def __init__(self, max_objects=128):
        self.max_objs = max_objects

    def __call__(self, results):
        gt_bbox = results['gt_bbox']
        num_classes = results['num_classes']
        clip_len = results['clip_len']
        new_h, new_w = results['img_shape']
        output_stride = results['output_stride']

        out_h, out_w = new_h // output_stride, new_w // output_stride
        hm = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, clip_len * 2), dtype=np.float32)
        mov = np.zeros((self.max_objs, clip_len * 2), dtype=np.float32)
        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, clip_len * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        for i in gt_bbox:
            for j in range(len(gt_bbox[i])):
                key = clip_len // 2
                key_h, key_w = gt_bbox[i][j][key, 3] - gt_bbox[i][j][key, 1], \
                    gt_bbox[i][j][key, 2] - gt_bbox[i][j][key, 0]
                radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
                radius = max(0, int(radius))

                center = np.array([(gt_bbox[i][j][key, 0] + 
                    gt_bbox[i][j][key, 2]) / 2,
                    (gt_bbox[i][j][key, 1] + gt_bbox[i][j][key, 3]) / 2],
                    dtype=np.float32)
                center_int = center.astype(np.int32)
                assert 0 <= center_int[0] and center_int[0] <= out_w and \
                    0 <= center_int[1] and center_int[1] <= out_h

                draw_umich_gaussian(hm[i], center_int, radius)

                for k in range(clip_len):
                    center_all = np.array([(gt_bbox[i][j][k, 0] + 
                        gt_bbox[i][j][k, 2]) / 2,
                        (gt_bbox[i][j][k, 1] + gt_bbox[i][j][k, 3]) / 2],
                        dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    wh[num_objs, i * 2: i * 2 + 2] = 1. * (gt_bbox[i][j][k, 2] -
                        gt_bbox[i][j][k, 0]), \
                        1. * (gt_bbox[i][j][k, 3] - gt_bbox[i][j][k, 1])
                    mov[num_objs, i * 2: i * 2 + 2] = (gt_bbox[i][j][k, 0] + 
                        gt_bbox[i][j][k, 2]) / 2 - center_int[0], \
                        (gt_bbox[i][j][k, 1] + gt_bbox[i][j][k, 3]) / 2 - \
                            center_int[1]
                    index_all[num_objs, i * 2: i * 2 + 2] = center_all_int[1] \
                        * out_w + center_all_int[0], \
                        center_all_int[1] * out_w + center_all_int[0]

                index[num_objs] = center_int[1] * out_w + center_int[0]
                mask[num_objs] = 1
                num_objs += 1
        
        results['hm'] = hm
        results['mov'] = mov
        results['wh'] = wh
        results['mask'] = mask
        results['index'] = index
        results['index_all'] = index_all
        return results