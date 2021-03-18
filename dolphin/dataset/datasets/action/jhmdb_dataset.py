import os
import pickle
import os.path as osp
import numpy as np
from collections import defaultdict

from dolphin.utils import Registers
from .moc_dataset import MOCDatasetMixIn
from ..base import BaseDataset


@Registers.data.register
class JHMDBDataset(BaseDataset, MOCDatasetMixIn):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 num_classes=None,
                 modality='RGB',
                 K=None,
                 filename_tmpl='{:0>5}.png',
                 split=1,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(JHMDBDataset, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            modality=modality,
            num_classes=num_classes,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        
        self.K = K
        self.split = split
        self.filename_tmpl = filename_tmpl

        self.indices = self.get_indices()
        
    def load_annotations(self):
        with open(self.ann_file, 'rb') as fid:
            ann = pickle.load(fid, encoding='iso-8859-1')
        for k in ann:
            setattr(self, ('_' if k != 'labels' else '') + k, ann[k])
        assert len(self.train_videos[self.split - 1]) + \
            len(self.test_videos[self.split - 1]) == len(self.nframes)
        if self.test_mode:
            video_list = self.test_videos[self.split - 1]
        else:
            video_list = self.train_videos[self.split - 1]
        
        return video_list

    @property
    def gttubes(self):
        return self._gttubes

    @property
    def nframes(self):
        return self._nframes
    
    @property
    def train_videos(self):
        return self._train_videos
    
    @property
    def test_videos(self):
        return self._test_videos
    
    @property
    def resolution(self):
        return self._resolution
    
    def get_indices(self):
        indices = []
        for v in self.data_infos:
            if not self.test_mode:
                vtubes = sum(self.gttubes[v].values(), [])
                indices += [(v, i) for i in range(1, self.nframes[v] + 2 -
                    self.K) if self.tubelet_in_out_tubes(vtubes, i, self.K) and
                    self.tubelet_has_gt(vtubes, i, self.K)]
            else:
                for i in range(1, 1 + self.nframes[v] - self.K + 1):
                    outfile = osp.join(
                        self.test_cfg['inference_dir'], v, 
                        '{:0>5}.pkl'.format(i))
                    if not osp.exists(outfile):
                        indices += [(v, i)]
        return indices

    def _import_ground_truths(self):
        gts = {}
        for ilabel, label in enumerate(self.labels):
            gt = defaultdict(list)
            
            for iv, v in enumerate(self.data_infos):
                tubes = self.gttubes[v]
                if ilabel not in tubes:
                    continue
                for tube in tubes[ilabel]:
                    for i in range(tube.shape[0]):
                        k = (iv, int(tube[i, 0]))
                        gt[k].append(tube[i, 1:5].tolist())
            
            for k in gt:
                gt[k] = np.array(gt[k])

            gts[ilabel] = gt

        return gts

    def prepare_test_frames(self, idx):
        video, frame = self.indices[idx]
        results = dict(video_name=video,
                       frame_idx=frame,
                       resolution=self.resolution[video],
                       filename_tmpl=self.filename_tmpl,
                       data_prefix=self.data_prefix,
                       clip_len=self.K,
                       gttubes=self.gttubes[video],
                       num_frames=self.nframes[video],
                       num_classes=self.num_classes)
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        video, frame = self.indices[idx]
        data_prefix = osp.join(self.data_prefix, video)
        results = dict(video_name=video,
                       frame_idx=frame,
                       resolution=self.resolution[video],
                       num_frames=self.nframes[video],
                       filename_tmpl=self.filename_tmpl,
                       gttubes=self.gttubes[video],
                       clip_len=self.K,
                       data_prefix=data_prefix,
                       num_classes=self.num_classes)
        return self.pipeline(results)

    def tubelet_in_out_tubes(self, tube_list, i, K):
        return all([self.tubelet_in_tube(tube, i, K) or 
            self.tubelet_out_tube(tube, i, K) for tube in tube_list])
        
    def tubelet_in_tube(self, tube, i, K):
        return all([j in tube[:, 0] for j in range(i, i + K)])
    
    def tubelet_out_tube(self, tube, i, K):
        return all([not j in tube[:, 0] for j in range(i, i + K)])
    
    def tubelet_has_gt(self, tube_list, i, K):
        return any([self.tubelet_in_tube(tube, i, K) for tube in tube_list])

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)