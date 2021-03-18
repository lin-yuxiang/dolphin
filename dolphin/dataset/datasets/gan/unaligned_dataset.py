import os
import os.path as osp
import random

from ..base import BaseDataset
from dolphin.utils import Registers, is_image_file


@Registers.data.register
class UnalignedDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 serial_batches=False,
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        
        super(UnalignedDataset, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        
        self.serial_batches = serial_batches

    def __len__(self):
        return max(self.A_size, self.B_size)
        
    def load_annotations(self):
        if self.test_mode:
            phase = 'test'
        else:
            phase = 'train'
        dir_A = osp.join(self.data_prefix, phase + 'A')
        dir_B = osp.join(self.data_prefix, phase + 'B')
        A_paths = self.get_data_files(dir_A)
        B_paths = self.get_data_files(dir_B)
        self.A_size = len(A_paths)
        self.B_size = len(B_paths)
        data_infos = dict(A_paths=A_paths, B_paths=B_paths)
        return data_infos

    def get_data_files(self, dir):
        images = []
        assert osp.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = osp.join(root, fname)
                    images.append(path)
        return sorted(images)

    def evaluate(self):
        pass
    
    def prepare_train_data(self, idx):
        A_path = self.data_infos['A_paths'][idx % self.A_size]
        if self.serial_batches:
            B_path = self.data_infos['B_paths'][idx % self.B_size]
        else:
            B_path = self.data_infos['B_paths'][
                random.randint(0, self.B_size - 1)]
        results = dict(filename=dict(A=A_path, B=B_path))
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        A_path = self.data_infos['A_paths'][idx]
        if self.serial_batches:
            B_path = self.data_infos['B_paths'][idx]
        else:
            B_path = self.data_infos['B_paths'][
                random.randint(0, self.B_size - 1)]
        results = dict(filename=dict(A=A_path, B=B_path))
        return self.pipelin(results)