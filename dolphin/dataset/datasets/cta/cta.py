import os
import os.path as osp
import random

from .base import BaseDataset
from dolphin.utils import Registers


@Registers.data.register
class CTA(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(CTA, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        
    def load_annotations(self):
        data_infos = []
        with open(osp.join(self.data_prefix, self.ann_file)) as f:
            split = f.readlines()
        split = [a.strip() for a in split]
        for s in split:
            info = dict()
            img_dir = osp.join(self.data_prefix, s)
            img_names = os.listdir(img_dir)
            info['img_dir'] = img_dir
            info['img_name'] = img_names
            data_infos.append(info)
        return data_infos

    def evaluate(self):
        pass

    def random_select(self, data_info):
        img_dir = data_info['img_dir']
        img_name = data_info['img_name']
        img_name = [a.split('_')[0] for a in img_name]
        num = dict()
        file_name = []
        for i in range(1, 15):
            num[str(i)] = 0
        for name in img_name:
            num[name] += 1
        for key, value in num.items():
            idx = random.randint(0, value - 1)
            img_file = '_'.join([key, str(idx)]) + '.png'
            file_name.append(osp.join(img_dir, img_file)) 
        return file_name

    def prepare_train_data(self, idx):
        results = dict()
        data_info = self.data_infos[idx]
        filename = self.random_select(data_info)
        label = int(data_info['img_dir'][-1])
        results['filename'] = filename
        results['label'] = label
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        results = dict()
        data_info = self.data_infos[idx]
        filename = self.random_select(data_info)
        label = int(data_info['img_dir'][-1])
        results['filename'] = filename
        results['label'] = label
        return self.pipeline(results)