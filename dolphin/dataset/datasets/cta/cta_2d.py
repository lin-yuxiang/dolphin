import os
import json
import os.path as osp
import random

from .base import BaseDataset
from dolphin.utils import Registers


@Registers.data.register
class CTA2d(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 max_box=None,
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        
        self.split = split
        self.max_box = max_box

        super(CTA2d, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.load_max_box_num()
    
    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as f:
            ann = json.load(f)
        with open(self.split, 'r') as f:
            split = f.readlines()
        split = [a.strip() for a in split]
        for person in split:
            info = dict()
            img_dir = osp.join(self.data_prefix, person)
            img_names = os.listdir(img_dir)
            info['img_dir'] = img_dir
            info['img_name'] = [osp.join(img_dir, name) for name in img_names]
            img_info = {}
            for img in img_names:
                couple, label = person.split('/')
                cat = img.split('.')[0]
                key = '_'.join([couple, label, cat])
                img_info[cat] = ann[key]
            info['img_info'] = img_info
            data_infos.append(info)
        return data_infos

    def evaluate(self):
        pass
    
    def load_max_box_num(self):
        with open(self.max_box, 'r') as f:
            self.box_num = json.load(f)

    def prepare_train_data(self, idx):
        results = dict()
        data_info = self.data_infos[idx]
        label = int(data_info['img_dir'][-1])
        results['filename'] = data_info['img_name']
        results['label'] = label
        results['box_num'] = self.box_num
        results['img_info'] = data_info['img_info']
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        results = dict()
        data_info = self.data_infos[idx]
        label = int(data_info['img_dir'][-1])
        results['filename'] = data_info['img_name']
        results['label'] = label
        results['box_num'] = self.box_num
        results['img_info'] = data_info['img_info']
        return self.pipeline(results)