import os.path as osp
import torch
import numpy as np
import torchvision
from PIL import Image

from dolphin.utils import Registers
from ..base import BaseDataset


@Registers.data.register
class MnistDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 num_classes=10,
                 init_labels=10000,
                 data_name='MNIST',
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        
        self.data_name = data_name
        self.init_labels = init_labels
        self.num_classes = num_classes
        super(MnistDataset, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        if not test_mode:
            self.init_train_dataset()

    def __len__(self):
        if not self.test_mode:
            return np.where(self.idx_labeled == True)[0].size
        else:
            return len(self.data_infos['imgs'])

    def update_label_set(self, idx):
        self.idx_labeled = ~self.idx_labeled
        self.idx_labeled[idx] = True
    
    def acquire_unlabeled_data(self):
        self.idx_labeled = ~self.idx_labeled
    
    def init_train_dataset(self):
        self.num_data = len(self.data_infos['imgs'])
        self.idx_labeled = np.zeros(self.num_data, dtype=bool)
        idx_tmp = np.arange(self.num_data)
        np.random.shuffle(idx_tmp)
        self.idx_labeled[idx_tmp[:self.init_labels]] = True

    def load_annotations(self):
        data_infos = dict()
        if hasattr(torchvision.datasets, self.data_name):
            data_path = osp.join(self.data_prefix, self.data_name)
            data = getattr(torchvision.datasets, self.data_name)
            if not self.test_mode:
                raw_data = data(data_path, train=True, download=True)
                data_infos['imgs'] = raw_data.train_data
                data_infos['label'] = raw_data.train_labels
            else:
                raw_data = data(data_path, train=False, download=True)
                data_infos['imgs'] = raw_data.test_data
                data_infos['label'] = raw_data.test_labels
        else:
            raise NotImplementedError
        return data_infos

    def prepare_train_data(self, idx):
        imgs = self.data_infos['imgs'][self.idx_labeled][idx]
        label = self.data_infos['label'][self.idx_labeled][idx]
        imgs = Image.fromarray(imgs.numpy(), mode='L')
        results = dict(imgs=np.array(imgs)[..., np.newaxis], label=np.array(label))
        results['imgs_cfg'] = dict()
        results['label_cfg'] = dict()
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        imgs = self.data_infos['imgs'][idx] 
        imgs = Image.fromarray(imgs.numpy(), mode='L')
        results = dict(imgs=np.array(imgs)[..., np.newaxis])
        results['imgs_cfg'] = dict()
        return self.pipeline(results)

    def evaluate(self, pred, logger=None):
        label = self.data_infos['label'].numpy()
        pred = pred.astype(label.dtype)
        acc = 1.0 * (label == pred).sum() / len(label)
        if logger is not None:
            logger.info(f'Testing accuracy: {acc}')