import matplotlib.pyplot as plot
import os.path as osp
import numpy as np
import h5py
import pdb
import copy

from ..base import BaseDataset
from dolphin.utils import Registers, mkdir_or_exist


@Registers.data.register
class NyuDepthDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_list=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        self.data_list = data_list
        super(NyuDepthDataset, self).__init__(
            ann_file,
            pipeline,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        
    def load_annotations(self):
        with open(self.data_list) as f:
            data_infos = f.readlines()
        data_infos = [int(i.strip()) - 1 for i in data_infos]

        self.ann = h5py.File(self.ann_file)
        self.images = self.ann['images']
        self.depths = self.ann['depths']
        return data_infos

    def evaluate(self, 
                 num, 
                 results, 
                 data_batch, 
                 num_sample,
                 thresh_1_25,
                 thresh_1_25_2,
                 thresh_1_25_3,
                 rmse_linear,
                 rmse_log,
                 rmse_log_scale_invariant,
                 ard,
                 srd):

        filename_tmpl = '{:05d}.png'
        if self.test_cfg['gt_dir'] is not None and \
            self.test_cfg['pred_dir'] is not None:
            gt_dir = self.test_cfg['gt_dir']
            pred_dir = self.test_cfg['pred_dir']
        else:
            gt_dir = './gt_dir'
            pred_dir = './pred_dir'
        mkdir_or_exist(gt_dir)
        mkdir_or_exist(pred_dir)
        gt_file = 'test_gt_depth_' + filename_tmpl.format(num)
        pred_file = 'test_pred_depth_' + filename_tmpl.format(num)

        gt_depth = data_batch['label']
        gt_depth = gt_depth[0].data.squeeze().cpu().numpy().astype(np.float32)
        pred_depth = results[0].data.squeeze().cpu().numpy().astype(np.float32)

        gt_depth /= np.max(gt_depth)
        pred_depth /= np.max(pred_depth)

        plot.imsave(osp.join(gt_dir, gt_file), gt_depth, cmap='viridis')
        plot.imsave(osp.join(pred_dir, pred_file), pred_depth, cmap='viridis')

        n = np.sum(gt_depth > 1e-3)

        indice = (gt_depth <= 1e-3)
        pred_depth[indice] = 1
        gt_depth[indice] = 1

        pred_d_gt = pred_depth / gt_depth
        pred_d_gt[indice] = 100
        gt_d_pred = gt_depth / pred_depth
        gt_d_pred[indice] = 100

        thresh_1_25 = (np.sum(
                np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n / num_sample)
        thresh_1_25_2 = (np.sum(
            np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n / num_sample)
        thresh_1_25_3 = (np.sum(
            np.maximum(pred_d_gt, gt_d_pred) < 1.25 ** 3) / n / num_sample)

        log_pred = np.log(pred_depth)
        log_gt = np.log(gt_depth)

        d_i = log_gt - log_pred

        rmse_linear = (np.sqrt(
            np.sum((pred_depth - gt_depth) ** 2) / n) / num_sample)
        rmse_log = np.sqrt(np.sum((log_pred - log_gt) ** 2) / n) / num_sample
        rmse_log_scale_invariant = (np.sum(d_i ** 2) / n + \
            (np.sum(d_i) ** 2) / (n ** 2) / num_sample)
        ard = (np.sum(
            np.abs((pred_depth - gt_depth)) / gt_depth) / n / num_sample)
        srd = (np.sum(
            ((pred_depth - gt_depth) ** 2) / gt_depth) / n / num_sample)
        return (thresh_1_25, thresh_1_25_2, thresh_1_25_3, rmse_linear, 
                    rmse_log, rmse_log_scale_invariant, ard, srd)

    def prepare_train_data(self, idx):
        results = dict()
        imgs = self.images[self.data_infos[idx]].astype(np.float32)
        depth = self.depths[self.data_infos[idx]].astype(np.float32)
        imgs_shape = (imgs.shape[2], imgs.shape[1])
        label_shape = (depth.shape[1], depth.shape[0])
        results['imgs'] = imgs
        results['label'] = depth
        results['imgs_cfg'] = dict()
        results['label_cfg'] = dict()
        results['imgs_cfg']['ori_shape'] = imgs_shape
        results['label_cfg']['ori_shape'] = label_shape
        results['imgs_cfg']['shape'] = imgs_shape
        results['label_cfg']['shape'] = label_shape
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        results = dict()
        imgs = self.images[self.data_infos[idx]].astype(np.float32)
        depth = self.depths[self.data_infos[idx]].astype(np.float32)
        imgs_shape = (imgs.shape[2], imgs.shape[1])
        label_shape = (depth.shape[1], depth.shape[0])
        results['imgs'] = imgs
        results['label'] = depth
        results['imgs_cfg'] = dict()
        results['label_cfg'] = dict()
        results['imgs_cfg']['ori_shape'] = imgs_shape
        results['label_cfg']['ori_shape'] = label_shape
        results['imgs_cfg']['shape'] = imgs_shape
        results['label_cfg']['shape'] = label_shape
        return self.pipeline(results)