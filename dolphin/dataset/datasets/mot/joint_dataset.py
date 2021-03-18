import os.path as osp
import numpy as np
import torch
from collections import OrderedDict
from sklearn import metrics
from scipy import interpolate

from ..base import BaseDataset
from dolphin.utils import Registers, mot_decode, ctdet_post_process
from dolphin.dataset.utils import xywh2xyxy, bbox_iou
from dolphin.dataset.evaluate import ap_per_class


@Registers.data.register
class JointDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        super(JointDataset, self).__init__(
            ann_file,
            pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

        if self.train_cfg is not None and 'num_classes' in self.train_cfg:
            self.num_classes = self.train_cfg['num_classes']
            # to be modified
        else:
            self.num_classes = None
    
    def __len__(self):
        if isinstance(self.num_files, list):
            return sum(self.num_files)
        elif isinstance(self.num_files, int):
            return self.num_files
        
    def load_annotations(self):
        if not isinstance(self.ann_file, dict):
            raise ValueError(f'format of ann_file must be dict.')
        img_files = OrderedDict()
        label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_idx = OrderedDict()

        for ds, path in self.ann_file.items():
            with open(path, 'r') as file:
                img_files[ds] = file.readlines()
                img_files[ds] = [
                    osp.join(self.data_prefix, x.strip())
                    for x in img_files[ds]]
                img_files[ds] = list(
                    filter(lambda x: len(x) > 0, img_files[ds]))
                
                label_files[ds] = [
                    x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                    for x in img_files[ds]]
        
        for ds, label_paths in label_files.items():
            max_idx = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_idx:
                    max_idx = img_max
            self.tid_num[ds] = max_idx + 1
        
        last_idx = 0
        for _, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_idx[k] = last_idx
            last_idx += v
        
        self.num_files = [len(x) for x in img_files.values()]
        self.cur_num = [
            sum(self.num_files[:i]) for i in range(len(self.num_files))]

        return dict(img_files=img_files, label_files=label_files)

    def evaluate(self, 
                 mode='test_emb',
                 embedding=None,
                 id_labels=None,
                 logger=None,
                 all_hm=None,
                 all_wh=None,
                 all_reg=None,
                 all_label=None,
                 metas=None,
                 **kwargs):
        assert mode in ['test_emb', 'test_det']
        if mode == 'test_emb':
            if len(embedding) < 1:
                logger.info('No embedding extracted.')
                return None
            embedding = torch.stack(embedding, dim=0).cuda()
            id_labels = torch.LongTensor(id_labels)
            n = len(id_labels)
            assert len(embedding) == n

            embedding = torch.nn.functional.normalize(embedding, dim=1)
            pdist = torch.mm(embedding, embedding.t()).cpu().numpy()
            gt = id_labels.expand(n, n).eq(id_labels.expand(n, n).t()).numpy()

            up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
            pdist = pdist[up_triangle]
            gt = gt[up_triangle]

            far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            far, tar, _ = metrics.roc_curve(gt, pdist)
            interp = interpolate.interp1d(far, tar)
            tar_at_far = [interp(x) for x in far_levels]
            for f, fa in enumerate(far_levels):
                logger.info('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
            return tar_at_far
        else:
            mean_AP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
            AP_accum, AP_accum_count = np.zeros(1), np.zeros(1)
            mAPs, mR, mP = [], [], []
            iou_thresh = self.test_cfg['iou_thresh']
            self.num_classes = self.test_cfg['num_classes']
            logger.info('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
            for hm, wh, reg, label, meta in zip(
                all_hm, all_wh, all_reg, all_label, metas):
                height, width = meta['original_shape']
                inp_height, inp_width = meta['resize_shape']
                c = np.array([width / 2., height / 2.], dtype=np.float32)
                s = max(
                    float(inp_width) / float(inp_height) * height, width) * 1.0
                meta = {
                    'c': c, 's': s,
                    'out_height': inp_height // self.test_cfg['down_ratio'],
                    'out_width': inp_width // self.test_cfg['down_ratio']}
                hm = hm.sigmoid_()
                detections, _ = mot_decode(
                    hm, wh, reg=reg, cat_spec_wh=self.test_cfg['cat_spec_wh'], 
                    K=self.test_cfg['max_objs'])
                for si, labels in enumerate(label):
                    seen += 1
                    dets = detections[si]
                    dets = dets.unsqueeze(0)
                    dets = self.post_process(dets, meta)
                    dets = self.merge_outputs([dets])[1]

                    if dets is None:
                        if labels.size(0) != 0:
                            mAPs.append(0), mR.append(0), mP.append(0)
                        continue

                    correct = []
                    if labels.size(0) == 0:
                        mAPs.append(0), mR.append(0), mP.append(0)
                        continue
                    else:
                        target_cls = labels[:, 0]
                        target_boxes = xywh2xyxy(labels[:, 2:6]) 
                        target_boxes[:, 0] *= width
                        target_boxes[:, 2] *= width
                        target_boxes[:, 1] *= height
                        target_boxes[:, 3] *= height

                        detected = []
                        for *pred_bbox, _ in dets:
                            obj_pred = 0
                            pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                            iou = bbox_iou(
                                pred_bbox, target_boxes, x1y1x2y2=True)[0]
                            best_i = np.argmax(iou)
                            if iou[best_i] > iou_thresh and \
                                obj_pred == labels[best_i, 0] and \
                                    best_i not in detected:
                                correct.append(1)
                                detected.append(best_i)
                            else:
                                correct.append(0)

                    AP, AP_class, R, P = ap_per_class(
                        tp=correct, conf=dets[:, 4], target_cls=target_cls,
                        pred_cls=np.zeros_like(dets[:, 4]))
                    AP_accum_count += np.bincount(AP_class, minlength=1)
                    AP_accum += np.bincount(AP_class, minlength=1, weights=AP)

                    mAPs.append(AP.mean())
                    mR.append(R.mean())
                    mP.append(P.mean())

                    mean_AP = np.sum(mAPs) / (AP_accum_count + 1e-16)
                    mean_R = np.sum(mR) / (AP_accum_count + 1e-16)
                    mean_P = np.sum(mP) / (AP_accum_count + 1e-16)
                
                logger.info(
                    ('%11s%11s' + '%11.3g' * 3) % (
                        seen, len(self), mean_P, mean_R, mean_AP))

            return mean_AP, mean_R, mean_P

    def prepare_train_data(self, idx):
        for i, c in enumerate(self.cur_num):
            if idx >= c:
                dataset = list(self.data_infos['label_files'].keys())[i]
                start_idx = c

        img_path = self.data_infos['img_files'][dataset][idx - start_idx]
        label_path = self.data_infos['label_files'][dataset][idx - start_idx]
        results = dict(img_path=img_path, label_path=label_path,
                       data_name=dataset, tid_start_index=self.tid_start_idx)
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        for i, c in enumerate(self.cur_num):
            if idx >= c:
                dataset = list(self.data_infos['label_files'].keys())[i]
                start_idx = c

        img_path = self.data_infos['img_files'][dataset][idx - start_idx]
        label_path = self.data_infos['label_files'][dataset][idx - start_idx]
        results = dict(img_path=img_path, label_path=label_path, 
                       data_name=dataset, tid_start_index=self.tid_start_idx)
        return self.pipeline(results)

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]
    
    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], 
                axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > 128:
            kth = len(scores) - 128
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results