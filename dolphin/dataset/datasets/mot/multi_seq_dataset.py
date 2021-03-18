import os.path as osp
import glob
import copy
import numpy as np
import motmetrics as mm
mm.lap.default_solver = 'lap'

from .base import BaseDataset
from dolphin.utils import Registers
from dolphin import utils


@Registers.data.register
class MultiSequenceDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=True,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):

        seqs = test_cfg['track']['sequences']
        if isinstance(seqs, str):
            self.seqs = [seqs]
        elif isinstance(seqs, (list, tuple)):
            self.seqs = seqs
        else:
            raise ValueError(f'The sequences in test_cfg should be type of '
                              'str or list')
        self.seq = self.seqs[0]
        
        super(MultiSequenceDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
    
    def load_annotations(self):
        data_infos = dict()
        for seq in self.seqs:
            data_path = osp.join(self.data_prefix, seq, 'img1')
            img_format = ['.jpg', 'jpeg', '.png', '.tif']
            img_files = sorted(glob.glob('%s/*.*' % data_path))
            img_files = list(
                filter(lambda x: osp.splitext(x)[1].lower() 
                in img_format, img_files))
            if len(img_files) <= 0:
                raise ValueError('No images found in ' + data_path)
            data_infos[seq] = img_files
        return data_infos

    def prepare_train_data(self, idx):
        raise ValueError(
        f'test_mode should be set to True when using MultiSequencesDataset.')
    
    def prepare_test_data(self, idx):
        if self.seq not in self.seqs:
            raise ValueError(f'seq not in assigned seqs.')
        img_files = self.data_infos[self.seq]
        num_frames = len(img_files)
        results=dict(
            sequence=self.seq, img_file=img_files[idx], num_frames=num_frames)
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos[self.seq])

    def evaluate(self, gt_path, results_path):
        self.gt_frame_dict = utils.read_mot_results(gt_path, is_gt=True)
        self.gt_ignore_frame_dict = utils.read_mot_results(gt_path, is_gt=False)
        self.acc = mm.MOTAccumulator(auto_id=True)
        result_frame_dict = utils.read_mot_results(results_path)
        frames = sorted(
            list(
                set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = utils.unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)
        
        return self.acc

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = utils.unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = utils.unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(
            ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(
                lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and \
            hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    def get_eval_summary(self,
                         accs, 
                         names, 
                         metrics=[
                             'meta', 'num_switches', 'idp', 'idr', 'idf1', 
                             'precision', 'recall']):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True)
        
        return summary
    
    def save_eval_summary(self, summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()