import copy
import os
import os.path as osp
import json
from ..base import BaseDataset
import numpy as np
from dolphin.utils import Registers
from dolphin.dataset.evaluate import average_recall_at_avg_proposals


@Registers.data.register
class ActivityNetDataset(BaseDataset):
    def __init__(self, 
                 ann_file, 
                 pipeline, 
                 data_prefix=None, 
                 test_mode=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(ActivityNetDataset, self).__init__(
            ann_file, 
            pipeline, 
            data_prefix=data_prefix, 
            test_mode=test_mode, 
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        video_infos = []
        with open(self.ann_file, 'r') as f:
            anno_database = json.load(f)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            video_info['video_name'] = video_name
            video_infos.append(video_info)
        return video_infos

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.data_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.data_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_infos)

    def _import_ground_truth(self):
        """Read ground truth data from video_infos."""
        ground_truth = {}
        for video_info in self.data_infos:
            video_id = video_info['video_name'][2:]
            this_video_ground_truths = []
            for ann in video_info['annotations']:
                t_start, t_end = ann['segment']
                label = ann['label']
                this_video_ground_truths.append([t_start, t_end, label])
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    def proposals2json(self, results):
        result_dict = {}
        # print('Convert proposals to json format')

        for result in results:
            video_name = result['video_name']
            result_dict[video_name[2:]] = result['proposal_list']
        return result_dict

    def _import_proposals(self, results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name'][2:]
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals

    def dump_results(self, results, out, output_format, version='VERSION 1.3'):
        """Dump data to json/csv files."""
        if output_format == 'json':
            result_dict = self.proposals2json(results)
            output_dict = {
                'version': version,
                'results': result_dict,
                'external_data': {}
            }
            with open(out, 'w') as f:
                json.dump(output_dict, f)
        elif output_format == 'csv':
            os.makedirs(out, exist_ok=True)
            header = 'action,start,end,tmin,tmax'
            for result in results:
                video_name, outputs = result
                output_path = osp.join(out, video_name + '.csv')
                np.savetxt(
                    output_path,
                    outputs,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

    def evaluate(self,
                 results,
                 metrics='AR@AN',
                 max_avg_proposals=100,
                 temporal_iou_thresholds=np.linspace(0.5, 0.95, 10)):
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['AR@AN']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        ground_truth = self._import_ground_truth()
        proposal, num_proposals = self._import_proposals(results)

        for metric in metrics:
            if metric == 'AR@AN':
                recall, _, _, auc = (
                    average_recall_at_avg_proposals(
                        ground_truth,
                        proposal,
                        num_proposals,
                        max_avg_proposals=max_avg_proposals,
                        temporal_iou_thresholds=temporal_iou_thresholds))
                eval_results['auc'] = auc
                eval_results['AR@1'] = np.mean(recall[:, 0])
                eval_results['AR@5'] = np.mean(recall[:, 4])
                eval_results['AR@10'] = np.mean(recall[:, 9])
                eval_results['AR@100'] = np.mean(recall[:, 99])

        return eval_results