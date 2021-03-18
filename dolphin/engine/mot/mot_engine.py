import torch
import os.path as osp
import os
import itertools
import time
import motmetrics as mm
mm.lap.default_solver = 'lap'

from ..base import ABCEngine
from dolphin.utils import Registers, Bar 
from dolphin import utils
from dolphin.dataset import PCLDataParallel
from dolphin.models.utils import load_checkpoint


@Registers.engine.register
class MOTEngine(ABCEngine):

    def __init__(self,
                 algorithm=None,
                 train_cfg=None,
                 test_cfg=None,
                 data=None,
                 runtime_cfg=None,
                 meta=None,
                 distributed=None):

        super(MOTEngine, self).__init__(
            algorithm=algorithm,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data=data,
            runtime_cfg=runtime_cfg,
            meta=meta,
            distributed=distributed)
    
    def test_emb(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode=mode)
        embedding, id_labels = [], []

        num_batch = len(data_loader)
        bar = Bar('Extracting Feature: ', max=(num_batch // self.vis_interval))
        self.meters.before_epoch()

        for i, data_batch in enumerate(data_loader):
            start_time = time.time()
            # during train iter
            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
            outputs = outputs['results']
            id_head = outputs['id']
            id_target = data_batch['ids'][data_batch['reg_mask'] > 0]

            for i in range(id_head.shape[0]):
                feat, label = id_head[i], id_target[i].long()
                if label != 1:
                    embedding.append(feat)
                    id_labels.append(label)
            
            end_time = time.time()
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.average()

            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format((i // self.vis_interval) + 1, 
                    num_batch // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.4f}'.format(name, val)
                Bar.suffix = print_str
                self.log.debug(f'Extracting Feature: {self.epoch}' + print_str)
                bar.next()
            self.meters.clear_output()
        
        bar.finish()
        self.log.info(f'Calculating Evaluation Results ...')
        if hasattr(data_loader.dataset, mode):
            data_loader.dataset.evaluate(
                mode='test_emb', embedding=embedding, 
                id_labels=id_labels, logger=self.log)
        else:
            raise ValueError(f'No {mode} method in dataset class found.')
    
    def test_det(self, data_loader, mode, **kwargs):
        self.model.set_mode(mode=mode)

        num_batch = len(data_loader)
        bar = Bar('Testing Det: ', max=(num_batch // self.vis_interval))
        self.meters.before_epoch()
        
        all_hm, all_wh, all_reg, all_label, metas = [], [], [], [], []
        for i, data_batch in enumerate(data_loader):
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model.test_step(data_batch, **kwargs)
            outputs = outputs['results']
            all_hm.append(outputs['hm'])
            all_wh.append(outputs['wh'])
            all_reg.append(outputs.get('reg', None))
            all_label.append(data_batch['label'])
            metas.append(data_batch['meta'])

            end_time = time.time()
            self.meters.update({'batch_time': end_time - start_time})
            self.meters.average()

            if i % self.vis_interval == 0:
                print_str = '[{}/{}]'.format((i // self.vis_interval) + 1, 
                    num_batch // self.vis_interval)
                for name, val in self.meters.output.items():
                    print_str += ' | {} {:.4f}'.format(name, val)
                Bar.suffix = print_str
                self.log.debug(f'Testing Det: {self.epoch}' + print_str)
                bar.next()
            self.meters.clear_output()
        
        bar.finish()
        self.log.info('Calculating Evaluation Results ...')
        if hasattr(data_loader.dataset, mode):
            data_loader.dataset.evaluate(
                mode='test_det', all_hm=all_hm, all_wh=all_wh, all_reg=all_reg, 
                all_label=all_label, metas=metas, logger=self.log)
        else:
            raise ValueError(f'No {mode} method in dataset class found.')

    def test_track(self, data_loader, mode, **kwargs):
        assert 'test_track' in self.workflow
        self.model.set_mode(mode)
        save_images = self.test_cfg['save_images']
        save_videos = self.test_cfg['save_videos']
        data_prefix = self.data_cfg['data_prefix']
        output_dir = osp.join(self.work_dir, 'outputs')
        utils.mkdir_or_exist(output_dir)

        seqs = self.test_cfg['sequences']
        accs = []
        for seq in seqs:

            frame_id = 0
            save_dir = osp.join(output_dir, seq)
            utils.mkdir_or_exist(save_dir)
            results = []
            num_batch = len(data_loader)
            self.meters.before_epoch()

            data_loader.dataset.seq = seq
            num_batch = len(data_loader.dataset)
            bar = Bar(f'Sequence: {seq}', max=(num_batch // self.vis_interval))
            for i, data_batch in enumerate(data_loader):
                start_time = time.time()

                outputs = self.model.test_track(data_batch, **kwargs)
                # after train iter
                online_tlwhs = outputs['online_tlwhs']
                online_ids = outputs['online_ids']

                end_time = time.time()
                self.meters.update({'batch_time': end_time - start_time})
                self.meters.average()

                results.append((frame_id + 1, online_tlwhs, online_ids))
                
                if save_images:
                    online_im = utils.plot_tracking(
                        data_batch['original_imgs'], online_tlwhs, 
                        online_ids, frame_id=frame_id)
                    utils.imwrite(
                        online_im, osp.join(save_dir, 
                        '{:05d}.jpg'.format(frame_id)))

                if i % self.vis_interval == 0:
                    print_str = '[{}/{}]'.format(
                        (i // self.vis_interval) + 1, 
                        num_batch // self.vis_interval)
                    for name, val in self.meters.output.items():
                        print_str += ' | {} {:.4f}'.format(name, val)
                    Bar.suffix = print_str
                    self.log.debug(f'Seqence: {seq}' + print_str)
                    bar.next()
                self.meters.clear_output()
                frame_id += 1
            
            bar.finish()
            results_path = osp.join(save_dir, 'results.txt')
            self.write_results(results_path, results)

            if save_videos:
                video_path = osp.join(save_dir, '{}.mp4'.format(seq))
                cmd = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
                    save_dir, video_path)
                os.system(cmd)
            
            gt_path = osp.join(data_prefix, seq, 'gt', 'gt.txt')
            if osp.isfile(gt_path):
                accs.append(
                    data_loader.dataset.evaluate(gt_path, results_path))
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = data_loader.dataset.get_eval_summary(accs, seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        self.log.info(strsummary)
        eval_summary_path = osp.join(self.work_dir, 'eval_summary_mot.xlsx')
        data_loader.dataset.save_eval_summary(summary, eval_summary_path)

    def write_results(self, filename, results):
        save_format = '{frame}, {id}, {x1}, {y1}, {w}, {h}, 1, -1, -1, -1\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(
                        frame=frame_id, id=track_id, x1=x1, y1=y1, 
                        x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
        self.log.info('save results to {}'.format(filename))