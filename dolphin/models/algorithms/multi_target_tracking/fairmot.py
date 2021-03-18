import numpy as np
import torch
import torch.nn.functional as F

from . import matching
from dolphin.base.base_algorithm import BaseAlgorithm
from .base_tracker import (BaseTrack, TrackState, STrack, joint_stracks, 
                           sub_stracks, remove_duplicate_stracks)
from dolphin.utils import (Registers, build_module_from_registers, mot_decode, 
                         ctdet_post_process, tranpose_and_gather_feat)


@Registers.algorithm.register
class FairMOT(BaseAlgorithm):
    def __init__(self,
                 pretrained=None,
                 pretrained_modules=None,
                 down_ratio=4,
                 num_classes=1,
                 frame_rate=30,
                 track_buffer=30,
                 conf_thres=0.6,
                 max_objs=128,
                 min_box_area=200,
                 backbone=None,
                 head=None,
                 filter=dict(type='KalmanFilter'),
                 train_cfg=None,
                 test_cfg=None):

        super(FairMOT, self).__init__(
            pretrained=pretrained,
            pretrained_modules=pretrained_modules)

        self.backbone = build_module_from_registers(
            backbone, module_name='backbone')
        self.head = build_module_from_registers(head, module_name='head')

        self.init_weights()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.min_box_area = min_box_area

        self.det_head_cfg = head['det_head']

        self.filter = build_module_from_registers(
            filter, module_name='filter')
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thres = conf_thres
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 128

    def forward_train(self, imgs, label, **kwargs):
        x = self.backbone(imgs)
        x = self.head(x, label)
        losses = self.head.loss(x, label)
        return losses
    
    def forward_test(self, imgs, label, **kwargs):
        x = self.backbone(imgs)
        x = self.head(x, label)
        return x

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
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, imgs=None, meta=None, **kwargs):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        height, width = meta['original_shape']
        inp_height, inp_width = meta['resize_shape']
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s, 'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.forward(imgs, return_loss=False)
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.det_head_cfg['reg_offset'] else None
            dets, inds = mot_decode(
                hm, wh, reg=reg, cat_spec_wh=self.det_head_cfg['cat_spec_wh'],
                K=self.max_objs)
            id_feature = tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.det_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 
                self.filter, 30) for (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.gate_cost_matrix(
        #           self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(
            self.filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track \
            if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one '''
        '''beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thres:
                continue
            track.activate(self.filter, self.frame_id)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)


        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
    
    def test_track(self, data_batch, **kwargs):
        online_targets = self.update(**data_batch)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        return dict(online_tlwhs=online_tlwhs, online_ids=online_ids)