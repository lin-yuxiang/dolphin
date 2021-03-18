import pickle
import os.path as osp
import numpy as np
from collections import defaultdict
import copy

from dolphin import utils
from dolphin.dataset.evaluate import pr_to_ap
from dolphin.dataset.utils import iou2d, iou3dt
from dolphin.utils.postprocess import nms2d, nms3dt, nms_tubelets


class MOCDatasetMixIn(object):

    def __init__(self):
        super().__init__()
        
        self.inference_dir = self.test_cfg['inference_dir']
        self.iou_thresh = self.test_cfg['iou_thresh']

    def build_frame_prediction(self):
        alldets = []
        for iv, v in enumerate(self.data_infos):
            vdets = {i: np.empty((0, 6), dtype=np.float32) 
                     for i in range(1, 1 + self.nframes[v])}
            for i in range(1, 1 + self.nframes[v] - self.K + 1):
                pkl = osp.join(self.inference_dir, v, '{:0>5}.pkl'.format(i))
                if not osp.isfile(pkl):
                    raise RuntimeError('Missing extracted tubelets ' + pkl)
                with open(pkl, 'rb') as fid:
                    dets = pickle.load(fid)

                for label in dets:
                    tubelets = dets[label]
                    labels = np.empty((tubelets.shape[0], 1), dtype=np.int32)
                    labels[:, 0] = label - 1
                    for k in range(self.K):
                        vdets[i + k] = np.concatenate((vdets[i + k], 
                        np.concatenate(
                            (tubelets[:, np.array(
                                [4 * k, 1 + 4 * k, 2 + 4 * k, 3 + 4 * k, -1])], 
                             labels),
                            axis=1)), axis=0)
            
            for i in vdets:
                num_objs = vdets[i].shape[0]
                for ilabel in range(len(self.labels)):
                    vdets[i] = vdets[i].astype(np.float32)
                    a = np.where(vdets[i][:, 5] == ilabel)[0]
                    if a.size == 0:
                        continue
                    vdets[i][vdets[i][:, 5] == ilabel, :5] = nms2d(
                        vdets[i][vdets[i][:, 5] == ilabel, :5], 0.6)
                alldets.append(np.concatenate((iv * np.ones((num_objs, 1), 
                    dtype=np.float32), i * np.ones((num_objs, 1),
                    dtype=np.float32), vdets[i][:, np.array([5, 4, 0, 1, 2, 3],
                    dtype=np.int32)]), axis=1))

        alldets = np.concatenate(alldets, axis=0)
        
        return alldets
    
    def build_tubes(self):
        for iv, v in enumerate(self.data_infos):
            outfile = osp.join(self.inference_dir, v + '_tubes.pkl')
            if osp.isfile(outfile):
                continue

            RES = {}
            nframes = self.nframes[v]

            VDets = {}
            for startframe in range(1, nframes + 2 - self.K):
                resname = osp.join(
                    self.inference_dir, v, '{:0>5}.pkl'.format(startframe))
                if not osp.isfile(resname):
                    raise ValueError('Missing extracted tubelets ' + resname)
                with open(resname, 'rb') as fid:
                    VDets[startframe] = pickle.load()
            for ilabel in range(len(self.labels)):
                FINISHED_TUBES = []
                CURRENT_TUBES = []

                def tubescore(tt):
                    return np.mean(
                        np.array([tt[i][1][-1] for i in range(len(tt))]))
                
                for frame in range(1, self.nframes[v] + 2 - self.K):
                    ltubelets = VDets[frame][ilabel + 1]
                    ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)

                    if frame == 1:
                        for i in range(ltubelets.shape[0]):
                            CURRENT_TUBES.append([1, ltubelets[i, :]])
                        continue
                        
                    avgscore = [tubescore(t) for t in CURRENT_TUBES]
                    argsort = np.argsort(-np.array(avgscore))
                    CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                    finished = []
                    for it, t in enumerate(CURRENT_TUBES):
                        last_frame, last_tubelet = t[-1]
                        ious = []
                        offset = frame - last_frame
                        if offset < self.K:
                            nov = self.K - offset
                            ious = sum(
                                [iou2d(ltubelets[:, 4 * iov: 4 * iov + 4],
                                last_tubelet[
                                    4 * (iov + offset): 4 * (iov + offset + 1)])
                                for iov in range(nov)]) / float(nov)
                        else:
                            ious = iou2d(
                                ltubelets[:, :4], 
                                last_tubelet[4 * self.K - 4: 4 * self.K])
                        valid = np.where(ious >= 0.5)[0]
                        if valid.size > 0:
                            idx = valid[np.argmax(ltubelets[valid, -1])]
                            CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                            ltubelets = np.delete(ltubelets, idx, axis=0)
                        else:
                            if offset >= self.K:
                                finished.append(it)

                    for it in finished[::-1]:
                        FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                        del CURRENT_TUBES[it]

                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(frame, ltubelets[i, :])])
                
                FINISHED_TUBES += CURRENT_TUBES

                output = []
                for t in FINISHED_TUBES:
                    score = tubescore(t)
                    if score < 0.005:
                        continue
                    beginframe = t[0][0]
                    endframe = t[-1][0] + self.K - 1
                    length = endframe - beginframe + 1

                    if length < 15:
                        continue
                    out = np.zeros((length, 6), dtype=np.float32)
                    out[:, 0] = np.arange(beginframe, endframe + 1)
                    n_per_frame = np.zeros((length, 1), dtype=np.int32)
                    for i in range(len(t)):
                        frame, box = t[i]
                        for k in range(self.K):
                            out[frame - beginframe + k, 1: 5] += \
                                box[4 * k: 4 * k + 4]
                            out[frame - beginframe + k, -1] += box[-1]
                            n_per_frame[frame - beginframe + k, 0] += 1
                    out[:, 1:] /= n_per_frame
                    output.append([out, score])
                
                RES[ilabel] = output
            
            with open(outfile, 'wb') as fid:
                pickle.dump(RES, fid)

    def evaluate(self, metric='frameAP', logger=None):
        if metric == 'frameAP' or metric == 'frameAP_error':
            metric_func = getattr(self, metric)
        elif metric == 'videoAP' or metric == 'videoAP_050_095':
            metric_func = getattr(self, metric)
        else:
            raise ValueError(f'No {metric} metric found.')
        metric_func(logger=logger)

    def dump_results(self, results, data_batch, num_samples):
        video_name = data_batch['video_meta']['video_name']
        frame = data_batch['video_meta']['frame_idx']
        for j in range(num_samples):
            outfile = osp.join(
                self.inference_dir, video_name[j], 
                '{:0>5}.pkl'.format(frame[j]))
            if not osp.isdir(osp.dirname(outfile)):
                utils.mkdir_or_exist(osp.dirname(outfile))        
        
            with open(outfile, 'wb') as f:
                pickle.dump(results[j], f)

    def frameAP(self, logger=None):
        frame_detections_file = osp.join(
            self.inference_dir, 'frame_detections.pkl')
        if osp.isfile(frame_detections_file):
            with open(frame_detections_file, 'rb') as fid:
                alldets = pickle.load(fid)
        else:
            alldets = self.build_frame_prediction()
            try:
                with open(frame_detections_file, 'wb') as fid:
                    pickle.dump(alldets, fid, protocol=4)
            except:
                raise RuntimeError(
            'OverflowError: cannot serialize a bytes object lagger than 4 GiB')
        results = {}
        for ilabel, label in enumerate(self.labels):
            detections = alldets[alldets[:, 2] == ilabel, :]

            gt = defaultdict(list)
            for iv, v in enumerate(self.data_infos):
                tubes = self.gttubes[v]
                if ilabel not in tubes:
                    continue
                for tube in tubes[ilabel]:
                    for i in range(tube.shape[0]):
                        k = (iv, int(tube[i, 0]))
                        gt[k].append(tube[i, 1:5].tolist())
            
            for k in gt:
                gt[k] = np.array(gt[k])
            
            pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0
            fn = sum([g.shape[0] for g in gt.values()])
            fp = 0
            tp = 0

            for i, j in enumerate(np.argsort(-detections[:, 3])):
                k = (int(detections[j, 0]), int(detections[j, 1]))
                box = detections[j, 4:8]
                is_positive = False

                if k in gt:
                    ious = iou2d(gt[k], box)
                    amax = np.argmax(ious)

                    if ious[amax] >= self.iou_thresh:
                        is_positive = True
                        gt[k] = np.delete(gt[k], amax, 0)

                        if gt[k].size == 0:
                            del gt[k]
                
                if is_positive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                pr[i + 1, 0] = float(tp) / float(tp + fp)
                pr[i + 1, 1] = float(tp) / float(tp + fn)
            
            results[label] = pr
        
        ap = 100 * np.array([pr_to_ap(results[label]) for label in self.labels])
        frame_ap_result = np.mean(ap)
        logger.info('Task_{} frameAP_{}\n'.format('MOC_det', self.iou_thresh))
        logger.info('{:20s} {:8.2f}'.format('mAP', frame_ap_result))

    def frameAP_error(self, logger=None):
        eval_file = osp.join(
            self.inference_dir, 
            "frameAP{:g}ErrorAnalysis.pkl".format(self.iou_thresh))

        if osp.isfile(eval_file):
            with open(eval_file, 'rb') as fid:
                res = pickle.load(fid)
        else:
            # load per- frame detections
            frame_detections_file = osp.join(
                self.inference_dir, 'frame_detections.pkl')
            if osp.isfile(frame_detections_file):
                with open(frame_detections_file, 'rb') as fid:
                    alldets = pickle.load(fid)
            else:
                alldets = self.build_frame_prediction()
                with open(frame_detections_file, 'wb') as fid:
                    pickle.dump(alldets, fid)
            res = {}
            # alldets: list of numpy array with 
            # <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
            # compute AP for each class
            for ilabel, label in enumerate(self.labels):
                # detections of this class
                detections = alldets[alldets[:, 2] == ilabel, :]

                gt = {}
                othergt = {}
                labellist = {}

                # iv,v : 0 Basketball/v_Basketball_g01_c01
                for iv, v in enumerate(self.data_infos):
                    # tubes: dict {ilabel: (list of)<frame number> 
                    #              <x1> <y1> <x2> <y2>}
                    tubes = self.gttubes[v]
                    # labellist[iv]: label list for v
                    labellist[iv] = tubes.keys()

                    for il in tubes:
                        # tube: list of <frame number> <x1> <y1> <x2> <y2>
                        for tube in tubes[il]:
                            for i in range(tube.shape[0]):
                                # k: (video_index, frame_index)
                                k = (iv, int(tube[i, 0]))
                                if il == ilabel:
                                    if k not in gt:
                                        gt[k] = []
                                    gt[k].append(tube[i, 1:5].tolist())
                                else:
                                    if k not in othergt:
                                        othergt[k] = []
                                    othergt[k].append(tube[i, 1:5].tolist())

                for k in gt:
                    gt[k] = np.array(gt[k])
                for k in othergt:
                    othergt[k] = np.array(othergt[k])

                dupgt = copy.deepcopy(gt)

                # pr will be an array containing precision-recall values and 
                # 4 types of errors:
                # localization, classification, timing, others
                pr = np.empty((detections.shape[0] + 1, 6), dtype=np.float32)  
                # precision, recall
                pr[0, 0] = 1.0
                pr[0, 1:] = 0.0

                fn = sum([g.shape[0] for g in gt.values()])  # false negatives
                fp = 0  # false positives
                tp = 0  # true positives
                EL = 0  # localization errors
                EC = 0  # cls error: overlap >=0.5 with another object
                EO = 0  # other errors
                ET = 0  
                # timing error: video contains action but not at this frame

                for i, j in enumerate(np.argsort(-detections[:, 3])):
                    k = (int(detections[j, 0]), int(detections[j, 1]))
                    box = detections[j, 4:8]
                    ispositive = False

                    if k in dupgt:
                        if k in gt:
                            ious = iou2d(gt[k], box)
                            amax = np.argmax(ious)
                        if k in gt and ious[amax] >= self.iou_thresh:
                            ispositive = True
                            gt[k] = np.delete(gt[k], amax, 0)
                            if gt[k].size == 0:
                                del gt[k]
                        else:
                            EL += 1

                    elif k in othergt:
                        ious = iou2d(othergt[k], box)
                        if np.max(ious) >= self.iou_thresh:
                            EC += 1
                        else:
                            EO += 1
                    elif ilabel in labellist[k[0]]:
                        ET += 1
                    else:
                        EO += 1
                    if ispositive:
                        tp += 1
                        fn -= 1
                    else:
                        fp += 1

                    pr[i + 1, 0] = float(tp) / float(tp + fp)  # precision
                    pr[i + 1, 1] = float(tp) / float(tp + fn)  # recall
                    pr[i + 1, 2] = float(EL) / float(tp + fp)
                    pr[i + 1, 3] = float(EC) / float(tp + fp)
                    pr[i + 1, 4] = float(ET) / float(tp + fp)
                    pr[i + 1, 5] = float(EO) / float(tp + fp)

                res[label] = pr

            # save results
            with open(eval_file, 'wb') as fid:
                pickle.dump(res, fid)

        # display results
        AP = 100 * np.array(
            [pr_to_ap(res[label][:, [0, 1]]) for label in self.labels])
        othersap = [100 * np.array(
            [pr_to_ap(res[label][:, [j, 1]]) for label in self.labels]) 
            for j in range(2, 6)]

        EL = othersap[0]
        EC = othersap[1]
        ET = othersap[2]
        EO = othersap[3]
        # missed detections = 1 - recall
        EM = 100 - 100 * np.array([res[label][-1, 1] for label in self.labels])

        LIST = [AP, EL, EC, ET, EO, EM]

        logger.info('Error Analysis: \n')

        logger.info("{:20s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(
            'label', '   AP   ', '  Loc.  ', '  Cls.  ', '  Time  ', ' Other ', 
            ' missed \n'))
        for il, label in enumerate(self.labels):
            logger.info("{:20s} ".format(label) + \
                " ".join(["{:8.2f}".format(L[il]) for L in LIST]))

        logger.info("\n{:20s} ".format("mean") + \
            " ".join(["{:8.2f}".format(np.mean(L)) for L in LIST]))

    def videoAP(self, logger=None):
        alldets = {ilabel: [] for ilabel in range(len(self.labels))}
        for v in self.data_infos:
            tubename = osp.join(self.inference_dir, v + '_tubes.pkl')
            if not osp.isfile(tubename):
                raise ValueError('Missing extracted tubes '+ tubename)
            with open(tubename, 'rb') as fid:
                tubes = pickle.load(fid)
            for ilabel in range(len(self.labels)):
                ltubes = tubes[ilabel]
                idx = nms3dt(ltubes, 0.3)
                alldets[ilabel] += [
                    (v, ltubes[i][1], ltubes[i][0]) for i in idx]
        res = {}
        for ilabel in range(len(self.labels)):
            detections = alldets[ilabel]
            gt = {}
            for v in self.data_infos:
                tubes = self.gttubes[v]
                if ilabel not in tubes:
                    continue
                gt[v] = tubes[ilabel]
                if len(gt[v]) == 0:
                    del gt[v]
            
            pr = np.empty((len(detections) + 1, 2), dtype=np.float32)
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0
            fn = sum([len(g) for g in gt.values()])
            fp = 0
            tp = 0
            for i, j in enumerate(
                np.argsort(-np.array( [dd[1] for dd in detections]))):
                v, score, tube = detections[i]
                is_positive = False

                if v in gt:
                    ious = [iou3dt(g, tube) for g in gt[v]]
                    amax = np.argmax(ious)
                    if ious[amax] >= self.iou_thresh:
                        is_positive = True
                        del gt[v][amax]
                        if len(gt[v]) == 0:
                            del gt[v]
                
                if is_positive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                
                pr[i + 1, 0] = float(tp) / float(tp + fp)
                pr[i + 1, 1] = float(tp) / float(tp + fn)
            
            res[self.labels[ilabel]] = pr
        ap = 100 * np.array([pr_to_ap(res[label]) for label in self.labels])
        video_ap_result = np.mean(ap)
        logger.info('Task_{} VideoAP_{}\n'.format('MOC_det', self.iou_thresh))
        logger.info('\n{:20s} {:8.2f}\n\n'.format('mAP', video_ap_result))

    def videoAP_050_095(self, logger=None):
        ap = 0
        for i in range(10):
            self.iou_thresh = 0.5 + 0.05 * i
            ap += self.videoAP()
        ap = ap / 10.0
        logger.info('\nTask_{} VideoAP_0.05:0.95 \n'.format('MOC_det'))
        logger.info('\n{:20s} {:8.2f}\n\n'.format('mAP', ap))