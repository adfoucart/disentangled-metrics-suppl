import os
import statistics
import pickle
import numpy as np
from skimage.morphology import erosion, disk
from skimage.metrics import hausdorff_distance
from typing import Tuple, List
from matching_ops import ImageMatches
from abc import ABC, abstractmethod


class ImageBaseMetric(ABC):
    """Abstract class for Image metrics"""

    def addAll(self, pkl_files: list) -> None:
        """Add .pkl files for the sub-images matches"""
        for pkl in pkl_files:
            self.add(pkl)

    def add(self, pkl_file: str) -> None:
        """Add the matches precomputed from a .pkl file"""
        with open(pkl_file, 'rb') as fp:
            matches = pickle.load(fp)

        self.addMatches(matches)

    @abstractmethod
    def addMatches(self, matches: ImageMatches) -> None:
        """Add the matches from a single sub-image to the aggregate components of the metric"""
        pass


class ImagePQMetric(ImageBaseMetric):
    """Compute & aggregate image PQ metrics from Pickle files"""

    def __init__(self):
        super().__init__()
        self.TPs = [0 for _ in range(4)]
        self.FPs = [0 for _ in range(4)]
        self.FNs = [0 for _ in range(4)]
        self.IoUs = [[] for _ in range(4)]

    def addMatches(self, matches: ImageMatches) -> None:
        """Add the matches from a single sub-image to the aggregate components of the metric"""
        TP_im = [0 for _ in range(4)]

        gt_count = [0 for _ in range(4)]
        pred_count = [0 for _ in range(4)]
        for idx, cl in matches.gt_idxs_class.items():
            gt_count[int(cl) - 1] += 1
        for idx, cl in matches.pred_idxs_class.items():
            pred_count[int(cl) - 1] += 1

        for match in matches.matches:
            if match.gt_class == match.pred_class:
                TP_im[match.gt_class - 1] += 1
                self.IoUs[match.gt_class - 1] += [match.iou]

        for i in range(4):
            self.TPs[i] += TP_im[i]
            self.FPs[i] += pred_count[i] - TP_im[i]
            self.FNs[i] += gt_count[i] - TP_im[i]

    def compute_PQc(self) -> list:
        """Return PQ per class if we want the detail"""
        PQc = [None for _ in range(4)]
        for cl in range(4):
            if self.TPs[cl] + self.FPs[cl] + self.FNs[cl] == 0:
                continue
            PQc[cl] = sum(self.IoUs[cl]) / (self.TPs[cl] + 0.5 * self.FPs[cl] + 0.5 * self.FNs[cl])

        return PQc

    def compute_PQ(self) -> float:
        """Compute avg PQ for the classes that are present in gt or pred"""
        PQc = []
        for cl in range(4):
            if self.TPs[cl] + self.FPs[cl] + self.FNs[cl] == 0:
                continue
            PQc.append(sum(self.IoUs[cl]) / (self.TPs[cl] + 0.5 * self.FPs[cl] + 0.5 * self.FNs[cl]))

        return statistics.mean(PQc)

    def compute_DQ_SQ(self) -> Tuple[float, float]:
        DQc = []
        SQc = []
        for cl in range(4):
            if self.TPs[cl] + self.FPs[cl] + self.FNs[cl] > 0:
                DQc.append(self.TPs[cl] / (self.TPs[cl] + 0.5 * self.FPs[cl] + 0.5 * self.FNs[cl]))
            if self.TPs[cl] > 0:
                SQc.append(sum(self.IoUs[cl]) / self.TPs[cl])

        return statistics.mean(DQc), statistics.mean(SQc)


class ImageSeparatedMetrics(ImageBaseMetric):
    """Compute & aggregate image classification/segmentation/detection metrics from Pickle files"""

    def __init__(self):
        super().__init__()
        self.cm = np.zeros((5, 5))  # main classification/detection confusion matrix
        self.ious = [[] for _ in range(4)]  # IOU for segmentation score

    def addMatches(self, matches: ImageMatches) -> None:
        """Add the matches from a single sub-image to the aggregate components of the metric"""
        gt_count = [0 for _ in range(4)]
        pred_count = [0 for _ in range(4)]
        im_cm = np.zeros((5, 5))
        for idx, cl in matches.gt_idxs_class.items():
            gt_count[int(cl) - 1] += 1
        for idx, cl in matches.pred_idxs_class.items():
            pred_count[int(cl) - 1] += 1

        for match in matches.matches:
            im_cm[match.gt_class, match.pred_class] += 1
            self.ious[match.gt_class - 1].append(match.iou)

        self.cm += im_cm

        for i in range(4):
            # add the missed gt & pred objects to the cm
            self.cm[0, i + 1] += pred_count[i] - im_cm[1:, i + 1].sum()  # predicted objects that have no match -> FPs
            self.cm[i + 1, 0] += gt_count[i] - im_cm[i + 1, 1:].sum()  # gt objects that have no match -> FNs

    def compute_detection_scores(self) -> dict:
        """Detection scores:

        Overall Acc/Precision/Recall/F1
        """
        results = {}

        TP = self.cm[1:, 1:].sum()  # objects that were found, even if the class is wrong
        FP = self.cm[0, :].sum()  # objects that were found even though there is no gt match
        FN = self.cm[:, 0].sum()  # gt objects that were not found at all

        results['acc'] = TP / (TP + FP + FN)
        results['prec'] = TP / (TP + FP)
        results['rec'] = TP / (TP + FN)
        results['f1'] = TP / (TP + 0.5 * FP + 0.5 * FN)

        return results

    def compute_classification_scores(self) -> dict:
        """Classification scores:
        
        Looking at the normalized confusion matrix only for the matched objects (cm[1:,1:]).
        * Overall accuracy
        * Per-class acc / prec / rec / f1
        """
        cm_class = self.cm[1:, 1:]
        cm_class_norm = np.zeros_like(cm_class)
        for classid in range(4):
            cm_class_norm[classid, :] = cm_class[classid, :] / np.maximum(1, cm_class[classid, :].sum())

        results = {'overall_acc': cm_class_norm.diagonal().sum() / cm_class_norm.sum(),
                   'class_prec': [None for _ in range(4)],
                   'class_rec': [None for _ in range(4)],
                   'class_f1': [None for _ in range(4)]}
        for i in range(4):
            TP = cm_class_norm[i, i]
            FP = cm_class_norm[:, i].sum() - TP
            FN = cm_class_norm[i, :].sum() - TP
            if TP + FP + FN > 0:
                results['class_f1'][i] = TP / (TP + 0.5 * FP + 0.5 * FN)
            if TP + FP > 0:
                results['class_prec'][i] = TP / (TP + FP)
            if TP + FN > 0:
                results['class_rec'][i] = TP / (TP + FN)
        results['NCM'] = cm_class_norm

        return results

    def compute_segmentation_scores(self) -> dict:
        """Segmentation scores:
        
        Looking at all the ious of a given gt class
        (independantly of whether the matching object was correctly classified)
        -> avg IoU per class
        -> overall avg IoU = avg-avg-IoU
        """

        results = {'IoUc': [None for _ in range(4)], 'aIoU': 0}
        for i in range(4):
            if len(self.ious[i]) > 0:
                results['IoUc'][i] = statistics.mean(self.ious[i])

        results['aIoU'] = statistics.mean([iouc for iouc in results['IoUc'] if iouc is not None])

        return results


def getTeamPQs(team_dir: str, ext: str) -> List[float]:
    """Get list of team's patient PQ for a given .pkl extension"""
    patients = [f for f in os.listdir(team_dir) if f.startswith("TCGA")]
    PQs = []
    for patient in patients:
        patient_dir = os.path.join(team_dir, patient)
        pkls = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(f'{ext}.pkl')]
        PQ_patient = ImagePQMetric()
        PQ_patient.addAll(pkls)
        PQs.append(PQ_patient.compute_PQ())
    return PQs


def getTeamDQSQs(team_dir: str, ext: str) -> List[Tuple[float, float]]:
    """Get list of team's patient DQ/SQ for a given .pkl extension"""
    patients = [f for f in os.listdir(team_dir) if f.startswith("TCGA")]
    DQSQs = []
    for patient in patients:
        patient_dir = os.path.join(team_dir, patient)
        pkls = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(f'{ext}.pkl')]
        PQ_patient = ImagePQMetric()
        PQ_patient.addAll(pkls)
        DQSQs.append(PQ_patient.compute_DQ_SQ())
    return DQSQs


def getTeamHausdorffDistances(team_dir: str, team_masks: dict, gt_nary_masks: dict, ext: str):
    hds = []
    hd_class = [[] for _ in range(4)]
    for idp, patient in enumerate(team_masks):
        print(f"Patient {idp + 1}/{len(team_masks)}", end="\r")
        patient_dir = os.path.join(team_dir, patient)
        patient_hds = []
        patient_hds_per_class = [[] for _ in range(4)]

        for im in team_masks[patient]:
            pred = team_masks[patient][im][..., 0].astype('int')
            gt = gt_nary_masks[patient][im][..., 0].astype('int')
            with open(os.path.join(patient_dir, f"{im}{ext}.pkl"), "rb") as fp:
                matches = pickle.load(fp)

            pred_contours = [pred == i - erosion(pred == i, disk(1)) for i in np.unique(pred) if i > 0]
            gt_contours = [gt == i - erosion(gt == i, disk(1)) for i in np.unique(gt) if i > 0]
            for match in matches.matches:
                hd = hausdorff_distance(gt_contours[match.gt_idx - 1], pred_contours[match.pred_idx - 1])
                patient_hds.append(hd)
                patient_hds_per_class[match.gt_class - 1].append(hd)

        hds.append(statistics.mean(patient_hds))
        for classid in range(4):
            if len(patient_hds_per_class[classid]) > 0:
                hd_class[classid].append(statistics.mean(patient_hds_per_class[classid]))

    return hds, hd_class


def compute_all_hausdorff_distances(teams_dir: str, teams_masks: dict, gt_nary_masks: dict, methods: dict, save_dir: str):
    all_hds = {}
    all_hds_per_class = {}
    teams = os.listdir(teams_dir)
    for idt, team in enumerate(teams):
        print(f"Team {idt + 1}")
        team_dir = os.path.join(teams_dir, team)
        all_hds[team] = {}
        all_hds_per_class[team] = {}
        for ext in methods:
            print(methods[ext])
            nary = ext.split('_')[1]

            hds, hds_class = getTeamHausdorffDistances(team_dir, teams_masks[team][nary], gt_nary_masks, ext)

            all_hds[team][ext] = hds
            all_hds_per_class[team][ext] = hds_class

    with open(os.path.join(save_dir, "all_hds.pkl"), "wb") as fp:
        pickle.dump(all_hds, fp)
    with open(os.path.join(save_dir, "all_hds_per_class.pkl"), "wb") as fp:
        pickle.dump(all_hds_per_class, fp)
