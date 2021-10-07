import pickle
import os
import numpy as np
from skimage.measure import regionprops


class Match:
    """Stores the informations of a matching object pair"""

    def __init__(self, gt_idx, pred_idx, gt_class, pred_class, iou):
        self.gt_idx = int(gt_idx)
        self.pred_idx = int(pred_idx)
        self.gt_class = int(gt_class)
        self.pred_class = int(pred_class)
        self.iou = iou

    def __str__(self):
        return f"Gt obj {self.gt_idx} (class {self.gt_class}) with " \
               f"pred obj {self.pred_idx} (class {self.pred_class})" \
               f" - IoU = {self.iou:.3f}"


class ImageMatches:
    """Stores the information about all the matches and non-matches in an image."""

    def __init__(self, gt_idxs_class: dict, pred_idxs_class: dict):
        self.gt_idxs_class = gt_idxs_class
        self.pred_idxs_class = pred_idxs_class
        self.matches = []

    def add(self, match):
        self.matches.append(match)


def match_strict_iou(gt: np.array, pred: np.array) -> ImageMatches:
    """Find matching pairs of objects between gt & pred mask using the iou>0.5 criterion.

    Returns ImageMatches which contains list of matches & list of existing idxs for easy PQ recomputation."""
    nonambiguous_mask = gt[..., 1] != 5
    gt_uid = gt[..., 0] * nonambiguous_mask
    pred_uid = pred[..., 0] * nonambiguous_mask

    gt_idxs = np.unique(gt_uid)
    gt_idxs = gt_idxs[gt_idxs > 0]
    pred_idxs = np.unique(pred_uid)
    pred_idxs = pred_idxs[pred_idxs > 0]

    # prepare quick reference from uid to class
    gt_idxs_class = {}
    for gt_idx in gt_idxs:
        gt_idxs_class[gt_idx] = gt[gt_uid == gt_idx, 1].max()
    pred_idxs_class = {}
    for pred_idx in pred_idxs:
        pred_idxs_class[pred_idx] = pred[pred_uid == pred_idx, 1].max()

    matched_instances = ImageMatches(gt_idxs_class, pred_idxs_class)

    # Find matched instances and add it to the list
    for gt_idx in gt_idxs:
        gt_obj_mask = gt_uid == gt_idx
        pred_in_obj = gt_obj_mask * pred_uid

        for pred_idx in [idx for idx in np.unique(pred_in_obj) if idx > 0]:
            pred_obj_mask = pred_uid == pred_idx
            intersection = (gt_obj_mask & pred_obj_mask).sum()
            union = (gt_obj_mask | pred_obj_mask).sum()
            IOU = intersection / union
            if IOU > 0.5:
                matched_instances.add(Match(gt_idx, pred_idx, gt_idxs_class[gt_idx], pred_idxs_class[pred_idx], IOU))

    return matched_instances


def match_centroid_rule(gt: np.array, pred: np.array) -> ImageMatches:
    """Find matching pairs of objects between gt & pred mask using the
    "best iou w/ centroid inside gt object" criterion"""
    nonambiguous_mask = gt[..., 1] != 5
    gt_uid = gt[..., 0] * nonambiguous_mask
    pred_uid = pred[..., 0] * nonambiguous_mask

    gt_idxs = np.unique(gt_uid)
    gt_idxs = gt_idxs[gt_idxs > 0]
    pred_idxs = np.unique(pred_uid)
    pred_idxs = pred_idxs[pred_idxs > 0]

    gt_idxs_class = {}
    for gt_idx in gt_idxs:
        gt_idxs_class[gt_idx] = gt[gt_uid == gt_idx, 1].max()
    pred_idxs_class = {}
    for pred_idx in pred_idxs:
        pred_idxs_class[pred_idx] = pred[pred_uid == pred_idx, 1].max()

    matched_instances = ImageMatches(gt_idxs_class, pred_idxs_class)

    # Find matched instances and add it to the list
    for gt_idx in gt_idxs:
        gt_obj_mask = gt_uid == gt_idx
        pred_in_obj = gt_obj_mask * pred_uid

        best_pred_idx = -1
        best_match_IOU = 0

        for pred_idx in [idx for idx in np.unique(pred_in_obj) if idx > 0]:
            pred_obj_mask = pred_uid == pred_idx

            intersection = (gt_obj_mask & pred_obj_mask).sum()
            union = (gt_obj_mask | pred_obj_mask).sum()
            IOU = intersection / union
            if IOU > best_match_IOU:
                best_match_IOU = IOU
                best_pred_idx = pred_idx

        if best_pred_idx > 0:
            pred_centroid = np.round(regionprops((pred_uid == best_pred_idx).astype('uint8'))[0].centroid).astype('int')
            if gt_uid[pred_centroid[0], pred_centroid[1]] == gt_idx:
                matched_instances.add(
                    Match(gt_idx, best_pred_idx, gt_idxs_class[gt_idx], pred_idxs_class[best_pred_idx], best_match_IOU)
                )

    return matched_instances


def compute_all_matches_for(team_dir, team_masks: dict, gt_masks: dict, nary_mode: str, match_mode: str) -> None:
    """Compute all matches for all images in the dataset, precomputed dictionary of nary-masks. 

    Store the results as pickled dictionaries per-image"""
    match_fn = {'strict-iou': match_strict_iou, 'centroid': match_centroid_rule}
    if match_mode not in match_fn:
        raise ValueError(f"Unknown match mode: {match_mode} (expected: {match_fn.keys()})")

    for idp, patient in enumerate(gt_masks):
        print(f"Patient {idp + 1}", end='\r')
        patient_dir = os.path.join(team_dir, patient)
        for im in gt_masks[patient]:
            gt = gt_masks[patient][im]
            pred = team_masks[nary_mode][patient][im]
            matches = match_fn[match_mode](gt, pred)

            with open(os.path.join(patient_dir, f'{im}_{nary_mode}_nary_{match_mode}.pkl'), 'wb') as fp:
                pickle.dump(matches, fp)


def compute_all_matches(teams_dir: str, ccgt_dir: str, teams_masks: dict, ccgt_masks: dict, gt_nary_masks: dict):
    teams = os.listdir(teams_dir)
    for idt, team in enumerate(teams):
        print(f"Computing all matching pairs of objects for team {idt+1}")
        team_dir = os.path.join(teams_dir, team)
        compute_all_matches_for(team_dir, teams_masks[team], gt_nary_masks, 'border-removed', 'strict-iou')
        compute_all_matches_for(team_dir, teams_masks[team], gt_nary_masks, 'border-dilated', 'strict-iou')
        compute_all_matches_for(team_dir, teams_masks[team], gt_nary_masks, 'border-removed', 'centroid')
        compute_all_matches_for(team_dir, teams_masks[team], gt_nary_masks, 'border-dilated', 'centroid')

    print(f"Computing all matching pairs of objects for the color-coded ground truth.")
    compute_all_matches_for(ccgt_dir, ccgt_masks, gt_nary_masks, 'border-removed', 'strict-iou')
    compute_all_matches_for(ccgt_dir, ccgt_masks, gt_nary_masks, 'border-dilated', 'strict-iou')
    compute_all_matches_for(ccgt_dir, ccgt_masks, gt_nary_masks, 'border-removed', 'centroid')
    compute_all_matches_for(ccgt_dir, ccgt_masks, gt_nary_masks, 'border-dilated', 'centroid')
