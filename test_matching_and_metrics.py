import random
import numpy as np
import statistics
from typing import Tuple
import unittest
from matching_ops import match_strict_iou, match_centroid_rule
from metrics_ops import ImagePQMetric, ImageSeparatedMetrics, add_hd_of_matches


def nary_from_cells(cells: list, offset: int = 0):
    base_image = np.zeros((200, 200, 2))
    possible_ids = [i for i in range(1, 1000)]
    random.shuffle(possible_ids)

    for y in range(10):
        for x in range(10):
            idc = y * 10 + x
            if cells[idc] != 0:
                base_image[5 + y * 20:15 + y * 20, offset + 5 + x * 20:offset + 15 + x * 20, 0] = possible_ids.pop()
                base_image[5 + y * 20:15 + y * 20, offset + 5 + x * 20:offset + 15 + x * 20, 1] = cells[idc]

    return base_image


def generate_base_nary(obj_per_class: Tuple[int, int, int, int]):
    """Generates a 200x200 n-ary mask with the required object. Each object will be a 10x10 square,
    within a 20x20 cell.."""
    n_cells = 10 * 10
    cells = []
    for classid, c in enumerate(obj_per_class):
        cells += [classid+1 for _ in range(c)]

    while len(cells) < n_cells:
        cells.append(0)
    random.shuffle(cells)

    base_image = nary_from_cells(cells)

    return base_image, cells


def generate_prediction(nary_base: np.array, cells: list, cm: np.array, offset: int):
    """Generate fake prediction with given CM"""
    new_cells = [c for c in cells] # copy cells
    has_flipped = [False for _ in cells] # so we don't re-flip a flipped cell
    # print(cells)

    # Check that we have enough room for the false positives
    assert sum([c == 0 for c in cells]) > cm[0, :].sum()

    for gt_classid in range(5):
        for pred_classid in range(5):
            if gt_classid == pred_classid:
                continue
            to_flip = cm[gt_classid, pred_classid]
            # print(f"Need to flip {to_flip} from class {gt_classid} to class {pred_classid}")
            # find a cell to flip:
            for cellid, c in enumerate(new_cells):
                if to_flip == 0:
                    break
                if c == gt_classid and not has_flipped[cellid]:
                    new_cells[cellid] = pred_classid  # flip
                    has_flipped[cellid] = True
                    # print(f"Flipped cell {cellid} from class {cells[cellid]} to class {new_cells[cellid]}")
                    to_flip -= 1

    # print(new_cells)
    return nary_from_cells(new_cells, offset)


def generate_nary_and_prediction(cm: np.array, offset: int):
    """Generate nary gt & fake prediction."""
    assert cm.shape == (5, 5)
    assert cm.sum() < 100

    obj_per_class = cm[1:].sum(axis=1)
    nary_base, cells = generate_base_nary(obj_per_class)
    pred = generate_prediction(nary_base, cells, cm, offset)
    return nary_base, pred


class TestGenerator(unittest.TestCase):

    def test_generator(self):
        """Test that our fake data generator produces the right number of objects with the right size"""
        test_nary, cells = generate_base_nary((15, 12, 6, 3))
        self.assertEqual(len(np.unique(test_nary[..., 0])), 15+12+6+3+1)
        self.assertEqual((test_nary[..., 0] > 0).sum(), (15+12+6+3)*100)
        self.assertEqual(len(np.unique(test_nary[..., 1])), 5)
        self.assertEqual(np.sum(test_nary[..., 0] > 0), (15+12+6+3)*100)

    def test_prediction_easy(self):
        """Test that our fake data generator correctly generates prediction n-ary masks
        when all predictions are correct"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 0)
        self.assertEqual(len(np.unique(nary[..., 0])), 21)
        self.assertEqual(len(np.unique(nary[..., 1])), 5)
        diff = np.array(nary[..., 1] != pred[..., 1])
        self.assertEqual(diff.sum(), 0)

    def test_prediction_with_fp(self):
        """Test that our fake data generator correctly generates prediction n-ary masks
        when there are false positives"""
        conf_mat = np.array([[0, 1, 2, 3, 4],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 0)
        self.assertEqual((pred[..., 1] == 1).sum(), 6*100)
        self.assertEqual((pred[..., 1] == 2).sum(), 7*100)
        self.assertEqual((pred[..., 1] == 3).sum(), 8*100)
        self.assertEqual((pred[..., 1] == 4).sum(), 9*100)

    def test_prediction_with_fn(self):
        """Test that our fake data generator correctly generates prediction n-ary masks
        when there are false negatives"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [1, 4, 0, 0, 0],
                             [2, 0, 3, 0, 0],
                             [3, 0, 0, 2, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 0)
        self.assertEqual((pred[..., 1] == 1).sum(), 4*100)
        self.assertEqual((pred[..., 1] == 2).sum(), 3*100)
        self.assertEqual((pred[..., 1] == 3).sum(), 2*100)
        self.assertEqual((pred[..., 1] == 4).sum(), 1*100)

    def test_prediction_with_conf(self):
        """Test that our fake data generator correctly generates prediction n-ary masks
        when there are inter-class confusions"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 4, 1, 0, 0],
                             [0, 2, 3, 0, 0],
                             [0, 0, 0, 2, 3],
                             [0, 0, 4, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 0)
        self.assertEqual((pred[..., 1] == 1).sum(), 6*100)
        self.assertEqual((pred[..., 1] == 2).sum(), 8*100)
        self.assertEqual((pred[..., 1] == 3).sum(), 2*100)
        self.assertEqual((pred[..., 1] == 4).sum(), 4*100)

    def test_prediction_with_iou(self):
        """Test that our fake data generator correctly generates prediction n-ary masks
        when there is an offset (assert that there is the right amount of pixels outside of overlap region)"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        self.assertEqual(((pred[..., 1] > 0) & (nary[..., 1] == 0)).sum(), 20*10)

    def test_match_iou_perfect(self):
        """Test that the strict-iou matching correctly finds all matches if there is just a 1px offset"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_strict_iou(nary, pred)
        self.assertEqual(len(matches.matches), 20)

    def test_match_iou_all_misses(self):
        """Test that the strict-iou matching correctly finds no matches with a 5px offset (IoU = 5/15 < 0.5)"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 5)
        matches = match_strict_iou(nary, pred)
        self.assertEqual(len(matches.matches), 0)

    def test_match_centroid_perfect(self):
        """Test that the centroid-rule matching correctly finds all matches if there is just a 1px offset"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        self.assertEqual(len(matches.matches), 20)

    def test_match_centroid_all_misses(self):
        """Test that the strict-iou matching correctly finds no matches with a 6px offset"""
        conf_mat = np.array([[0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 0],
                             [0, 0, 5, 0, 0],
                             [0, 0, 0, 5, 0],
                             [0, 0, 0, 0, 5]])
        nary, pred = generate_nary_and_prediction(conf_mat, 6)
        matches = match_centroid_rule(nary, pred)
        self.assertEqual(len(matches.matches), 0)

    def test_match_with_fp_fn(self):
        """Test that the centroid matching correctly finds matches when there are FPs and FNs"""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 0, 0, 0],
                             [3, 0, 2, 0, 0],
                             [0, 0, 0, 5, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        self.assertEqual(len(matches.matches), 13)
        matches = match_strict_iou(nary, pred)
        self.assertEqual(len(matches.matches), 13)

    def test_match_with_conf(self):
        """Test that the centroid matching correctly finds matches when classes are not the same"""
        conf_mat = np.array([[0, 0, 0, 1, 0],
                             [0, 4, 1, 0, 0],
                             [0, 2, 3, 0, 0],
                             [2, 0, 0, 2, 3],
                             [0, 0, 4, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        self.assertEqual(len(matches.matches), 20)
        matches = match_strict_iou(nary, pred)
        self.assertEqual(len(matches.matches), 20)

    def test_pq_det(self):
        """Test that the PQ computation correctly counts TP/FP/FNs"""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 0, 0, 0],
                             [3, 0, 2, 0, 0],
                             [0, 0, 0, 5, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        pq = ImagePQMetric()
        pq.addMatches(matches)
        self.assertEqual(pq.TPs[0], 5)
        self.assertEqual(pq.TPs[1], 2)
        self.assertEqual(pq.TPs[2], 5)
        self.assertEqual(pq.TPs[3], 1)
        self.assertEqual(pq.FPs[0], 1)
        self.assertEqual(pq.FPs[1], 0)
        self.assertEqual(pq.FPs[2], 2)
        self.assertEqual(pq.FPs[3], 0)
        self.assertEqual(pq.FNs[0], 0)
        self.assertEqual(pq.FNs[1], 3)
        self.assertEqual(pq.FNs[2], 0)
        self.assertEqual(pq.FNs[3], 4)

    def test_pq_pqc(self):
        """Test that the PQc are correctly computed"""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 0, 0, 0],
                             [3, 0, 2, 0, 0],
                             [0, 0, 0, 5, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        pq = ImagePQMetric()
        pq.addMatches(matches)
        pqc = pq.compute_PQc()
        avg_iou = 9/11
        self.assertAlmostEqual(pqc[0], avg_iou * (5/5.5))
        self.assertAlmostEqual(pqc[1], avg_iou * (2/3.5))
        self.assertAlmostEqual(pqc[2], avg_iou * (5/6))
        self.assertAlmostEqual(pqc[3], avg_iou * (1/3))
        self.assertAlmostEqual(pq.compute_PQ(), avg_iou * ((1/3)+(5/6)+(2/3.5)+(5/5.5))/4)

    def test_dq_sq(self):
        """Test that the DQs and SQs are correctly computed"""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 0, 0, 0],
                             [3, 0, 2, 0, 0],
                             [0, 0, 0, 5, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        pq = ImagePQMetric()
        pq.addMatches(matches)

        dq_sq = pq.compute_DQ_SQ()
        avg_iou = 9/11
        self.assertAlmostEqual(dq_sq[1], avg_iou)
        self.assertAlmostEqual(dq_sq[0], ((1/3)+(5/6)+(2/3.5)+(5/5.5))/4)

    def test_sm_detect(self):
        """Test that the Detection metrics of the Separated Metrics are correctly computed"""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 0, 0, 0],
                             [3, 0, 2, 0, 0],
                             [0, 0, 0, 5, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        sm = ImageSeparatedMetrics()
        sm.addMatches(matches)
        results = sm.compute_detection_scores()
        self.assertAlmostEqual(results['acc'], 13/(13+3+7))
        self.assertAlmostEqual(results['prec'], 13/(13+3))
        self.assertAlmostEqual(results['rec'], 13/(13+7))
        self.assertAlmostEqual(results['f1'], 13/(13+1.5+3.5))

    def test_sm_classif_perfect(self):
        """Test that the Classification metrics of the Separated Metrics are correctly computed when all classif
        are correct (and there are some detection errors)"""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 0, 0, 0],
                             [3, 0, 2, 0, 0],
                             [0, 0, 0, 5, 0],
                             [4, 0, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        sm = ImageSeparatedMetrics()
        sm.addMatches(matches)
        results = sm.compute_classification_scores()
        self.assertAlmostEqual(results['overall_acc'], 1)

    def test_sm_classif_imperfect(self):
        """Test that the Classification metrics of the Separated Metrics are correctly computed in the presence of
        classification errors."""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 2, 0, 0],
                             [2, 1, 2, 0, 0],
                             [0, 1, 0, 5, 2],
                             [0, 1, 0, 0, 1]])
        ncm = np.array([[5/7, 2/7, 0, 0],
                        [1/3, 2/3, 0, 0],
                        [1/8, 0, 5/8, 2/8],
                        [1/2, 0, 0, 1/2]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        sm = ImageSeparatedMetrics()
        sm.addMatches(matches)
        results = sm.compute_classification_scores()
        self.assertTrue(np.allclose(ncm, results['NCM']))
        self.assertAlmostEqual(results['class_f1'][0], (5/7)/(5/7+1/7+1/6+1/16+1/4))
        self.assertAlmostEqual(results['class_prec'][0], (5/7)/(5/7+1/3+1/8+1/2))
        self.assertAlmostEqual(results['class_rec'][0], (5/7)/(5/7+2/7))

    def test_sm_seg_iou(self):
        """Test that the segmentation avg-IoU is correctly computed."""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 2, 0, 0],
                             [2, 1, 2, 0, 0],
                             [0, 1, 0, 5, 2],
                             [0, 1, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        sm = ImageSeparatedMetrics()
        sm.addMatches(matches)
        results = sm.compute_segmentation_scores()
        self.assertAlmostEqual(results['aIoU'], 9/11)

    def test_sm_seg_hd(self):
        """Test that the segmentation HD is correctly computed."""
        conf_mat = np.array([[0, 1, 0, 2, 0],
                             [0, 5, 2, 0, 0],
                             [2, 1, 2, 0, 0],
                             [0, 1, 0, 5, 2],
                             [0, 1, 0, 0, 1]])
        nary, pred = generate_nary_and_prediction(conf_mat, 1)
        matches = match_centroid_rule(nary, pred)
        hd = []
        hd_class = [[] for _ in range(4)]
        add_hd_of_matches(nary[..., 0].astype('int'), pred[..., 0].astype('int'), matches, hd, hd_class)
        self.assertAlmostEqual(statistics.mean(hd), 1)


if __name__ == "__main__":
    unittest.main()
