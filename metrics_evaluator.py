#  @Date    : 2023-03-19
#  @Editor  : Mikhail Kiselyov
#  @E-mail  : kiselev.0353@gmail.com
#  Provided as is

import os
from pathlib import Path
import json

from sklearn.metrics import jaccard_score, f1_score, accuracy_score, mean_absolute_error, balanced_accuracy_score
from PIL import Image
import numpy as np


class MetricsEvaluator:
    def __init__(self,
                 prediction_dir: Path,
                 ground_truth_dir: Path,
                 output_path: Path):
        self.prediction_dir = prediction_dir
        self.ground_truth_dir = ground_truth_dir
        self.output_path = output_path

        self.pred_files = os.listdir(prediction_dir)
        self.gt_files = os.listdir(ground_truth_dir)
        assert all([pred_name == gt_name for pred_name, gt_name in zip(self.pred_files, self.gt_files)])

    def _evaluate_pair(self, pred_file, gt_file):
        pred = np.array(Image.open(self.prediction_dir / pred_file).convert('1')).flatten()
        gt = np.array(Image.open(self.ground_truth_dir / gt_file).convert('1')).flatten()
        return {
            'iou': jaccard_score(gt, pred),
            'f1_score': f1_score(gt, pred),
            'accuracy': accuracy_score(gt, pred),
            'mean_absolute_error': mean_absolute_error(gt.astype(np.int8), pred.astype(np.int8)),
            'balanced_error_rate': 1 - balanced_accuracy_score(gt, pred)
        }

    def evaluate(self):
        result = {}

        for pred_file, gt_file in zip(self.pred_files, self.gt_files):
            result[pred_file] = self._evaluate_pair(pred_file, gt_file)

        with open(self.output_path, 'w') as output_file:
            json.dump(result, output_file)
