"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

from ...core import register
from ...misc import dist_utils

__all__ = [
    "CocoEvaluator",
]


@register()
class CocoEvaluator(FasterCocoEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_results = {iou_type: [] for iou_type in self.iou_types}

    def update(self, predictions):
        for iou_type in self.iou_types:
            self.coco_results[iou_type].extend(self.prepare(predictions, iou_type))
        super().update(predictions)

    def synchronize_between_processes(self):
        super().synchronize_between_processes()
        for iou_type in self.iou_types:
            by_image = {}
            for rank_results in dist_utils.all_gather(self.coco_results[iou_type]):
                rank_by_image = {}
                for result in rank_results:
                    rank_by_image.setdefault(result["image_id"], []).append(result)
                for image_id, results in rank_by_image.items():
                    by_image.setdefault(image_id, results)
            self.coco_results[iou_type] = [
                result for image_id in sorted(by_image) for result in by_image[image_id]
            ]
