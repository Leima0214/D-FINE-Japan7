#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets
from tidecv.data import Data


def evaluate(coco_gt, coco_dt, category_id=None):
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    if category_id is not None:
        evaluator.params.catIds = [category_id]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
    }


def tide_ground_truth(path):
    payload = json.loads(path.read_text())
    ground_truth = Data(path.stem, max_dets=100)
    for image in payload["images"]:
        ground_truth.add_image(image["id"], image["file_name"])
    for category in payload["categories"]:
        ground_truth.add_class(category["id"], category["name"])
    for annotation in payload["annotations"]:
        add = (
            ground_truth.add_ignore_region
            if annotation.get("iscrowd", 0)
            else ground_truth.add_ground_truth
        )
        add(annotation["image_id"], annotation["category_id"], annotation["bbox"])
    return ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    if not isinstance(predictions, list) or any(
        not {"image_id", "category_id", "bbox", "score"} <= prediction.keys()
        for prediction in predictions
    ):
        raise ValueError("predictions must be a COCO result list")

    coco_gt = COCO(str(args.annotations))
    coco_dt = coco_gt.loadRes(str(args.predictions))
    report = {"overall": evaluate(coco_gt, coco_dt), "per_class": {}}
    for category in coco_gt.loadCats(coco_gt.getCatIds()):
        report["per_class"][category["name"]] = evaluate(coco_gt, coco_dt, category["id"])

    print(json.dumps(report, indent=2))

    tide = TIDE()
    tide.evaluate(
        tide_ground_truth(args.annotations),
        datasets.COCOResult(str(args.predictions)),
        mode=TIDE.BOX,
        name=args.predictions.parent.name,
    )
    tide.summarize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
