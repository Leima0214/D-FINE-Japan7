#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    tide = TIDE()
    tide.evaluate(
        datasets.COCO(str(args.annotations)),
        datasets.COCOResult(str(args.predictions)),
        mode=TIDE.BOX,
        name=args.predictions.parent.name,
    )
    tide.summarize()


if __name__ == "__main__":
    main()
