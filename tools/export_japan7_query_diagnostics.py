#!/usr/bin/env python3
"""Export final-layer D-FINE queries and simulated Hungarian assignments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
from src.zoo.dfine.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
)


QUERY_FIELDS = (
    "image_id",
    "file_name",
    "query_index",
    "top_class_id",
    "top_class_name",
    "top_score",
    "bbox_norm_cx",
    "bbox_norm_cy",
    "bbox_norm_width",
    "bbox_norm_height",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "hungarian_matched",
    "matched_gt_index",
    "matched_gt_class_id",
    "matched_gt_class_name",
    "matched_gt_class_score",
    "matched_gt_iou",
    "matched_total_cost",
    "matched_class_cost",
    "matched_bbox_l1_cost",
    "matched_giou_cost",
    "nearest_gt_index",
    "nearest_gt_class_id",
    "nearest_gt_class_name",
    "nearest_gt_iou",
)

GT_FIELDS = (
    "image_id",
    "file_name",
    "gt_index",
    "gt_class_id",
    "gt_class_name",
    "gt_bbox_x",
    "gt_bbox_y",
    "gt_bbox_width",
    "gt_bbox_height",
    "detected_iou50",
    "detection_iou50_query_index",
    "detection_iou50_score",
    "detection_iou50",
    "detected_iou75",
    "detection_iou75_query_index",
    "detection_iou75_score",
    "detection_iou75",
    "hungarian_query_index",
    "hungarian_class_score",
    "hungarian_iou",
    "hungarian_total_cost",
    "best_iou_query_index",
    "best_iou",
    "best_iou_query_class_score",
    "best_class_score_query_index",
    "best_class_score",
    "best_class_score_query_iou",
    "best_total_cost_query_index",
    "best_total_cost",
    "hungarian_total_cost_gap",
    "queries_iou50",
    "queries_iou50_score_ge_0_10",
    "queries_iou50_score_ge_0_50",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "min": float(tensor.min()),
        "p25": float(torch.quantile(tensor, 0.25)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p75": float(torch.quantile(tensor, 0.75)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


def matcher_targets(targets, input_height, input_width):
    scale = torch.tensor(
        [input_width, input_height, input_width, input_height],
        dtype=torch.float32,
        device=targets[0]["boxes"].device,
    )
    return [
        {
            "labels": target["labels"],
            "boxes": box_xyxy_to_cxcywh(target["boxes"].to(torch.float32)) / scale,
        }
        for target in targets
    ]


def pairwise_diagnostics(logits, boxes, target, matcher):
    probabilities = logits.sigmoid() if matcher.use_focal_loss else logits.softmax(-1)
    labels = target["labels"]
    target_boxes = target["boxes"]
    if not len(target_boxes):
        empty = boxes.new_empty((len(boxes), 0))
        return probabilities, {
            "iou": empty,
            "class_cost": empty,
            "bbox_cost": empty,
            "giou_cost": empty,
            "total_cost": empty,
        }

    target_probabilities = probabilities[:, labels]
    if matcher.use_focal_loss:
        negative = (
            (1 - matcher.alpha)
            * target_probabilities**matcher.gamma
            * (-(1 - target_probabilities + 1e-8).log())
        )
        positive = (
            matcher.alpha
            * (1 - target_probabilities) ** matcher.gamma
            * (-(target_probabilities + 1e-8).log())
        )
        class_cost = positive - negative
    else:
        class_cost = -target_probabilities

    bbox_cost = torch.cdist(boxes, target_boxes, p=1)
    predicted_xyxy = box_cxcywh_to_xyxy(boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    iou = box_iou(predicted_xyxy, target_xyxy)[0]
    giou_cost = -generalized_box_iou(predicted_xyxy, target_xyxy)
    total_cost = (
        matcher.cost_class * class_cost
        + matcher.cost_bbox * bbox_cost
        + matcher.cost_giou * giou_cost
    )
    return probabilities, {
        "iou": iou,
        "class_cost": class_cost,
        "bbox_cost": bbox_cost,
        "giou_cost": giou_cost,
        "total_cost": total_cost,
    }


def greedy_top100_matches(probabilities, iou, labels, threshold):
    scores, flat_indices = torch.topk(
        probabilities.flatten(), min(100, probabilities.numel())
    )
    class_count = probabilities.shape[1]
    matched_gt = set()
    matches = {}
    for score, flat_index in zip(scores, flat_indices):
        query_index = int(flat_index // class_count)
        class_id = int(flat_index % class_count)
        candidates = [
            gt_index
            for gt_index, label in enumerate(labels.tolist())
            if label == class_id and gt_index not in matched_gt
        ]
        if not candidates:
            continue
        candidate_ious = iou[query_index, candidates]
        best_position = int(candidate_ious.argmax())
        gt_index = candidates[best_position]
        value = float(candidate_ious[best_position])
        if value >= threshold:
            matched_gt.add(gt_index)
            matches[gt_index] = {
                "query_index": query_index,
                "score": float(score),
                "iou": value,
            }
    return matches


def absolute_xywh(normalized_cxcywh, original_width, original_height):
    scale = normalized_cxcywh.new_tensor(
        [original_width, original_height, original_width, original_height]
    )
    xyxy = box_cxcywh_to_xyxy(normalized_cxcywh) * scale
    return (
        float(xyxy[0]),
        float(xyxy[1]),
        float(xyxy[2] - xyxy[0]),
        float(xyxy[3] - xyxy[1]),
    )


def value(matrix, row, column):
    return float(matrix[row, column])


def process_image(
    outputs,
    target,
    normalized_target,
    matched_indices,
    categories,
):
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    matcher = outputs["matcher"]
    probabilities, pairwise = pairwise_diagnostics(
        logits, boxes, normalized_target, matcher
    )
    labels = normalized_target["labels"]
    gt_count = len(labels)
    image_id = int(target["image_id"].item())
    file_name = Path(target["image_path"]).name
    original_width, original_height = map(int, target["orig_size"].tolist())
    assignments = {
        int(query_index): int(gt_index)
        for query_index, gt_index in zip(*matched_indices)
    }
    gt_assignments = {gt_index: query_index for query_index, gt_index in assignments.items()}
    detections50 = greedy_top100_matches(
        probabilities, pairwise["iou"], labels, 0.50
    )
    detections75 = greedy_top100_matches(
        probabilities, pairwise["iou"], labels, 0.75
    )

    query_rows = []
    top_scores, top_classes = probabilities.max(dim=1)
    for query_index in range(len(boxes)):
        top_class = int(top_classes[query_index])
        row = {
            "image_id": image_id,
            "file_name": file_name,
            "query_index": query_index,
            "top_class_id": top_class,
            "top_class_name": categories[top_class],
            "top_score": float(top_scores[query_index]),
            "bbox_norm_cx": float(boxes[query_index, 0]),
            "bbox_norm_cy": float(boxes[query_index, 1]),
            "bbox_norm_width": float(boxes[query_index, 2]),
            "bbox_norm_height": float(boxes[query_index, 3]),
            "hungarian_matched": query_index in assignments,
        }
        (
            row["bbox_x"],
            row["bbox_y"],
            row["bbox_width"],
            row["bbox_height"],
        ) = absolute_xywh(boxes[query_index], original_width, original_height)

        if gt_count:
            nearest_gt = int(pairwise["iou"][query_index].argmax())
            nearest_class = int(labels[nearest_gt])
            row.update(
                {
                    "nearest_gt_index": nearest_gt,
                    "nearest_gt_class_id": nearest_class,
                    "nearest_gt_class_name": categories[nearest_class],
                    "nearest_gt_iou": value(pairwise["iou"], query_index, nearest_gt),
                }
            )
        if query_index in assignments:
            gt_index = assignments[query_index]
            class_id = int(labels[gt_index])
            row.update(
                {
                    "matched_gt_index": gt_index,
                    "matched_gt_class_id": class_id,
                    "matched_gt_class_name": categories[class_id],
                    "matched_gt_class_score": float(
                        probabilities[query_index, class_id]
                    ),
                    "matched_gt_iou": value(
                        pairwise["iou"], query_index, gt_index
                    ),
                    "matched_total_cost": value(
                        pairwise["total_cost"], query_index, gt_index
                    ),
                    "matched_class_cost": value(
                        pairwise["class_cost"], query_index, gt_index
                    ),
                    "matched_bbox_l1_cost": value(
                        pairwise["bbox_cost"], query_index, gt_index
                    ),
                    "matched_giou_cost": value(
                        pairwise["giou_cost"], query_index, gt_index
                    ),
                }
            )
        query_rows.append(row)

    gt_rows = []
    for gt_index, class_tensor in enumerate(labels):
        class_id = int(class_tensor)
        class_scores = probabilities[:, class_id]
        gt_ious = pairwise["iou"][:, gt_index]
        total_costs = pairwise["total_cost"][:, gt_index]
        query_index = gt_assignments[gt_index]
        best_iou_query = int(gt_ious.argmax())
        best_score_query = int(class_scores.argmax())
        best_cost_query = int(total_costs.argmin())
        geometric = gt_ious >= 0.50
        assigned_cost = float(total_costs[query_index])
        gt_box = absolute_xywh(
            normalized_target["boxes"][gt_index], original_width, original_height
        )
        row = {
            "image_id": image_id,
            "file_name": file_name,
            "gt_index": gt_index,
            "gt_class_id": class_id,
            "gt_class_name": categories[class_id],
            "gt_bbox_x": gt_box[0],
            "gt_bbox_y": gt_box[1],
            "gt_bbox_width": gt_box[2],
            "gt_bbox_height": gt_box[3],
            "detected_iou50": gt_index in detections50,
            "detected_iou75": gt_index in detections75,
            "hungarian_query_index": query_index,
            "hungarian_class_score": float(class_scores[query_index]),
            "hungarian_iou": float(gt_ious[query_index]),
            "hungarian_total_cost": assigned_cost,
            "best_iou_query_index": best_iou_query,
            "best_iou": float(gt_ious[best_iou_query]),
            "best_iou_query_class_score": float(class_scores[best_iou_query]),
            "best_class_score_query_index": best_score_query,
            "best_class_score": float(class_scores[best_score_query]),
            "best_class_score_query_iou": float(gt_ious[best_score_query]),
            "best_total_cost_query_index": best_cost_query,
            "best_total_cost": float(total_costs[best_cost_query]),
            "hungarian_total_cost_gap": assigned_cost
            - float(total_costs[best_cost_query]),
            "queries_iou50": int(geometric.sum()),
            "queries_iou50_score_ge_0_10": int(
                (geometric & (class_scores >= 0.10)).sum()
            ),
            "queries_iou50_score_ge_0_50": int(
                (geometric & (class_scores >= 0.50)).sum()
            ),
        }
        for threshold, matches in ((50, detections50), (75, detections75)):
            if gt_index in matches:
                match = matches[gt_index]
                row.update(
                    {
                        f"detection_iou{threshold}_query_index": match[
                            "query_index"
                        ],
                        f"detection_iou{threshold}_score": match["score"],
                        f"detection_iou{threshold}": match["iou"],
                    }
                )
        gt_rows.append(row)

    image_row = {
        "image_id": image_id,
        "file_name": file_name,
        "gt_count": gt_count,
        "query_count": len(boxes),
        "detected_gt_iou50": len(detections50),
        "fn_iou50": gt_count - len(detections50),
        "detected_gt_iou75": len(detections75),
        "fn_iou75": gt_count - len(detections75),
        "unmatched_query_count": len(boxes) - len(assignments),
        "unmatched_query_score_ge_0_50": sum(
            row["top_score"] >= 0.50 and not row["hungarian_matched"]
            for row in query_rows
        ),
    }
    return query_rows, gt_rows, image_row


def build_summary(gt_rows, query_score_groups, image_count, matcher):
    per_class = defaultdict(Counter)
    for row in gt_rows:
        counts = per_class[row["gt_class_name"]]
        counts["gt"] += 1
        counts["detected_iou50"] += int(row["detected_iou50"])
        counts["fn_iou50"] += int(not row["detected_iou50"])
        if not row["detected_iou50"]:
            counts["fn_with_geometric_query"] += int(row["best_iou"] >= 0.50)
            counts["fn_with_geometric_query_score_ge_0_10"] += int(
                row["queries_iou50_score_ge_0_10"] > 0
            )
            counts["fn_with_geometric_query_score_ge_0_50"] += int(
                row["queries_iou50_score_ge_0_50"] > 0
            )
        counts["hungarian_iou_gap_ge_0_10"] += int(
            row["best_iou"] - row["hungarian_iou"] >= 0.10
        )

    fn_rows = [row for row in gt_rows if not row["detected_iou50"]]
    return {
        "status": "raw query and simulated final-layer Hungarian audit complete",
        "method": {
            "decoder_layer": "final",
            "query_count_per_image": 300,
            "diagnostic_detection_max_dets": 100,
            "diagnostic_detection_iou": [0.50, 0.75],
            "query_score": "sigmoid foreground class probability",
            "matcher_weights": {
                "class": matcher.cost_class,
                "bbox_l1": matcher.cost_bbox,
                "giou": matcher.cost_giou,
            },
            "note": (
                "Hungarian assignments are simulated on final eval outputs and current "
                "GT. They diagnose the training matcher but are not used at inference."
            ),
        },
        "counts": {
            "images": image_count,
            "gt": len(gt_rows),
            "fn_iou50": len(fn_rows),
            "fn_with_any_query_iou50": sum(
                row["best_iou"] >= 0.50 for row in fn_rows
            ),
            "fn_with_query_iou50_score_ge_0_10": sum(
                row["queries_iou50_score_ge_0_10"] > 0 for row in fn_rows
            ),
            "fn_with_query_iou50_score_ge_0_50": sum(
                row["queries_iou50_score_ge_0_50"] > 0 for row in fn_rows
            ),
        },
        "score_distributions": {
            name: quantiles(values) for name, values in query_score_groups.items()
        },
        "per_class": {name: dict(counts) for name, counts in per_class.items()},
    }


def write_csv(path, rows, fieldnames, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary):
    counts = summary["counts"]
    lines = [
        "# Japan7 B0 final-layer query/Hungarian diagnostic",
        "",
        f"- Images: `{counts['images']}`",
        f"- GT: `{counts['gt']}`",
        f"- Diagnostic FN at IoU 0.50: `{counts['fn_iou50']}`",
        f"- FN with any raw query IoU >= 0.50: `{counts['fn_with_any_query_iou50']}`",
        (
            "- FN with IoU >= 0.50 query and class score >= 0.10: "
            f"`{counts['fn_with_query_iou50_score_ge_0_10']}`"
        ),
        (
            "- FN with IoU >= 0.50 query and class score >= 0.50: "
            f"`{counts['fn_with_query_iou50_score_ge_0_50']}`"
        ),
        "",
        "| class | GT | detected@50 | FN@50 | FN with geometric query | "
        "FN geometric score>=0.10 | Hungarian IoU gap>=0.10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary["per_class"].items():
        lines.append(
            f"| {name} | {row.get('gt', 0)} | {row.get('detected_iou50', 0)} | "
            f"{row.get('fn_iou50', 0)} | {row.get('fn_with_geometric_query', 0)} | "
            f"{row.get('fn_with_geometric_query_score_ge_0_10', 0)} | "
            f"{row.get('hungarian_iou_gap_ge_0_10', 0)} |"
        )
    lines.extend(
        [
            "",
            "This report is diagnostic only. It does not justify BAQS, DDE, DEIM, "
            "or another module without reviewing the exported rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test():
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 20.0]]),
            "labels": torch.tensor([0]),
        }
    ]
    normalized = matcher_targets(targets, 20, 20)[0]["boxes"]
    assert torch.allclose(normalized, torch.tensor([[0.25, 0.50, 0.50, 1.00]]))

    probabilities = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    iou = torch.tensor([[0.8, 0.0], [0.0, 0.7]])
    labels = torch.tensor([0, 1])
    matches = greedy_top100_matches(probabilities, iou, labels, 0.5)
    assert matches[0]["query_index"] == 0
    assert matches[1]["query_index"] == 1
    print("self-test passed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-r", "--resume")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not all((args.config, args.resume, args.output)):
        parser.error("--config, --resume and --output are required")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    checkpoint = Path(args.resume)
    config = Path(args.config)
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.name}-", dir=args.output.parent
    ) as temporary:
        temporary = Path(temporary)
        results_dir = temporary / "results"
        results_dir.mkdir()
        updates = yaml_utils.parse_cli(args.update)
        updates.update(
            {
                "device": args.device,
                "resume": str(checkpoint),
                "output_dir": str(temporary / "solver"),
                "seed": args.seed,
            }
        )
        cfg = YAMLConfig(str(config), **updates)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        model = solver.ema.module if solver.ema else solver.model
        matcher = solver.criterion.matcher
        model.eval()
        matcher.eval()
        categories = solver.val_dataloader.dataset.category2name

        gt_rows = []
        image_rows = []
        score_groups = {
            "hungarian_matched_gt_class_score": [],
            "hungarian_unmatched_top_score": [],
        }
        query_path = results_dir / "query_records.csv.gz"
        with gzip.open(query_path, "wt", encoding="utf-8", newline="") as handle:
            query_writer = csv.DictWriter(
                handle, fieldnames=QUERY_FIELDS, extrasaction="ignore"
            )
            query_writer.writeheader()
            image_count = 0
            for samples, targets in solver.val_dataloader:
                samples = samples.to(solver.device)
                targets = [
                    {
                        key: value.to(solver.device)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]
                outputs = model(samples)
                normalized_targets = matcher_targets(
                    targets, samples.shape[-2], samples.shape[-1]
                )
                matched = matcher(outputs, normalized_targets)["indices"]
                for batch_index, target in enumerate(targets):
                    image_outputs = {
                        "pred_logits": outputs["pred_logits"][batch_index],
                        "pred_boxes": outputs["pred_boxes"][batch_index],
                        "matcher": matcher,
                    }
                    query_rows, image_gt_rows, image_row = process_image(
                        image_outputs,
                        target,
                        normalized_targets[batch_index],
                        matched[batch_index],
                        categories,
                    )
                    query_writer.writerows(query_rows)
                    gt_rows.extend(image_gt_rows)
                    image_rows.append(image_row)
                    for row in query_rows:
                        group = (
                            "hungarian_matched_gt_class_score"
                            if row["hungarian_matched"]
                            else "hungarian_unmatched_top_score"
                        )
                        score_groups[group].append(
                            row.get("matched_gt_class_score", row["top_score"])
                        )
                    image_count += 1
                    if image_count % 100 == 0:
                        print(f"processed {image_count} images", flush=True)
                    if args.max_images and image_count >= args.max_images:
                        break
                if args.max_images and image_count >= args.max_images:
                    break

        summary = build_summary(gt_rows, score_groups, image_count, matcher)
        summary["inputs"] = {
            "config": str(config.resolve()),
            "config_sha256": sha256(config),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "command": sys.argv,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else str(solver.device),
        }
        write_csv(results_dir / "gt_query_diagnostics.csv", gt_rows, GT_FIELDS)
        write_csv(
            results_dir / "image_query_summary.csv",
            image_rows,
            tuple(image_rows[0]),
        )
        (results_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(results_dir / "summary.md", summary)
        manifest = [
            f"{sha256(path)}  {path.name}"
            for path in sorted(results_dir.iterdir(), key=lambda item: item.name)
        ]
        (results_dir / "manifest.sha256").write_text(
            "\n".join(manifest) + "\n", encoding="utf-8"
        )
        results_dir.rename(args.output)

    print(json.dumps({"output": str(args.output), **summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
