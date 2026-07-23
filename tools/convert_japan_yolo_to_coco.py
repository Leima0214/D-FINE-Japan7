#!/usr/bin/env python3
"""Convert a Japan YOLO detection dataset to the COCO layout used by D-FINE."""

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image


TARGET_CLASSES = ["D00", "D10", "D20", "D40", "D43", "D44", "D50"]
SOURCE_TEN_CLASSES = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D50", "Repair"]
SOURCE_TO_TARGET = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_DIR_NAMES = {"images", "image", "imgs"}
LABEL_DIR_NAMES = {"labels", "label", "annotations", "annots"}
SPLIT_NAMES = {"train": {"train", "training"}, "val": {"val", "valid", "validation"}}


def find_split_dir(root: Path, kind_names: set[str], split: str) -> Path:
    split_names = SPLIT_NAMES[split]
    directories = [root]
    for child in root.iterdir():
        if child.is_dir():
            directories.append(child)
            directories.extend(path for path in child.iterdir() if path.is_dir())

    candidates = {
        path
        for path in directories
        if (
            path.name.lower() in split_names and path.parent.name.lower() in kind_names
        )
        or (
            path.name.lower() in kind_names and path.parent.name.lower() in split_names
        )
    }
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in sorted(candidates))
        raise ValueError(f"cannot uniquely find {split} {sorted(kind_names)} directories: {found or 'none'}")
    return candidates.pop()


def discover_layout(root: Path) -> dict[str, dict[str, Path]]:
    if not root.is_dir():
        raise ValueError(f"source directory does not exist: {root}")
    return {
        split: {
            "images": find_split_dir(root, IMAGE_DIR_NAMES, split),
            "labels": find_split_dir(root, LABEL_DIR_NAMES, split),
        }
        for split in ("train", "val")
    }


def read_class_configs(root: Path) -> Optional[dict[int, str]]:
    mappings = set()
    for path in sorted([*root.glob("*.yaml"), *root.glob("*.yml")]):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = data.get("names")
            if isinstance(names, list):
                mapping = {index: str(name) for index, name in enumerate(names)}
            elif isinstance(names, dict):
                mapping = {int(index): str(name) for index, name in names.items()}
            else:
                continue
            mappings.add(tuple(sorted(mapping.items())))
        except (OSError, ValueError, yaml.YAMLError):
            continue
    if len(mappings) > 1:
        raise ValueError("conflicting class mappings found in source YAML files")
    return dict(next(iter(mappings))) if mappings else None


def collect_label_ids(layout: dict[str, dict[str, Path]]) -> tuple[set[int], list[str]]:
    class_ids, malformed = set(), []
    for paths in layout.values():
        for label_path in paths["labels"].rglob("*.txt"):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    malformed.append(f"{label_path}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    malformed.append(f"{label_path}:{line_number}: non-numeric label")
                    continue
                if not all(math.isfinite(value) for value in values):
                    malformed.append(f"{label_path}:{line_number}: non-finite coordinate")
                    continue
                class_ids.add(class_id)
    return class_ids, malformed


def choose_mapping(
    class_ids: set[int], class_config: Optional[dict[int, str]]
) -> tuple[str, dict[int, Optional[int]]]:
    direct = {index: index for index in range(7)}
    ten = {index: SOURCE_TO_TARGET.get(index) for index in range(10)}
    if class_config is not None:
        names = [class_config.get(index) for index in range(len(class_config))]
        if names == TARGET_CLASSES:
            if not class_ids.issubset(direct):
                raise ValueError(f"labels conflict with Japan7 class config: {sorted(class_ids)}")
            return "direct Japan7 ids 0-6", direct
        if names == SOURCE_TEN_CLASSES:
            if not class_ids.issubset(ten):
                raise ValueError(f"labels conflict with ten-class source config: {sorted(class_ids)}")
            return "ten-class mapping; dropped D01/D11/Repair", ten
        raise ValueError(f"unsupported source YAML class config: {class_config}")
    if class_ids.issubset(direct):
        return "direct Japan7 ids 0-6", direct
    if class_ids.issubset(ten) and any(class_id > 6 for class_id in class_ids):
        return "ten-class mapping; dropped D01/D11/Repair", ten
    raise ValueError(f"unsupported class ids without a usable YAML class config: {sorted(class_ids)}")


def image_index(image_dir: Path) -> dict[Path, Path]:
    images = {
        image.relative_to(image_dir): image
        for image in image_dir.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    }
    output_names = [path.name for path in images]
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"duplicate image names under {image_dir}; cannot flatten output safely")
    return images


def parse_annotations(
    label_path: Path,
    image_path: Path,
    class_mapping: dict[int, Optional[int]],
    report: Counter,
) -> list[dict]:
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    annotations = []
    if not label_path.is_file():
        report["missing_label_files"] += 1
        return annotations

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        x_center, y_center, width, height = (float(value) for value in fields[1:])
        target_id = class_mapping[class_id]
        if target_id is None:
            report[f"dropped_source_id_{class_id}"] += 1
            continue
        if width <= 0 or height <= 0:
            report["invalid_bboxes"] += 1
            continue

        x = (x_center - width / 2) * image_width
        y = (y_center - height / 2) * image_height
        box_width = width * image_width
        box_height = height * image_height
        if x < 0 or y < 0 or x + box_width > image_width or y + box_height > image_height:
            report["clipped_out_of_bounds_bboxes"] += 1
            right = min(float(image_width), x + box_width)
            bottom = min(float(image_height), y + box_height)
            x = max(0.0, x)
            y = max(0.0, y)
            box_width = right - x
            box_height = bottom - y
        if box_width <= 0 or box_height <= 0:
            report["invalid_bboxes"] += 1
            continue
        annotations.append(
            {
                "category_id": target_id,
                "bbox": [x, y, box_width, box_height],
                "area": float(box_width * box_height),
                "iscrowd": 0,
                "source": f"{label_path}:{line_number}",
            }
        )
    return annotations


def prepare_split(
    split: str,
    paths: dict[str, Path],
    class_mapping: dict[int, Optional[int]],
    report: Counter,
) -> list[dict]:
    images = image_index(paths["images"])
    labels = {label.relative_to(paths["labels"]) for label in paths["labels"].rglob("*.txt")}
    expected_labels = {relative.with_suffix(".txt") for relative in images}
    report[f"{split}_label_only_files"] += len(labels - expected_labels)
    report[f"{split}_image_only_files"] += len(expected_labels - labels)

    records = []
    for relative_path, image_path in sorted(images.items()):
        label_path = paths["labels"] / relative_path.with_suffix(".txt")
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            {
                "source": image_path,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "annotations": parse_annotations(label_path, image_path, class_mapping, report),
            }
        )
    return records


def write_split(records: list[dict], image_dir: Path) -> dict:
    image_dir.mkdir(parents=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": index, "name": name} for index, name in enumerate(TARGET_CLASSES)],
    }
    annotation_id = 1
    for image_id, record in enumerate(records, 1):
        shutil.copy2(record["source"], image_dir / record["file_name"])
        coco["images"].append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for annotation in record["annotations"]:
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation["category_id"],
                    "bbox": annotation["bbox"],
                    "area": annotation["area"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return coco


def validate_coco(coco: dict, image_dir: Path) -> Counter:
    errors = Counter()
    image_sizes = {image["id"]: (image["width"], image["height"]) for image in coco["images"]}
    for image in coco["images"]:
        if not (image_dir / image["file_name"]).is_file():
            errors["missing_output_images"] += 1
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        image_width, image_height = image_sizes.get(annotation["image_id"], (0, 0))
        if annotation["category_id"] not in range(7):
            errors["invalid_category_ids"] += 1
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
            errors["invalid_output_bboxes"] += 1
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Japan YOLO labels to D-FINE-compatible COCO.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Inspect only; do not create output.")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace a non-empty output directory.")
    args = parser.parse_args()

    layout = discover_layout(args.source)
    class_ids, malformed = collect_label_ids(layout)
    if malformed:
        raise ValueError("malformed YOLO labels:\n" + "\n".join(malformed[:20]))
    class_config = read_class_configs(args.source)
    mapping_name, class_mapping = choose_mapping(class_ids, class_config)
    report = Counter()
    records = {
        split: prepare_split(split, layout[split], class_mapping, report)
        for split in ("train", "val")
    }

    print("source layout:")
    for split in ("train", "val"):
        print(f"  {split}: images={layout[split]['images']} labels={layout[split]['labels']} samples={len(records[split])}")
    print(f"source class ids: {sorted(class_ids)}")
    print(f"class mapping: {mapping_name}")
    if args.dry_run:
        print("DRY RUN PASS")
        return 0

    if args.output.exists() and any(args.output.iterdir()):
        if not args.replace:
            raise ValueError(f"refusing to overwrite non-empty output directory: {args.output}")
        print(f"replacing existing output directory: {args.output}")
        shutil.rmtree(args.output)
    if args.output.exists():
        print(f"using existing empty output directory: {args.output}")
    else:
        args.output.mkdir(parents=True)

    train_coco = write_split(records["train"], args.output / "images" / "train")
    val_coco = write_split(records["val"], args.output / "images" / "val")
    annotation_dir = args.output / "annotations"
    annotation_dir.mkdir()
    for split, coco in (("train", train_coco), ("val", val_coco)):
        path = annotation_dir / f"instances_{split}.json"
        path.write_text(json.dumps(coco, ensure_ascii=False) + "\n", encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        report.update(validate_coco(loaded, args.output / "images" / split))

    category_counts = Counter(
        annotation["category_id"]
        for coco in (train_coco, val_coco)
        for annotation in coco["annotations"]
    )
    print(f"output: {args.output}")
    print(f"train images={len(train_coco['images'])} annotations={len(train_coco['annotations'])}")
    print(f"val images={len(val_coco['images'])} annotations={len(val_coco['annotations'])}")
    print("per-class annotations:", {TARGET_CLASSES[index]: category_counts[index] for index in range(7)})
    print(
        "dropped D01/D11/Repair:",
        {name: report[f"dropped_source_id_{index}"] for index, name in ((1, "D01"), (3, "D11"), (9, "Repair"))},
    )
    print("anomalies:", dict(sorted(report.items())))
    if report["missing_output_images"] or report["invalid_category_ids"] or report["invalid_output_bboxes"]:
        raise ValueError(f"post-conversion validation failed: {dict(report)}")
    print("CONVERSION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
