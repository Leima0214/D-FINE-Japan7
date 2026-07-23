#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


EXPECTED_CATEGORIES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "D43",
    5: "D44",
    6: "D50",
}


def check_split(root: Path, split: str) -> dict:
    image_dir = root / "images" / split
    annotation_file = root / "annotations" / f"instances_{split}.json"
    errors = []

    if not image_dir.is_dir():
        errors.append(f"missing image directory: {image_dir}")
    if not annotation_file.is_file():
        errors.append(f"missing annotation file: {annotation_file}")
        return {"split": split, "errors": errors}

    try:
        with annotation_file.open(encoding="utf-8") as handle:
            coco = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"split": split, "errors": [f"invalid JSON: {exc}"]}

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    image_ids = [image.get("id") for image in images]
    image_id_set = set(image_ids)
    category_map = {category.get("id"): category.get("name") for category in categories}

    if len(image_ids) != len(image_id_set):
        errors.append("duplicate image ids")
    if len(categories) != len(EXPECTED_CATEGORIES) or category_map != EXPECTED_CATEGORIES:
        errors.append(f"categories must be {EXPECTED_CATEGORIES}, got {category_map}")

    missing_files = []
    for image in images:
        file_name = image.get("file_name")
        relative_path = Path(file_name) if isinstance(file_name, str) else None
        if (
            relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not (image_dir / relative_path).is_file()
        ):
            missing_files.append(file_name)
    if missing_files:
        errors.append(f"{len(missing_files)} image files missing; first={missing_files[0]!r}")

    annotation_ids = set()
    for annotation in annotations:
        annotation_id = annotation.get("id")
        if annotation_id in annotation_ids:
            errors.append(f"duplicate annotation id: {annotation_id}")
            break
        annotation_ids.add(annotation_id)

        if annotation.get("image_id") not in image_id_set:
            errors.append(f"annotation {annotation_id} references unknown image")
            continue
        if annotation.get("category_id") not in EXPECTED_CATEGORIES:
            errors.append(f"annotation {annotation_id} has invalid category_id")

        bbox = annotation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            errors.append(f"annotation {annotation_id} has invalid bbox")

    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "categories": category_map,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the read-only Japan7 COCO dataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result = {
        "root": str(args.root.resolve()),
        "splits": [check_split(args.root, split) for split in ("train", "val")],
    }
    result["ok"] = all(not split["errors"] for split in result["splits"])

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
