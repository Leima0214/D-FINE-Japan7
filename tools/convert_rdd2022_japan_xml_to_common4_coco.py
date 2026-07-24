#!/usr/bin/env python3
"""Convert the annotated RDD2022 Japan VOC/XML archive to Common4 COCO.

Only the official ``train/images`` and ``train/annotations/xmls`` pair is
used.  The archive's ``test/images`` has no XML annotations and is therefore
deliberately outside this supervised-detection conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CATEGORIES = [
    {"id": 0, "name": "D00"},
    {"id": 1, "name": "D10"},
    {"id": 2, "name": "D20"},
    {"id": 3, "name": "D40"},
]
COMMON4 = {category["name"]: category["id"] for category in CATEGORIES}
KNOWN_RDD2022_LABELS = {
    "D00",
    "D01",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
    "Repair",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    """Read JPEG/PNG dimensions without a third-party dependency."""
    with path.open("rb") as handle:
        header = handle.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            return struct.unpack(">II", header[16:24])
        if not header.startswith(b"\xff\xd8"):
            raise ValueError(f"Unsupported or corrupt image: {path}")
        handle.seek(2)
        while True:
            byte = handle.read(1)
            while byte == b"\xff":
                byte = handle.read(1)
            if not byte:
                break
            marker = byte[0]
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            size_bytes = handle.read(2)
            if len(size_bytes) != 2:
                break
            segment_size = struct.unpack(">H", size_bytes)[0]
            if segment_size < 2:
                break
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                data = handle.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            handle.seek(segment_size - 2, 1)
    raise ValueError(f"Could not read image dimensions: {path}")


def collect_images(directory: Path, recursive: bool = False) -> dict[str, Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    result: dict[str, Path] = {}
    for path in files:
        if path.name in result:
            raise ValueError(f"Duplicate image filename under {directory}: {path.name}")
        result[path.name] = path
    return result


def parse_voc(xml_path: Path, image_path: Path) -> tuple[int, int, list[dict[str, Any]]]:
    root = ET.parse(xml_path).getroot()
    try:
        xml_width = int(root.findtext("size/width", ""))
        xml_height = int(root.findtext("size/height", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid XML image size: {xml_path}") from exc
    if xml_width <= 0 or xml_height <= 0:
        raise ValueError(f"Non-positive XML image size: {xml_path}")
    actual_width, actual_height = image_size(image_path)
    if (xml_width, xml_height) != (actual_width, actual_height):
        raise ValueError(
            f"XML/image size mismatch for {image_path.name}: "
            f"XML={xml_width}x{xml_height}, image={actual_width}x{actual_height}"
        )
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not label or box is None:
            raise ValueError(f"Malformed object in {xml_path}")
        try:
            xmin = float(box.findtext("xmin", ""))
            ymin = float(box.findtext("ymin", ""))
            xmax = float(box.findtext("xmax", ""))
            ymax = float(box.findtext("ymax", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid bbox values in {xml_path}") from exc
        objects.append(
            {"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        )
    return actual_width, actual_height, objects


def reuse_split_by_hash(source_images: dict[str, Path], reuse_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    split_dirs = {split: reuse_root / "images" / split for split in ("train", "val")}
    if not all(directory.is_dir() for directory in split_dirs.values()):
        raise ValueError(f"Missing images/train or images/val under reuse root: {reuse_root}")
    source_hashes: dict[str, str] = {}
    for name, path in source_images.items():
        digest = sha256(path)
        if digest in source_hashes:
            raise ValueError(f"Source images are not hash-unique: {name} and {source_hashes[digest]}")
        source_hashes[digest] = name
    assignment: dict[str, str] = {}
    current_count = 0
    for split, directory in split_dirs.items():
        for path in collect_images(directory, recursive=True).values():
            current_count += 1
            digest = sha256(path)
            if digest not in source_hashes:
                raise ValueError(f"Existing split contains image absent from official source: {path}")
            source_name = source_hashes[digest]
            if source_name in assignment:
                raise ValueError(f"Existing split maps multiple files to source image: {source_name}")
            assignment[source_name] = split
    missing = sorted(set(source_images) - set(assignment))
    if current_count != len(source_images) or missing:
        raise ValueError(
            f"Existing split is not a complete source match: current={current_count}, "
            f"source={len(source_images)}, missing_source={len(missing)}"
        )
    return assignment, {
        "strategy": "reuse_existing_japan7_coco_membership",
        "sha256_verified": True,
        "source_images": len(source_images),
        "existing_images": current_count,
        "existing_split_root": str(reuse_root),
    }


def fallback_scene_split(names: list[str], seed: int, block_size: int) -> tuple[dict[str, str], dict[str, Any]]:
    if block_size < 2:
        raise ValueError("fallback scene block size must be at least 2")
    numbered: list[tuple[int, str]] = []
    for name in names:
        match = re.search(r"(\d+)(?=\.[^.]+$)", name)
        if match is None:
            raise ValueError(f"Cannot infer numeric RDD frame id for fallback split: {name}")
        numbered.append((int(match.group(1)), name))
    numbered.sort()
    groups: list[list[str]] = []
    group: list[str] = []
    previous: int | None = None
    for frame_id, name in numbered:
        if previous is None or (frame_id == previous + 1 and len(group) < block_size):
            group.append(name)
        else:
            groups.append(group)
            group = [name]
        previous = frame_id
    if group:
        groups.append(group)
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    target_val = round(len(names) * 0.2)
    val: set[str] = set()
    for group in shuffled:
        if abs((len(val) + len(group)) - target_val) <= abs(len(val) - target_val):
            val.update(group)
    assignment = {name: ("val" if name in val else "train") for name in names}
    return assignment, {
        "strategy": "fallback_contiguous_numeric_frame_blocks",
        "seed": seed,
        "scene_block_size": block_size,
        "group_count": len(groups),
        "target_val_images": target_val,
        "actual_val_images": len(val),
    }


def make_coco(split: str, assignment: dict[str, str], source_images: dict[str, Path], xml_dir: Path) -> tuple[dict[str, Any], Counter[str], Counter[str], int]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    included = Counter()
    discarded = Counter()
    clipped_boxes = 0
    annotation_id = 1
    for image_id, name in enumerate(sorted(name for name, value in assignment.items() if value == split), start=1):
        image_path = source_images[name]
        xml_path = xml_dir / f"{image_path.stem}.xml"
        if not xml_path.is_file():
            raise ValueError(f"Missing XML for image: {image_path}")
        width, height, objects = parse_voc(xml_path, image_path)
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        for obj in objects:
            label = obj["label"]
            if label not in KNOWN_RDD2022_LABELS:
                raise ValueError(f"Unexpected RDD2022 label {label!r} in {xml_path}")
            if label not in COMMON4:
                discarded[label] += 1
                continue
            # Pascal VOC coordinates are 1-based inclusive; COCO is 0-based [x, y, w, h].
            raw_x0, raw_y0 = obj["xmin"] - 1.0, obj["ymin"] - 1.0
            raw_x1, raw_y1 = obj["xmax"], obj["ymax"]
            x0, y0 = max(0.0, raw_x0), max(0.0, raw_y0)
            x1, y1 = min(float(width), raw_x1), min(float(height), raw_y1)
            if (x0, y0, x1, y1) != (raw_x0, raw_y0, raw_x1, raw_y1):
                clipped_boxes += 1
            box_width, box_height = x1 - x0, y1 - y0
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"Non-positive Common4 bbox after clipping in {xml_path}: {obj}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": COMMON4[label],
                    "bbox": [round(x0, 6), round(y0, 6), round(box_width, 6), round(box_height, 6)],
                    "area": round(box_width * box_height, 6),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            included[label] += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}, included, discarded, clipped_boxes


def validate_coco(output: Path, split: str) -> dict[str, Any]:
    annotation_path = output / "annotations" / f"instances_{split}.json"
    with annotation_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("categories") != CATEGORIES:
        raise ValueError(f"Unexpected categories in {annotation_path}")
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    image_by_id = {image["id"]: image for image in images}
    if len(image_by_id) != len(images):
        raise ValueError(f"Duplicate image ids in {annotation_path}")
    names = [image["file_name"] for image in images]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate image names in {annotation_path}")
    category_counts = Counter()
    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            raise ValueError(f"Annotation points to missing image in {annotation_path}")
        category_id = annotation.get("category_id")
        if category_id not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid category id in {annotation_path}: {category_id}")
        x, y, width, height = annotation.get("bbox", [None] * 4)
        if width is None or width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            raise ValueError(f"Invalid bbox in {annotation_path}: {annotation}")
        if abs(annotation.get("area", -1) - width * height) > 1e-4:
            raise ValueError(f"Invalid area in {annotation_path}: {annotation}")
        if annotation.get("iscrowd") != 0:
            raise ValueError(f"Invalid iscrowd in {annotation_path}: {annotation}")
        category_counts[category_id] += 1
    image_dir = output / "images" / split
    missing_images = [name for name in names if not (image_dir / name).is_file()]
    if missing_images:
        raise ValueError(f"COCO references missing copied images in {annotation_path}: {missing_images[:5]}")
    manifest = output / "splits" / f"japan_common4_{split}.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    expected = [f"images/{split}/{name}" for name in names]
    if lines != expected:
        raise ValueError(f"Split manifest does not match COCO images: {manifest}")
    return {"images": len(images), "annotations": len(annotations), "category_counts": dict(sorted(category_counts.items()))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="RDD2022 Japan archive root")
    parser.add_argument("--output", type=Path, required=True, help="New Common4 COCO output directory")
    parser.add_argument(
        "--reuse-splits-from",
        type=Path,
        default=None,
        help="Existing Japan7-COCO root; membership is reused only after full SHA-256 equality validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fallback split seed")
    parser.add_argument("--fallback-scene-block-size", type=int, default=50, help="Contiguous numeric frames kept together by fallback")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    image_dir = source / "train" / "images"
    xml_dir = source / "train" / "annotations" / "xmls"
    if not image_dir.is_dir() or not xml_dir.is_dir():
        raise ValueError(f"Expected RDD2022 paths missing: {image_dir} and/or {xml_dir}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    source_images = collect_images(image_dir)
    if not source_images:
        raise ValueError(f"No images found in {image_dir}")
    xml_stems = {path.stem for path in xml_dir.glob("*.xml")}
    image_stems = {path.stem for path in source_images.values()}
    if xml_stems != image_stems:
        raise ValueError(
            f"Image/XML correspondence failed: missing_xml={len(image_stems - xml_stems)}, "
            f"orphan_xml={len(xml_stems - image_stems)}"
        )
    if args.reuse_splits_from is not None:
        assignment, split_metadata = reuse_split_by_hash(source_images, args.reuse_splits_from.resolve())
    else:
        assignment, split_metadata = fallback_scene_split(
            sorted(source_images), args.seed, args.fallback_scene_block_size
        )

    tmp_parent = output.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.partial-", dir=tmp_parent) as tmp_name:
        tmp_output = Path(tmp_name)
        for split in ("train", "val"):
            (tmp_output / "images" / split).mkdir(parents=True)
        (tmp_output / "annotations").mkdir(parents=True)
        (tmp_output / "splits").mkdir(parents=True)
        split_reports: dict[str, Any] = {}
        total_included: Counter[str] = Counter()
        total_discarded: Counter[str] = Counter()
        total_clipped = 0
        for split in ("train", "val"):
            coco, included, discarded, clipped = make_coco(split, assignment, source_images, xml_dir)
            for image in coco["images"]:
                shutil.copy2(source_images[image["file_name"]], tmp_output / "images" / split / image["file_name"])
            with (tmp_output / "annotations" / f"instances_{split}.json").open("w", encoding="utf-8") as handle:
                json.dump(coco, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            (tmp_output / "splits" / f"japan_common4_{split}.txt").write_text(
                "".join(f"images/{split}/{image['file_name']}\n" for image in coco["images"]), encoding="utf-8"
            )
            split_reports[split] = validate_coco(tmp_output, split)
            total_included.update(included)
            total_discarded.update(discarded)
            total_clipped += clipped
        split_metadata.update(
            {
                "source": str(source),
                "annotated_source_image_count": len(source_images),
                "source_test_images_excluded_no_xml": len(collect_images(source / "test" / "images")) if (source / "test" / "images").is_dir() else 0,
                "categories": CATEGORIES,
                "included_common4_objects": dict(sorted(total_included.items())),
                "discarded_non_common4_objects": dict(sorted(total_discarded.items())),
                "clipped_voc_boxes": total_clipped,
                "validation": split_reports,
            }
        )
        with (tmp_output / "splits" / "japan_common4_split_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(split_metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if output.exists():  # Defensive: never replace an output created mid-run.
            raise FileExistsError(f"Output appeared during conversion; refusing to replace it: {output}")
        tmp_output.rename(output)
        # Prevent TemporaryDirectory from trying to delete the moved directory.
        tmp_name = ""
    print(json.dumps(split_metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
