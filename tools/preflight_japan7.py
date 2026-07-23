#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import YAMLConfig
from src.solver import DetSolver


def dataloader_config(root: Path, split: str, batch_size: int, workers: int) -> dict:
    return {
        "total_batch_size": batch_size,
        "num_workers": workers,
        "dataset": {
            "img_folder": str(root / "images" / split),
            "ann_file": str(root / "annotations" / f"instances_{split}.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build D-FINE-S and one Japan7 batch per split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dfine/custom/dfine_hgnetv2_s_japan7.yml"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    cfg = YAMLConfig(
        str(args.config),
        num_classes=7,
        remap_mscoco_category=False,
        HGNetv2={"pretrained": False},
        train_dataloader=dataloader_config(
            args.root, "train", args.batch_size, args.workers
        ),
        val_dataloader=dataloader_config(
            args.root, "val", args.batch_size, args.workers
        ),
    )
    assert cfg.yaml_cfg["num_classes"] == 7
    assert cfg.yaml_cfg["remap_mscoco_category"] is False

    model = cfg.model
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "ema" in checkpoint:
            source_name = "ema.module"
            source_state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            source_name = "model"
            source_state = checkpoint["model"]
        else:
            raise KeyError("checkpoint must contain ema.module or model")
        model_state = model.state_dict()
        matched_parameters = sum(
            model_state[name].numel()
            for name, value in source_state.items()
            if name in model_state and value.shape == model_state[name].shape
        )
        total_parameters = sum(value.numel() for value in model_state.values())
        coverage = matched_parameters / total_parameters
        print(f"checkpoint={source_name} parameter_coverage={coverage:.2%}")
        assert coverage > 0.9
        solver = DetSolver(cfg)
        solver.model = model
        solver.load_tuning_state(str(args.checkpoint))

    model.to(torch.device(args.device))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model=DFINE-S parameters={parameter_count:,} device={args.device}")

    for split, loader in (
        ("train", cfg.train_dataloader),
        ("val", cfg.val_dataloader),
    ):
        images, targets = next(iter(loader))
        labels = torch.cat([target["labels"] for target in targets])
        if labels.numel():
            assert labels.min() >= 0 and labels.max() < 7
        print(
            f"{split}: dataset={len(loader.dataset)} "
            f"batch={tuple(images.shape)} labels={labels.numel()}"
        )

    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
