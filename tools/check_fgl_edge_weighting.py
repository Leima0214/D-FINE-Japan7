#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zoo.dfine.dfine_criterion import fgl_edge_weights


def self_check():
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.8, 0.2],
            [0.5, 0.5, 0.2, 0.8],
        ]
    )
    for mode in ("fixed", "sensitivity", "sqrt_sensitivity"):
        weights = fgl_edge_weights(boxes, mode=mode)
        assert torch.allclose(weights.mean(dim=1), torch.ones(3))
        assert torch.allclose(weights[0], torch.ones(4))
        assert weights[1, 1] > weights[1, 0] and weights[1, 3] > weights[1, 2]
        assert weights[2, 0] > weights[2, 1] and weights[2, 2] > weights[2, 3]
    print("FGL edge-weight self-check PASS")


def audit_annotations(path, mode):
    payload = json.loads(path.read_text())
    boxes = torch.tensor(
        [[0.0, 0.0, annotation["bbox"][2], annotation["bbox"][3]]
         for annotation in payload["annotations"]]
    )
    weights = fgl_edge_weights(boxes, mode=mode)
    quantiles = torch.quantile(
        weights.flatten(), torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    )
    print(f"annotations={len(boxes)} mode={mode}")
    print(
        "edge_weight "
        f"min={weights.min().item():.6f} "
        f"p01={quantiles[0].item():.6f} "
        f"p25={quantiles[1].item():.6f} "
        f"p50={quantiles[2].item():.6f} "
        f"p75={quantiles[3].item():.6f} "
        f"p99={quantiles[4].item():.6f} "
        f"max={weights.max().item():.6f} "
        f"mean={weights.mean().item():.6f}"
    )
    aspect = torch.maximum(boxes[:, 2] / boxes[:, 3], boxes[:, 3] / boxes[:, 2])
    saturated = ((weights.min(dim=1).values <= 0.500001)
                 | (weights.max(dim=1).values >= 1.499999)).float().mean()
    print(f"aspect_ratio_p50={aspect.median().item():.6f} saturated={saturated.item():.6%}")
    for name, index in (
        ("near_square", torch.argmin(torch.abs(aspect - 1.0))),
        ("horizontal", torch.argmax(boxes[:, 2] / boxes[:, 3])),
        ("vertical", torch.argmax(boxes[:, 3] / boxes[:, 2])),
    ):
        box = boxes[index]
        print(
            f"{name}: width={box[2].item():.3f} height={box[3].item():.3f} "
            f"weights={weights[index].tolist()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--mode",
        choices=("fixed", "sensitivity", "sqrt_sensitivity"),
        default="sqrt_sensitivity",
    )
    args = parser.parse_args()
    self_check()
    if args.annotations:
        audit_annotations(args.annotations, args.mode)


if __name__ == "__main__":
    main()
