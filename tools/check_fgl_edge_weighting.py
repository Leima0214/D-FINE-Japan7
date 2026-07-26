#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zoo.dfine.dfine_criterion import fgl_edge_weights


def main():
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.8, 0.2],
            [0.5, 0.5, 0.2, 0.8],
        ]
    )
    for mode in ("fixed", "sensitivity"):
        weights = fgl_edge_weights(boxes, mode=mode)
        assert torch.allclose(weights.mean(dim=1), torch.ones(3))
        assert torch.allclose(weights[0], torch.ones(4))
        assert weights[1, 1] > weights[1, 0] and weights[1, 3] > weights[1, 2]
        assert weights[2, 0] > weights[2, 1] and weights[2, 2] > weights[2, 3]
    print("FGL edge-weight self-check PASS")


if __name__ == "__main__":
    main()
