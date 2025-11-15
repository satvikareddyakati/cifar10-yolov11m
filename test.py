#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import torch
from ultralytics import YOLO


def detect_device(cli_device: str | None) -> str | int:
    if cli_device:
        return cli_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11 classification model on CIFAR-10 test split and save metrics/visualizations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/satvikaakati/Desktop/yolo_project/runs/classify/train2/weights/best.pt",
        help="Path to trained .pt weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/Users/satvikaakati/datasets/cifar10",
        help="Dataset root (Ultralytics will resolve splits inside).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=160,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'mps', 'cpu', or CUDA index like '0'. If omitted, auto-select.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)

    model = YOLO(args.model)
    results = model.val(data=args.data, split="test", imgsz=args.imgsz, device=device)

    metrics = {"top1": float(results.top1), "top5": float(results.top5)}
    print(metrics)

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, "test_metrics.json")
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Ultralytics automatically saves visualization files like confusion matrices into save_dir.
        # We surface their expected locations for convenience.
        print("Results saved to:", save_dir)
        for name in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "val_batch0_pred.jpg",
            "val_batch1_pred.jpg",
            "val_batch2_pred.jpg",
        ):
            path = os.path.join(save_dir, name)
            if os.path.exists(path):
                print("Visualization:", path)


if __name__ == "__main__":
    main()


