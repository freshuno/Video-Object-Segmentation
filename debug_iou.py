import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

LABEL_MAP = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5,
    "train": 6, "truck": 7, "boat": 8, "traffic light": 9, "fire hydrant": 10,
    "stop sign": 11, "parking meter": 12, "bench": 13, "bird": 14, "cat": 15,
    "dog": 16, "horse": 17, "sheep": 18, "cow": 19, "elephant": 20, "bear": 21,
    "zebra": 22, "giraffe": 23
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_gt_mask_from_json(json_path, shape):
    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros(shape[:2], dtype=np.uint8)
    for shape_obj in data['shapes']:
        label = shape_obj['label'].strip()
        class_id = LABEL_MAP.get(label, None)
        if class_id is None:
            continue

        if shape_obj['shape_type'] == 'rectangle':
            (x1, y1), (x2, y2) = shape_obj['points']
            points = np.array([
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)]
            ], dtype=np.int32)
        else:
            points = np.array(shape_obj['points'], dtype=np.int32)

        cv2.fillPoly(mask, [points], class_id + 1)
    return mask


def visualize_comparison(gt_mask, pred_mask, title):
    diff = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    for class_id in np.unique(np.concatenate((gt_mask.flatten(), pred_mask.flatten()))):
        if class_id == 0:
            continue  # skip background

        gt = gt_mask == class_id
        pr = pred_mask == class_id

        tp = np.logical_and(gt, pr)
        fp = np.logical_and(pr, ~gt)
        fn = np.logical_and(gt, ~pr)

        diff[tp] = [0, 255, 0]    # green = true positive
        diff[fp] = [255, 0, 0]    # red = false positive
        diff[fn] = [0, 0, 255]    # blue = false negative

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(gt_mask, cmap='nipy_spectral')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='nipy_spectral')
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title("Difference (TP=green, FP=red, FN=blue)")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main(pred_folder, gt_folder, limit=5):
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    pred_files = sorted(pred_folder.glob("mask_*.png"))
    gt_files = sorted(gt_folder.glob("*.json"))

    if len(pred_files) != len(gt_files):
        print(f"⚠️ Liczba masek ({len(pred_files)}) ≠ liczba JSON-ów ({len(gt_files)}) — porównuję najmniejszy wspólny zbiór.")

    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        if i >= limit:
            break

        pred_mask = np.array(Image.open(pred_file))
        shape = pred_mask.shape
        gt_mask = load_gt_mask_from_json(gt_file, shape)

        title = f"Frame {i} — {gt_file.name} vs {pred_file.name}"
        visualize_comparison(gt_mask, pred_mask, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wizualne porównanie masek GT i predykcyjnych")
    parser.add_argument("--pred", required=True, help="Folder z maskami predykcyjnymi (.png)")
    parser.add_argument("--gt", required=True, help="Folder z LabelMe JSON")
    parser.add_argument("--limit", type=int, default=5, help="Liczba klatek do porównania")
    args = parser.parse_args()

    main(args.pred, args.gt, args.limit)
