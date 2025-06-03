import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# COCO: class name -> class ID
LABEL_MAP = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5,
    "train": 6, "truck": 7, "boat": 8, "traffic light": 9, "fire hydrant": 10,
    "stop sign": 11, "parking meter": 12, "bench": 13, "bird": 14, "cat": 15,
    "dog": 16, "horse": 17, "sheep": 18, "cow": 19, "elephant": 20, "bear": 21,
    "zebra": 22, "giraffe": 23
    # Dodaj więcej jeśli potrzebujesz
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def load_gt_mask_from_json(json_path, shape):
    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros(shape[:2], dtype=np.uint8)
    for shape_obj in data['shapes']:
        label = shape_obj['label']
        class_id = LABEL_MAP.get(label, None)
        if class_id is None:
            continue
        points = np.array(shape_obj['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], class_id + 1)  # +1 to match model output
    return mask

def compute_class_iou(gt_mask, pred_mask, class_id):
    gt = (gt_mask == class_id + 1)
    pred = (pred_mask == class_id + 1)
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union > 0 else None

def evaluate(pred_folder, gt_folder):
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    pred_files = sorted(pred_folder.glob("mask_*.png"))
    gt_files = sorted(gt_folder.glob("*.json"))

    if len(pred_files) != len(gt_files):
        print(f"⚠️ Uwaga: liczba masek ({len(pred_files)}) ≠ liczba plików GT ({len(gt_files)})")

    class_scores = {i: [] for i in LABEL_MAP.values()}

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=min(len(pred_files), len(gt_files)), desc="🔍 Obliczanie IoU"):
        pred_mask = np.array(Image.open(pred_file))
        shape = pred_mask.shape
        gt_mask = load_gt_mask_from_json(gt_file, shape)

        for class_id in LABEL_MAP.values():
            iou = compute_class_iou(gt_mask, pred_mask, class_id)
            if iou is not None:
                class_scores[class_id].append(iou)

    print("\n📊 Średnie IoU dla każdej klasy:")
    print(f"{'Klasa':<20} {'IoU (%)':>10}")
    print("-" * 32)
    for class_id, scores in class_scores.items():
        if scores:
            avg_iou = 100 * np.mean(scores)
            print(f"{ID_TO_LABEL[class_id]:<20} {avg_iou:>10.2f}")
        else:
            print(f"{ID_TO_LABEL[class_id]:<20} {'–':>10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Porównanie masek modelu z LabelMe GT")
    parser.add_argument("--pred", required=True, help="Folder z maskami predykcyjnymi (.png)")
    parser.add_argument("--gt", required=True, help="Folder z LabelMe JSON")
    args = parser.parse_args()

    evaluate(args.pred, args.gt)
