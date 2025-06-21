import os
import argparse
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm

def load_gt_mask_from_mat(mat_path: Path):
    mat = scipy.io.loadmat(mat_path)
    gt = mat['groundTruth'][0][0]
    mask = gt['Segmentation'][0][0]
    return mask.astype(np.uint8)

def compute_binary_iou(gt_mask, pred_mask):
    gt_binary = gt_mask > 0
    pred_binary = pred_mask > 0
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    return intersection / union if union > 0 else None

def evaluate_category(pred_path: Path, gt_path: Path):
    pred_files = sorted(pred_path.glob("*.png"))
    gt_files = sorted(gt_path.glob("*.mat"))

    n = min(len(pred_files), len(gt_files))
    if n == 0:
        return None, 0

    iou_scores = []

    for i in range(n):
        pred_mask = np.array(Image.open(pred_files[i]))
        gt_mask = load_gt_mask_from_mat(gt_files[i])

        if gt_mask.shape != pred_mask.shape:
            pred_mask = np.array(Image.open(pred_files[i]).resize(gt_mask.shape[::-1], Image.NEAREST))

        iou = compute_binary_iou(gt_mask, pred_mask)
        if iou is not None:
            iou_scores.append(iou)

    if not iou_scores:
        return None, 0

    return np.mean(iou_scores), len(iou_scores)

def evaluate_all(pred_root: str, gt_root: str):
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)

    total_scores = []
    total_count = 0

    gt_categories = [p for p in gt_root.iterdir() if p.is_dir()]

    for gt_cat_path in sorted(gt_categories):
        category = gt_cat_path.name
        gt_path = gt_cat_path
        pred_path = pred_root / "motion" / category / "mask"  # <- tu dodajemy motion

        if not pred_path.exists():
            print(f"⚠️ Pomijam kategorię '{category}' — brak folderu: {pred_path}")
            continue

        avg_iou, count = evaluate_category(pred_path, gt_path)
        if avg_iou is not None:
            print(f"✅ {category:30s} — IoU: {avg_iou * 100:.2f}% na {count} próbkach")
            total_scores.extend([avg_iou] * count)
            total_count += count
        else:
            print(f"⚠️ {category:30s} — brak wspólnych masek.")

    if total_scores:
        global_avg = 100 * np.mean(total_scores)
        print(f"\n📊 ŚREDNIE GLOBALNE IoU (obiekt vs tło): {global_avg:.2f}% na {total_count} próbkach")
    else:
        print("\n❌ Nie znaleziono żadnych wspólnych masek do oceny.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Porównanie masek predykcji PNG z GT w formacie MAT.")
    parser.add_argument("pred_folder", type=str, help="Folder z maskami PNG (predykcja).")
    parser.add_argument("gt_folder", type=str, help="Folder z maskami MAT (ground truth).")
    args = parser.parse_args()

    evaluate_all(args.pred_folder, args.gt_folder)


