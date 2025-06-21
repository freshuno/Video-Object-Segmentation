import os
import argparse
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm

# -----------------------------
# 📥 Wczytaj maskę GT z .mat (dostosuj wg swojej struktury .mat)
# -----------------------------
def load_gt_mask_from_mat(mat_path: Path):
    mat = scipy.io.loadmat(mat_path)
    # Przykład dostępu (dostosuj do swojego pliku):
    # zakładam, że maska jest w mat['groundTruth'][0][0]['Segmentation']
    gt = mat['groundTruth'][0][0]
    mask = gt['Segmentation'][0][0]
    mask = mask.astype(np.uint8)
    return mask

# -----------------------------
# 📏 Obliczanie IoU binarnego (obiekt vs tło)
# -----------------------------
def compute_binary_iou(gt_mask, pred_mask):
    gt_binary = (gt_mask > 0)
    pred_binary = (pred_mask > 0)
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    return intersection / union if union > 0 else None

# -----------------------------
# 🛠️ Pomocnik do wyciągnięcia numeru ramki z nazwy pliku
# -----------------------------
def extract_number(name: str):
    import re
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else -1

# -----------------------------
# 🚀 Główna funkcja
# -----------------------------
def evaluate(pred_folder, gt_folder):
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    pred_files = sorted(pred_folder.glob("*.png"))
    gt_files = sorted(gt_folder.glob("*.mat"))

    if len(pred_files) != len(gt_files):
        print(f"⚠️ Uwaga: liczba plików różna: predykcji {len(pred_files)} vs GT {len(gt_files)}")

    n = min(len(pred_files), len(gt_files))

    iou_scores = []

    for i in tqdm(range(n), desc="🔍 Obliczanie IoU"):
        pred_mask = np.array(Image.open(pred_files[i]))
        gt_mask = load_gt_mask_from_mat(gt_files[i])

        if gt_mask.shape != pred_mask.shape:
            print(f"⚠️ Dopasowanie rozmiaru: {gt_files[i].name} ↔ {pred_files[i].name}")
            pred_mask = np.array(Image.open(pred_files[i]).resize(gt_mask.shape[::-1], Image.NEAREST))

        iou = compute_binary_iou(gt_mask, pred_mask)
        if iou is not None:
            iou_scores.append(iou)

    avg_iou = 100 * np.mean(iou_scores) if iou_scores else 0
    print(f"\n📊 Średnie IoU (obiekt vs tło): {avg_iou:.2f}%")


# -----------------------------
# 🏃‍♂️ Uruchomienie
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Porównanie masek predykcji PNG z GT w formacie MAT.")
    parser.add_argument("pred_folder", type=str, help="Folder z maskami PNG (predykcja).")
    parser.add_argument("gt_folder", type=str, help="Folder z maskami MAT (ground truth).")
    args = parser.parse_args()

    evaluate(args.pred_folder, args.gt_folder)

