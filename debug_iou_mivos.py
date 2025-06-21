import os
import numpy as np
import scipy.io
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_gt_mask_from_mat(mat_path: Path):
    mat = scipy.io.loadmat(mat_path)
    gt = mat['groundTruth'][0]
    first_gt = gt[0]
    segmentation_field = first_gt['Segmentation']
    mask = segmentation_field[0][0]
    return (mask > 0).astype(np.uint8)

def visualize_comparison(gt_mask, pred_mask, title):
    diff = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    tp = np.logical_and(gt_mask == 1, pred_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(gt_mask == 1, pred_mask == 0)

    diff[tp] = [0, 255, 0]
    diff[fp] = [255, 0, 0]
    diff[fn] = [0, 0, 255]

    plt.figure(figsize=(15, 6))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title("Różnice (TP/FP/FN)")
    plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def main(pred_root, gt_root, limit=5):
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)

    gt_categories = [f for f in gt_root.iterdir() if f.is_dir()]

    for category_path in sorted(gt_categories):
        category = category_path.name

        gt_folder = category_path
        pred_folder = pred_root / "motion" / category / "mask"

        if not pred_folder.exists():
            print(f"⚠️ Pomijam kategorię '{category}' — brak folderu: {pred_folder}")
            continue

        gt_files = sorted(gt_folder.glob("*.mat"))
        pred_files = sorted(pred_folder.glob("*.png"))

        num_pairs = min(len(gt_files), len(pred_files), limit)
        if num_pairs == 0:
            print(f"⚠️ Pomijam kategorię '{category}' — brak wspólnych plików.")
            continue

        print(f"✅ Kategoria '{category}': porównuję {num_pairs} par...")

        for i in range(num_pairs):
            gt_file = gt_files[i]
            pred_file = pred_files[i]

            pred_mask = (np.array(Image.open(pred_file).convert("L")) > 0).astype(np.uint8)
            gt_mask = load_gt_mask_from_mat(gt_file)

            if gt_mask.shape != pred_mask.shape:
                print(f"⚠️ Rozmiar nie pasuje ({gt_file.name} vs {pred_file.name}) — skaluję.")
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            visualize_comparison(gt_mask, pred_mask, f"[{category}] {pred_file.name} vs {gt_file.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Folder z predykcjami (zawiera podfoldery kategorii)")
    parser.add_argument("--gt", required=True, help="Folder z ground truth (zawiera podfoldery kategorii)")
    parser.add_argument("--limit", type=int, default=5, help="Maksymalna liczba przykładów na kategorię")
    args = parser.parse_args()

    main(args.pred, args.gt, args.limit)
