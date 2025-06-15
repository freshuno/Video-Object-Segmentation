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

def main(pred_folder, gt_folder, limit=5):
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    # Posortowane listy plików
    gt_files = sorted(gt_folder.glob("*.mat"))
    pred_files = sorted(pred_folder.glob("*.png"))

    # Dopasowanie po indeksie sortowania
    num_pairs = min(len(gt_files), len(pred_files), limit)
    if num_pairs == 0:
        print("🚫 Brak wspólnych par do porównania.")
        return

    for i in range(num_pairs):
        gt_file = gt_files[i]
        pred_file = pred_files[i]

        pred_mask = (np.array(Image.open(pred_file).convert("L")) > 0).astype(np.uint8)
        gt_mask = load_gt_mask_from_mat(gt_file)

        # Dopasuj rozmiar jeśli nie pasuje
        if gt_mask.shape != pred_mask.shape:
            print(f"⚠️ Rozmiar nie pasuje dla pary {gt_file.name} ↔ {pred_file.name}, dopasowuję...")
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        visualize_comparison(gt_mask, pred_mask, f"{pred_file.name} vs {gt_file.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    main(args.pred, args.gt, args.limit)
