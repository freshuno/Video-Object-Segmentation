import argparse
import os
from PIL import Image
import numpy as np

def load_mask(path, object_id=1):
    mask = np.array(Image.open(path))
    return (mask == object_id).astype(np.uint8)

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def main(gt_dir, pred_dir, object_id):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    common = set(gt_files) & set(pred_files)

    if not common:
        print("❌ Brak wspólnych plików PNG w obu folderach.")
        return

    ious = []

    for fname in sorted(common):
        gt_mask = load_mask(os.path.join(gt_dir, fname), object_id)
        pred_mask = load_mask(os.path.join(pred_dir, fname), object_id)
        iou = compute_iou(gt_mask, pred_mask)
        ious.append(iou)
        print(f"{fname} - IOU: {iou:.4f}")

    mean_iou = sum(ious) / len(ious)
    print(f"\n📊 Średnie IOU: {mean_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True, help='Folder z ground truth maskami')
    parser.add_argument('--pred_dir', required=True, help='Folder z przewidywanymi maskami')
    parser.add_argument('--object_id', type=int, default=1, help='ID obiektu do porównania (domyślnie 1)')
    args = parser.parse_args()

    main(args.gt_dir, args.pred_dir, args.object_id)
