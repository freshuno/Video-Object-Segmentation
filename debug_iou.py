import os
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# 👤 1. KONFIGURACJA KLAS COCO
# -----------------------------
LABEL_MAP = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5,
    "train": 6, "truck": 7, "boat": 8, "traffic light": 9, "fire hydrant": 10,
    "stop sign": 11, "parking meter": 12, "bench": 13, "bird": 14, "cat": 15,
    "dog": 16, "horse": 17, "sheep": 18, "cow": 19, "elephant": 20, "bear": 21,
    "zebra": 22, "giraffe": 23
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# -------------------------------------------------------
# 👤 2. FUNKCJA: wczytanie GT z LabelMe do maski klasowej
# -------------------------------------------------------

def _decode_embedded_mask(mask_b64: str) -> np.ndarray:
    """Dekoduje base64 z LabelMe (shape_type == 'mask') do binarnej maski 0/1."""
    mask_bytes = base64.b64decode(mask_b64)
    img = Image.open(BytesIO(mask_bytes)).convert("L")  # grayscale
    arr = np.array(img)
    return (arr > 0).astype(np.uint8)  # binarna


def load_gt_mask_from_json(json_path: Path, full_shape):
    """Zamienia jeden plik .json z LabelMe na maskę klas (0 = tło, 1+ = klasy)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros(full_shape[:2], dtype=np.uint8)

    for shape_obj in data.get("shapes", []):
        label = shape_obj.get("label", "").strip()
        class_id = LABEL_MAP.get(label)
        if class_id is None:
            # pomijamy etykiety spoza mapy
            continue

        shape_type = shape_obj.get("shape_type", "polygon")
        points = np.array(shape_obj.get("points", []), dtype=np.int32)

        # -----------------------
        # 🔲 RECTANGLE  (2 punkty)
        # -----------------------
        if shape_type == "rectangle":
            if len(points) != 2:
                print(f"⚠️ Rectangle w {json_path.name} ma {len(points)} punktów, oczekiwano 2.")
                continue
            (x1, y1), (x2, y2) = points
            points = np.array([
                [int(x1), int(y1)], [int(x2), int(y1)],
                [int(x2), int(y2)], [int(x1), int(y2)]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [points], class_id + 1)

        # -----------------------------------
        # 🟦 POLYGON (>=3 punktów) / POLYLINE
        # -----------------------------------
        elif shape_type in {"polygon", "polyline"}:
            if len(points) < 3:
                print(f"⚠️ Pominięto {label} – za mało punktów (len={len(points)}) w {json_path.name}")
                continue
            cv2.fillPoly(mask, [points], class_id + 1)

        # -----------------------
        # 🖼️  MASK (embedded PNG)
        # -----------------------
        elif shape_type == "mask":
            mask_b64 = shape_obj.get("mask")
            if mask_b64 is None or len(points) != 2:
                print(f"⚠️ Mask bez danych lub bez bbox w {json_path.name}, pominięto.")
                continue
            # Dekoduj osadzoną maskę 0/1
            small_mask = _decode_embedded_mask(mask_b64)
            (x1, y1), (x2, y2) = points
            x1, y1 = map(int, (x1, y1))
            x2, y2 = map(int, (x2, y2))
            # Dopasuj rozmiar maski do bbox (LabelMe zapisuje mask w oryginalnym rozmiarze bboxa)
            target_w, target_h = x2 - x1, y2 - y1
            if (small_mask.shape[1], small_mask.shape[0]) != (target_w, target_h):
                small_mask = cv2.resize(small_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            # Nałóż maskę na właściwe miejsce
            mask[y1:y2, x1:x2][small_mask > 0] = class_id + 1

        else:
            print(f"⚠️ Nieobsługiwany shape_type: {shape_type} (plik {json_path.name})")

    return mask


# ------------------------------------------------------------
# 👁️ 3. WIZUALIZACJA GT vs PRED, różnice (TP/FP/FN kolorami)
# ------------------------------------------------------------

def visualize_comparison(gt_mask: np.ndarray, pred_mask: np.ndarray, title: str):
    """Pokazuje trzy obrazki: GT, predykcję i różnice (TP/FP/FN)."""
    diff = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

    all_classes = np.unique(np.concatenate((gt_mask.flatten(), pred_mask.flatten())))
    for class_id in all_classes:
        if class_id == 0:
            continue  # pomiń tło
        gt = gt_mask == class_id
        pr = pred_mask == class_id
        tp = np.logical_and(gt, pr)
        fp = np.logical_and(pr, ~gt)
        fn = np.logical_and(gt, ~pr)
        diff[tp] = [0, 255, 0]      # zielony → TP
        diff[fp] = [255, 0, 0]      # czerwony → FP
        diff[fn] = [0, 0, 255]      # niebieski → FN

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(gt_mask, cmap="nipy_spectral")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="nipy_spectral")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title("Diff (TP=green, FP=red, FN=blue)")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------
# 🚀 4. GŁÓWNA PĘTLA – porównanie n klatek
# ---------------------------------------

def main(pred_folder: str, gt_folder: str, limit: int = 5):
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    pred_files = sorted(pred_folder.glob("mask_*.png"))
    gt_files = sorted(gt_folder.glob("*.json"))

    if not pred_files or not gt_files:
        raise FileNotFoundError("Brak masek lub plików JSON w podanych folderach.")

    if len(pred_files) != len(gt_files):
        print(f"⚠️ Liczba masek ({len(pred_files)}) ≠ liczba JSON-ów ({len(gt_files)}) – używam min wspólnego.")

    for idx, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        if idx >= limit:
            break

        pred_mask = np.array(Image.open(pred_file))
        gt_mask = load_gt_mask_from_json(gt_file, pred_mask.shape)

        title = f"Frame {idx} → {gt_file.name} vs {pred_file.name}"
        visualize_comparison(gt_mask, pred_mask, title)


# -----------------------------
# 🏃‍♂️ 5. CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wizualne porównanie masek GT i predykcyjnych")
    parser.add_argument("--pred", required=True, help="Folder z maskami predykcyjnymi (.png)")
    parser.add_argument("--gt", required=True, help="Folder z LabelMe JSON")
    parser.add_argument("--limit", type=int, default=5, help="Ile pierwszych klatek porównać")
    args = parser.parse_args()

    main(args.pred, args.gt, args.limit)
