import cv2
import torch
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Wyciszenie ostrzeżeń
warnings.filterwarnings("ignore", message="torch.meshgrid")


def load_mask_rcnn_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def apply_colormap(mask):
    """Ta sama funkcja co w DeepLabV3 dla spójności"""
    colormap = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ])
    mask_color = colormap[mask % len(colormap)]
    return mask_color.astype(np.uint8)


def create_combined_mask(outputs, height, width):
    """Tworzy połączoną maskę wszystkich instancji"""
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for mask, class_id in zip(masks, classes):
        combined_mask[mask > 0] = class_id + 1  # +1 aby uniknąć klasy 0 (tło)

    return combined_mask


def run_mask_rcnn_on_video(input_video_filename):
    base_folder = Path("data/raw")
    input_video_path = base_folder / input_video_filename

    if not input_video_path.exists():
        print(f"🚫 Plik {input_video_path} nie istnieje.")
        return

    predictor, cfg = load_mask_rcnn_model()
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    cap = cv2.VideoCapture(str(input_video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Przygotowanie folderów wyjściowych
    output_video_folder = Path("results/mask_rcnnVideo")
    output_video_folder.mkdir(parents=True, exist_ok=True)

    output_masks_folder = Path("results/mask_rcnn") / input_video_path.stem
    output_masks_folder.mkdir(parents=True, exist_ok=True)

    # Video writer setup
    output_filename = input_video_path.stem + "_segmented.mp4"
    output_path = output_video_folder / output_filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("🔁 Wstępne przygotowanie modelu...")
    ret, warmup_frame = cap.read()
    if ret:
        _ = predictor(warmup_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset do początku

    pbar = tqdm(total=total_frames, desc="Przetwarzanie wideo", ncols=100)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = predictor(frame)

        # Tworzenie połączonej maski (1-kanałowej)
        combined_mask = create_combined_mask(outputs, height, width)
        mask_color = apply_colormap(combined_mask)

        # Zapis masek
        cv2.imwrite(str(output_masks_folder / f"mask_{frame_count:05d}.png"), combined_mask)
        cv2.imwrite(str(output_masks_folder / f"mask_color_{frame_count:05d}.png"), mask_color)

        # Wizualizacja i zapis wideo
        v = Visualizer(frame[:, :, ::-1], metadata, scale=1.0)
        out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
        out.write(out_frame)

        frame_count += 1
        pbar.update(1)

    cap.release()
    out.release()
    print(f"\n✅ Wyniki zapisane w:")
    print(f"- Wideo z segmentacją: {output_path}")
    print(f"- Maski (oryginalne): {output_masks_folder}/mask_*.png")
    print(f"- Maski (kolorowe): {output_masks_folder}/mask_color_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchom Mask R-CNN na pliku wideo.")
    parser.add_argument("input_video", type=str, help="Nazwa pliku wideo z folderu data/raw")
    args = parser.parse_args()

    run_mask_rcnn_on_video(args.input_video)