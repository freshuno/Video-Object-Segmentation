import os
import cv2
import torch
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ✅ Wyciszenie ostrzeżenia torch.meshgrid
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

def run_mask_rcnn_on_folder(input_folder):
    input_folder = Path(input_folder)
    folder_name = input_folder.name
    output_folder = Path("results/mask_rcnn") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    predictor, cfg = load_mask_rcnn_model()

    # Lista plików JPG
    frame_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".jpg"))

    if not frame_files:
        print("🚫 Brak plików .jpg w podanym folderze.")
        return

    # ✅ Warm-up modelu – unikamy warninga w trakcie paska
    warmup_image = cv2.imread(str(input_folder / frame_files[0]))
    _ = predictor(warmup_image)

    # ✅ Pasek postępu
    for filename in tqdm(frame_files, desc="Przetwarzanie obrazów", ncols=100):
        path = input_folder / filename
        image = cv2.imread(str(path))
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1],
                       MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                       scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        output_path = output_folder / filename
        cv2.imwrite(str(output_path), out.get_image()[:, :, ::-1])

    print(f"✅ Segmentacja zakończona. Wyniki zapisano w: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchom Mask R-CNN na folderze z obrazami.")
    parser.add_argument("input_folder", type=str, help="Ścieżka do folderu z obrazami wejściowymi")
    args = parser.parse_args()

    run_mask_rcnn_on_folder(args.input_folder)
