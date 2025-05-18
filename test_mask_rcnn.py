import os
import cv2
import torch
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def run_mask_rcnn_on_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Konfiguracja modelu
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

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            path = os.path.join(input_folder, filename)
            image = cv2.imread(path)
            outputs = predictor(image)

            v = Visualizer(image[:, :, ::-1],
                           MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                           scale=1.0)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print(f"✅ Segmentacja zakończona. Wyniki zapisano w: {output_folder}")

# === Przykład użycia:
run_mask_rcnn_on_folder("data/frames/film1", "results/mask_rcnn/film1")
