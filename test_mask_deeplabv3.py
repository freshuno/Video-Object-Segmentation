import os
import cv2
import torch
import numpy as np
import sys
from torchvision import models, transforms
from tqdm import tqdm
from pathlib import Path

def load_deeplab_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def segment_frame(model, image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    prediction = output.argmax(0).byte().cpu().numpy()
    return prediction

def apply_colormap(mask):
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

def run_deeplab_on_folder(input_folder):
    input_folder = Path(input_folder)
    folder_name = input_folder.name
    output_folder = Path("results/deeplabv3") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    model = load_deeplab_model()
    frame_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".jpg"))

    for filename in tqdm(frame_files, desc="Przetwarzanie klatek"):
        input_path = input_folder / filename
        image = cv2.imread(str(input_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = segment_frame(model, image_rgb)
        mask_color = apply_colormap(mask)

        overlay = cv2.addWeighted(image, 0.5, mask_color, 0.5, 0)
        output_path = output_folder / filename
        cv2.imwrite(str(output_path), overlay)

    print(f"✅ Segmentacja zakończona. Wyniki zapisano w: {output_folder}")

run_deeplab_on_folder(sys.argv[1])
