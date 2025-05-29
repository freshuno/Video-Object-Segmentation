import cv2
import torch
import numpy as np
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import sys


def load_deeplab_model():
    model = deeplabv3_mobilenet_v3_large(pretrained=True)
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


def run_deeplab_on_video(input_video_filename):
    base_folder = Path("data/raw")
    input_video_path = base_folder / input_video_filename

    if not input_video_path.exists():
        print(f"❗ Plik {input_video_path} nie istnieje.")
        return

    model = load_deeplab_model()

    cap = cv2.VideoCapture(str(input_video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output folders
    output_video_folder = Path("results") / "deeplabv3Video"
    output_video_folder.mkdir(parents=True, exist_ok=True)

    output_masks_folder = Path("results") / "deeplabv3" / input_video_path.stem
    output_masks_folder.mkdir(parents=True, exist_ok=True)

    # Video writer setup
    output_filename = input_video_path.stem + "_segmented.mp4"
    output_path = output_video_folder / output_filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Przetwarzanie wideo")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = segment_frame(model, image_rgb)
        mask_color = apply_colormap(mask)
        overlay = cv2.addWeighted(frame, 0.5, mask_color, 0.5, 0)

        # Save the overlay video frame
        out.write(overlay)

        # Save both versions of the mask
        cv2.imwrite(str(output_masks_folder / f"mask_{frame_count:05d}.png"), mask)  # Original 1-channel mask
        cv2.imwrite(str(output_masks_folder / f"mask_color_{frame_count:05d}.png"), mask_color)  # Color visualization

        frame_count += 1
        pbar.update(1)

    cap.release()
    out.release()
    print(f"\n✅ Wyniki zapisane w:")
    print(f"- Wideo z segmentacją: {output_path}")
    print(f"- Maski (oryginalne): {output_masks_folder}/mask_*.png")
    print(f"- Maski (kolorowe): {output_masks_folder}/mask_color_*.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Użycie: python segment_video.py <nazwa_pliku_wideo>")
        print("Przykład: python segment_video.py test.mp4")
    else:
        run_deeplab_on_video(sys.argv[1])