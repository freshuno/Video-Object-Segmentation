import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image)

def visualize(image, masks):
    plt.imshow(image.permute(1, 2, 0))
    for mask in masks:
        plt.imshow(mask, alpha=0.3, cmap='Reds')
    plt.axis("off")
    plt.show()

def main():
    print("Ładowanie modelu Mask R-CNN...")
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    image_path = "data/sample.jpg"
    image = load_image(image_path)
    with torch.no_grad():
        prediction = model([image])[0]

    masks = [m for i, m in enumerate(prediction['masks']) if prediction['scores'][i] > 0.8]
    masks = [m[0].numpy() > 0.5 for m in masks]

    visualize(image, masks)

if __name__ == "__main__":
    main()
