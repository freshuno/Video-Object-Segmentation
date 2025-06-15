import os
import torch
import numpy as np
from PIL import Image
import scipy.io
from tqdm import tqdm
import argparse

from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from inference_core import InferenceCore

def images_to_torch(frames, device='cpu'):
    """
    Converts a list of HxWx3 numpy arrays to a torch tensor Bx3xHxW normalized to [0,1]
    """
    frame_tensors = []
    for frame in frames:
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensors.append(frame)
    return torch.stack(frame_tensors).to(device)

def load_mask_from_mat(mat_path):
    mat = scipy.io.loadmat(mat_path)
    gt = mat['groundTruth'][0]
    first_gt = gt[0]
    segmentation_field = first_gt['Segmentation']
    print(f"Segmentation field shape: {segmentation_field.shape}")
    print(f"Segmentation field type: {type(segmentation_field)}")
    mask = segmentation_field[0][0]
    print(f"Mask shape: {mask.shape}")
    return mask.astype(np.uint8)


def save_mask(mask, path):
    mask_img = Image.fromarray(mask)
    mask_img.save(path)

def main(args):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # wymuszamy CPU
    
    frame_names = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    frames = [np.array(Image.open(os.path.join(args.frames_dir, f))) for f in frame_names]
    frames_tensor = images_to_torch(frames, device=device)
    frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, 3, H, W)

    # Load models
    print("🔁 Loading models from saves/")
    prop_model = PropagationNetwork().to(device).eval()
    prop_model.load_state_dict(torch.load('saves/propagation_model.pth', map_location=device))

    fusion_model = FusionNet().to(device).eval()
    fusion_model.load_state_dict(torch.load('saves/fusion.pth', map_location=device))

    s2m_model = S2M().to(device).eval()
    s2m_model.load_state_dict(torch.load('saves/s2m.pth', map_location=device))

    # processor = InferenceCore(prop_model, fusion_model, frames_tensor, num_objects=0)
    # print(dir(processor))
    # Load initial mask
    init_mask_name = os.path.splitext(frame_names[0])[0] + '.mat'
    init_mask_path = os.path.join(args.masks_dir, init_mask_name)
    init_mask_np = load_mask_from_mat(init_mask_path)

    init_mask_tensor = torch.tensor(init_mask_np, dtype=torch.uint8, device=device).unsqueeze(0)
    print("Init mask tensor shape:", init_mask_tensor.shape)
    #processor.set_all_labels(init_mask_tensor)
    if init_mask_tensor.ndim == 3:
    	init_mask_tensor = init_mask_tensor.unsqueeze(1)  # (1, 1, H, W)

    print("Mask shape before interact:", init_mask_tensor.shape)

    num_objects = init_mask_tensor.shape[1] - 1  # jeśli init_mask_tensor ma shape (1, C, H, W), C to liczba obiektów + tło
    processor = InferenceCore(prop_model, fusion_model, frames_tensor, num_objects=num_objects)

    # Propagate
    print(f"🚀 Propagating from {frame_names[0]}")
    all_masks = processor.interact(init_mask_tensor, 0)  # zwraca numpy array (num_frames, H, W)
    print("all_masks.shape:", all_masks.shape)
    # processor.interact(None, frame_idx=0)

    os.makedirs(args.output_dir, exist_ok=True)
    for i, f in tqdm(enumerate(frame_names), total=len(frame_names)):
        #mask = processor.prob_to_mask(i)
        mask = all_masks[i]
        out_path = os.path.join(args.output_dir, f)
        save_mask(mask, out_path)

    print(f"✅ Done. Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True, help='Folder with input frames')
    parser.add_argument('--masks_dir', required=True, help='Folder with .mat masks')
    parser.add_argument('--output_dir', required=True, help='Where to save predicted masks')
    args = parser.parse_args()
    main(args)

