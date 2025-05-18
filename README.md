# 🎥 Object Segmentation in Video – Project (VIRAT + Mask R-CNN)

A system for object segmentation and tracking in video sequences using selected deep learning models (Mask R-CNN, DeepLab, MiVOS, PReMVOS). Currently tested on the **VIRAT** dataset.

---

## 🧠 Applications
- Surveillance
- Traffic analysis
- Object tracking in video footage

---

## ⚙️ System Requirements

- Python 3.10 (recommended)
- Anaconda (recommended)
- Windows / Linux
- Optional: GPU with CUDA (for acceleration)

---

## 🧪 Tested Models

- ✅ Mask R-CNN (`detectron2`)
- 🔜 DeepLab v3+
- 🔜 MiVOS
- 🔜 PReMVOS

---

## 🛠️ Installation (Anaconda)

### 1. Install [Anaconda](https://www.anaconda.com/products/distribution)

### 2. Create environment
```bash
a) conda env create -f environment.yml
   conda activate segmentacja
b) (if a doesn't work)
   conda create -n segmentation python=3.10 -y
   conda activate segmentation
```

### 3. Install required libraries
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib scikit-learn numpy tqdm ffmpeg-python imageio imageio-ffmpeg
```

### 4. Install Detectron2 (from pre-built `.whl`)
Download [detectron2-0.6-cp310-cp310-win_amd64.whl](https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/detectron2-0.6-cp310-cp310-win_amd64.whl) and install it:
```bash
pip install detectron2-0.6-cp310-cp310-win_amd64.whl
```

---

## 📂 Project Structure

```
segmentation/
├── data/
│   ├── raw/            # original videos (.mpg)
│   ├── frames/         # extracted video frames
│   └── annotations/    # masks (if available)
├── results/            # generated masks and visualizations
├── models/             # (for additional models)
├── scripts/            # (for training/testing)
├── test_mask_rcnn.py   # Mask R-CNN test script on custom frames
├── extract_frames.py   # script to extract frames from .mpg
└── README.md
```

---

## ▶️ How to Run

### 1. Extract frames from `.mpg` videos
```bash
python extract_frames.py
```

### 2. Run Mask R-CNN on the extracted frames
```bash
python test_mask_rcnn.py
```

Results will be saved in:
```
results/mask_rcnn/film1/
```

---

## 🧪 Check if GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---
