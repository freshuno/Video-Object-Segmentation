# 🎥 Object Segmentation in Video

A system for object segmentation and tracking in video sequences using selected deep learning models (Mask R-CNN, DeepLab, MiVOS, PReMVOS).

---

## 🧠 Applications
- Surveillance
- Traffic analysis
- Object tracking in video footage

---

## ⚙️ System Requirements

- Python
- Anaconda
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

### 5. Prepare dataset
Download your video dataset (e.g., .mpg files) and place them inside the data/raw/ folder. 

---

## 📂 Project Structure

```
segmentation/
├── data/
│   ├── raw/                # original videos (.mpg)
│   ├── frames/             # extracted video frames
│   └── annotations/        # masks (if available)
├── results/                # generated masks and visualizations
├── models/                 # (for additional models)
├── scripts/                # (for training/testing)
├── test_mask_deeplabv3.py  # DeepLab v3+ test script on custom frames
├── test_mask_rcnn.py       # Mask R-CNN test script on custom frames
├── extract_frames.py       # script to extract frames from .mpg
└── README.md
```

---

## ▶️ How to Run

Set paths using command line arguments as needed (check `-h` or `--help`).


### 1. Extract frames from `.mpg` videos
```bash
python extract_frames.py <path/to/file.mpg> <data/frames/file>
```

### 2. Run models on the extracted frames
```bash
python test_mask_rcnn.py <data/frames/file>
```

```bash
python test_mask_deeplabv3.py <data/frames/file>
```

For example results will be saved in:
```
results/<model>/file/
```

---

## 🧪 Check if GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---
