# 🎥 Object Segmentation in Video

A system for object segmentation and tracking in video sequences using selected deep learning models (Mask R-CNN, DeepLab, MiVOS, PReMVOS).

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

### 5. Prepare dataset
Download your video dataset (e.g., .mpg files) and place them inside the data/raw/ folder. 

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

To change the input video (or folder) and the output folder where processed results are saved, you need to edit the paths directly in the scripts:

In extract_frames.py modify the path to the video file you want to process (usually near the top of the script).

In test_mask_rcnn.py change the input folder with extracted frames and the output folder where results are saved.

Make sure to update these paths before running the scripts to process different videos or save results to different locations.


### 1. Extract frames from `.mpg` videos
```bash
python extract_frames.py
```

### 2. Run Mask R-CNN on the extracted frames
```bash
python test_mask_rcnn.py
```

For example results will be saved in:
```
results/mask_rcnn/film1/
```

---

## 🧪 Check if GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---
