
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
Download your video dataset and place them inside the `data/raw/` folder. 

---

## 📂 Project Structure

```
segmentation/
├── data/
│   ├── raw/                # original videos
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

## A) Frame-by-Frame Segmentation
### 1. Extract frames from videos
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

Example results will be saved in:
```
results/<model>/file/
```

MiVOS model:

a) getting all the required models:
```bash
python download_model.py
```
b) running gui:
```bash
python interactive_gui.py --video <path to video>
```
or
```bash
python interactive_gui.py --images <path to a folder of images>
```

---

## B) Video Segmentation
### 1. Run models on videos (file must be in `data/raw` directory)

```bash
python deeplabv3_video.py <file>
```

Example results will be saved in:
```
results/<model>Video/file/
```

---

## 🧰 Creating Ground Truth Masks (Using LabelMe)

To generate ground truth annotations from video frames:

### 1. Install LabelMe
```bash
pip install labelme
```

### 2. Launch LabelMe tool
```bash
labelme
```

### 3. Select folder with extracted video frames  
- Navigate to `data/frames/<video_name>`  
- Annotate each frame by drawing polygons around objects of interest  
- Save each annotation — this will create a `.json` file per image

The annotations will be used later for IoU evaluation against model predictions.

---

## 📏 Evaluate IoU between Model Predictions and Ground Truth

### Use the following command to TEST IoU evaluation:

```bash
python debug_iou.py --pred results/mask_rcnn/<video_folder> --gt data/frames/<video_folder>
```

### Use the following command to run IoU evaluation:

```bash
python evaluate_iou.py --pred results/mask_rcnn/<video_folder> --gt data/frames/<video_folder>
```

- `--pred` is the folder with the model's predicted masks (e.g., `mask_00001.png`)  
- `--gt` is the folder containing ground truth annotations in LabelMe JSON format  

The script will:
- Convert LabelMe JSONs to class masks
- Align class indices with the COCO dataset (e.g., person=1, car=3, etc.)
- Compute per-class IoU and display a summary table

---

## 🧪 Check if GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
