# 🎥 Segmentacja Obiektów na Wideo (Wersja 2.0)

**Dane wejściowe:**  
Przykładowy zbiór danych do segmentacji możesz pobrać ze strony:  
https://lmb.informatik.uni-freiburg.de/resources/datasets/vsb.en.html

System do segmentacji obiektów na klatkach wideo z użyciem wielu modeli deep learning (Mask R-CNN, DeepLab, ...), obsługujący wiele zestawów danych (np. "motion", "general"). System posiada wygodne GUI do wyboru datasetu/modelu oraz wizualizacji wyników i ewaluacji IoU.

---

## 🧠 Zastosowania
- Monitoring, analiza ruchu, systemy śledzenia
- Analiza ruchu drogowego
- Segmentacja i tracking w filmach

---

## ⚙️ Wymagania systemowe

- Python 3.10
- Anaconda
- Windows / Linux
- (Opcjonalnie) GPU z CUDA

---

## 🧪 Obsługiwane modele

- ✅ Mask R-CNN (Detectron2)
- ✅ DeepLab v3+ (torchvision)
- 🔜 Możliwość rozbudowy o kolejne modele (np. MiVOS, PReMVOS)

---

## 🛠️ Instalacja (Anaconda)

### 1. Zainstaluj [Anacondę](https://www.anaconda.com/products/distribution)

### 2. Utwórz środowisko
```bash
conda create -n segmentation python=3.10 -y
conda activate segmentation
```

### 3. Zainstaluj wymagane biblioteki
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python matplotlib scikit-learn numpy tqdm ffmpeg-python imageio imageio-ffmpeg
```

### 4. Zainstaluj Detectron2 (gotowy `.whl`)
Pobierz [detectron2-0.6-cp310-cp310-win_amd64.whl](https://github.com/carlosedubarreto/CEB_4d_Humans/blob/main/detectron2-0.6-cp310-cp310-win_amd64.whl) i zainstaluj:
```bash
pip install detectron2-0.6-cp310-cp310-win_amd64.whl
```

---

## 📂 Struktura katalogów projektu

```
project_root/
├── Data/
│   ├── motion/
│   │   ├── Images/       # klatki video pogrupowane w foldery (po filmach)
│   │   └── Groundtruth/  # ground truth maski w .mat (po filmach)
│   └── general/
│       ├── Images/
│       └── Groundtruth/
├── Results/
│   ├── motion/
│   │   └── Images/
│   │       ├── Rcnn/        # Wyniki segmentacji Mask R-CNN
│   │       └── Deeplab/     # Wyniki segmentacji DeepLab
│   └── general/
│       └── Images/
│           ├── Rcnn/
│           └── Deeplab/
├── debug_iou.py           # GUI do debugowania masek i IoU
├── evaluate.py        # GUI do zbiorczego podsumowania IoU dla modeli i datasetów
├── mask_rcnn.py           # Segmentacja Mask R-CNN (wszystkie filmy, dowolny dataset)
├── deeplab.py   # Segmentacja DeepLab (wszystkie filmy, dowolny dataset)
├── prepare_folders.py   # Przygotowanie folderów
└── README.md
```

---

## 📥 Przygotowanie własnego datasetu

1. Pobierz klatki i maski z: https://lmb.informatik.uni-freiburg.de/resources/datasets/vsb.en.html  
2. Umieść je w strukturze:
```
Data/
└── <nazwa_datasetu>/
    ├── Images/
    │   └── <nazwa_filmu>/
    │       └── frame_00001.jpg
    └── Groundtruth/
        └── <nazwa_filmu>/
            └── frame_00001.mat
```
3. Datasety mogą nazywać się np. `motion`, `general` itd.

---

## ▶️ Jak uruchomić?

Wszystkie uruchomienia odbywają się przez **GUI**:
1. Program zapyta o dataset ("motion", "general", ...), a następnie o model ("Rcnn", "Deeplab", ...).
2. Ścieżki oraz struktura folderów wykrywane są automatycznie.

### 1. Segmentacja Mask R-CNN
```bash
python mask_rcnn.py
```
### 2. Segmentacja DeepLab
```bash
python deeplab.py
```
### 3. Debugowanie masek, podgląd błędów i IoU
```bash
python debug_iou.py
```
### 4. Zbiorcza ocena IoU (tabela, podsumowanie)
```bash
python evaluate.py
```

---

## 🧪 Sprawdzenie dostępności GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
