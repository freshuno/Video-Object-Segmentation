import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from pathlib import Path
import warnings

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QProgressBar, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

warnings.filterwarnings("ignore", category=UserWarning)

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
    input_tensor = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    prediction = output.argmax(0).byte().cpu().numpy()
    binary_mask = (prediction != 0).astype(np.uint8) * 255
    return binary_mask

def get_available_datasets():
    base = Path("Data")
    return [f.name for f in base.iterdir() if f.is_dir()]

class DatasetSelector(QWidget):
    def __init__(self, on_select_callback):
        super().__init__()
        self.setWindowTitle("Wybierz dataset")
        self.on_select_callback = on_select_callback

        label = QLabel("Wybierz dataset:")
        self.combo = QComboBox()
        datasets = get_available_datasets()
        if not datasets:
            QMessageBox.critical(self, "Błąd", "Nie znaleziono żadnych datasetów w Data/")
            sys.exit(1)
        self.combo.addItems(datasets)
        btn = QPushButton("Dalej")
        btn.clicked.connect(self.select)

        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.addWidget(self.combo)
        vbox.addWidget(btn)
        self.setLayout(vbox)

    def select(self):
        dataset = self.combo.currentText()
        self.on_select_callback(dataset)
        self.close()

class Worker(QThread):
    progress_signal = pyqtSignal(int, int)    # (film_idx, progress)
    done_signal = pyqtSignal(int)             # film_idx
    error_signal = pyqtSignal(int, str)       # film_idx, error

    def __init__(self, dataset, films):
        super().__init__()
        self.dataset = dataset
        self.films = films

    def run(self):
        input_base = Path("Data") / self.dataset / "Images"
        output_base = Path("Results") / self.dataset / "Images" / "Deeplab"
        model = load_deeplab_model()

        for idx, folder in enumerate(self.films):
            try:
                image_files = sorted(folder.glob("*.*"))
                if not image_files:
                    self.error_signal.emit(idx, f"Folder {folder.name} pusty")
                    continue

                output_folder = output_base / folder.name / "masks"
                output_folder.mkdir(parents=True, exist_ok=True)

                for i, img_path in enumerate(image_files):
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mask = segment_frame(model, image_rgb)

                    mask_filename = img_path.stem + ".png"
                    cv2.imwrite(str(output_folder / mask_filename), mask)

                    if len(image_files) > 0:
                        self.progress_signal.emit(idx, int((i + 1) / len(image_files) * 100))

                self.done_signal.emit(idx)
            except Exception as e:
                self.error_signal.emit(idx, f"Error: {str(e)}")

class ProgressWindow(QWidget):
    def __init__(self, dataset):
        super().__init__()
        self.setWindowTitle(f"Segmentacja Deeplab – {dataset}")
        self.dataset = dataset

        self.input_base = Path("Data") / dataset / "Images"
        self.films = [f for f in self.input_base.iterdir() if f.is_dir()]
        if not self.films:
            QMessageBox.critical(self, "Błąd", f"Brak folderów z filmami w {self.input_base}")
            sys.exit(1)

        self.progress_bars = []
        vbox = QVBoxLayout()
        label = QLabel(f"Przetwarzanie datasetu: <b>{dataset}</b>")
        vbox.addWidget(label)

        for folder in self.films:
            hbox = QHBoxLayout()
            film_label = QLabel(folder.name)
            progress = QProgressBar()
            progress.setValue(0)
            hbox.addWidget(film_label)
            hbox.addWidget(progress)
            vbox.addLayout(hbox)
            self.progress_bars.append(progress)

        spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Expanding)
        vbox.addSpacerItem(spacer)
        self.setLayout(vbox)

        self.worker = Worker(dataset, self.films)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.done_signal.connect(self.mark_done)
        self.worker.error_signal.connect(self.mark_error)
        self.worker.start()

    def update_progress(self, film_idx, value):
        self.progress_bars[film_idx].setValue(value)

    def mark_done(self, film_idx):
        self.progress_bars[film_idx].setValue(100)
        self.progress_bars[film_idx].setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")

    def mark_error(self, film_idx, msg):
        self.progress_bars[film_idx].setFormat(f"Błąd: {msg}")
        self.progress_bars[film_idx].setStyleSheet("QProgressBar::chunk { background-color: red; }")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    progress_win = None  # <- ważne!

    def start_segmentation(selected_dataset):
        global progress_win
        progress_win = ProgressWindow(selected_dataset)
        progress_win.resize(600, 60 + 40 * len(progress_win.films))
        progress_win.show()

    selector = DatasetSelector(on_select_callback=start_segmentation)
    selector.resize(300, 100)
    selector.show()
    sys.exit(app.exec_())
