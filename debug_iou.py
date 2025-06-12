import sys
import random
from pathlib import Path

import numpy as np
import cv2
from scipy.io import loadmat
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_available_datasets():
    base = Path("Data")
    return [f.name for f in base.iterdir() if f.is_dir()]

def get_available_models(dataset):
    models_base = Path("Results") / dataset / "Images"
    if not models_base.exists():
        return []
    return [f.name for f in models_base.iterdir() if f.is_dir()]

def random_mask_pairs(dataset, model_name, num_samples=5):
    pred_base = Path("Results") / dataset / "Images" / model_name
    gt_base = Path("Data") / dataset / "Groundtruth"

    film_folders = [f for f in pred_base.iterdir() if f.is_dir()]
    if not film_folders:
        return []

    chosen_samples = []
    attempts = 0
    max_attempts = num_samples * 8

    while len(chosen_samples) < num_samples and attempts < max_attempts:
        folder = random.choice(film_folders)
        pred_mask_folder = folder / "masks"
        gt_folder = gt_base / folder.name
        pred_masks = sorted(pred_mask_folder.glob("*.png"))
        if not pred_masks:
            attempts += 1
            continue
        pred_path = random.choice(pred_masks)
        mat_filename = pred_path.stem + ".mat"
        gt_path = gt_folder / mat_filename
        if not gt_path.exists():
            attempts += 1
            continue
        chosen_samples.append((pred_path, gt_path, folder.name))
        attempts += 1

    return chosen_samples

def load_and_prepare_masks(pred_path, gt_path):
    pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    pred_mask = (pred_mask > 127).astype(np.uint8)

    mat = loadmat(str(gt_path))
    gt_struct = mat["groundTruth"]
    mask_gt = gt_struct[0, 0]["Segmentation"][0, 0]
    mask_gt = (mask_gt > 0).astype(np.uint8)

    if mask_gt.shape != pred_mask.shape:
        mask_gt = cv2.resize(mask_gt, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    return pred_mask, mask_gt

def make_error_map(pred_mask, mask_gt):
    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    tp = np.logical_and(pred_mask == 1, mask_gt == 1)
    fp = np.logical_and(pred_mask == 1, mask_gt == 0)
    fn = np.logical_and(pred_mask == 0, mask_gt == 1)
    error_map[tp] = [0, 255, 0]       # Zielony
    error_map[fp] = [255, 0, 0]       # Czerwony
    error_map[fn] = [0, 0, 255]       # Niebieski
    return error_map

def to_pixmap(mask, is_color=False):
    if not is_color:
        mask = (mask * 255).astype(np.uint8)
        h, w = mask.shape
        qimg = QImage(mask.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, c = mask.shape
        qimg = QImage(mask.tobytes(), w, h, w * c, QImage.Format_RGB888)
    qimg = qimg.scaled(320, 180, Qt.KeepAspectRatio)
    return QPixmap.fromImage(qimg)

class DatasetModelSelector(QWidget):
    def __init__(self, on_select_callback):
        super().__init__()
        self.setWindowTitle("Wybierz dataset i model")
        self.on_select_callback = on_select_callback

        self.dataset_combo = QComboBox()
        self.model_combo = QComboBox()
        self.datasets = get_available_datasets()
        if not self.datasets:
            QMessageBox.critical(self, "Błąd", "Nie znaleziono żadnych datasetów w Data/")
            sys.exit(1)
        self.dataset_combo.addItems(self.datasets)
        self.dataset_combo.currentTextChanged.connect(self.update_models)
        self.update_models(self.dataset_combo.currentText())

        btn = QPushButton("Dalej")
        btn.clicked.connect(self.select)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Dataset:"))
        layout.addWidget(self.dataset_combo)
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_combo)
        layout.addWidget(btn)
        self.setLayout(layout)

    def update_models(self, dataset):
        self.model_combo.clear()
        models = get_available_models(dataset)
        if not models:
            self.model_combo.addItem("Brak modeli")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.addItems(models)
            self.model_combo.setEnabled(True)

    def select(self):
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()
        if not model or model == "Brak modeli":
            QMessageBox.warning(self, "Błąd", "Wybierz poprawny model!")
            return
        self.on_select_callback(dataset, model)
        self.close()

class DebugIoUWindow(QWidget):
    def __init__(self, dataset, model_name, num_samples=5):
        super().__init__()
        self.setWindowTitle(f"Debug IoU – {dataset} / {model_name}")
        self.dataset = dataset
        self.model_name = model_name
        self.samples = random_mask_pairs(dataset, model_name, num_samples)
        self.idx = 0

        self.label_info = QLabel("")
        self.label_pred = QLabel("")
        self.label_gt = QLabel("")
        self.label_errormap = QLabel("")

        self.btn_next = QPushButton("Następny przykład")
        self.btn_next.clicked.connect(self.show_next)
        self.btn_prev = QPushButton("Poprzedni przykład")
        self.btn_prev.clicked.connect(self.show_prev)
        self.btn_reload = QPushButton("Wylosuj nowe 5 przykładów")
        self.btn_reload.clicked.connect(self.reload_examples)

        hbox = QHBoxLayout()
        vbox_pred = QVBoxLayout()
        vbox_gt = QVBoxLayout()
        vbox_errormap = QVBoxLayout()

        vbox_pred.addWidget(QLabel("Maska predykcji (model)"))
        vbox_pred.addWidget(self.label_pred)

        vbox_gt.addWidget(QLabel("Maska ground truth"))
        vbox_gt.addWidget(self.label_gt)

        vbox_errormap.addWidget(QLabel("Mapa błędów (TP/FP/FN)"))
        vbox_errormap.addWidget(self.label_errormap)

        hbox.addLayout(vbox_pred)
        hbox.addLayout(vbox_gt)
        hbox.addLayout(vbox_errormap)

        btnbox = QHBoxLayout()
        btnbox.addWidget(self.btn_prev)
        btnbox.addWidget(self.btn_next)
        btnbox.addWidget(self.btn_reload)

        vbox_main = QVBoxLayout()
        vbox_main.addWidget(self.label_info)
        vbox_main.addLayout(hbox)
        vbox_main.addLayout(btnbox)

        self.setLayout(vbox_main)

        self.show_example()

    def show_example(self):
        if not self.samples:
            self.label_info.setText("Brak przykładów do wyświetlenia.")
            self.label_pred.clear()
            self.label_gt.clear()
            self.label_errormap.clear()
            return

        pred_path, gt_path, film_name = self.samples[self.idx]
        pred_mask, mask_gt = load_and_prepare_masks(pred_path, gt_path)
        iou = calculate_iou(pred_mask, mask_gt)

        errormap = make_error_map(pred_mask, mask_gt)

        self.label_info.setText(
            f"<b>Dataset:</b> {self.dataset} &nbsp;&nbsp; <b>Model:</b> {self.model_name} "
            f"&nbsp;&nbsp; <b>Film:</b> {film_name} &nbsp;&nbsp; <b>Klatka:</b> {pred_path.stem} "
            f"&nbsp;&nbsp; <b>IoU:</b> {iou:.3f}"
        )
        self.label_pred.setPixmap(to_pixmap(pred_mask))
        self.label_gt.setPixmap(to_pixmap(mask_gt))
        self.label_errormap.setPixmap(to_pixmap(errormap, is_color=True))

    def show_next(self):
        if not self.samples:
            return
        self.idx = (self.idx + 1) % len(self.samples)
        self.show_example()

    def show_prev(self):
        if not self.samples:
            return
        self.idx = (self.idx - 1) % len(self.samples)
        self.show_example()

    def reload_examples(self):
        self.samples = random_mask_pairs(self.dataset, self.model_name, 5)
        self.idx = 0
        self.show_example()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    debug_win = None  # zapobiega znikaniu okna!

    def launch_debug_window(dataset, model):
        global debug_win
        debug_win = DebugIoUWindow(dataset, model, num_samples=5)
        debug_win.resize(1100, 400)
        debug_win.show()

    selector = DatasetModelSelector(on_select_callback=launch_debug_window)
    selector.resize(350, 150)
    selector.show()
    sys.exit(app.exec_())
