import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont


def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def binarize_mask(mask_gt):
    unique, counts = np.unique(mask_gt, return_counts=True)
    background_value = unique[np.argmax(counts)]
    return (mask_gt != background_value).astype(np.uint8)


def is_box_mask(mask_gt, min_fill=0.95):
    ys, xs = np.where(mask_gt > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    box = mask_gt[y0:y1 + 1, x0:x1 + 1]
    fill_ratio = box.sum() / box.size
    return fill_ratio > min_fill


def get_available_datasets():
    base = Path("Results")
    return [f.name for f in base.iterdir() if f.is_dir()]


def get_model_list(dataset):
    models_base = Path("Results") / dataset / "Images"
    if not models_base.exists():
        return []
    return [f.name for f in models_base.iterdir() if f.is_dir()]


def evaluate_model_iou(dataset, model_name):
    pred_base = Path("Results") / dataset / "Images" / model_name
    gt_base = Path("Data") / dataset / "Groundtruth"

    if not pred_base.exists() or not gt_base.exists():
        return None, None, 0, 0, None

    folders = [f for f in pred_base.iterdir() if f.is_dir()]
    all_ious = []
    film_to_ious = {}
    total_masks = 0
    box_masks = 0

    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for folder in folders:
        pred_mask_folder = folder / "masks"
        gt_folder = gt_base / folder.name
        if not pred_mask_folder.exists() or not gt_folder.exists():
            continue

        pred_masks = sorted(pred_mask_folder.glob("*.png"))
        if not pred_masks:
            continue

        ious = []

        for pred_path in pred_masks:
            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            pred_mask = (pred_mask > 127).astype(np.uint8)

            mat_filename = pred_path.stem + ".mat"
            gt_path = gt_folder / mat_filename
            if not gt_path.exists():
                continue

            try:
                mat = loadmat(str(gt_path))
                if "groundTruth" not in mat:
                    continue

                gt_struct = mat["groundTruth"]
                mask_gt = gt_struct[0, 0]["Segmentation"][0, 0]
                mask_gt = binarize_mask(mask_gt)

                if mask_gt.shape != pred_mask.shape:
                    mask_gt = cv2.resize(mask_gt, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            except Exception:
                continue

            total_masks += 1
            if is_box_mask(mask_gt):
                box_masks += 1

            iou = calculate_iou(pred_mask, mask_gt)
            ious.append(iou)
            all_ious.append(iou)

            confusion["TP"] += np.logical_and(pred_mask == 1, mask_gt == 1).sum()
            confusion["FP"] += np.logical_and(pred_mask == 1, mask_gt == 0).sum()
            confusion["TN"] += np.logical_and(pred_mask == 0, mask_gt == 0).sum()
            confusion["FN"] += np.logical_and(pred_mask == 0, mask_gt == 1).sum()

        if ious:
            film_to_ious[folder.name] = np.mean(ious) * 100

    if all_ious:
        overall_mean_iou = np.mean(all_ious) * 100
    else:
        overall_mean_iou = None

    return film_to_ious, overall_mean_iou, total_masks, box_masks, confusion


class DatasetModelSelector(QWidget):
    def __init__(self, on_select_callback):
        super().__init__()
        self.setWindowTitle("Wybierz dataset i model")
        self.on_select_callback = on_select_callback

        self.dataset_combo = QComboBox()
        self.model_combo = QComboBox()
        self.datasets = get_available_datasets()
        if not self.datasets:
            QMessageBox.critical(self, "Błąd", "Nie znaleziono żadnych datasetów w Results/")
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
        models = get_model_list(dataset)
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


class ResultsWindow(QWidget):
    def __init__(self, dataset, model_name, film_to_ious, overall_iou, total_masks, box_masks, confusion):
        super().__init__()
        self.setWindowTitle(f"Wyniki IoU – {dataset} / {model_name}")
        vbox = QVBoxLayout()

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Film", "IoU [%]"])
        table.setRowCount(len(film_to_ious))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)

        for row, (film, iou) in enumerate(sorted(film_to_ious.items())):
            table.setItem(row, 0, QTableWidgetItem(film))
            iou_item = QTableWidgetItem(f"{iou:.2f}")
            if iou < 30:
                iou_item.setForeground(QColor("white"))
                iou_item.setBackground(QColor("red"))
            elif iou < 50:
                iou_item.setBackground(QColor("yellow"))
            table.setItem(row, 1, iou_item)

        table.setEditTriggers(QTableWidget.NoEditTriggers)
        vbox.addWidget(table)

        if confusion is not None:
            conf_table = QTableWidget()
            conf_table.setRowCount(2)
            conf_table.setColumnCount(2)
            conf_table.setHorizontalHeaderLabels(["GT: Pozytyw", "GT: Negatyw"])
            conf_table.setVerticalHeaderLabels(["Pred: Pozytyw", "Pred: Negatyw"])
            conf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            conf_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

            font_bold = QFont()
            font_bold.setBold(True)

            tp = confusion["TP"]
            fp = confusion["FP"]
            fn = confusion["FN"]
            tn = confusion["TN"]

            for i, value in enumerate([[tp, fp], [fn, tn]]):
                for j, val in enumerate(value):
                    item = QTableWidgetItem(str(val))
                    item.setFont(font_bold)
                    conf_table.setItem(i, j, item)

            conf_table.setEditTriggers(QTableWidget.NoEditTriggers)

            vbox.addWidget(QLabel("<b>Macierz pomyłek (Confusion Matrix):</b>"))
            vbox.addWidget(conf_table)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            stats_label = QLabel(
                f"<b>Precision:</b> {precision * 100:.2f}%<br>"
                f"<b>Recall:</b> {recall * 100:.2f}%<br>"
                f"<b>F1-score:</b> {f1 * 100:.2f}%"
            )
            stats_label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(stats_label)

        info = ""
        if total_masks > 0 and box_masks > 0:
            percent = 100 * box_masks / total_masks
            info = f"<br> <span style='color:red;'>Uwaga:</span> ok. {percent:.1f}% masek ground truth to prostokąty (bounding box) –<br>wynik IoU może być zawyżony/zaniżony!"
            info = info.replace(".", ",")

        if overall_iou is not None:
            summary = QLabel(
                f"<b>Ogólne średnie IoU dla <span style='color:blue'>{dataset}</span> / <span style='color:blue'>{model_name}</span>: "
                f"<span style='color:green'>{overall_iou:.2f}%</span></b>{info}"
            )
        else:
            summary = QLabel("<b>Brak danych do podsumowania ogólnego.</b>")
        summary.setAlignment(Qt.AlignCenter)
        vbox.addWidget(summary)

        self.setLayout(vbox)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    results_win = None

    def launch_results_window(dataset, model_name):
        film_to_ious, overall_iou, total_masks, box_masks, confusion = evaluate_model_iou(dataset, model_name)
        if film_to_ious is None:
            QMessageBox.critical(None, "Błąd", f"Brak danych do oceny IoU dla: {dataset} / {model_name}")
            sys.exit(1)
        global results_win
        results_win = ResultsWindow(dataset, model_name, film_to_ious, overall_iou, total_masks, box_masks, confusion)
        results_win.resize(600, 500)
        results_win.show()

    selector = DatasetModelSelector(on_select_callback=launch_results_window)
    selector.resize(350, 150)
    selector.show()
    sys.exit(app.exec_())
