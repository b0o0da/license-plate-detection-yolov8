# 🚘 License Plate Detection — YOLOv8s

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A YOLOv8s-based object detection pipeline for **automatic license plate detection**, including dataset exploration, bounding-box analysis, model training with custom hyperparameters, and evaluation on validation/test sets.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline Steps](#pipeline-steps)
- [Training Configuration](#training-configuration)
- [How to Run](#how-to-run)
- [Results / Evaluation](#results--evaluation)
- [Notes](#notes)
- [License](#license)

---

## 🔍 Overview

This notebook trains a **YOLOv8s** model to detect license plates in images. It covers the full workflow: dataset inspection, exploratory data analysis on bounding boxes, model training with tuned augmentation/optimizer settings, validation on both `val` and `test` splits, and visual inference on sample test images.

---

## 📊 Dataset

- **Source:** [License Plate Detection Dataset — 10,125 images](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images) (Kaggle)
- **Format:** YOLO format (`data.yaml` + `images/` and `labels/` per split)
- **Splits:** `train`, `valid`, `test`
- **Label structure:** Each label file contains `class x_center y_center width height` (normalized YOLO format)
- **Classes:** License plate (single class)

---

## 🛠️ Tech Stack

- **Model:** [Ultralytics YOLOv8s](https://github.com/ultralytics/ultralytics)
- **Core Libraries:** `ultralytics`, `opencv-python`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Other:** `pyyaml`, `pathlib`, `scikit-learn` (`train_test_split`)

---

## 🧩 Pipeline Steps

1. **Imports & setup** — install `ultralytics`, load core libraries
2. **Dataset inspection** — read `data.yaml`, count images/labels per split (`train`/`valid`/`test`)
3. **Visualize training samples** — draw bounding boxes on random training images
4. **Crop license plates** — extract cropped plate regions from labeled images
5. **Build a stats DataFrame** — collect bounding box `width`, `height`, and `area` for analysis
6. **EDA on bounding boxes**
   - Box area distribution (histogram)
   - Width vs. height relationship (scatter plot)
7. **Model training** — train YOLOv8s with custom hyperparameters (see below)
8. **Validation** — evaluate the best checkpoint on the `val` split
9. **Test evaluation** — evaluate on the held-out `test` split
10. **Sample predictions** — run inference on random test images and visualize detections

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Patience (early stopping) | 10 |
| Optimizer | AdamW |
| Initial LR (`lr0`) | 0.001 |
| Final LR factor (`lrf`) | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Augmentations | HSV (h/s/v), rotation, translation, scale, shear, perspective, flip (lr), mosaic, mixup, random erasing |
| Seed | 42 |

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python pandas numpy matplotlib seaborn pyyaml scikit-learn
   ```

2. **Set the dataset path** — update `DATA_YAML` to point to your local `data.yaml`:
   ```python
   DATA_YAML = "path/to/data.yaml"
   ```

3. **Run the notebook cells in order**:
   - Dataset inspection → EDA → Training → Validation → Test evaluation → Sample predictions

4. **Training checkpoints** are saved under:
   ```
   runs/detect/train/weights/best.pt
   ```

---

## 📈 Results / Evaluation

The model is evaluated using standard object detection metrics computed via `model.val()`, on **both** the validation and test splits:

- **Precision**
- **Recall**
- **mAP50** (mean Average Precision at IoU 0.50)
- **mAP50-95** (mean Average Precision averaged over IoU 0.50–0.95)

> Exact metric values depend on the training run and are printed at the end of the validation/test cells — update this section with your final numbers after training completes.

---

## 📝 Notes

- Originally built to run on Kaggle (`/kaggle/input/...`, `/kaggle/working/...` paths) — update paths if running locally or elsewhere.
- Confidence threshold used for validation/inference: `conf=0.50`, IoU threshold: `iou=0.45`.
- Bounding box EDA (area distribution, width vs. height) helps understand plate size variability before training.

---

## 📄 License

This project is open-source. Add your preferred license (e.g., MIT) here.
