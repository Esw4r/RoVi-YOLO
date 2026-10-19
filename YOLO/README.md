# Gopher-AI — Road Damage Detection using YOLOv11

> A fine-tuned **YOLOv11n** object-detection model that identifies five categories of road surface damage from dashcam imagery, trained on the **RDD2022** dataset.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [📥 Download Pre-trained Model](#-download-pre-trained-model)
3. [Detected Classes](#detected-classes)
4. [Repository Structure](#repository-structure)
5. [Hardware & Software Requirements](#hardware--software-requirements)
6. [Quick Start & Inference](#quick-start--inference)
7. [Step-by-Step: Reproducing the Training](#step-by-step-reproducing-the-training)
   - [Step 1 — Clone & Set Up the Environment](#step-1--clone--set-up-the-environment)
   - [Step 2 — Download the RDD2022 Dataset](#step-2--download-the-rdd2022-dataset)
   - [Step 3 — Filter Non-Dashcam Data](#step-3--filter-non-dashcam-data)
   - [Step 4 — Prune Background Images](#step-4--prune-background-images)
   - [Step 5 — Resize Images](#step-5--resize-images)
   - [Step 6 — Verify the Dataset Config](#step-6--verify-the-dataset-config)
   - [Step 7 — Download the Pretrained Backbone](#step-7--download-the-pretrained-backbone)
   - [Step 8 — Train the Model](#step-8--train-the-model)
   - [Step 9 — Resume a Crashed / Paused Run](#step-9--resume-a-crashed--paused-run)
   - [Step 10 — Evaluate Results](#step-10--evaluate-results)
8. [Training Hyperparameters Explained](#training-hyperparameters-explained)
9. [License](#license)

---

## Project Overview

Gopher-AI uses **Ultralytics YOLOv11n** (the nano variant) to detect road damage in real-time from dashcam footage. It is intentionally sized to run on modest hardware (4 GB VRAM), making it practical for edge deployments.

The model was trained on the publicly available **RDD2022** dataset, with a focused preprocessing pipeline to keep only dashcam-perspective imagery (removing drone and motorbike views).

---

## 📥 Download Pre-trained Model

Pre-trained weights are available via the official GitHub Release:

[![GitHub Release](https://img.shields.io/badge/Release-v1.0.0-blue?logo=github)](https://github.com/Esw4r/gopher-ai/releases/tag/v1.0.0)
[![Model Weights](https://img.shields.io/badge/Weights-gopher--ai__v1.pt%20(21.2MB)-brightgreen)](https://github.com/Esw4r/gopher-ai/releases/download/v1.0.0/gopher-ai_v1.pt)

| Model File | Size | Base Architecture | Release Link |
|---|---|---|---|
| **`gopher-ai_v1.pt`** | ~21.2 MB | YOLOv11n (Fine-tuned on RDD2022) | [Download v1.0.0](https://github.com/Esw4r/gopher-ai/releases/download/v1.0.0/gopher-ai_v1.pt) |

### Quick CLI Download:
```bash
# Using curl (Linux / macOS / Windows Git Bash):
curl -L -o gopher-ai_v1.pt https://github.com/Esw4r/gopher-ai/releases/download/v1.0.0/gopher-ai_v1.pt

# Using PowerShell (Windows):
Invoke-WebRequest -Uri "https://github.com/Esw4r/gopher-ai/releases/download/v1.0.0/gopher-ai_v1.pt" -OutFile "gopher-ai_v1.pt"

# Using wget:
wget https://github.com/Esw4r/gopher-ai/releases/download/v1.0.0/gopher-ai_v1.pt
```

---

## Detected Classes

| ID | Class Name           | Description                          |
|----|----------------------|--------------------------------------|
| 0  | Longitudinal Crack   | Cracks running along the road length |
| 1  | Transverse Crack     | Cracks running across the road       |
| 2  | Alligator Crack      | Interconnected, web-like cracking    |
| 3  | Other Corruption     | Misc. surface degradation            |
| 4  | Pothole              | Bowl-shaped road depression          |

---

## Repository Structure

```
Gopher-AI/
├── rdd2022/                    # Dataset root (excluded from git — download manually)
│   ├── data.yaml               # YOLO dataset config
│   └── RDD_SPLIT/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── runs/                       # Training outputs (excluded from git)
│   └── detect/
│       └── train-N/
│           ├── weights/
│           │   ├── best.pt     # Best checkpoint
│           │   └── last.pt     # Latest checkpoint
│           └── results.csv
├── Afilter_dash.py             # Step 3: Filter drone/motorbike images
├── prune_backgrounds.py        # Step 4: Prune excess background images
├── resize_images.py            # Step 5: Resize images to max 640px
├── rdd2022_kaggle_dwnld_script.py  # Step 2: Download dataset from Kaggle
├── new_train.py                # Step 8: Start training
└── resume_train.py             # Step 9: Resume interrupted training
```

---

## Hardware & Software Requirements

### Hardware
| Component | Minimum | Used for Training |
|-----------|---------|-------------------|
| GPU | 4 GB VRAM (NVIDIA) | NVIDIA GTX 1650 (4 GB) |
| RAM | 8 GB | 16 GB |
| Storage | ~15 GB free | Dataset + checkpoints |

### Software
```
Python        >= 3.10
ultralytics   >= 8.3          # pip install ultralytics
torch         >= 2.0          # Follow https://pytorch.org for CUDA builds
Pillow        >= 9.0          # pip install Pillow
kaggle        >= 1.6          # pip install kaggle
```

---

## Step-by-Step: Reproducing the Training

### Step 1 — Clone & Set Up the Environment

```bash
git clone https://github.com/Esw4r/Gopher-AI.git
cd Gopher-AI

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install ultralytics Pillow kaggle
```

> **Note:** For GPU acceleration, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) **before** installing ultralytics.

---

### Step 2 — Download the RDD2022 Dataset

The dataset is hosted on Kaggle. You need a free Kaggle account and an API token.

1. Go to [kaggle.com → Account → API → Create New Token](https://www.kaggle.com/settings).
2. Download `kaggle.json` and place it at:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`
3. Run the download script:

```bash
python rdd2022_kaggle_dwnld_script.py
```

This downloads and extracts the dataset into the `./rdd2022/` directory (~10 GB).

> **Dataset source:** [aliabdelmenam/rdd-2022 on Kaggle](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022)

---

### Step 3 — Filter Non-Dashcam Data

The RDD2022 dataset includes images captured from drones and motorbikes in addition to standard dashcams. These perspectives are unsuitable for a dashcam-focused model, so we remove them.

```bash
# Dry run first — shows what would be deleted, touches nothing:
python Afilter_dash.py --dry-run

# If the preview looks correct, run the actual cleanup:
python Afilter_dash.py --yes
```

**What this removes:** All files whose filenames begin with `China_Drone` or `China_MotorBike` across all splits (`train`, `val`, `test`) and sub-folders (`images`, `labels`).

After this step, your dataset contains only ground-level dashcam imagery.

---

### Step 4 — Prune Background Images

Background images (road scenes with no damage) are important for preventing false positives, but too many of them cause class imbalance. We keep **1 000** background images and move the rest to a backup folder.

```bash
python prune_backgrounds.py
```

**What this does:**
- Scans `rdd2022/RDD_SPLIT/train/labels/` for empty `.txt` files (empty = no annotations = background).
- Keeps the first 1 000 background pairs (image + label).
- Moves all excess background pairs to `rdd2022/RDD_SPLIT/train/backgrounds_backup/`.

> Nothing is deleted — the backups are preserved if you want to undo this step.

---

### Step 5 — Resize Images

Large images slow down training significantly. We cap all training and validation images at **640 × 640 pixels** while preserving aspect ratio.

```bash
python resize_images.py
```

**What this does:**
- Processes all `.jpg` images in `train/images/` and `val/images/` in parallel (8 threads).
- Images smaller than 640px on their longest side are left untouched.
- Images are re-saved as JPEG at quality 85.

> After this step, training images should be ≤ 640px on each side. This saved roughly **36% in FLOPs** per batch compared to training at 640-native.

---

### Step 6 — Verify the Dataset Config

Open `rdd2022/data.yaml` and make sure the paths are correct relative to where you run training from:

```yaml
path: ./rdd2022/RDD_SPLIT   # path to dataset root
train: train/images
val:   val/images
test:  test/images

names:
  0: longitudinal crack
  1: transverse crack
  2: alligator crack
  3: other corruption
  4: Pothole
```

The `path` field is resolved relative to the working directory where you launch `new_train.py`. If you run training from the repo root, `./rdd2022/RDD_SPLIT` is correct.

---

### Step 7 — Download the Pretrained Backbone

YOLOv11n is initialized from COCO pretrained weights. Ultralytics downloads this automatically on first run, but you can also pre-download it:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

This saves `yolo11n.pt` (~5.4 MB) to your current directory.

---

### Step 8 — Train the Model

```bash
python new_train.py
```

Training will start and logs will be printed to the console. Checkpoints are saved to `runs/detect/train-N/weights/` where `N` increments automatically.

**Key hyperparameters used and why:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `epochs` | 100 | Maximum epochs; early stopping kicks in at 10 stagnant epochs |
| `patience` | 10 | Stops training if mAP doesn't improve for 10 consecutive epochs |
| `imgsz` | 512 | Smaller than default 640 → fewer FLOPs, fits in 4 GB VRAM |
| `batch` | 16 | Safe limit for 4 GB VRAM with FP32 tensors |
| `workers` | 3 | Data loader threads (tuned to avoid CPU bottleneck) |
| `cache` | True | Caches dataset in RAM (~1.5 GB); eliminates disk latency per batch |
| `amp` | False | Mixed precision disabled — required for GTX 1650 stability |
| `optimizer` | AdamW | Better convergence than SGD on small datasets |
| `lr0` | 0.0015 | Slightly conservative initial learning rate |
| `close_mosaic` | 10 | Disables mosaic augmentation in the final 10 epochs for fine-tuning |
| `degrees` | 0.0 | No rotation augmentation (road images have fixed orientation) |
| `flipud` | 0.0 | No vertical flip (sky and road should not be swapped) |
| `fliplr` | 0.5 | Horizontal flip is fine (left-right symmetry on roads) |

---

### Step 9 — Resume a Crashed / Paused Run

If training is interrupted (power cut, crash, etc.), you can resume from the last checkpoint:

1. Open `resume_train.py` and update the path to point to the correct `last.pt`:

```python
model = YOLO(r"runs\detect\train-N\weights\last.pt")  # ← update N
model.train(resume=True)
```

2. Run it:

```bash
python resume_train.py
```

Ultralytics will read all the original hyperparameters from the checkpoint and continue from the exact epoch it stopped at. **Do not change any hyperparameters when resuming** — they are baked into the checkpoint.

---

### Step 10 — Evaluate Results

After training, results are stored in `runs/detect/train-N/`:

| File | Contents |
|------|----------|
| `results.csv` | Per-epoch metrics: box loss, cls loss, mAP@50, mAP@50-95 |
| `weights/best.pt` | Weights from the epoch with the highest mAP@50 |
| `weights/last.pt` | Weights from the final epoch |
| `labels.jpg` | Class distribution visualisation |
| `train_batch*.jpg` | Sample augmented training batches |

To run validation on the best weights:

```bash
yolo detect val model=runs/detect/train-N/weights/best.pt data=rdd2022/data.yaml
```

---

## Training Hyperparameters Explained

A full copy of every parameter used is saved automatically at `runs/detect/train-N/args.yaml`. This is useful for exact reproducibility — if you need to retrain, cross-reference this file.

Key augmentation settings kept from defaults:
- **HSV jitter** (`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`) — simulates lighting variation.
- **Mosaic** (`mosaic=1.0`) — combines 4 images per batch for richer context; disabled in last 10 epochs.
- **Random erasing** (`erasing=0.4`) — randomly blanks patches to improve occlusion robustness.
- **Scale** (`scale=0.5`) — random scale up/down by ±50%.

---

## Quick Start & Inference

You can run inference using the pre-trained **`gopher-ai_v1.pt`** weights (downloaded from the [v1.0.0 Release](https://github.com/Esw4r/gopher-ai/releases/tag/v1.0.0)) or your own trained checkpoint from `runs/detect/train-N/weights/best.pt`.

### CLI Inference

```bash
# Detect on an image
yolo detect predict model=gopher-ai_v1.pt source=path/to/image.jpg imgsz=512 conf=0.25

# Real-time webcam / dashcam feed
yolo detect predict model=gopher-ai_v1.pt source=0 imgsz=512 conf=0.25

# Process a dashcam video
yolo detect predict model=gopher-ai_v1.pt source=path/to/video.mp4 imgsz=512 conf=0.25
```

Predicted frames with bounding boxes are automatically saved to `runs/detect/predict/`.

### Python API

```python
from ultralytics import YOLO

# Load the model
model = YOLO("gopher-ai_v1.pt")

# Predict on an image or video
results = model.predict(source="path/to/road_image.jpg", conf=0.25, imgsz=512)

# Display or save results
for r in results:
    r.show()  # Display annotated image
    r.save(filename="output.jpg")
```

---

## License

This project is released for academic and research purposes.  
The **RDD2022 dataset** is subject to its own terms — refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022) for details.
