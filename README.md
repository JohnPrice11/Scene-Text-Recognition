# Scene Text Detection & Prediction Pipeline (Detectron2 + TrOCR)

This project implements a complete **Scene Text Detection & Recognition** system using:

- **Detectron2 Mask‑RCNN** for text detection  
- **TrOCR (Transformer OCR)** for text recognition   

The pipeline is fully compatible with systems **without sudo access** and supports **GPU acceleration**.

---

#  Environment Setup (Conda)

### 1. Create Conda Environment
```bash
conda create -n d2_stable python=3.9 -y
conda activate d2_stable
```

### 2. Install Correct PyTorch (CUDA 11.8)
```bash
pip install torch==2.1.2 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies
```bash
pip install     pandas     shapely     transformers     tqdm     pillow     opencv-python
```

### install the 1.24.4 version of numpy only
```bash
pip install numpy==1.24.4
```

---

# Install Detectron2 From Source (Without sudo)

Clone your Detectron2 code into a folder (example: `detectron2/`)

Then install **without build isolation**:
```bash
cd detectron2
python -m pip install -e . --no-build-isolation
```

If you want Detectron2 importable everywhere:
```bash
export PYTHONPATH=/path/to/detectron2:$PYTHONPATH
```

---

#  Pipeline Overview

##  Train Mask‑RCNN Text Detector
Runs Detectron2 training and produces your detection model.

```bash
python Stage1-model1-Mask-CNN.py
```

 Outputs model checkpoints to:

```
output_detectron2/
```

---

##  Run Bounding Box Inference

Uses your trained Mask‑RCNN to predict bounding boxes.

```bash
python inference_draw.py
```

 Saves **ONLY bounding‑boxed images** to:

```
results_test_polygon_improv/
```

(No text extraction yet.)

---

##  Crop Text Regions for TrOCR Training

Reads images + bounding boxes from dataset and generates cropped text patches.

```bash
python Stage2_data_conv.py
```

 Cropped images saved to:

```
train_data/
```

These cropped patches are used to fine‑tune TrOCR.

---

##  Stage 4 — Fine‑Tune TrOCR on Cropped Text Images

```bash
python Stage2-trocr-train.py
```

 Trained TrOCR model saved to:

```
trocr_finetuned_full/
```

---

##  Stage 5 Recognize Text

Runs TrOCR pipeline.

```bash
python extract_text_improv.py
```

 Outputs:

```
result_final_text/   -> recognized text files
result_final_vis/    -> images with boxes + text labels
```

This is the **final output** for the entire system.

---

#  Folder Structure (Final)

```
project/
│
├── dataset
    ├── images
    ├── coco_annotations_fixed
├── detectron2/                  # Source installation
├── Stage1-model1-Mask-CNN.py
├── inference_draw.py
├── Stage2_data_conv.py
├── Stage2-trocr-train.py
├── extract_text_improv.py
│
├── output_detectron2/           # Stage 1 model
├── results_test_polygon_improv/ # Stage 2 outputs
├── train_data/                  # Stage 3 crops
├── trocr_finetuned_full/        # Stage 4 model
├── result_final_text/           # Stage 5 text
└── result_final_vis/            # Stage 5 visualization
```

---

#  Troubleshooting

###  Detectron2 fails to build: `No module named torch`
Install PyTorch **before** installing Detectron2.

###  Numpy dtype error
Use a stable version:
```bash
pip install numpy==1.24.4
```

###  “_C could not be loaded”
Make sure PYTHONPATH is set correctly:
```bash
export PYTHONPATH=/path/to/detectron2:$PYTHONPATH
```

---


#  Pipeline Completed

### you can download the datasets and models from below link

### Dataset
```
https://drive.google.com/drive/folders/13VUdQynQ90nkvLfHYBUENiU-8D6qSZ8R?usp=drive_link
```

### Trained models(both for text detection and recognition)
```
https://drive.google.com/drive/folders/1joz1u-SdjqpGwGyYbzJt6GnrzUOVFZZt?usp=drive_link
```

### Backend
app.py and model.py are for backend

### To start server 
## run from root of project 
```
uvicorn app:app --host 0.0.0.0 --port 8000
```
