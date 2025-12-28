# Homework 1: Image Classification, Object Detection, Segmentation & Tracking

Comprehensive computer vision project implementing core deep learning tasks on wildlife/animal datasets.

## Overview

This homework encompasses the complete pipeline of modern computer vision:
1. **Image Classification** - Categorizing images into predefined classes
2. **Object Detection** - Localizing and identifying objects in images
3. **Instance Segmentation** - Pixel-level object delineation
4. **Multi-Object Tracking** - Following objects across video frames

## Project Structure

### [Classification/](Classification) - Image Classification
Classification of wildlife/animal images using deep neural networks.

**Files:**
- `dataset_class.py` - PyTorch Dataset class for loading and preprocessing images
- `model_class.py` - CNN architecture definitions
- `weights/` - Pre-trained model weights:
  - `resnet.pth` - ResNet model (baseline)
  - `resnet_aug.pth` - ResNet with data augmentation
  - `convnet.pth` - Custom ConvNet architecture

**Key Features:**
- Multi-class classification
- Data augmentation pipeline
- Transfer learning with ResNet
- Custom CNN implementation

---

### [Detection/](Detection) - Object Detection
Object detection using YOLO or similar framework to identify and localize objects in images.

**Files:**
- `Detection-Starter-Notebook.ipynb` - Tutorial and starter code
- `coco predictions.json` - Model predictions in COCO format
  - Contains bounding boxes and confidence scores
  - Suitable for evaluation with standard metrics

**Key Features:**
- Multi-scale object detection
- Bounding box generation
- Confidence score predictions
- COCO format compatibility

---

### [Segmentation/](Segmentation) - Instance & Semantic Segmentation
Pixel-level segmentation using DeepLabV3 architecture for detailed object boundary detection.

**Files:**
- `dataset_class.py` - Custom Dataset loader for segmentation tasks
- `model_class.py` - DeepLabV3 model wrapper
- `deeplabv3.pth` - Pre-trained DeepLabV3 weights
- `decoder.pth` - Decoder weights for upsampling

**Key Features:**
- Dense pixel prediction
- Multi-class semantic segmentation
- Atrous spatial pyramid pooling (ASPP)
- High-resolution output maps

---

### [Tracking/](Tracking) - Multi-Object Tracking
Video tracking system to maintain object identities across frames using IOU-based association.

**Files:**
- `tracking.py` - Main tracking pipeline
- `IOU_Tracker.py` - Intersection-over-Union based tracking algorithm

**Key Features:**
- Frame-by-frame object tracking
- Identity persistence
- IOU-based data association
- Video sequence processing

**Advanced Tracking (Bonus):**
- [bonus code dump/](Code_Dump/bonus%20code%20dump) - BYTE Tracker implementation
  - `byte_tracker.py` - Advanced tracking with track confidence
  - `kalman_filter.py` - Kalman filtering for motion prediction
  - `matching.py` - Hungarian algorithm for association
  - MOT17 benchmark datasets

---

### [Code_Dump/](Code_Dump) - Complete Implementation Files

**Main Scripts:**
- `Classification.ipynb` / `Classification.py` - Full classification pipeline
- `code_2022254_segmentation.py` - Segmentation implementation
- `detection_starter_notebook.py` - Detection boilerplate

**Bonus Tracking:**
- BYTE Tracker with Kalman filtering
- MOT17 dataset evaluation
- IOU-based tracking comparison

---

## Model Architectures

### Classification
- **ResNet-50**: Pre-trained on ImageNet, fine-tuned for target classes
- **Custom ConvNet**: Lightweight architecture optimized for dataset

### Detection
- YOLO-based architecture with multi-scale predictions

### Segmentation
- **DeepLabV3**: Atrous convolutions for dense prediction with context aggregation

### Tracking
- **IOU Tracker**: Simple but effective IoU-based association
- **BYTE Tracker**: Advanced tracking with confidence-based matching and Kalman filtering

## Technical Stack
- **Framework**: PyTorch
- **Dataset**: WildlifeDataset (custom)
- **Libraries**: torchvision, OpenCV, scikit-learn
- **Evaluation**: COCO metrics, MOT metrics

## Key Features Across Tasks

| Task | Input | Output | Metric |
|------|-------|--------|--------|
| Classification | Image | Class label | Accuracy |
| Detection | Image | Bounding boxes | mAP, IoU |
| Segmentation | Image | Pixel-wise masks | mIoU, Dice |
| Tracking | Video frames | Tracked objects | MOTA, IDF1 |

## Usage

Each subdirectory contains:
- Python implementation (`.py`)
- Jupyter notebook for interactive development (`.ipynb`)
- Pre-trained weights for inference

## Report
See `Report.pdf` for detailed results, performance metrics, and analysis across all tasks.
