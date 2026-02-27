# ResLKA-UNet: Two-Stage U-Net with Large Kernel Attention for Liver Tumor Segmentation

## Overview

This repository presents the official implementation of:

**“Two-Stage U-Net for Liver and Liver Tumor Segmentation in CT Images”**

We propose a **coarse-to-fine two-stage segmentation framework** for automatic liver and liver tumor segmentation from CT scans.

The pipeline consists of:

- **Stage 1 – Liver Segmentation**
  - ResNet50-based U-Net backbone
  - Group Normalization for small-batch stability
  - Whole-slice liver localization

- **Stage 2 – Tumor Segmentation**
  - ROI cropping based on liver prediction
  - ResNet50 encoder with Large Kernel Attention (LKA)
  - Fine-grained tumor segmentation inside liver region

Performance on LiTS17:
- **98.49% Dice** (Liver)
- **84.18% Dice** (Tumor)

---

## Motivation

Liver tumor segmentation is challenging due to:

- Severe class imbalance
- Small tumor size
- Low contrast boundaries
- High anatomical variability

To address these issues, we introduce:

- A **two-stage ROI-based framework**
- **Large Kernel Attention (LKA)** for global spatial modeling
- A **Hybrid Loss (Dice + BCE + Focal)**

---

## Model Architecture

### 1. Two-Stage Framework

Stage 1:
- Input: Full CT slice
- Output: Liver mask

Stage 2:
- Input: Cropped liver ROI
- Output: Tumor mask

This reduces background interference and improves tumor sensitivity.

---

### 2. Encoder Backbone

- ResNet-50
- BatchNorm replaced by **Group Normalization**
- Residual connections improve training stability

---

### 3. Large Kernel Attention (LKA)

- Expands receptive field efficiently
- Captures long-range spatial dependencies
- Lightweight alternative to full Transformer attention

---

### 4. Loss Function

Hybrid Loss:
- Binary Cross Entropy (BCE)
- Dice Loss
- Focal Loss
---

## Dataset

### LiTS17 (Liver Tumor Segmentation Challenge)

- 131 contrast-enhanced abdominal CT volumes
- Manual liver and tumor annotations
- 7,190 tumor-containing slices extracted
- Train/Val/Test split: 6:2:2

You can using our preprocessed dataset: **[Preprecessed dataset](https://www.kaggle.com/datasets/thnhlngtrng/lits17)** 

---

## Results

| Task | Dice Score | IoU |
|------|------------|-----|
| Liver Segmentation | **98.49%** |**97.08%**|
| Tumor Segmentation | **84.18%** |**77.11%**|

The proposed method improves tumor boundary delineation and small lesion detection compared to baseline U-Net variants.

___
## Pretrained Weight

You can download our weights: **[Models Weight](https://drive.google.com/drive/folders/1wMYRC2TOLUSjnYCqrmxQ_juzThzk6Bn3?usp=sharing)** 



