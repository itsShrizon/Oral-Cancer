# Multi-Task Oral Cancer Classification Pipeline

## Overview

This repository implements a robust, multi-task deep learning pipeline for oral pathology image classification. Built primarily with PyTorch from the ground up, the system is designed to perform simultaneous binary disease detection (benign vs. malignant) and fine-grained pathology subtype classification. 

The project emphasizes rigorous machine learning engineering principles, focusing heavily on solving complex underlying dataset flaws, leveraging modern deep learning architectures, and establishing highly reliable training loops.

## Core Engineering Achievements & Problem Solving

### 1. Rigorous Data Engineering & Leakage Prevention
Real-world datasets often present hidden flaws that can invalidate machine learning models in production. During the initial data analysis phase, a severe data leakage anomaly was identified: pre-existing validation and test sets exhibited mathematically identical class distributions, indicating severe duplication that would result in artificially inflated accuracy metrics.

**Solution:** Engineered a robust data ingestion pipeline (`data/dataset.py`) that rejected the compromised splits, merged the raw data, and algorithmically applied strict stratified sampling to rebuild a mathematically sound 60/20/20 train/validation/test split. This ensured the model accurately reflects true generalization capability on unseen data. (Detailed in `DATASET_STRATEGY.md`).

### 2. Multi-Task Deep Learning Architecture
Designed and implemented a neural network architecture capable of joint optimization across multiple distinct clinical objectives.
- **Architecture (`models/architecture.py`):** Utilized robust vision foundational models (via `timm`) and extended them with a custom multi-head architecture for joint classification.
- **Optimization:** Designed a custom `MultiTaskLoss` function tailored to calculate and balance gradient flows across binary and multi-class objectives simultaneously during backpropagation.

### 3. Advanced Training Dynamics
Built a highly stable training engine designed to autonomously manage convergence and mitigate overfitting on medical imagery data.
- **Adaptive Scheduling:** Implemented an extensible training configuration supporting dynamic learning rate decay, including `CosineAnnealingLR` and `ReduceLROnPlateau`.
- **Early Stopping & Metrics:** Engineered a rigorous validation tracker that halts training based on minimum delta decay thresholds, dynamically saving the true optimum weights to minimize inference time and compute waste. (Detailed in `SCHEDULER_EARLY_STOPPING.md`).

## Technical Stack & AI/ML Competencies
- **Core ML Framework:** PyTorch, Torchvision
- **Computer Vision:** Timm (PyTorch Image Models)
- **Data Engineering:** NumPy, Pandas, Scikit-learn (Stratification, Metrics)
- **Applied Skills:** Data cleaning, deep learning architecture design, hyperparameter optimization, learning rate scheduling, rigorous statistical evaluation (F1-score, Precision, Recall).

## Directory Structure & System Design

- `models/`: Contains the dynamic network architectures and customized joint-loss functions.
- `engine/`: Modularized forward/backward propagation loops with step-aware learning rate adjustments.
- `data/`: Custom PyTorch `Dataset` classes, handling on-the-fly augmentations and tensor transformations.
- `configs/`: Centralized hyperparameter store for managing global configurations across training sweeps.
- `utils/`: Rigorous evaluation frameworks strictly utilizing held-out methodology.

## Getting Started

1. **Environment Setup:**
```bash
pip install -r requirements.txt
```

2. **Model Training:**
Execute the core training engine. The pipeline automatically structures the DataLoaders, provisions the optimizer, and initiates the multi-task learning process.
```bash
python train.py
```

3. **Inference & Evaluation:**
Perform rigorous statistical assessment using the strictly isolated test set. The engine automatically outputs a comprehensive classification report highlighting per-class precision and recall.
```bash
python evaluate_final.py
```

