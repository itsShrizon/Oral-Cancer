# Oral Cancer Classification Project

## Overview
Multi-model oral cancer classification project comparing 9 CNN architectures on:
- **Binary task:** Benign vs Malignant
- **Subtype task:** 7-class (CaS, CoS, Gum, MC, OC, OLP, OT)

## Models (9 total)
resnet50, densenet121, convnext_tiny, swin_t, efficientnet_b0, efficientnet_v2b2, efficientnet_v2b3, efficientnet_v2s, custom_efficientnet_v2

## Project Structure
```
Oral-Cancer-main/
├── configs/config.py          # All configuration (paths, hyperparams, model options)
├── models/
│   ├── architecture.py        # OralCancerModel: dual-head (binary + subtype) classifier
│   ├── loss.py                # Loss functions
│   └── custom_efficientnet/   # Custom EfficientNet V2 implementation
├── engine/trainer.py          # Training & validation loop (train_one_epoch, validate)
├── utils/
│   ├── common.py              # Data loading, transforms, common utilities
│   └── evaluation.py          # Eval utilities (has unused plot_confusion_matrix)
├── train.py                   # Standard training script (all models except custom)
├── evaluate_final.py          # Test-set evaluation: ACC, Precision, Recall, F1
├── compute_model_metrics.py   # Performance metrics: FLOPs, timing, memory, GradCAM, SHAP
├── custom_efficientnet_colab.py  # Dedicated train+eval for custom_efficientnet_v2
├── run_all_models.py          # Master runner: train -> eval -> viz -> metrics (all 9)
├── combined_main.ipynb        # Notebook version
├── Dataset 1/                 # Binary dataset (benign_lesions / malignant_lesions)
├── Dataset 2/                 # Subtype dataset (Training/Validation/Testing x 7 classes)
└── results/{backbone}/        # Per-model outputs
    ├── best_model.pth
    ├── evaluation_results.txt
    ├── performance_metrics.json
    ├── latency_distribution.png
    ├── gradcam_binary.png / gradcam_subtype.png
    └── shap_binary.png / shap_subtype.png
```

## Pipeline (run_all_models.py)
1. `train.py --backbone <model>` → `results/<model>/best_model.pth`
2. `evaluate_final.py --backbone <model>` → `results/<model>/evaluation_results.txt`
3. `visualize_predictions.py --backbone <model>` → prediction samples
4. `compute_model_metrics.py --backbone <model>` → `results/<model>/performance_metrics.json`

## Key Config (configs/config.py)
- IMG_SIZE: 224, BATCH_SIZE: 32, NUM_WORKERS: 0
- DS2_CLASSES: ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
- MALIGNANT_SUBTYPES: ['MC', 'OC', 'CaS']

## Optional Dependencies
`pip install thop grad-cam shap codecarbon psutil`

## Conventions
- All models share the same `OralCancerModel` dual-head architecture (models/architecture.py)
- custom_efficientnet_v2 has its own training script with warmup + TTA + Kaiming init
- `run_all_models.py` skips steps whose outputs already exist; use `--force` to rerun
- GradCAM target layers are backbone-specific (defined in compute_model_metrics.py)

## Output Files Per Model (results/{backbone}/)
- `best_model.pth` — saved model weights
- `evaluation_results.txt` — human-readable classification metrics
- `classification_metrics.json` — ACC/Precision/Recall/F1 for binary + subtype (JSON)
- `confusion_matrices.png` — binary + subtype confusion matrices side-by-side
- `training_time.json` — total training time, per-epoch times, epochs completed
- `performance_metrics.json` — FLOPs, timing, memory, latency, energy, carbon
- `latency_distribution.png` — histogram + CDF of inference latency
- `gradcam_binary.png` / `gradcam_subtype.png` — GradCAM++ visualizations
- `shap_binary.png` / `shap_subtype.png` — SHAP gradient explanations

## Known Remaining Gaps
- Training batch time is measured on test-phase inference in compute_model_metrics.py, not during actual training
