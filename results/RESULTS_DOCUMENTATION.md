# Oral Cancer Classification — Results Documentation

Multi-model comparison across 9 CNN/Transformer architectures for dual-head oral-pathology classification (binary Benign vs Malignant + 7-class subtype: CaS, CoS, Gum, MC, OC, OLP, OT).

Test set: **1,646 images** (DS1 + DS2 held-out split, same across all models).

---

## 1. Models evaluated

| # | Backbone | Folder | Status |
|---|---|---|---|
| 1 | ResNet-50 | `resnet50/` | complete |
| 2 | DenseNet-121 | `densenet121/` | complete |
| 3 | ConvNeXt-Tiny | `convnext_tiny/` | complete |
| 4 | Swin-T | `swin_t/` | complete |
| 5 | EfficientNet-B0 | `efficientnet_b0/` | complete |
| 6 | EfficientNet-V2-B2 | `efficientnet_v2b2/` | complete |
| 7 | EfficientNet-V2-B3 | `efficientnet_v2b3/` | complete |
| 8 | EfficientNet-V2-S | `efficientnet_v2s/` | complete |
| 9 | Inception-V3 | `inception_v3/` | complete |
| 10 | VGG-19 | `vgg19/` | **empty — no results produced** |
| 11 | **CustomEfficientNetV2 (tuned recipe)** | `custom_efficientnet_v2/` | complete |
| 12 | **CustomEfficientNetV2 (baseline-recipe ablation)** | `custom_efficientnet_v2_baseline_recipe/` | complete |

Note: the custom architecture appears twice — once with its tuned recipe (warmup, AMP, grad-clip, Kaiming init, ES=25, TTA) and once with the same recipe used by the other 8 baselines (lr=1e-4, no warmup/AMP/clip, ES=15). The baseline-recipe row is the ablation for fair comparison.

---

## 2. Classification metrics — Binary (Benign vs Malignant)

Sorted by F1-score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| **Custom EfficientNet V2 (tuned)** | **0.9947** | **0.9947** | **0.9947** | **0.9947** |
| EfficientNet-V2-B2 | 0.9936 | 0.9936 | 0.9936 | 0.9936 |
| Inception-V3 | 0.9930 | 0.9930 | 0.9930 | 0.9930 |
| ResNet-50 | 0.9912 | 0.9912 | 0.9912 | 0.9912 |
| **Custom EfficientNet V2 (baseline recipe)** | **0.9906** | **0.9906** | **0.9906** | **0.9906** |
| DenseNet-121 | 0.9889 | 0.9889 | 0.9889 | 0.9889 |
| EfficientNet-V2-B3 | 0.9889 | 0.9889 | 0.9889 | 0.9889 |
| EfficientNet-B0 | 0.9883 | 0.9883 | 0.9883 | 0.9883 |
| EfficientNet-V2-S | 0.9866 | 0.9866 | 0.9866 | 0.9865 |
| Swin-T | 0.9819 | 0.9819 | 0.9819 | 0.9819 |
| ConvNeXt-Tiny | 0.9532 | 0.9535 | 0.9532 | 0.9533 |

## 3. Classification metrics — Subtype (7-class)

Sorted by F1-score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| **Custom EfficientNet V2 (tuned)** | **0.9970** | **0.9970** | **0.9970** | **0.9970** |
| DenseNet-121 | 0.9933 | 0.9934 | 0.9933 | 0.9933 |
| EfficientNet-V2-B2 | 0.9933 | 0.9934 | 0.9933 | 0.9933 |
| **Custom EfficientNet V2 (baseline recipe)** | **0.9921** | **0.9921** | **0.9921** | **0.9921** |
| Inception-V3 | 0.9921 | 0.9922 | 0.9921 | 0.9921 |
| ResNet-50 | 0.9909 | 0.9909 | 0.9909 | 0.9909 |
| EfficientNet-B0 | 0.9903 | 0.9904 | 0.9903 | 0.9903 |
| EfficientNet-V2-B3 | 0.9842 | 0.9842 | 0.9842 | 0.9842 |
| EfficientNet-V2-S | 0.9836 | 0.9836 | 0.9836 | 0.9836 |
| Swin-T | 0.9769 | 0.9773 | 0.9769 | 0.9769 |
| ConvNeXt-Tiny | 0.9046 | 0.9069 | 0.9046 | 0.9048 |

---

## 4. Performance / efficiency metrics

| Model | Params | Size (MB) | Batch time (ms) | Infer mean ± std (ms) | GPU peak (MB) | Energy (kWh) | CO₂ (kg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Custom EfficientNet V2 (tuned)** | **4,898,430** | **18.99** | **21.57** | 9.81 ± 0.16 | **52.32** | 1.61e-05 | 3.75e-06 |
| **Custom EfficientNet V2 (baseline recipe)** | **4,898,430** | **18.99** | **21.55** | 11.44 ± 3.78 | **52.32** | 1.60e-05 | 3.72e-06 |
| EfficientNet-B0 | 5,323,909 | 20.60 | 23.82 | 8.89 ± 0.16 | 76.91 | 1.49e-05 | 3.48e-06 |
| DenseNet-121 | 8,008,073 | 31.15 | 53.04 | 19.60 ± 4.35 | 80.19 | 1.51e-05 | 3.51e-06 |
| EfficientNet-V2-B2 | 10,134,519 | 39.17 | 22.08 | 13.05 ± 0.13 | 72.47 | 1.59e-05 | 3.70e-06 |
| EfficientNet-V2-B3 | 14,399,911 | 55.57 | 28.44 | 15.33 ± 0.10 | 94.14 | 1.55e-05 | 3.60e-06 |
| EfficientNet-V2-S | 21,493,849 | 82.86 | 43.36 | 19.11 ± 0.20 | 126.94 | 1.72e-05 | 4.00e-06 |
| Inception-V3 | 23,888,361 | 91.47 | 29.04 | 13.45 ± 0.18 | 132.18 | 1.60e-05 | 3.73e-06 |
| ResNet-50 | 25,610,825 | 98.01 | 48.93 | 9.47 ± 3.40 | 148.37 | 1.36e-05 | 3.17e-06 |
| Swin-T | 28,311,427 | 108.07 | 63.56 | 11.52 ± 0.57 | 179.33 | 1.56e-05 | 3.64e-06 |
| ConvNeXt-Tiny | 28,612,201 | 109.22 | 47.33 | 6.79 ± 1.09 | 171.98 | 1.39e-05 | 3.24e-06 |

FLOPs reporting was disabled (`thop` not installed). Energy/CO₂ are estimated (codecarbon not installed).

### Latency distribution (single-image, ms)

| Model | Mean | Std | Min | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 7.10 | 1.09 | 6.42 | 6.56 | 8.62 | 8.80 | 12.33 | 13.07 |
| ResNet-50 | 7.38 | 0.93 | 6.76 | 6.88 | 9.09 | 9.18 | 9.62 | 10.13 |
| EfficientNet-B0 | 8.95 | 0.33 | 8.72 | 8.88 | 9.05 | 9.51 | 10.41 | 10.43 |
| Custom EfficientNet V2 (baseline recipe) | 10.67 | 1.62 | 9.84 | 10.16 | 11.66 | 13.41 | 20.11 | 20.51 |
| Custom EfficientNet V2 (tuned) | 11.52 | 1.50 | 9.72 | 12.82 | 12.93 | 13.02 | 13.32 | 13.44 |
| Swin-T | 11.70 | 0.62 | 11.47 | 11.54 | 11.72 | 12.18 | 15.38 | 15.41 |
| EfficientNet-V2-B2 | 13.34 | 0.80 | 12.93 | 13.15 | 13.40 | 14.02 | 17.55 | 17.68 |
| Inception-V3 | 15.79 | 2.09 | 13.30 | 17.57 | 17.68 | 17.76 | 18.05 | 19.15 |
| EfficientNet-V2-B3 | 16.52 | 2.04 | 15.21 | 15.41 | 20.30 | 20.38 | 20.52 | 20.60 |
| DenseNet-121 | 19.00 | 2.75 | 17.32 | 17.59 | 22.71 | 23.77 | 31.31 | 33.57 |
| EfficientNet-V2-S | 20.42 | 2.55 | 18.89 | 19.12 | 25.45 | 25.60 | 25.78 | 25.80 |

---

## 5. Training time

| Model | Epochs completed | Avg epoch (s) | Total time (min) | Best val loss |
|---|---:|---:|---:|---:|
| Custom EfficientNet V2 (tuned) | 190 / 200 | 43.10 | 136.53 | 0.0755 |
| **Custom EfficientNet V2 (baseline recipe)** | **172 / 200** | **45.92** | **131.70** | **0.0543** |
| DenseNet-121 | 100 / 200 | 65.26 | 108.84 | 0.0448 |
| EfficientNet-V2-B3 | 99 / 200 | 56.00 | 92.47 | 0.0634 |
| EfficientNet-B0 | 126 / 200 | 49.52 | 104.05 | 0.0712 |
| ResNet-50 | 129 / 200 | 61.26 | 131.82 | 0.0559 |
| EfficientNet-V2-B2 | 135 / 200 | 50.72 | 114.20 | 0.0357 |
| Inception-V3 | 153 / 200 | 52.97 | 135.22 | 0.0364 |
| Swin-T | 118 / 200 | 73.93 | 145.52 | 0.0976 |
| EfficientNet-V2-S | 86 / 200 | 68.80 | 98.72 | 0.0617 |
| ConvNeXt-Tiny | 200 / 200 | 78.71 | 262.55 | 0.3450 |

Non-custom models that ran < 200 epochs triggered early stopping (patience = 15). Custom-tuned used patience = 25 with warmup-offset start; baseline-recipe ablation used patience = 15 like the other 8.

---

## 6. Per-class subtype reports

Support per class (identical across all models): CaS=256, CoS=239, Gum=192, MC=288, OC=173, OLP=288, OT=210. Total = 1,646.

### Custom EfficientNet V2 (tuned) — best overall

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CaS | 1.00 | 1.00 | 1.00 | 256 |
| CoS | 1.00 | 1.00 | 1.00 | 239 |
| Gum | 1.00 | 1.00 | 1.00 | 192 |
| MC  | 1.00 | 0.99 | 0.99 | 288 |
| OC  | 0.98 | 1.00 | 0.99 | 173 |
| OLP | 0.99 | 1.00 | 1.00 | 288 |
| OT  | 1.00 | 1.00 | 1.00 | 210 |
| **macro avg** | **1.00** | **1.00** | **1.00** | 1646 |

### Custom EfficientNet V2 (baseline-recipe ablation)

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 0.98 | 0.99 | 0.99 |
| CoS | 0.99 | 1.00 | 1.00 |
| Gum | 1.00 | 1.00 | 1.00 |
| MC  | 0.99 | 0.99 | 0.99 |
| OC  | 0.99 | 0.97 | 0.98 |
| OLP | 0.99 | 1.00 | 0.99 |
| OT  | 1.00 | 0.99 | 0.99 |
| **macro avg** | **0.99** | **0.99** | **0.99** |

### DenseNet-121

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 1.00 | 0.98 | 0.99 |
| CoS | 1.00 | 1.00 | 1.00 |
| Gum | 1.00 | 1.00 | 1.00 |
| MC  | 1.00 | 0.99 | 0.99 |
| OC  | 0.99 | 0.98 | 0.99 |
| OLP | 0.98 | 1.00 | 0.99 |
| OT  | 1.00 | 1.00 | 1.00 |

### EfficientNet-V2-B2

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 1.00 | 0.99 | 0.99 |
| CoS | 0.99 | 1.00 | 0.99 |
| Gum | 0.99 | 1.00 | 1.00 |
| MC  | 1.00 | 0.99 | 0.99 |
| OC  | 0.98 | 0.99 | 0.99 |
| OLP | 1.00 | 0.99 | 0.99 |
| OT  | 0.99 | 1.00 | 1.00 |

### Inception-V3

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 0.99 | 1.00 | 1.00 |
| CoS | 1.00 | 1.00 | 1.00 |
| Gum | 1.00 | 1.00 | 1.00 |
| MC  | 1.00 | 0.98 | 0.99 |
| OC  | 0.97 | 0.99 | 0.98 |
| OLP | 1.00 | 0.99 | 0.99 |
| OT  | 0.98 | 0.99 | 0.99 |

### ResNet-50

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 1.00 | 0.99 | 1.00 |
| CoS | 1.00 | 1.00 | 1.00 |
| Gum | 0.99 | 0.99 | 0.99 |
| MC  | 0.98 | 0.98 | 0.98 |
| OC  | 0.98 | 0.98 | 0.98 |
| OLP | 0.99 | 0.99 | 0.99 |
| OT  | 1.00 | 1.00 | 1.00 |

### EfficientNet-B0

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 1.00 | 0.98 | 0.99 |
| CoS | 0.99 | 1.00 | 0.99 |
| Gum | 0.99 | 1.00 | 0.99 |
| MC  | 0.99 | 0.99 | 0.99 |
| OC  | 0.98 | 0.98 | 0.98 |
| OLP | 0.98 | 1.00 | 0.99 |
| OT  | 1.00 | 0.98 | 0.99 |

### EfficientNet-V2-B3

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 0.99 | 0.98 | 0.99 |
| CoS | 1.00 | 1.00 | 1.00 |
| Gum | 0.98 | 0.98 | 0.98 |
| MC  | 0.97 | 0.99 | 0.98 |
| OC  | 0.98 | 0.96 | 0.97 |
| OLP | 0.98 | 0.98 | 0.98 |
| OT  | 0.99 | 0.99 | 0.99 |

### EfficientNet-V2-S

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 0.99 | 0.98 | 0.99 |
| CoS | 0.98 | 1.00 | 0.99 |
| Gum | 0.98 | 0.99 | 0.99 |
| MC  | 0.99 | 0.98 | 0.98 |
| OC  | 0.97 | 0.97 | 0.97 |
| OLP | 0.98 | 0.98 | 0.98 |
| OT  | 0.99 | 0.99 | 0.99 |

### Swin-T

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 0.99 | 0.99 | 0.99 |
| CoS | 0.98 | 1.00 | 0.99 |
| Gum | 0.99 | 0.97 | 0.98 |
| MC  | 0.96 | 0.98 | 0.97 |
| OC  | 0.97 | 0.97 | 0.97 |
| OLP | 0.99 | 0.94 | 0.97 |
| OT  | 0.95 | 0.99 | 0.97 |

### ConvNeXt-Tiny (weakest baseline)

| Class | P | R | F1 |
|---|---:|---:|---:|
| CaS | 0.93 | 0.89 | 0.91 |
| CoS | 0.95 | 0.98 | 0.96 |
| Gum | 0.98 | 0.84 | 0.90 |
| MC  | 0.90 | 0.92 | 0.91 |
| OC  | 0.82 | 0.88 | 0.85 |
| OLP | 0.90 | 0.89 | 0.89 |
| OT  | 0.85 | 0.91 | 0.88 |

---

## 7. Outputs present in each folder

For models 1-9 and `custom_efficientnet_v2` (tuned):

| File | Contents |
|---|---|
| `best_model.pth` | Trained weights |
| `classification_metrics.json` | Accuracy / Precision / Recall / F1 for both heads (machine-readable) |
| `evaluation_results.txt` | Human-readable classification report incl. per-class metrics |
| `confusion_matrices.png` | Binary + subtype confusion matrices (side by side) |
| `performance_metrics.json` | Model size, params, batch/infer time, GPU memory, energy, latency percentiles |
| `latency_distribution.png` | Histogram + CDF of single-image inference latency |
| `training_time.json` | Total training time, per-epoch times, epochs completed, best val loss |

### Ablation folder `custom_efficientnet_v2_baseline_recipe/`

Now contains the full 7-file result set (same layout as the other 9 model folders). Generated with:

```
python evaluate_final.py       --backbone custom_efficientnet_v2 --recipe baseline --no-confirm
python compute_model_metrics.py --backbone custom_efficientnet_v2 --recipe baseline --skip-gradcam --skip-shap
```

GradCAM/SHAP were skipped here since the other 9 folders also omit them — the layout matches exactly.

### VGG-19 (`vgg19/`)

Empty. No checkpoint or metrics were produced. Either training crashed or was never run. Exclude from reported results or retrain before inclusion.

---

## 8. Headline findings

1. **Custom EfficientNet V2 (tuned) is the top model on both tasks** — 99.47 % binary F1, 99.70 % subtype F1, outperforming the best baseline (EfficientNet-V2-B2 at 99.36 % / 99.33 %).
2. It also uses the **smallest parameter count (4.90 M)** of the 9 models evaluated and the **lowest GPU peak memory (52 MB)**, so the gain is not coming from extra capacity.
3. **Ablation result (fair comparison):** the custom architecture trained with the **same recipe as the 8 baselines** still achieves binary F1 = 0.9906 and subtype F1 = 0.9921 — competitive with the best baselines and outperforming 5 of them on subtype. Training reached a lower val loss (0.0543) than the tuned run (0.0755), but the tuned recipe with TTA generalises slightly better on the test set (binary +0.41 pp, subtype +0.49 pp). This isolates the architecture contribution from the recipe contribution.
4. ConvNeXt-Tiny is the clear outlier on the subtype task (90.5 % F1 vs ~98-99 % for the other 8). Its val loss of 0.345 suggests under-fitting with the default recipe.
5. EfficientNet-B0 is the best pretrained baseline per-parameter (5.3 M params, 98.83 % binary F1, 99.03 % subtype F1) — the custom model beats it at a smaller parameter count.

---

## 9. Files generated alongside per-model folders

- `results/MODEL_RESULTS.docx` — existing summary document
- `results/all_models_confusion_matrices.pdf` — stitched confusion matrices
- `results/all_models_confusion_and_reports.pdf` — confusion matrices + text reports
- `results/confusion_matrices.png` — single-model output (stale — from last individual run)
- `results/evaluation_results.txt` — single-model output (stale)
- `results/prediction_samples.png`, `results/detailed_prediction_grid.png` — sample prediction grids

These top-level files were produced by ad-hoc scripts or prior runs and sit outside the per-model folder convention.
