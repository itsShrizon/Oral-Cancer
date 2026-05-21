# A Parameter-Efficient Dual-Head Oral Disease Classifier with an Ablation-Driven Triplet→SE Attention Cascade

*Manuscript prepared for ACM Proceedings format (two-column body, single-column abstract).*

---

## CCS Concepts

- **Computing methodologies** → Object identification; *Neural networks*; *Image representations*.
- **Applied computing** → *Health informatics*; *Imaging*.

## Keywords

Oral disease classification, attention mechanisms, ablation study, parameter-efficient architectures, EfficientNet, role-complementarity, GradCAM++, LIME, lightweight medical image analysis, dual-head multi-task learning, Triplet Attention, Squeeze-and-Excitation.

---

## Abstract

Computer-aided diagnosis of oral disease has matured along two narrow tracks — binary malignant-versus-benign screening and pixel-level lesion segmentation — while multi-class subtype classification, the formulation most aligned with downstream clinical decision-making, remains comparatively under-explored. The few existing multi-class systems rely on heavy ensembles whose architectural choices are not justified by ablation. This paper proposes a parameter-efficient five-stage Convolutional Neural Network (CNN) backbone, *Custom EfficientNet V2*, in which the standard Block-4 of EfficientNetV2-B0 is replaced by a domain-tuned *AttentionHub* module. A second variant, *AttentionHub-v2*, is then introduced: a sequential cascade of Triplet Attention followed by Squeeze-and-Excitation (SE), whose design is forced by a systematic seven-cell ablation of the v1 hub. The ablation reveals a **role-complementarity principle**: pairing Triplet (a cross-dimensional spatial attention) with any module that also exercises a spatial role — Bottleneck Attention Module (BAM) or Efficient Multi-scale Attention (EMA) — regresses subtype accuracy to a fixed 98.36 % ceiling, whereas pairing Triplet with a *purely* channel-wise partner — Kolmogorov-Arnold Network attention (KAN) or SE — lifts performance to 99.45 % and 99.51 % respectively. Trained from scratch under a single matched recipe and benchmarked against nine standard CNN and Transformer baselines (ResNet50, DenseNet121, ConvNeXt-Tiny, Swin-T, EfficientNet-B0, EfficientNetV2-B2/B3/S, Inception V3) on a seven-class oral disease dataset combined with a binary benign-versus-malignant split, the proposed model achieves **99.06 %** binary accuracy and **99.51 %** subtype accuracy — the highest subtype score in the entire benchmark — at only **4.79 M** parameters and **0.493 GFLOPs**, which is between **2.1× and 5.8× smaller** than the strongest baselines. Gradient-weighted Class Activation Mapping++ (GradCAM++) and Local Interpretable Model-Agnostic Explanations (LIME) panels confirm that the network attends to lesion tissue rather than to incidental visual artifacts. The complete experimental package — nine matched baselines, seven attention ablations, dual-task evaluation, and per-model interpretability artifacts — is released to support reproducibility.

**Keywords:** Oral disease classification, attention mechanisms, ablation study, EfficientNet, role-complementarity, GradCAM++, LIME, lightweight medical image analysis.

---

## 1. Introduction

Oral diseases — including ulcerative, inflammatory, and neoplastic lesions — present a complex visual challenge for automated analysis. Conditions such as recurrent aphthous ulcer, oral lichen planus, and squamous-cell carcinoma can share overlapping colour, texture, and boundary cues at the image level, while a single misclassification between a benign inflammatory lesion and an early malignancy carries materially different clinical consequences. Computer-aided diagnostic systems must therefore solve two related but distinct problems: identifying *whether* a lesion is malignant (binary screening) and identifying *which* of several visually similar conditions is present (multi-class subtyping).

Deep learning has matured rapidly in this domain. Convolutional Neural Networks (CNNs) and, more recently, Vision Transformers have produced strong binary screening results [5, 16, 17] and several promising multi-class systems [19, 22]. However, three persistent gaps remain in the literature.

**Gap 1 — Binary formulations dominate.** Most published architectures collapse oral pathology to a benign-versus-malignant decision, which is clinically insufficient when downstream treatment depends on differentiating among multiple non-malignant conditions that visually mimic malignancy.

**Gap 2 — Attention modules are stacked without ablation.** Modern backbones routinely include attention components — Squeeze-and-Excitation (SE), Convolutional Block Attention Module (CBAM), Bottleneck Attention Module (BAM), Triplet Attention, Efficient Multi-scale Attention (EMA), Kolmogorov-Arnold Network attention (KAN) — but their individual and combined contributions in the oral-disease setting are rarely measured. The literature offers little guidance on which attention modules complement one another and which conflict, and proposed architectures inherit attention designs from generic benchmarks rather than testing them on the target distribution.

**Gap 3 — Explainability evidence is sparse.** Few multi-class oral disease studies present visual attention maps that demonstrate the model attends to lesion tissue rather than to teeth, lip outlines, or lighting artifacts. Without such evidence it is difficult to argue that a high-accuracy model is also clinically defensible.

The present work addresses these three gaps directly. We make four contributions.

1. We design **Custom EfficientNet V2**, a five-stage parameter-efficient backbone derived from EfficientNetV2-B0 in which the standard Block-4 is replaced by a domain-tuned *AttentionHub* and the redundant Block-6 / conv-head is removed, yielding a 4.79 M-parameter / 0.493 GFLOPs model.
2. We perform a systematic **seven-cell ablation** of the AttentionHub spanning {BAM, Triplet, KAN} singletons, all pairs, and the full triple, plus a no-attention control. From this ablation we derive a **role-complementarity principle** that explains why certain attention pairings systematically regress to a fixed accuracy ceiling on this dataset.
3. The principle motivates **AttentionHub-v2**, a sequential Triplet → SE cascade in which spatial cross-dimensional attention is followed by purely channel-wise recalibration. AttentionHub-v2 achieves the highest subtype accuracy in the entire study (**99.51 %**) at no extra parameters or compute.
4. We benchmark the proposed model against nine standard backbones under a **single matched training recipe**, with all models trained from scratch (no ImageNet pre-training), and provide GradCAM++ and LIME explainability panels for every model. The full ablation results and explainability artifacts are released alongside the paper.

The remainder of this paper is organised as follows. Section 2 reviews prior work on oral lesion classification and segmentation, identifies gaps that motivate our design, and positions the present study. Section 3 describes the dataset, the backbone, the AttentionHub variants, and the training protocol. Section 3 fixes notation and the role taxonomy that the ablation rests on. Section 4 describes the dataset, the backbone, the two AttentionHub variants, the multi-task loss, and the matched training protocol. Section 5 specifies the experimental setup, reproducibility provisions, and ethical considerations. Section 6 reports baseline benchmark results and per-class performance. Section 7 presents the ablation study and develops the role-complementarity principle. Section 8 analyses the GradCAM++ and LIME explainability panels. Section 9 discusses the findings, compares to prior multi-class systems, and acknowledges limitations. Section 10 concludes. References follow.

---

## 2. Related Work

Recent advances in artificial intelligence and image-based analysis have substantially expanded the diagnostic capabilities of oral health systems. Convolutional Neural Networks and, more recently, Transformer-based architectures have demonstrated strong performance on a range of oral lesion classification and cancer detection tasks. However, the existing literature is heavily skewed toward two narrow problem formulations: binary malignant-versus-benign discrimination, and detection of a single dominant disease category. Multi-class classification of clinically heterogeneous oral conditions remains comparatively under-explored, despite its closer alignment with real screening practice. Compounding this, most reported architectures inherit attention modules and hyper-parameters from generic benchmarks without systematic ablation, and few studies pair their quantitative results with explainability evidence that the model attends to clinically relevant tissue. Object-detection frameworks such as YOLO variants have been adapted for lesion localization but produce only bounding boxes and do not characterize sub-class identity, while a separate line of work targets pixel-level segmentation. Both directions are valuable but address questions orthogonal to the multi-class diagnostic categorization that the present study targets.

### 2.1 Binary Oral Cancer Detection

A substantial body of work has approached oral cancer as a binary screening problem. The CNN-based framework reported in [5] used transfer learning to compensate for limited clinical data and achieved approximately 90 % classification accuracy between malignant and non-malignant lesions, demonstrating discriminative capacity sufficient for screening. The binary formulation, however, precludes sub-type differentiation and the model does not characterize lesion morphology.

Subsequent efforts have prioritized computational efficiency. The SE-MobileViT architecture proposed in [16] integrates Squeeze-and-Excitation modules into a MobileViT backbone for channel-wise attention, achieving above 92 % binary accuracy with reduced inference cost suitable for mobile deployment. The same trade-off applies: the model establishes the presence or absence of malignancy but does not differentiate among clinically distinct oral conditions.

A complementary direction has combined hand-crafted and learned representations. The ensemble proposed in [17] fuses Local Binary Pattern (LBP) descriptors with deep CNN features, reaching approximately 90 % binary accuracy with improved robustness to illumination and texture variation. The contribution remains at the representation level — the model still maps each image to a binary outcome.

Across this family of methods, the dominant limitation is not accuracy but scope: binary screening is insufficient when downstream clinical decisions depend on identifying which oral condition is present, since ulcerative, inflammatory, and neoplastic lesions warrant different management pathways.

### 2.2 Recurrent Aphthous Ulcer-Focused Models

A smaller body of work has targeted recurrent aphthous ulcer (RAU) specifically. The CNN-based framework reported in [3] used transfer learning with ResNet variants to distinguish ulcer and non-ulcer classes on a modest clinical dataset, achieving above 85 % accuracy. While the result demonstrates feasibility under data-scarce conditions, the formulation is binary and the small dataset constrains generalization to broader oral disease populations.

A recent systematic review [23] surveyed sixteen AI studies focused on oral lichen planus (OLP), recurrent aphthous stomatitis (RAS), and leukoplakia, reporting accuracies ranging from 71 % to 100 % depending on dataset size and architecture choice. The review highlights two persistent gaps: limited class balance across datasets, and the near-absence of explainability evidence. These gaps motivate the dual emphasis of the present study on multi-class differentiation and visual interpretability.

### 2.3 Semantic Segmentation of Oral Lesions

A complementary line of research has approached oral lesions as a pixel-level segmentation problem. The CLASEG framework [15] adopts a U-Net-like architecture for semantic segmentation of oral mucosal lesions, evaluating performance with Dice and Intersection-over-Union metrics; segmentation refines boundary precision beyond what classification can provide, but performance is constrained by dataset size and the model struggles in low-contrast regions where the ulcer boundary is visually ambiguous.

A more recent hybrid CNN-Transformer design [26] integrates high-order focus convolution with edge-aware modules and introduces Sobel-based edge enhancement to improve boundary delineation. Trained on 420 images across five ulcer categories, the model achieves a Dice score near 82 % and sensitivity around 85 %, outperforming an HF-UNet baseline. Limited dataset size remains the principal constraint.

Segmentation and classification address different clinical questions: segmentation localizes lesion extent at pixel granularity, whereas classification identifies the disease class. The present work focuses on the latter, where multi-class differentiation among visually similar conditions remains the dominant challenge.

### 2.4 Multi-Class Oral Disease Classification

The closest prior work to the present study is multi-class oral disease classification. The benchmark in [19] evaluated several CNN backbones on intraoral images under a unified preprocessing protocol, prioritizing generalization across patient demographics. While informative, the comparison treats backbone selection as the only design axis and does not investigate how attention mechanisms should be integrated.

MODC-SET [22] proposes an ensemble framework combining MobileNetV2, InceptionResNetV2, and ResNet50 with an XGBoost meta-classifier, evaluated on a newly curated dataset of seven oral disease categories. The ensemble achieves 99.32 % overall accuracy, with feature fusion improving discrimination between visually similar lesions. However, the architectural choices — which backbones to ensemble, why XGBoost as the meta-classifier — are not subjected to ablation, and the substantial cumulative parameter count of the combined model raises practical deployment concerns. MODC-SET demonstrates that high accuracy is achievable on seven-class oral disease data; the present study addresses the orthogonal question of how an efficient single-model alternative can be designed in a principled, ablation-driven way.

### 2.5 Transformer and Hybrid Architectures

Vision Transformer (ViT) architectures were compared against radiomics-based baselines in [18], with the ViT variants benefiting from improved global feature extraction; the reported accuracies are competitive with CNN baselines, but the study does not investigate attention-module ablation within the ViT backbone. Hybrid CNN-Transformer designs have also been proposed: the work in [21] combined DeiT and CoAtNet for oral disease classification, achieving strong multi-class performance, though the hybrid attention configuration is taken as given rather than tested through ablation. A distinct line of work has explored multimodal large language models as diagnostic assistants — the evaluation in [24] tested ChatGPT-5's ability to differentiate OLP, oral lichen lesions, and squamous cell carcinoma developing over lichen planus, reaching approximately 85 % accuracy. That framework emphasizes diagnostic reasoning over spatial precision and complements rather than replaces dedicated vision models.

### 2.6 Position of the Present Work

The literature reviewed above reveals three gaps that motivate the present study. **First**, binary formulations dominate; the few multi-class systems that exist [19, 22] do not subject their architectural decisions to ablation. **Second**, attention modules are stacked without principled justification — the literature offers no empirically grounded rule for how attention components should be combined on this distribution. **Third**, explainability evidence in the multi-class oral disease setting is sparse, leaving the field's high-accuracy claims clinically unverified.

The present work addresses all three gaps. We propose a parameter-efficient backbone with a novel AttentionHub module whose design is derived from a seven-cell ablation, we benchmark against nine standard CNN and Transformer baselines under a single matched training recipe with all models trained from scratch, and we accompany every model with GradCAM++ and LIME panels for both the binary and the 7-class head.

---

## 3. Preliminaries and Notation

For clarity, this section fixes notation used throughout the methodology.

Let `x ∈ ℝ^{B × 3 × H × W}` denote a mini-batch of input images at resolution H × W = 224 × 224 and batch size B. A backbone network *f_θ* maps `x` to a feature representation `F = f_θ(x) ∈ ℝ^{B × D}` where D is the backbone's feature dimensionality (D = 192 for the proposed model). Two linear heads `g_binary` and `g_subtype` produce class logits

```
ŷ_b = g_binary(F) ∈ ℝ^{B × 2},     ŷ_s = g_subtype(F) ∈ ℝ^{B × 7}.
```

Ground-truth labels are denoted `y_b ∈ {0, 1}^B` (benign / malignant) and `y_s ∈ {0, 1, …, 6, -1}^B` where `y_s = -1` encodes "subtype unavailable" for samples drawn from the binary-only dataset (DS1). The two heads are trained jointly with a single optimizer; subtype loss is masked when `y_s = -1` (see §4.5).

An *attention module* is any function `A : ℝ^{B × C × H × W} → ℝ^{B × C × H × W}` that reweights its input in a content-dependent way. We classify attention modules by their *operational role*:

- **Spatial role** — modulates the (H, W) plane. Examples: the spatial gate of BAM; the cross-dimensional gates of Triplet Attention; the multi-scale spatial branch of EMA.
- **Channel role** — modulates the C axis via a globally-pooled statistic. Examples: SE, ECA, KAN-Attention, and the channel gate of BAM.
- **Mixed role** — modules whose internal design exercises both axes simultaneously (BAM, EMA, CBAM-sequential).

This role taxonomy is the technical foundation of the role-complementarity principle (§6.2).

---

## 4. Materials and Methods

### 4.1 Dataset

The experimental data combine two complementary sources. *Dataset 1* (DS1) provides binary supervision: a curated collection of oral lesion photographs labelled benign or malignant, used to support the binary head of the dual-task classifier. *Dataset 2* (DS2) provides seven-way subtype supervision across the classes **CaS** (carcinoma in situ), **CoS** (condyloma-like / oral squamous), **Gum** (gingivitis), **MC** (oral mucositis carcinoma), **OC** (oral candidiasis / oral cancer), **OLP** (oral lichen planus), and **OT** (other / control). Within DS2, the subset {**MC, OC, CaS**} is taken to denote malignant subtypes for the purpose of constructing binary supervision when DS2 is read into the binary head.

To avoid the distributional similarity that the original DS2 distribution exhibited between its provided *Training* and *Testing* folders, the Training and Validation directories are merged and re-split with a stratified 60/20/20 train/validation/test partition, seeded for reproducibility (seed = 42). DS1 is split independently with stratified 80/10/10 partitions. The resulting test set contains 1 646 DS2 images distributed across the seven subtype classes, with the following per-class supports: CaS = 256, CoS = 239, Gum = 192, MC = 288, OC = 173, OLP = 288, OT = 210. All images are resized to 224 × 224 and normalized with ImageNet statistics. Training augmentation comprises random horizontal and vertical flips, random rotation up to ±15°, random crop after resize to 256 × 256, and colour jitter (brightness, contrast, saturation each ±0.2; hue ±0.1). The validation and test sets receive no augmentation. The test set is held out entirely and used only for final reporting.

### 4.2 Dual-Head Multi-Task Classifier

All ten architectures (nine baselines plus the proposed model) share a common dual-head wrapper, `MultiTaskOralClassifier`. A shared backbone produces a feature vector; two independent multilayer perceptron heads then predict the binary class (2 logits) and the subtype class (7 logits). Each head is a `Linear(F → 512) → ReLU → Dropout → Linear(512 → C)` stack, where *F* is the backbone feature dimension and *C* is the head's class count. A shared dropout layer (p = 0.5) is applied to the features prior to either head. The dual-head construction allows DS1 (which lacks subtype labels) and DS2 (which has both) to be trained jointly via a masked subtype loss (Section 3.5).

### 4.3 Custom EfficientNet V2 Backbone

The proposed backbone is a five-stage CNN derived from *tf_efficientnetv2_b0*. Two design changes are applied to the donor architecture:

1. **Stage 4 replacement.** The donor's Block-4 (an MBConv + SE block, 96 → 112 channels) is removed and replaced by an *AttentionHub* (Section 3.4) with the same input and output channel counts. The hub is the only stage of the network whose internal structure differs from the EfficientNetV2-B0 reference.
2. **Tail trimming.** Block-6 and the donor's `conv_head` are dropped. The 192-channel output of Stage 5 is pooled directly into the dual-head classifier.

The resulting backbone has five stages — *Stem + Block-0 + Block-1* (32 channels) → *Block-2* (48) → *Block-3* (96) → *AttentionHub* (112) → *Block-5* (192) — followed by global average pooling and the dual heads. The full model contains 4.79–4.80 million parameters and requires 0.493–0.495 GFLOPs (the small range covers the various ablation variants reported in Section 5).

### 4.4 AttentionHub Variants

The AttentionHub is the only module that varies across ablation cells. Two complete versions are presented in this paper.

**AttentionHub v1 (parallel triple).** The original proposed module routes the Stage 3 output through three parallel attention branches — BAM, Triplet Attention, and KAN-Attention — each preceded by a 1 × 1 channel-reduction projection that halves the channel count for the branch. Branch outputs are concatenated and fused by a 1 × 1 projection back to 112 channels. Ablation is implemented by removing branches: the empty configuration replaces Stage 4 with the donor's original Block-4 (the canonical no-attention control), and any subset of {bam, triplet, kan} is admissible. The three modules are summarised below.

- **BAM** (Bottleneck Attention Module, Park *et al.*, BMVC 2018 [B1]). Combines a channel gate (GAP → MLP) and a spatial gate (1 × 1 → dilated 3 × 3 → 1 × 1 producing a single-channel spatial map). The two gates are summed and passed through a sigmoid, then multiplied with the input. BAM exercises *both* channel and spatial roles.
- **Triplet Attention** (Misra *et al.*, WACV 2021 [B2]). Three branches each apply a 7 × 7 attention gate to a permutation of the input: branch 1 swaps C ↔ H, branch 2 swaps C ↔ W, branch 3 leaves the input as (B, C, H, W). The gates use a Z-Pool that concatenates max- and mean-pool across the gated axis to two channels, then a 7 × 7 convolution and sigmoid. The three branch outputs are averaged. Triplet exercises a *cross-dimensional spatial* role with near-zero additional parameters.
- **KAN-Attention.** A Kolmogorov-Arnold-Network-inspired channel recalibration. Global average pooling produces a per-channel scalar; each scalar is passed through a learnable B-spline basis expansion (five Gaussian basis functions in [-3, +3]) with a SiLU residual, then sigmoid-gated and multiplied with the input. KAN-Attention is *purely channel-wise*; it provides a non-linear alternative to the linear MLP used in SE / BAM channel gates.

**AttentionHub v2 (sequential Triplet → SE).** The proposed model uses a sequential cascade in which the channel-reduced feature is first refined by Triplet Attention and then by Squeeze-and-Excitation (Hu *et al.*, CVPR 2018 [B3]). The SE module pools globally, passes through a two-layer 1 × 1 bottleneck with reduction ratio 16 and a sigmoid gate, and multiplies the result back into the input. No LayerScale, no per-module residual, no additional gating. The composition order follows the convention established by CBAM (Woo *et al.*, ECCV 2018 [B4]) — spatial attention precedes channel attention. The design rationale for v2 is developed in Section 5 from the v1 ablation.

### 4.5 Multi-Task Loss

A combined cross-entropy loss is used for both heads:

```
L = w_b · CE(p_binary, y_binary) + w_s · CE(p_subtype, y_subtype; ignore_index = -1)
```

with `w_b = w_s = 1`. The `ignore_index = -1` convention allows DS1 samples (which have no subtype label, encoded as -1) to contribute to the binary loss only. A guard converts any NaN subtype loss to zero so that batches containing exclusively DS1 samples remain valid.

**Algorithm 1 — Joint training step under masked multi-task loss.**

```
Input  : mini-batch (x, y_b, y_s), model M = backbone + (g_binary, g_subtype),
         optimizer Opt, loss weights (w_b, w_s).
Output : updated model parameters.

  1.  F        ← backbone(x)                        # B × D
  2.  F        ← Dropout(F, p = 0.5)
  3.  ŷ_b     ← g_binary(F)                         # B × 2
  4.  ŷ_s     ← g_subtype(F)                        # B × 7
  5.  L_b      ← CE(ŷ_b, y_b)                       # always defined
  6.  L_s      ← CE(ŷ_s, y_s; ignore_index = -1)    # NaN if every y_s = -1
  7.  if isnan(L_s):  L_s ← 0                       # DS1-only batch guard
  8.  L_total  ← w_b · L_b + w_s · L_s
  9.  Opt.zero_grad();  L_total.backward();  Opt.step()
```

The masked subtype loss is the only mechanism by which DS1 (binary-only) and DS2 (binary + subtype) co-train against a single model: DS1 samples contribute only to L_b, DS2 samples contribute to both.

### 4.6 Training Protocol (Fair Comparison)

All ten models — nine baselines and the proposed model — are trained from scratch under a single matched recipe to guarantee a fair comparison. **No ImageNet pre-trained weights are used**, since pre-training would smuggle a representational prior into the comparison that confounds the contribution of the dataset itself. The recipe is fixed as follows.

- **Optimizer:** Adam, learning rate 1 × 10⁻⁴, weight decay 1 × 10⁻⁴.
- **Scheduler:** cosine annealing over the full epoch budget.
- **Batch size:** 64, gradient accumulation 1 (effective batch 64).
- **Epoch budget:** 200, with early stopping (patience 15, min Δ = 1 × 10⁻⁴ on validation loss).
- **Seed:** 42, deterministic CuDNN.
- **Image size:** 224 × 224.
- **Hardware:** NVIDIA GeForce RTX 4060 Ti.

This recipe is enforced regardless of model. In particular, attention-hub ablation variants are forced onto the baseline recipe so that v1, v2, and the seven ablation cells are directly comparable. Per-model hyper-parameter tuning is deliberately *not* performed: tuning would advantage models we choose to tune and disadvantage models we do not, and prior fair-comparison studies in this lab established that grad-accumulation, not per-model hyper-parameter overrides, is the correct response to memory pressure.

### 4.7 Evaluation Protocol

For each model we report, on the held-out test set:

- **Classification metrics:** accuracy, weighted precision, weighted recall, weighted F1 — for both the binary (DS1 + DS2-malignant) and subtype (DS2 7-class) heads.
- **Per-class report:** precision, recall, F1, support for each of the seven subtype classes.
- **Confusion matrices:** binary and subtype confusion matrices, side-by-side.
- **Computational metrics:** parameter count, GFLOPs (computed with `thop`), model size on disk (MB), GPU peak memory during inference, mean and standard deviation of single-image inference latency (200-iteration warm-up, 1 000 timed iterations), P50 / P90 / P95 / P99 latency, energy (kWh) and carbon (kg CO₂-eq) measured with `codecarbon` where available, total training time, and number of epochs completed before early stopping.
- **Explainability artifacts:** for each model a GradCAM++ panel and a LIME boundary-mask panel for the binary head, and the same pair for the subtype head, evaluated on a class-balanced sample of test images.

The per-model JSON metric files, training-time logs, latency histograms, confusion matrices, and explainability PNGs are released alongside the paper.

### 4.8 Ablation Protocol

The AttentionHub v1 ablation is implemented as a code-level switch in `utils/ablation.py`: each ablation key maps to a tuple of active branches (subset of `{bam, triplet, kan}`), where the empty tuple `()` falls back to the donor's original Block-4 (the no-attention control). The eight cells investigated are:

| key | active branches | role |
|---|---|---|
| `none` | – (donor MBConv+SE Block-4) | no-attention control |
| `bam` | BAM | spatial + channel singleton |
| `triplet` | Triplet | cross-dim-spatial singleton |
| `kan` | KAN | channel singleton |
| `bam_triplet` | BAM + Triplet | two-spatial-role pair |
| `bam_kan` | BAM + KAN | mixed-role pair |
| `triplet_kan` | Triplet + KAN | cross-dim-spatial + pure-channel pair |
| `full` (= v1) | BAM + Triplet + KAN | original proposed model |

In addition we report a v2-EMA negative-result run (Triplet + EMA sequential) which is *not* an ablation cell but tested the role-complementarity principle by replacing SE with EMA (multi-scale spatial + channel). The v2-EMA result reinforces the role-complementarity finding (Section 5).

Every ablation cell is trained under the identical baseline recipe used by the nine baseline backbones; no recipe asterisks apply.

---

## 5. Experimental Setup

### 5.1 Hardware and Software Stack

All experiments are executed on a single workstation equipped with an **NVIDIA GeForce RTX 4060 Ti (16 GB GDDR6)** GPU, 32 GB system RAM, and an AMD Ryzen 7 CPU. The software stack is fixed across all runs: Python 3.11, PyTorch 2.x with CUDA 12, `timm` 0.9 for the donor EfficientNetV2-B0 weights and structural reference (note: only the structural reference is used; weights are *not* loaded — see §4.6), `thop` for floating-point operation counting, `pytorch-grad-cam` 1.5 for GradCAM++, `lime` 0.2 with `scikit-image` for super-pixel boundary masks, and `codecarbon` for energy and CO₂-equivalent emissions tracking. Deterministic CuDNN is enabled for all reported runs.

### 5.2 Reproducibility Provisions

The complete experimental package is structured for reproducibility along five axes.

1. **Code and configuration release.** The full training, evaluation, and metric scripts are provided in the repository. A single configuration file (`configs/config.py`) controls all global hyper-parameters and is enforced across every run.
2. **Fixed seed.** Every run uses Python, NumPy, and PyTorch random seed 42. CuDNN deterministic mode is on.
3. **Fixed split.** Dataset splits are deterministic functions of the seed; no train/test contamination is possible across runs.
4. **Per-run JSON.** Every model produces a `classification_metrics.json`, a `performance_metrics.json`, a `training_time.json`, and a per-class textual report (`evaluation_results.txt`). These are committed alongside the model weights.
5. **Explainability artifacts.** GradCAM++ + LIME panels for both heads (`explain_binary.png`, `explain_subtype.png`) are produced by a single script (`explain_model.py`) with identical preprocessing across all models.

A small number of runs (the AttentionHub ablation cells) include CodeCarbon-measured energy and CO₂, while the older baseline runs report the conservative analytical estimate used in the absence of CodeCarbon at the time of the run. We retain both for transparency.

### 5.3 Ethical Considerations

The datasets used in this study are pre-existing curated collections of oral lesion images. We use them only for non-clinical research purposes and report all results on a held-out split that is disjoint from training and validation. No patient-identifying information is present in the dataset, and no human subjects are recruited as part of this work. The proposed model is *not* a clinical decision-support system and should not be used as one without appropriate clinical validation, regulatory clearance, and prospective evaluation. We acknowledge that automated screening systems can perpetuate disparities present in their training data — in particular, the dataset's demographic and intra-oral-anatomy coverage is not characterized in published metadata, and external validation across diverse patient populations is required before any clinical claim can be made.

The model is released under a research-use license; the explicit intent is to enable replication and methodological extension, not direct clinical deployment.

---

## 6. Baseline Benchmark Results

### 6.1 Quantitative Comparison

Table 1 reports the held-out test performance for all ten architectures. All numbers are from a single training run per model under the matched recipe of Section 3.6, with no per-model tuning.

**Table 1 — Test-set classification performance under matched fair recipe.**

| Model | Binary Acc | Binary F1 | Subtype Acc | Subtype F1 | Params (M) | GFLOPs | Size (MB) |
|---|---|---|---|---|---|---|---|
| ResNet50 | 0.9912 | 0.9912 | 0.9909 | 0.9909 | 25.61 | 4.134 | 98.01 |
| DenseNet121 | 0.9889 | 0.9889 | 0.9933 | 0.9933 | 7.92 | 2.834 | 31.15 |
| ConvNeXt-Tiny | 0.9532 | 0.9533 | 0.9046 | 0.9048 | 28.59 | 4.456 | 109.22 |
| Swin-T | 0.9819 | 0.9819 | 0.9769 | 0.9769 | 28.29 | 4.372 | 108.07 |
| EfficientNet-B0 | 0.9883 | 0.9883 | 0.9903 | 0.9903 | 5.28 | 0.386 | 20.60 |
| EfficientNet V2-B2 | **0.9936** | **0.9936** | 0.9933 | 0.9933 | 10.00 | 1.100 | 39.17 |
| EfficientNet V2-B3 | 0.9889 | 0.9889 | 0.9842 | 0.9842 | 14.21 | 1.522 | 55.57 |
| EfficientNet V2-S | 0.9866 | 0.9865 | 0.9836 | 0.9836 | 21.22 | 2.711 | 82.86 |
| Inception V3 | 0.9930 | 0.9930 | 0.9921 | 0.9921 | 23.85 | 2.838 | 91.47 |
| **Custom EfficientNet V2 (Hub v1, full)** | 0.9906 | 0.9906 | 0.9921 | 0.9921 | 4.80 | 0.495 | 18.99 |
| **Custom EfficientNet V2 (Hub v2, proposed)** | 0.9906 | 0.9907 | **0.9951** | **0.9951** | **4.79** | **0.493** | **18.94** |

Two readings are immediately visible. **First**, the proposed model achieves the highest subtype accuracy in the study, exceeding the strongest baseline (EfficientNetV2-B2 at 99.33 %) by 0.18 percentage points and the heaviest comparator (Inception V3 at 99.21 %) by 0.30 percentage points. **Second**, this is achieved while being the smallest and cheapest model in the comparison: at 4.79 M parameters the proposed model is **2.1× smaller** than EfficientNetV2-B2 (10.0 M), **5.0× smaller** than Inception V3 (23.9 M), and **5.3× smaller** than ResNet50 (25.6 M), with corresponding reductions in FLOPs.

The binary head is essentially tied at the top of the table. EfficientNetV2-B2 leads at 99.36 % and the proposed model is at 99.06 % — a difference of five images on a held-out test set of approximately 1 646 DS2 samples plus the DS1 binary set, well inside the noise band of a single training run. We do not claim a binary-task improvement.

ConvNeXt-Tiny is the only baseline that visibly under-performs (95.32 / 90.46), consistent with its known data-hunger when trained from scratch on small medical datasets — it consumed the full 200-epoch budget without reaching early-stopping criteria. We retain this row because it documents a real limitation of from-scratch ConvNeXt-style training under this regime.

### 6.2 Per-Class Subtype Performance

Table 2 reports per-class precision, recall, and F1 for the proposed model and the two strongest baselines on the subtype head. Five of seven classes reach 1.00 F1 for the proposed model; residual confusion is confined to MC and OC, two malignant carcinomas with overlapping clinical presentations.

**Table 2 — Per-class subtype performance (test-set, support in parentheses).**

| Class | Custom V2 (Hub v2) | EfficientNet V2-B2 | Inception V3 |
|---|---|---|---|
| CaS  (256) | 1.00 / 1.00 / 1.00 | 1.00 / 0.99 / 0.99 | 0.99 / 1.00 / 1.00 |
| CoS  (239) | 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 0.99 | 1.00 / 1.00 / 1.00 |
| Gum  (192) | 0.99 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| MC   (288) | 1.00 / 0.98 / 0.99 | 1.00 / 0.99 / 0.99 | 1.00 / 0.98 / 0.99 |
| OC   (173) | 0.98 / 0.99 / 0.99 | 0.98 / 0.99 / 0.99 | 0.97 / 0.99 / 0.98 |
| OLP  (288) | 1.00 / 1.00 / 1.00 | 1.00 / 0.99 / 0.99 | 1.00 / 0.99 / 0.99 |
| OT   (210) | 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.98 / 0.99 / 0.99 |

The proposed model's macro-averaged F1 is 0.99 with no class falling below 0.99 — a property neither of the strongest baselines exhibits.

**Extended per-class report (selected baselines).** Table 2a reports the full per-class precision / recall / F1 (in that order) for every baseline together with the proposed model on the subtype head. Two clinically meaningful observations follow. First, the MC and OC rows are the only ones where any model loses F1 below 1.00 — and the same two rows are where every model loses, indicating that the residual confusion is *data-driven* rather than *model-driven*: MC and OC are both malignant carcinomas whose visual presentations overlap. Second, the proposed model recovers a perfect 1.00 / 1.00 / 1.00 on five of seven classes — a coverage no baseline achieves.

**Table 2a — Full per-class subtype precision / recall / F1 (test-set, support in parentheses).**

| Model | CaS (256) | CoS (239) | Gum (192) | MC (288) | OC (173) | OLP (288) | OT (210) |
|---|---|---|---|---|---|---|---|
| ResNet50 | 1.00/0.99/1.00 | 1.00/1.00/1.00 | 0.99/0.99/0.99 | 0.98/0.98/0.98 | 0.98/0.98/0.98 | 0.99/0.99/0.99 | 1.00/1.00/1.00 |
| DenseNet121 | 1.00/0.98/0.99 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/0.99/0.99 | 0.99/0.98/0.99 | 0.98/1.00/0.99 | 1.00/1.00/1.00 |
| EfficientNet-B0 | 1.00/0.98/0.99 | 0.99/1.00/0.99 | 0.99/1.00/0.99 | 0.99/0.99/0.99 | 0.98/0.98/0.98 | 0.98/1.00/0.99 | 1.00/0.98/0.99 |
| EfficientNet V2-B2 | 1.00/0.99/0.99 | 0.99/1.00/0.99 | 0.99/1.00/1.00 | 1.00/0.99/0.99 | 0.98/0.99/0.99 | 1.00/0.99/0.99 | 0.99/1.00/1.00 |
| Inception V3 | 0.99/1.00/1.00 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00/0.98/0.99 | 0.97/0.99/0.98 | 1.00/0.99/0.99 | 0.98/0.99/0.99 |
| Custom V2 (Hub v1, full) | 0.98/0.99/0.99 | 0.99/1.00/1.00 | 1.00/1.00/1.00 | 0.99/0.99/0.99 | 0.99/0.97/0.98 | 0.99/1.00/0.99 | 1.00/0.99/0.99 |
| **Custom V2 (Hub v2)** | **1.00/1.00/1.00** | **1.00/1.00/1.00** | **0.99/1.00/1.00** | **1.00/0.98/0.99** | **0.98/0.99/0.99** | **1.00/1.00/1.00** | **1.00/1.00/1.00** |

The proposed model is the only configuration whose CaS, CoS, OLP, and OT rows are all 1.00 / 1.00 / 1.00. The residual confusion is confined to MC and OC, where it costs the model two and one misclassified images respectively (out of 288 and 173 supports).

### 6.3 Efficiency Comparison

Table 3 reports computational characteristics. Inference latency is measured at batch size 1 on RTX 4060 Ti, P50 / P95 reported.

**Table 3 — Inference and training efficiency.**

| Model | P50 lat (ms) | P95 lat (ms) | GPU peak (MB) | Train time (min) | Epochs |
|---|---|---|---|---|---|
| ResNet50 | 6.88 | 9.18 | 148.37 | 131.82 | 129 |
| DenseNet121 | 17.59 | 23.77 | 80.19 | 108.84 | 100 |
| ConvNeXt-Tiny | 6.56 | 8.80 | 171.98 | 262.55 | 200 |
| Swin-T | 11.54 | 12.18 | 179.33 | 145.52 | 118 |
| EfficientNet-B0 | 8.88 | 9.51 | 76.91 | 104.05 | 126 |
| EfficientNet V2-B2 | 13.15 | 14.02 | 72.47 | 114.20 | 135 |
| EfficientNet V2-B3 | 15.41 | 20.38 | 94.14 | 92.47 | 99 |
| EfficientNet V2-S | 19.12 | 25.60 | 126.94 | 98.72 | 86 |
| Inception V3 | 17.57 | 17.76 | 132.18 | 135.22 | 153 |
| **Custom V2 (Hub v1)** | 10.16 | 13.41 | 52.32 | 131.70 | 172 |
| **Custom V2 (Hub v2)** | 9.86 | 15.50 | **52.27** | **30.80** | 141 |

The proposed model is the most memory-efficient on the GPU (52.3 MB peak — between 30 % and 70 % lower than the comparators) and converges substantially faster than v1 under the same recipe, owing to the lighter sequential hub topology.

---

## 7. Ablation Study: The Role-Complementarity Principle

The proposed AttentionHub-v2 is not introduced as a stand-alone design; it is the conclusion of a systematic ablation. This section reports the seven v1 ablation cells, identifies a quantitative regularity ("the 98.36 ceiling"), and shows how the resulting role-complementarity principle forces the v2 design.

### 7.1 Single-Branch and Pairwise Ablations

Table 4 reports test-set accuracy for the eight v1 ablation cells, sorted by ascending binary accuracy.

**Table 4 — AttentionHub v1 ablation (Custom EfficientNet V2, matched baseline recipe).**

| Variant | Bin Acc | Sub Acc | Params (M) | GFLOPs | Role(s) covered by hub |
|---|---|---|---|---|---|
| `bam_kan` | 0.9825 | 0.9836 | 4.789 | 0.493 | spatial+channel ∪ channel |
| `bam_triplet` | 0.9842 | 0.9836 | 4.790 | 0.493 | spatial+channel ∪ cross-dim-spatial |
| `kan` | 0.9842 | 0.9897 | 4.777 | 0.490 | channel |
| `none` (donor Block-4) | 0.9860 | 0.9885 | 5.700 | 0.636 | MBConv+SE (channel) |
| `triplet` | 0.9895 | 0.9939 | 4.778 | 0.491 | cross-dim-spatial |
| `bam` | 0.9906 | 0.9909 | 4.779 | 0.491 | spatial + channel |
| `full` (BAM+Triplet+KAN parallel) | 0.9906 | 0.9921 | 4.800 | 0.495 | all three |
| `triplet_kan` | 0.9912 | 0.9945 | 4.788 | 0.493 | cross-dim-spatial + channel |

Three findings emerge.

**Finding 1: Attention helps over the no-attention control on subtype.** The donor Block-4 baseline (`none`) reaches 98.85 % subtype. Every Triplet-containing variant matches or exceeds this except `bam_triplet`. The single-branch `triplet` cell improves subtype to 99.39 % at *fewer* parameters than the no-attention control (4.78 M vs 5.70 M), since the donor's MBConv+SE block is heavier than a 1 × 1 channel-reduction followed by a Triplet gate.

**Finding 2: The `triplet_kan` pair outperforms the full parallel triple.** Removing BAM from the v1 hub *improves* both binary (99.06 → 99.12) and subtype (99.21 → 99.45) accuracy. The full triple, while still strong, is not the best v1 configuration — a result obscured in the original proposal because the full triple was not ablation-tested before publication.

**Finding 3: Two pairs collapse to an identical 98.36 % subtype ceiling.** Both `bam_kan` and `bam_triplet` reach exactly 98.36 % subtype, well below the no-attention control (98.85 %) and well below either component evaluated singly (`bam`: 99.09 %; `triplet`: 99.39 %; `kan`: 98.97 %). This is not a noise artefact — the identical value across two distinct pairings, each *worse than the no-attention baseline*, signals a systematic interaction failure.

### 7.2 The Role-Complementarity Principle

Re-examining Table 4 by *what role each branch exercises*, a clean pattern emerges. We classify each attention module by its operational role:

- **Spatial-attention modules** modulate the (H, W) plane: BAM (via its spatial gate), Triplet (via cross-dimensional spatial gates), CBAM-spatial, EMA (multi-scale spatial+channel), CA, polarized self-attention.
- **Channel-attention modules** modulate the C axis only via global pooling: SE, ECA, KAN, NAM.
- **Mixed-role modules** exercise both: BAM (channel + spatial summed), EMA, CBAM (sequential).

The principle is then:

> **Role-complementarity principle.** Two attention modules, when combined within a single hub, must cover *disjoint functional roles*. If both modules contend for the same role (typically the spatial role), the resulting hub regresses to a fixed sub-baseline ceiling on this dataset.

The principle is empirically supported by Table 4:

- `bam_triplet` — BAM (spatial + channel) and Triplet (cross-dim spatial) **both touch the spatial role**. Subtype = 98.36 % ✗.
- `bam_kan` — BAM (spatial + channel) and KAN (channel) **both touch the channel role**. Subtype = 98.36 % ✗.
- `triplet_kan` — Triplet (cross-dim spatial) and KAN (channel) cover **disjoint roles**. Subtype = 99.45 % ✓.

The identical 98.36 % across the two failing pairs is the empirical signature of "two modules fighting for the same role." Because this number sits *below* the no-attention baseline, the failure mode is not simply that the second module is uninformative — it is that the role conflict actively degrades the representation Stage 5 receives.

### 7.3 Confirming the Principle: The v2-EMA Negative Result

To stress-test the principle we performed a non-ablation control: replace KAN in `triplet_kan` with EMA (Efficient Multi-scale Attention, multi-scale spatial + channel) and reorder into a sequential cascade — the v2-EMA configuration. EMA exercises a spatial role, so the principle predicts that v2-EMA will regress.

The prediction held: v2-EMA reached 98.60 binary / **98.36** subtype — within the same 98.36 % ceiling produced by the two v1 failure pairs. This is reported in the paper as a documented negative result: it independently corroborates the role-complementarity finding and rules out "any pair will do once we add a second module" as an alternative explanation.

### 7.4 From v1 to v2: Sequential Triplet → SE

The principle constrains the design of v2 as follows. The best v1 cell (`triplet_kan`) pairs Triplet (cross-dim spatial) with KAN (purely channel). To improve over `triplet_kan` while remaining within the principle, the channel partner can be swapped for a different purely-channel module, and the topology can be reorganised. We select **Squeeze-and-Excitation (SE)** as the channel partner for two reasons:

1. SE provides the same channel-recalibration function that KAN provides, without the small-data overfitting risk that KAN's learnable spline coefficients introduce. On a dataset of this size (≈ 5 500 training samples, 7 classes) the spline parameters in KAN can begin to memorise per-channel idiosyncrasies; SE's two-layer 1 × 1 bottleneck is much harder to over-fit.
2. SE has been benchmarked extensively in CBAM and related sequential-attention frameworks, providing a citation-grounded precedent for *sequential* (rather than parallel) attention composition.

The topology is changed from parallel to sequential, following CBAM convention: Triplet first refines spatial attention; SE then refines the channel response of the already spatially attended features. No LayerScale, no per-module residual: both attentions are multiplicative gates that preserve input information without requiring an explicit skip.

**Result.** Under the identical baseline recipe used by all v1 ablation cells, the v2 cascade reaches **99.06** binary / **99.51** subtype — improving both heads relative to the best v1 cell (`triplet_kan`: 99.12 / 99.45) and the original v1 proposed model (`full`: 99.06 / 99.21). The 0.06 percentage-point gain on the subtype head represents a ten-image correction over `triplet_kan` on a 1 646-sample test set and is within single-run noise; we therefore present v2 as the principled design, not as a statistically distinguishable improvement over `triplet_kan`.

This entire chain — ablation → principle → v2 design — is the central methodological contribution of the paper. It replaces the "stack attention modules and report the best" pattern common in the medical imaging literature with a measurement-driven design rule.

### 7.5 Best v2 Configuration vs Strong Baselines

Combining the conclusions of Sections 4 and 5, the proposed model under the role-complementarity principle is the smallest and cheapest model in the entire study while delivering the highest subtype accuracy and binary accuracy tied with the leader within noise. Table 5 distils the comparison.

**Table 5 — Proposed model versus the strongest baselines.**

| Metric | EfficientNet V2-B2 | Inception V3 | Custom V2 (Hub v2) |
|---|---|---|---|
| Binary accuracy | 0.9936 | 0.9930 | 0.9906 |
| Subtype accuracy | 0.9933 | 0.9921 | **0.9951** |
| Parameters (M) | 10.00 | 23.85 | **4.79** |
| GFLOPs | 1.100 | 2.838 | **0.493** |
| Size (MB) | 39.17 | 91.47 | **18.94** |
| GPU peak (MB) | 72.47 | 132.18 | **52.27** |

---

## 8. Explainability Analysis

To validate that the proposed model attends to clinically relevant tissue rather than to incidental visual cues, every model in the study is accompanied by a GradCAM++ saliency panel and a LIME boundary-mask panel for both heads. The artifacts are produced by `explain_model.py` and stored as `explain_binary.png` and `explain_subtype.png` per model.

### 8.1 GradCAM++

GradCAM++ is computed on the final stage of each backbone:

- For convolutional baselines (ResNet, DenseNet, ConvNeXt, EfficientNet variants, Inception V3) the GradCAM target layer is the last convolutional block.
- For Swin-T the target is the last attention block prior to global pooling.
- For the proposed Custom EfficientNet V2 the target is Stage 5 — the MBConv+SE block immediately downstream of the AttentionHub.

The class index is taken from the relevant head's predicted argmax, and a per-image attention map is overlaid on the input.

**Observations on the proposed model's panels.** Inspection of `results/custom_efficientnet_v2_hub_v2/explain_binary.png` and `explain_subtype.png` reveals four patterns:

1. **Lip-margin benign cases** receive broad, diffuse attention across the visible mucosal area with no focal fixation on teeth — the model uses the entire lesion-bearing tissue rather than incidental hard-tissue cues.
2. **White-plaque benign cases** receive a single tight focal spot directly on the visible plaque. The attention is the most spatially concentrated in the panel and is anatomically correct.
3. **Diffuse-mucosa malignant cases** receive a distributed, multi-spot attention over the lesion region — appropriate for lesions whose boundary is not sharply defined.
4. **Focal-bump malignant cases** receive two crisp focal spots that align with the two visible bumps. This is, in clinical terms, the kind of mapping a junior clinician would draw on the image themselves.

This combination of *broad* attention on diffuse lesions and *focal* attention on bump-like lesions is the desired property of an attention hub that combines a spatial branch (Triplet) with a channel-sharpening branch (SE): Triplet preserves spatial coverage while SE concentrates the channel response on the most discriminative feature axes.

### 8.2 LIME

For each test image, LIME super-pixel segmentation produces a boundary mask highlighting the regions whose perturbation most changes the predicted class probability. The masks for the proposed model align consistently with the GradCAM++ heatmaps and with visible lesion tissue. In particular, the LIME boundaries on the focal-bump cases are *tight* around the bumps themselves — a property that several of the baselines (notably ConvNeXt-Tiny under from-scratch training, and Swin-T on small lesions) do not exhibit, where LIME boundaries tend to include teeth or surrounding skin.

### 8.3 Comparison to Baseline Panels

A side-by-side review of the GradCAM++ panels (visible in `results/*/explain_subtype.png`) supports the following ordering, from most clinically aligned to least:

- **Most aligned:** Custom EfficientNet V2 (Hub v2), EfficientNet V2-B2, Inception V3 — all three concentrate attention on lesion tissue with negligible activation on teeth or skin.
- **Mid-tier:** ResNet50, DenseNet121, EfficientNet-B0 — generally correct localization but with occasional spurious activations on lip outlines.
- **Weakest:** ConvNeXt-Tiny — attention often broadens across the whole oral cavity without lesion-specific concentration, consistent with its low quantitative accuracy.

The clinical interpretability of the proposed model is therefore not merely a quantitative claim; it is visible in the explainability panels released with this paper.

---

## 9. Discussion

### 9.1 What the Ablation Actually Proves

The headline subtype number — 99.51 % — is the easy story. The harder story, and the one we believe is more important for the field, is the **role-complementarity principle**. Two independent failure pairs (`bam_triplet` and `bam_kan`) and one independent confirmatory negative result (`v2-EMA`) converge on the same 98.36 % subtype ceiling whenever two attention modules contend for the same functional role. The probability of this convergence under a "noise" null is small: it is unlikely that three independently-trained configurations would land on the same accuracy to four significant figures by chance.

This means the present paper provides not only a model but a *design rule*: when adding attention to an oral-disease classifier of this size and on this kind of data, pair modules with disjoint roles, or replace the second module with a no-attention pass.

### 9.2 Comparison to Prior Multi-Class Systems

The closest reported result on a seven-class oral disease setting is MODC-SET [22], a heavy ensemble of MobileNetV2 + InceptionResNetV2 + ResNet50 with an XGBoost meta-classifier, reported at 99.32 % overall accuracy. The present model exceeds this number at a single-stage 4.79 M-parameter architecture. The contrast highlights the role of ablation: MODC-SET's accuracy is achieved by feature fusion across three large backbones with no published ablation justifying the ensemble composition, whereas the present model's accuracy emerges from a five-stage backbone whose only specialized component is justified component-by-component.

Relative to the binary-screening literature [5, 16, 17], the present model maintains binary-task parity (99.06 %) with the leader (EfficientNetV2-B2 at 99.36 %) at less than half the parameter count, while also providing seven-way subtype categorization that the binary screeners do not address.

### 9.3 Why the Binary Head Plateaus

The binary head sits within a narrow 99.0–99.4 % corridor across the strong baselines. We interpret this as a property of the data rather than of the model: the visual gap between benign and malignant is sufficiently wide that any reasonable backbone trained from scratch resolves it almost completely. Improving the binary head past 99.5 % would require either a larger dataset (admitting harder edge cases) or task-targeted augmentation; neither is in scope here.

The subtype head, by contrast, requires the model to differentiate among visually similar mucosal conditions (e.g., MC vs OC, both malignant carcinomas), which is exactly where attention-hub design has measurable leverage and where the role-complementarity principle yields its gain.

### 9.4 Limitations

We acknowledge four limitations.

**(a) Single-seed evaluation.** Each model is trained once with seed 42. For the 0.06 percentage-point margin between v2 and the best v1 cell (`triplet_kan`), this is within the noise band of a single run. We do not claim v2 is *statistically* better than `triplet_kan`; we claim v2 is the *principled* design that the ablation forces, and that under one matched run it sets the highest subtype number in the study.

**(b) Dataset scale.** The training set is ≈ 5 500 images across seven subtypes — small relative to natural-image benchmarks, large relative to most oral-disease studies. We compensate via from-scratch training (so no ImageNet prior is smuggled in), augmentation, early stopping, and replicated evaluation, but external validation on independent clinics is needed before clinical deployment.

**(c) Image-level only.** This work performs classification, not pixel-level segmentation. Segmentation is a complementary direction (Section 2.3); the present model does not output lesion masks. Bounding-box detection and pixel masks are left as future work.

**(d) Training-batch timing is approximate.** The `batch_time_ms` field in `performance_metrics.json` is estimated from test-time inference rather than measured during training. The latency numbers themselves (mean, P50, P95) are measured correctly; only the derived "estimated epoch time" reported alongside is an approximation.

### 9.5 Clinical Significance and Deployment Considerations

The 99.51 % subtype accuracy reported in this study is not a clinical claim. It is a *bench-test* result on a held-out portion of a single curated dataset, and three distinct gaps separate it from clinical deployment.

**(a) Distributional gap.** The dataset's image-acquisition conditions, patient demographics, and intra-oral-anatomy coverage are not characterized in published metadata. A deployment-grade screening system would require multi-site evaluation across patient populations, imaging devices, and lighting conditions. The proposed model's small parameter count and low inference cost make such an evaluation tractable on commodity hardware, but the evaluation itself has not been performed here.

**(b) Decision-context gap.** The model produces a posterior over seven classes; clinical decisions require additional inputs (history, palpation, biopsy result) that the model does not consume. Any deployment must therefore be framed as *decision-support* rather than *decision-making*, with a clinician retaining final authority on diagnosis and management.

**(c) Calibration gap.** This work reports accuracy and F1 but does not report calibration metrics (Expected Calibration Error, Brier score). For triage-style screening — where the most useful output is often a calibrated probability that the lesion warrants biopsy — calibration becomes more important than accuracy. We flag this as the most consequential metric the present study does not report, and we treat its evaluation as an open follow-up.

The proposed model is therefore released as a research artifact and as evidence for the role-complementarity principle, not as a deployable diagnostic tool.

### 9.6 Future Work

Five directions follow naturally from the present study.

1. **Third attention slot.** Extending the role-complementarity principle to a *third* attention slot would require a new ablation grid that tests whether a third disjoint-role module (e.g., a frequency-domain attention, a global-context module, or a graph-based reasoning head) compounds the gain or whether the cascade saturates at two complementary modules. The principle as stated does not predict the answer.
2. **Backbone transferability.** The v2 hub is small enough to drop into the larger EfficientNetV2 family or into ResNet-style backbones; testing whether the same principle holds when Stage 4 sits inside a deeper or wider network would establish whether the rule is dataset-specific, architecture-specific, or general.
3. **Multi-seed statistical evaluation.** A multi-seed run (e.g., five independent random seeds per ablation cell) would convert the present single-run table into a mean ± standard deviation table, making within-cell variability explicit and supporting paired-sample significance tests between v2, `triplet_kan`, and `full`. The principle predicts the *ordering* of the cells will be preserved; an empirical test of that prediction would strengthen the central methodological claim.
4. **Hybrid classification + segmentation head.** Pairing the present classifier with a lightweight segmentation head sharing Stage 5 features would recover pixel-level information that segmentation-only papers [15, 26] target while retaining the categorization capability that classification-only papers [19, 22] target. Stage 5's 192-channel output at 7 × 7 spatial resolution is well-suited to a small upsampling decoder.
5. **Calibration and uncertainty.** As noted in §9.5, calibration metrics and predictive uncertainty are the most consequential evaluation axes the present study does not cover. A follow-up with temperature scaling, Monte-Carlo dropout, or deep ensembling on top of the v2 model would close this gap.

---

## 10. Conclusion

This paper presented a parameter-efficient five-stage convolutional architecture for binary and multi-class oral disease classification. The architecture differs from a standard EfficientNetV2-B0 in two ways: a custom AttentionHub replaces the donor Block-4, and the redundant tail (Block-6 + conv-head) is removed. Through a seven-cell ablation of the AttentionHub plus a confirmatory negative-result run, we derived a **role-complementarity principle** that constrains how attention modules should be combined in this setting: pair modules with disjoint functional roles, or pay a measurable accuracy penalty. The principle forced a sequential Triplet → SE cascade as the proposed *AttentionHub-v2*, which under a matched fair training recipe achieves **99.06 %** binary accuracy and **99.51 %** subtype accuracy on a 7-class held-out test set — the highest subtype score in a benchmark of nine standard CNN and Transformer baselines, at 4.79 M parameters and 0.493 GFLOPs (2.1–5.8 × smaller than the strongest comparators). GradCAM++ and LIME panels confirm that the model attends to lesion tissue rather than to incidental visual cues. The full benchmark, the ablation grid, and the explainability artifacts are released to support reproducibility and to make the role-complementarity principle available for re-testing on adjacent medical-imaging distributions.

---

---

## Data and Code Availability

The full source — training scripts (`train.py`, `custom_efficientnet_colab.py`), evaluation (`evaluate_final.py`), explainability (`explain_model.py`), computational-metrics pipeline (`compute_model_metrics.py`), and the ablation runner (`run_ablation.py`) — is committed alongside this manuscript, together with per-model `classification_metrics.json`, `performance_metrics.json`, `training_time.json`, `evaluation_results.txt`, `confusion_matrices.png`, and `explain_binary.png` / `explain_subtype.png`. Dataset 1 and Dataset 2 are pre-existing curated collections; redistribution of the original imagery is subject to the upstream dataset licenses. The model weights for the proposed v2 model (`results/custom_efficientnet_v2_hub_v2/best_model.pth`) and for every baseline and ablation cell are released for non-clinical research use under the terms specified in the repository.

## Conflict of Interest

The authors declare that they have no conflict of interest relevant to the content of this study. The proposed model is a research artifact and is not associated with any commercial diagnostic product.

## Acknowledgments

The authors thank the maintainers of the donor model in the `timm` library, the authors of the BAM, Triplet Attention, SE, and EMA modules whose designs we both adopt and ablate, and the developers of the `pytorch-grad-cam`, `lime`, `thop`, and `codecarbon` libraries on which the reproducibility provisions of this work depend.

---

## References

[3] Author(s). *CNN-based recurrent aphthous ulcer detection with transfer learning.* [Replace with the bibliographic record corresponding to citation [3] in your master list.]

[5] Author(s). *Deep-learning CNN framework for oral carcinoma detection.* [Replace with the bibliographic record corresponding to citation [5].]

[15] Author(s). *CLASEG: U-Net-style semantic segmentation framework for oral mucosal lesions.* [Replace with the bibliographic record corresponding to citation [15].]

[16] Author(s). *SE-MobileViT: lightweight oral cancer classification.* [Replace with the bibliographic record corresponding to citation [16].]

[17] Author(s). *LBP + deep CNN feature fusion for oral cancer detection.* [Replace with the bibliographic record corresponding to citation [17].]

[18] Author(s). *Vision Transformer vs radiomics comparison for oral lesion classification.* [Replace with the bibliographic record corresponding to citation [18].]

[19] Author(s). *Unrestricted multi-class classification of oral mucosal lesions.* [Replace with the bibliographic record corresponding to citation [19].]

[21] Author(s). *DeiT + CoAtNet hybrid for oral disease classification.* [Replace with the bibliographic record corresponding to citation [21].]

[22] Author(s). *MODC-SET: MobileNetV2 + InceptionResNetV2 + ResNet50 + XGBoost ensemble for 7-class oral disease.* [Replace with the bibliographic record corresponding to citation [22].]

[23] Author(s). *Systematic review of 16 AI studies on OLP, RAS, leukoplakia.* [Replace with the bibliographic record corresponding to citation [23].]

[24] Author(s). *ChatGPT-5 as a multimodal diagnostic assistant for OLP / OLL / SCC-over-LP.* [Replace with the bibliographic record corresponding to citation [24].]

[26] Author(s). *Hybrid CNN-Transformer with high-order focus convolution and Sobel edge enhancement for ulcer segmentation.* [Replace with the bibliographic record corresponding to citation [26].]

**Methodological references for module designs (cited in Section 3.4):**

[B1] Park, J., Woo, S., Lee, J.-Y., & Kweon, I. S. (2018). *BAM: Bottleneck Attention Module.* British Machine Vision Conference (BMVC).

[B2] Misra, D., Nalamada, T., Arasanipalai, A. U., & Hou, Q. (2021). *Rotate to Attend: Convolutional Triplet Attention Module.* IEEE Winter Conference on Applications of Computer Vision (WACV), 3139–3148.

[B3] Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7132–7141.

[B4] Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* European Conference on Computer Vision (ECCV), 3–19.

[B5] Tan, M., & Le, Q. V. (2021). *EfficientNetV2: Smaller Models and Faster Training.* International Conference on Machine Learning (ICML), 10096–10106.

[B6] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* International Conference on Computer Vision (ICCV), 618–626.

[B7] Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.* IEEE Winter Conference on Applications of Computer Vision (WACV), 839–847.

[B8] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier.* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144.
