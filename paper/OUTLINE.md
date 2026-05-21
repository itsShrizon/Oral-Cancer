# Paper Outline — IEEE Journal Manuscript (execution guide)

**Title:** A Parameter-Efficient Dual-Head Oral Disease Classifier with an
Ablation-Driven Triplet→SE Attention Cascade
**Class:** `\documentclass[journal,a4paper]{IEEEtran}` · two-column · ~14–18 pp.
**Content/number source of truth:** `docs/PAPER.md` (relative to repo root:
`../docs/PAPER.md`). This file governs *structure, figures, labels, conventions*.

---

## Global conventions (ALL writer agents must follow)

- **One file per section** in `sections/`, already `\input` by `main.tex` in order.
  Do not touch `main.tex`'s preamble; the abstract block inside `main.tex` *is*
  edited by the writer.
- **Numbers are sacred.** Every accuracy, F1, parameter count, GFLOPs, latency,
  epoch count, support count is transcribed *verbatim* from `docs/PAPER.md` /
  its tables. Never round, never invent, never "improve" a number. Preserve every
  hedge ("within single-run noise", "we do not claim…").
- **Substantive rewrite is allowed** for prose, paragraph order, transitions, and
  framing — but the scientific claims, the contribution list, the role-
  complementarity principle, and all quantitative results stay exactly as in
  `PAPER.md`.
- **Citations:** use `\cite{key}` with keys defined in `paper.bib` (see its header
  comment block for the key↔description map). Do not invent keys.
- **Cross-references:** `\label`/`\ref` for every figure, table, section, equation.
  Label scheme below. Refer to floats as "Fig.~\ref{...}" and "Table~\ref{...}"
  (IEEE style, non-breaking space).
- **Acronyms:** define on first use — CNN, GradCAM++, LIME, SE, BAM, KAN, EMA,
  CBAM, MBConv, DS1/DS2, etc.
- **Tone:** formal IEEE academic prose; topic sentence per paragraph; quantified
  claims. Follow `.agents/skills/research-paper-writer/references/writing_style_guide.md`.
- Figures live in `figures/` (already set via `\graphicspath`); include WITHOUT
  extension, e.g. `\includegraphics[width=\columnwidth]{fig03_pareto_params}`.
- Wide floats (span both columns) use starred envs `figure*` / `table*` placed `[!t]`.

### Section labels
`sec:intro` `sec:related` `sec:prelim` `sec:methods` `sec:setup` `sec:results`
`sec:ablation` `sec:xai` `sec:discussion` `sec:conclusion`

---

## Figure inventory (15 figures — all in `paper/figures/`)

| Order | File (no ext) | `\label` | Shows | Section | Width |
|--|--|--|--|--|--|
| 1 | `fig02_dataset_sample_grid` | `fig:dataset` | sample images, 2 binary + 7 subtype classes | IV-A | `figure*` |
| 2 | `fig02b_class_distribution` | `fig:classdist` | per-class train/val/test counts | IV-A | column |
| 3 | `fig01_custom_efficientnet_v2_arch` | `fig:arch` | 5-stage Custom EfficientNet V2 + dual head | IV-C | `figure*` |
| 4 | `fig01d_attention_modules_internals` | `fig:modules` | BAM / Triplet / KAN internals | IV-D | `figure*` |
| 5 | `fig01b_attentionhub_v2_detail` | `fig:hubv2` | Triplet→SE sequential cascade | IV-D | column |
| 6 | `fig01c_attentionhub_v1_vs_v2` | `fig:v1v2` | v1 parallel vs v2 sequential | IV-D | `figure*` |
| 7 | `fig03_pareto_params` | `fig:pareto` | subtype acc vs params Pareto frontier | VI-A | column |
| 8 | `fig03d_proposed_vs_baselines_radar` | `fig:radar` | proposed vs top baselines radar | VI-A | column |
| 9 | `fig04_per_class_f1_heatmap` | `fig:heatmap` | per-class F1, 10 models × 7 classes | VI-B | `figure*` |
| 10 | `fig04c_confusion_matrices_proposed` | `fig:confusion` | binary + subtype confusion matrices | VI-B | column |
| 11 | `fig05_ablation_bars_with_ceiling` | `fig:ablbars` | 8 ablation cells + 98.36 % ceiling | VII-A | column |
| 12 | `fig05b_role_complementarity_matrix` | `fig:rolematrix` | role-pair → outcome matrix | VII-B | column |
| 13 | `fig05d_v1_to_v2_progression` | `fig:progression` | v1 → principle → v2 design path | VII-D | column |
| 14 | `fig06c_proposed_explain_panel_landscape` | `fig:xai_proposed` | GradCAM++ & LIME, proposed model | VIII | `figure*` |
| 15 | `fig06_gradcam_cross_model_composite` | `fig:xai_cross` | GradCAM++ across top-3 models | VIII-C | `figure*` |

Writers should `Read` each figure they place to caption it accurately.

## Table inventory (transcribe exact values from `PAPER.md`)

| `\label` | Content | PAPER.md | Width |
|--|--|--|--|
| `tab:benchmark` | Table 1 — 10-model test classification (Bin/Sub Acc+F1, Params, GFLOPs, Size) | §6.1 | `table*` |
| `tab:perclass` | Table 2 — per-class P/R/F1, proposed + 2 baselines | §6.2 | column |
| `tab:perclass_full` | Table 2a — full per-class P/R/F1, all models | §6.2 | `table*` |
| `tab:efficiency` | Table 3 — latency P50/P95, GPU peak, train time, epochs | §6.3 | column |
| `tab:ablation` | Table 4 — 8 v1 ablation cells (Bin/Sub Acc, Params, GFLOPs, roles) | §7.1 | column |
| `tab:proposed_vs_base` | Table 5 — proposed vs EffV2-B2 vs Inception V3 | §7.5 | column |
| `tab:ablkeys` | ablation-key table (8 keys → branches → role) | §4.8 | column |
| `alg:train` | Algorithm 1 — joint masked multi-task training step | §4.5 | column float, framed `\tt` listing (no algorithm package) |

Numeric table columns: use the `d{N}` column type (defined in `main.tex`) for
decimal-point alignment, e.g. `d{4}` for 4-decimal accuracies. Rules: `\hline`
only (no booktabs). Bold the proposed-model row and best-in-column values.

---

## Section-by-section

### Abstract (in `main.tex`) + Index Terms
Single paragraph, ~220–260 words. Arc: oral CAD matured on binary screening +
segmentation → multi-class subtyping under-explored, attention stacked without
ablation → propose Custom EfficientNet V2 + AttentionHub → 7-cell ablation yields
the role-complementarity principle → v2 = Triplet→SE cascade → 99.06 % binary /
99.51 % subtype at 4.79 M params / 0.493 GFLOPs, beats 9 baselines, 2.1–5.8×
smaller → GradCAM++/LIME confirm lesion focus → package released. Index Terms
already in `main.tex`.

### I. Introduction — `01_introduction.tex` (source: PAPER.md §1)
Open with `\IEEEPARstart`. Motivation (oral lesions, binary vs multi-class clinical
need). Three gaps: (1) binary formulations dominate, (2) attention stacked without
ablation, (3) explainability sparse. Four numbered contributions (custom backbone;
7-cell ablation → role-complementarity principle; AttentionHub-v2; 9-baseline
matched benchmark + XAI release). Roadmap paragraph. Reference `Fig.~\ref{fig:arch}`.

### II. Related Work — `02_related_work.tex` (source: PAPER.md §2)
Subsections II-A Binary detection · II-B RAU-focused · II-C Segmentation ·
II-D Multi-class · II-E Transformer/hybrid · II-F Positioning. Thematic grouping,
explicit "unlike [X]…" comparisons, gap statements. Cite `ref3, ref5, ref15-ref26`.
NOTE: check `CITATIONS_TODO.md` — where a cited paper's real content differs from
PAPER.md's description (e.g. CLASEG), describe the *real* paper.

### III. Preliminaries & Notation — `03_preliminaries.tex` (source: PAPER.md §3)
Notation: input `x`, backbone `f_θ`, feature `F` (D=192), heads `g_binary`,
`g_subtype`, labels, `y_s=-1` masking. Attention **role taxonomy**: spatial /
channel / mixed — this is the foundation the ablation rests on. Keep tight (~½ col).

### IV. Materials & Methods — `04_methods.tex` (source: PAPER.md §4)
- IV-A Dataset — DS1 binary + DS2 7-class (CaS/CoS/Gum/MC/OC/OLP/OT), the
  Train+Val merge & stratified 60/20/20 re-split, per-class test supports, augment.
  Place `Fig.~\ref{fig:dataset}`, `Fig.~\ref{fig:classdist}`.
- IV-B Dual-Head Multi-Task Classifier — `MultiTaskOralClassifier`, shared backbone
  + two MLP heads, dropout 0.5.
- IV-C Custom EfficientNet V2 Backbone — 5 stages, Stage-4 = AttentionHub, tail
  trimmed; 4.79–4.80 M params / 0.493–0.495 GFLOPs. Place `Fig.~\ref{fig:arch}`.
- IV-D AttentionHub Variants — v1 parallel triple (BAM/Triplet/KAN, describe each),
  v2 sequential Triplet→SE. Place `Fig.~\ref{fig:modules}`, `\ref{fig:hubv2}`,
  `\ref{fig:v1v2}`.
- IV-E Multi-Task Loss — masked CE, `ignore_index=-1`; Algorithm 1 (`alg:train`).
- IV-F Training Protocol — single matched recipe, from scratch (no ImageNet),
  Adam 1e-4, cosine, batch 64, 200 epochs + early stop, seed 42. State the fair-
  comparison rationale.
- IV-G Evaluation Protocol — metrics list, per-class report, confusion matrices,
  computational metrics, XAI artifacts.
- IV-H Ablation Protocol — code-level branch switch; place `tab:ablkeys`.

### V. Experimental Setup — `05_experimental_setup.tex` (source: PAPER.md §5)
V-A Hardware/Software (RTX 4060 Ti, Python 3.11, PyTorch 2.x, timm, thop,
pytorch-grad-cam, lime, codecarbon). V-B Reproducibility (5 provisions). V-C Ethics
(no clinical claim, needs external validation).

### VI. Baseline Benchmark Results — `06_baseline_results.tex` (source: PAPER.md §6)
- VI-A Quantitative comparison — `tab:benchmark`; readings (highest subtype acc;
  smallest model; binary tie within noise; ConvNeXt under-performs). Place
  `Fig.~\ref{fig:pareto}`, `Fig.~\ref{fig:radar}`.
- VI-B Per-class subtype performance — `tab:perclass`, `tab:perclass_full`; MC/OC
  residual confusion is data-driven. Place `Fig.~\ref{fig:heatmap}`,
  `Fig.~\ref{fig:confusion}`.
- VI-C Efficiency — `tab:efficiency`; memory + convergence-speed advantage.

### VII. Ablation Study — `07_ablation.tex` (source: PAPER.md §7)
- VII-A Single/pairwise ablations — `tab:ablation`; 3 findings; the 98.36 % ceiling.
  Place `Fig.~\ref{fig:ablbars}`.
- VII-B The Role-Complementarity Principle — stated as a named principle (blockquote
  or emphasized). Place `Fig.~\ref{fig:rolematrix}`.
- VII-C v2-EMA negative result — confirmatory.
- VII-D From v1 to v2: Triplet→SE — why SE over KAN, sequential topology. Place
  `Fig.~\ref{fig:progression}`.
- VII-E Best v2 vs strong baselines — `tab:proposed_vs_base`.

### VIII. Explainability Analysis — `08_explainability.tex` (source: PAPER.md §8)
VIII-A GradCAM++ (target layers; 4 observed patterns on proposed model). Place
`Fig.~\ref{fig:xai_proposed}`. VIII-B LIME. VIII-C Cross-model comparison —
alignment ranking. Place `Fig.~\ref{fig:xai_cross}`.

### IX. Discussion — `09_discussion.tex` (source: PAPER.md §9)
IX-A what the ablation proves · IX-B vs prior multi-class (MODC-SET) · IX-C binary
plateau · IX-D limitations (4) · IX-E clinical significance (3 gaps) · IX-F future
work (5). Honest, no over-claim.

### X. Conclusion — `10_conclusion.tex` (source: PAPER.md §10)
One tight paragraph: problem, the two architectural changes, ablation → principle →
v2, headline numbers, XAI, release. ~½ column.

### Back matter — `11_backmatter.tex` (source: PAPER.md end)
`\section*{Data and Code Availability}`, `\section*{Acknowledgment}` (IEEE spelling,
singular). Conflict-of-interest may fold into a `\thanks` or a short paragraph.
`\bibliography` is already in `main.tex` — do NOT add it here.
