# Referee Re-Review — Round 3

**Manuscript:** "A Parameter-Efficient Dual-Head Oral Disease Classifier with an Ablation-Driven Triplet→SE Attention Cascade"
**Format:** IEEE journal (IEEEtran, two-column). `paper/build/main.pdf` = 24 pages, compiles cleanly (`build3.log`: "Output written on main.pdf (24 pages").
**Reviewer role:** Senior researcher / journal referee, medical-imaging deep learning.
**Scope:** Verification re-review of the major revision against `REVIEW_round2.md`. Numbers re-checked against the released `classification_metrics.json` ground truth.
**Recommendation:** **Accept with minor revision** — conditional only on author-only actions (placeholder citations, author names/affiliations) plus three small text fixes listed in §3. The Section VII factual error that blocked Round 2 is fully and correctly resolved, and the corrected "spatial-spatial" story is consistent end to end.

---

## 1. Verification Table

JSON ground truth (re-confirmed this round): `bam_triplet` 0.9825/0.9836; `bam_kan` 0.9906/0.9915; `bam` 0.9906/0.9909; `triplet` 0.9895/0.9939; `kan` 0.9842/0.9897; `none` 0.986/0.9885; `triplet_kan` 0.9912/0.9945; `full` (baseline_recipe) 0.9906/0.9921; `hub_v2` 0.9906/0.9951.

| Item | Sev | Status | Evidence |
|---|---|---|---|
| **A1** Table 4 wrong rows | BLOCKER | **RESOLVED** | `07_ablation.tex` Table `tab:ablation` (l.50–57): all 8 rows now match the JSONs exactly — `bam_triplet` 0.9825/0.9836, `bam_kan` 0.9906/0.9915, sorted by ascending binary acc. Verified cell-by-cell. |
| **A2** Finding 3 "two pairs collapse" | BLOCKER | **RESOLVED** | `07_ablation.tex` l.84–97: retitled "pairing two spatial-role modules collapses…"; within-v1 evidence is the single `bam_triplet` cell; `bam_kan` explicitly stated as 99.15 % "comfortably above the control"; ceiling replication forwarded to VII-C (Triplet+EMA). |
| **A3** VII-B principle + bullets misclassify failure | BLOCKER | **RESOLVED** | `07_ablation.tex` l.121–141: principle reworded to spatial-role conflict; three bullets now correct — `bam_triplet` 98.36 % (×), `bam_kan` 99.15 % (✓), `triplet_kan` 99.45 % (✓). "two failing pairs" sentence replaced (l.146–150). |
| **A4** Fig. 5 caption "two failing v1 pairs" | BLOCKER | **RESOLVED** | `fig:ablbars` caption (l.102–107): "the spatial-spatial pair `bam_triplet` lands exactly on it, while every role-complementary configuration — including `bam_kan` — sits above it." Caption matches corrected data. *Author confirmation still required that the PDF asset was re-rendered (see §3, MINOR-1).* |
| **A5** Fig. 12 role-matrix caption two conflict pairs | MAJOR | **RESOLVED** | `fig:rolematrix` caption (l.161–167): only BAM+Triplet highlighted as conflict cell; BAM+KAN described as 99.15 % above control. *Re-render confirmation still required (§3, MINOR-1).* |
| **A6** VII-C v2-EMA wording | MAJOR | **RESOLVED** | `07_ablation.tex` l.173–196: opening reworded ("A single failing cell within the v1 grid (`bam_triplet`) invites two competing explanations…"); closing "reached twice, from two disjoint module sets — BAM+Triplet … and Triplet+EMA". v2-EMA = 98.60/98.36 retained, correct. |
| **B1** IX-A "two independent failure pairs" | BLOCKER | **RESOLVED** | `09_discussion.tex` l.5–23: convergence is now `bam_triplet` + v2-EMA only; probability claim softened to "probability … is low" and the mechanistic argument (below no-attention control) is foregrounded. |
| **B2** IX-A design rule over-general | MAJOR | **RESOLVED** | `09_discussion.tex` l.25–34: rule restated as "do not both exercise the spatial role"; matches corrected VII-B. |
| **B5** IX-B MODC-SET over-claim caveat | MAJOR | **RESOLVED** | `09_discussion.tex` l.41–45: "Because MODC-SET is evaluated on a different seven-class collection, this is not a controlled head-to-head comparison … the 0.19-point delta should not be read as a controlled accuracy advantage." (Caveat present; but ref22 still a placeholder — see D2.) |
| **C2** Abstract "fixed ceiling" over-states | MAJOR | **RESOLVED** | `00_abstract.tex` l.1: now "regresses subtype accuracy to a recurring 98.36 % sub-baseline ceiling" — "recurring" replaces "fixed" as prescribed. |
| **C3** Intro contribution 2 must match corrected body | MAJOR | **RESOLVED** | `01_introduction.tex` l.54–60: "pairing two spatial-role attention modules regresses the hub below its no-attention control … supported by the v1 ablation grid and an independent negative-result run." No "systematically"/"fixed". |
| **D1** II-D conflates ref19 / MODC-SET; ref19 accuracy dropped | MAJOR | **RESOLVED (text)** | `02_related_work.tex` l.100–127: ref19 (Rashid et al.) now reported with "approximately 99.5 % accuracy" and a literature-proximity acknowledgement (l.108–113); MODC-SET clearly separated as a different-dataset ensemble. Citation identity of ref22 still author-pending (D2). |
| **D2** ref22 / MODC-SET unconfirmed placeholder | MAJOR | **PARTIAL** | Text now hedges MODC-SET fully (II-D l.115–127; IX-B l.41–45 carry the not-a-controlled-comparison caveat). But `paper.bib` ref22 still `[PLACEHOLDER — unconfirmed]`. Author must confirm or delete the reference; the manuscript is structured so deletion is now low-cost (no claim depends on the number). |
| **D3** ref3/ref17/ref21 placeholders; ref21 dental-vs-oral | MAJOR | **PARTIAL** | ref21 sentence in II-E (l.137–140) was rewritten to the neutral "image-based disease classification" — no longer asserts "oral disease," so the factual-error risk is removed. ref3 (II-B l.54–60) and ref17 (II-A l.41–45) descriptions softened ("above 85 % accuracy"; "fusion … has also been explored," no number). Residual: all three still `[PLACEHOLDER]` in `paper.bib` — author-only resolution. |
| **E1** III-B example list cites unused modules | MAJOR | **RESOLVED** | `03_preliminaries.tex` l.36–49: examples trimmed to modules actually used (BAM spatial gate, Triplet, EMA branch / SE, KAN, BAM channel gate). ECA, CA, NAM, polarized self-attention, CBAM-sequential removed. |
| **E2** `none` cell 5.70 M contradicts 4.79–4.80 M envelope | MAJOR | **RESOLVED** | `04_methods.tex` IV-C l.133–139: "between 4.79 M and 4.80 M parameters … across the attention-bearing ablation variants; the no-attention control … is the exception at 5.70 M parameters and 0.636 GFLOPs (Table IV)." |
| **E3** Binary test-set size never stated | MAJOR | **RESOLVED (re-scoped)** | `06_baseline_results.tex` now states the binary head is evaluated on the 1,646-image DS2 test set (l.121, l.227–230) and the confusion matrices are "both computed over the 1,646-image DS2 test set so that the denominator is verifiable." 16 errors / 1646 = 99.03 % ≈ 99.06 % — consistent. The earlier "DS1 + DS2-malignant" ambiguity is gone. *(See §3 MINOR-3: IV-G evaluation-protocol bullet still says "DS1 together with the DS2 malignant subset" for the binary head — a residual seam.)* |
| **E6** v2 vs v1 3.5× per-epoch training-time gap | MAJOR | **RESOLVED** | `06_baseline_results.tex` l.316–330: training time now explicitly "indicative rather than … a controlled measurement"; the 3.5× per-epoch gap is named, attributed to shared-workstation background load, "we do not attribute the full 3.5× per-epoch gap to architecture," and no efficiency headline is drawn from it. `05_experimental_setup.tex` l.14–20 adds the CPU/training-time caveat. The "cheapest to train" headline is removed. |
| **F1** XAI purely qualitative; "three-tier ordering" unfalsifiable | MAJOR | **PARTIAL (acceptable)** | No quantitative interpretability metric added (deletion/insertion AUC not computed). However the language is now honestly downgraded: `08_explainability.tex` VIII-C l.124–129 explicitly states the ordering "is a visual judgement … rather than a quantitative interpretability measurement … should therefore be read as indicative"; Abstract/Conclusion "confirm"→"indicate" (see G2). This is the prescribed minimum-honest fix; acceptable for this revision. |
| **F2** Grad-CAM++ at Stage 5 blind to the Stage-4 hub | MAJOR | **RESOLVED (option b)** | `08_explainability.tex` l.59–69: new paragraph — "we make no causal attribution of these Stage-5 patterns to specific hub modules"; the old "Triplet preserves spatial coverage while SE concentrates the channel response" causal claim is gone, replaced by "qualitatively consistent … without being treated as direct evidence." |
| **F3** Figures referenced with thin interpretation | MAJOR | **RESOLVED** | `fig:heatmap` now has the prescribed unique-contribution sentence ("what the heat map adds over Table … is the visual gestalt …", `06_baseline_results.tex` l.221–225). `fig:progression` monotonicity holds with corrected numbers (none 98.85 → full 99.21 → triplet_kan 99.45 → v2 99.51). |
| **G1** "Highest subtype accuracy" qualifier | MAJOR | **RESOLVED** | Literature-proximity acknowledgement added once (II-D l.108–113; IX-B l.54–58). Spot-check of VII-E (l.264–265) and IX: every instance reads "highest subtype accuracy among the nine baselines / in this benchmark." No bare instance found. |
| **G2** "Confirm" too strong for XAI | MAJOR | **RESOLVED** | `00_abstract.tex` l.1 "panels indicate that the network attends to lesion tissue"; `10_conclusion.tex` l.19 "panels indicate … qualitative and is consistent with, rather than a formal verification of." |
| **G5** "2.1×–5.8× smaller than the strongest baselines" | MAJOR | **RESOLVED** | Abstract, Intro, Conclusion all now read "2.1× smaller than EfficientNet V2-B2 and 5.0× smaller than Inception V3, the two strongest baselines." The unsupported 5.8× endpoint is eliminated everywhere (verified by grep — no "5.8" remains in `sections/`). |
| **H1** Intro roadmap clean in .tex | MAJOR | **RESOLVED** | `01_introduction.tex` l.82–96: roadmap is correct (Sections II–X, proper `\ref`); the garbled PAPER.md "Section 3 … Section 3" duplication did not propagate. |
| **H2** "Dual-Head" promised but not isolated | MAJOR | **RESOLVED** | New IX-B paragraph (`09_discussion.tex` l.92–102) states the dual-head design is a "data-utilization mechanism," explicitly disclaims a subtype-accuracy claim from multi-tasking; Limitation (f) added (l.142–148). Limitations list expanded from four to six. |
| **A7** Finding 1 survives correction | MINOR | **RESOLVED** | `07_ablation.tex` l.66–74: Triplet-containing cells `triplet` 99.39, `triplet_kan` 99.45, `full` 99.21 all ≥ 98.85; only `bam_triplet` 98.36 below. Claim true. |
| **A8** VII-B taxonomy duplicates III-B | MINOR | **RESOLVED** | `07_ablation.tex` l.114–117: bullet list replaced by one sentence "Recall the role taxonomy of Section III-B: BAM is a mixed … module …". |
| **A9** Table 4 role abbreviations unexpanded | NIT | **RESOLVED** | `tab:ablation` caption (l.33–36) now defines "sp = spatial, ch = channel, cd-sp = cross-dimensional spatial; ∪ denotes the union of roles." |
| **B3** IX-C "99.0–99.4 % corridor" excludes table models | MINOR | **RESOLVED** | `09_discussion.tex` l.73–75: now "Across the strongest baselines — ResNet50, Inception V3, and EfficientNet V2-B2 — the binary head sits within a narrow 99.1–99.4 % corridor." Scoped correctly. |
| **B4** IX-B repeats MODC-SET composition | MINOR | **RESOLVED** | `09_discussion.tex` l.36–48: IX-B now opens with a one-line recall and pivots to the methodological contrast; no near-verbatim re-description. |
| **C1** Abstract not harmonized down to wrong body | MINOR (watch) | **RESOLVED** | Abstract retains the correct "BAM or EMA" framing and was not degraded; it now also carries the C2 hedge ("recurring"). |
| **C4** "no extra parameters or compute" loose | MINOR | **RESOLVED** | `01_introduction.tex` l.64–66: "at a parameter and compute budget (4.79 M / 0.493 GFLOPs) essentially identical to the v1 hub and well below every baseline." |
| **C5** Abstract sentence length | NIT | **RESOLVED** | `00_abstract.tex`: role-complementarity material is now two sentences ("…regresses … ceiling. Pairing Triplet with a purely channel-wise partner … instead lifts …"). |
| **D4** SE-MobileViT "above 98 %" → 98.39 % | MINOR | **RESOLVED** | `02_related_work.tex` l.36–39: "achieving 98.39 % binary accuracy." |
| **D5** ref17 "approximately 90 %" unverified | MINOR | **RESOLVED** | `02_related_work.tex` l.41–45: softened to "fusion of Local Binary Pattern descriptors with deep CNN features has also been explored" — no number, no robustness claim. |
| **D6** YOLO named but uncited | MINOR | **RESOLVED** | `02_related_work.tex` l.15: "Object-detection frameworks have been adapted for lesion localization" — "YOLO" name removed. |
| **D7** ChatGPT-5 "≈85 %" metric | NIT | **RESOLVED** | `02_related_work.tex` l.141–146: now "finding competitive but sub-expert performance that lagged human experts at the Top-1 differential" — the unverified 85 % figure dropped. |
| **E4** Subtype class glosses clinically dubious | MINOR | **RESOLVED** | `04_methods.tex` IV-A l.18–26: invented expansions removed — "we retain the dataset's class abbreviations verbatim rather than assigning clinical expansions that the source release does not document." Malignant-set incoherence (OC) gone. |
| **E5** baseline `classification_metrics.json` traceability | MINOR | **RESOLVED** | `05_experimental_setup.tex` l.51–58: "Every one of the ten models … emits a `classification_metrics.json` … committed alongside the trained model weights, so that every … number … can be independently regenerated for both the baseline rows and the proposed model." |
| **E7** "Pareto-optimal" wording | MINOR | **RESOLVED** | `06_baseline_results.tex` l.80–83: "it strictly dominates every baseline in the subtype-accuracy versus parameter-count plane — simultaneously the most accurate and the smallest." |
| **E8** CPU SKU not pinned | NIT | **RESOLVED** | `05_experimental_setup.tex` l.14–20: "The exact Ryzen 7 SKU was not recorded; this does not affect the latency and GPU-memory figures … but it is relevant to the wall-clock training time of Table 3." |
| **F4** `fig:dataset` 9×4 grid caption accounting | MINOR | **RESOLVED (text)** | `04_methods.tex` `fig:dataset` caption (l.59–66): "arranged as a 9-row by 4-column grid: one class per row, with four randomly sampled examples per class." Internally consistent; assumes the asset is 9×4 (asset not inspected here). |
| **F5** Table 5 duplicates Tables 1/3 | MINOR | **RESOLVED** | `tab:proposed_vs_base` caption (l.270–272): "Digest of Tables I and III for the proposed model and the two strongest baselines." Signalled as a digest. |
| **F6** "GradCAM++" vs "Grad-CAM++" | NIT | **RESOLVED** | Grep of `sections/` for "GradCAM" (no hyphen): zero matches. All occurrences are "Grad-CAM++". |
| **F7** GFLOPs vs MACs ambiguity | NIT | **RESOLVED** | `04_methods.tex` IV-G footnote (l.380): "Compute is reported as the multiply-accumulate (MAC) count returned by the thop profiler … we use the label GFLOPs for these MAC-count values for consistency with the comparison literature." One convention, stated once. |
| **G4** v2 0.06 pp "within single-run noise" hedge kept | MINOR (watch) | **RESOLVED** | `07_ablation.tex` l.233–237 and `09_discussion.tex` l.108–114: hedge intact, not strengthened. |
| **H3** Limitations should add XAI-qualitative + dual-head gaps | MINOR | **RESOLVED** | `09_discussion.tex` IX-D: limitations (e) interpretability-qualitative and (f) dual-head-not-isolated both added; list now six. |
| **H4** Acronym re-definitions (CNN, MBConv) | MINOR | **PARTIAL** | "CNN" is expanded in the Abstract and re-expanded in IV-C l.106 ("five-stage CNN" — uses the acronym, acceptable) and `03_preliminaries`/`10_conclusion` use "convolutional" unabbreviated — acceptable. "MBConv" expanded in IV-C l.114–115 and the term "MBConv+SE" recurs in VIII-A l.35–37 *with a re-expansion* ("Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation (MBConv+SE)"). Minor residual — see §3 NIT-1. |
| **H8** COI placement | MINOR | **RESOLVED** | `11_backmatter.tex` l.21: COI is now its own `\section*{Conflict of Interest}`, separate from Data/Code Availability. |

**Tally:** 27 BLOCKER/MAJOR items tracked → **22 RESOLVED, 5 PARTIAL, 0 NOT RESOLVED.** All 6 BLOCKERs (A1, A2, A3, A4, B1, and the G3 cluster) are RESOLVED. The 5 PARTIAL items (D2, D3, F1, E3-residual, H4) are either author-only citation actions or minor text seams, none blocking. Of the additional MINOR/NIT items reviewed, all are RESOLVED except H4 (PARTIAL).

---

## 2. Consistency Check — the corrected ablation story, end to end

The four parallel section-group revisions are **consistent**. The corrected "spatial-spatial" narrative is uniform across all six required loci:

| Locus | Statement | Consistent? |
|---|---|---|
| **Abstract** (`00_abstract.tex`) | "pairing Triplet with any module that also exercises a spatial role, namely BAM or EMA, regresses subtype accuracy to a recurring 98.36 % sub-baseline ceiling" | ✓ one v1 pair + EMA; "recurring" not "fixed" |
| **Introduction, contribution 2** (`01_introduction.tex` l.54–60) | "pairing two spatial-role attention modules regresses the hub below its no-attention control … supported by the v1 ablation grid and an independent negative-result run" | ✓ |
| **Section VII Table 4** (`07_ablation.tex` l.50–57) | `bam_triplet` 0.9825/0.9836 (on ceiling); `bam_kan` 0.9906/0.9915 (working) | ✓ matches JSON |
| **Section VII Finding 3** (l.84–97) | one cell (`bam_triplet`) on ceiling; `bam_kan` 99.15 % "comfortably above the control"; replication via VII-C | ✓ |
| **Section VII-B principle** (l.121–141) | spatial-spatial regresses; spatial+channel complementary; bullets correct | ✓ |
| **Section VII-C** (l.173–196) | "reached twice, from two disjoint module sets — BAM+Triplet … and Triplet+EMA" | ✓ |
| **Section IX-A** (`09_discussion.tex` l.5–23) | "`bam_triplet` cell and the independent v2-EMA negative result — two pairings of spatial-role attention modules" | ✓ |
| **Conclusion** (`10_conclusion.tex` l.8–12) | "seven-cell ablation … reinforced by a confirmatory negative-result run … pair modules with disjoint functional roles" | ✓ |

Specific confirmations requested in the brief:
- **Only ONE v1 pair on the ceiling:** ✓ `bam_triplet` (0.9836). `bam_kan` is 0.9915 everywhere it appears. Grep for "two failing pairs" / "two pairs collapse" / "5.8" across `sections/`: **zero matches** — the old story is fully purged.
- **`bam_kan` = 99.15 % subtype is a working configuration:** ✓ stated as such in Table 4, Finding 3, VII-B bullet (✓ mark), `fig:rolematrix` caption, and IX-A.
- **Ceiling reproduced by the v2-EMA negative result:** ✓ VII-C; v2-EMA 98.60/98.36 retained and correctly framed as the *independent* (second) instance.
- **Principle is spatial-spatial:** ✓ uniform; no channel-channel-collision language survives (the old wrong "`bam_kan` … both touch the channel role" bullet is gone).

**Terminology / notation:** consistent. "AttentionHub-v1" / "AttentionHub-v2" / "v2-EMA" used uniformly; "Triplet→SE" rendered via the `\arrowto` macro throughout; "Grad-CAM++" hyphenated everywhere; role abbreviations (sp/ch/cd-sp) defined in the Table 4 caption and used consistently; `tab:ablkeys` (IV) and `tab:ablation` (VII) both label `bam_triplet` "two-spatial-role pair" and `bam_kan` "mixed-role pair" — agreement between the methods table and the results table. The eight-cell count, the 1,646-image DS2 test set, per-class supports (sum 1646), and the 4.79 M/0.493 GFLOPs proposed-model figures are consistent across Sections IV, VI, VII, IX, and the Abstract/Conclusion.

**Seams from parallel authorship:** only one minor residual — see RESIDUAL MINOR-3 (the IV-G evaluation-protocol bullet still describes the binary head as "DS1 together with the DS2 malignant subset," whereas the writer of Section VI re-scoped the binary evaluation to the 1,646-image DS2 test set). No contradictory *numbers*, no duplicated claims beyond what the prescribed fixes already addressed.

---

## 3. Residual Issues

No BLOCKERs and no MAJORs remain. The following are minor and do not block acceptance.

**[MINOR-1] Figure assets — re-render confirmation required.**
Files: `figures/fig05_ablation_bars_with_ceiling.pdf`, `figures/fig05b_role_complementarity_matrix.pdf` (and, for safety, `fig05d_v1_to_v2_progression.pdf`).
The Round-2 prescriptions A4/A5 required these three plotted assets to be **re-rendered** from corrected JSON, not merely re-captioned. The captions are now correct, but a binary referee cannot verify the pixels. If `fig05` still draws the `bam_kan` bar at 98.36 % (old data), it will visibly contradict the corrected Table 4, and `fig05b` would still highlight BAM+KAN as a conflict cell. **Fix:** the authors must confirm in the revision letter that all three figures were regenerated after the JSON correction; if not, regenerate them so the `bam_kan` bar sits at 99.06 % binary / 99.15 % subtype and the role matrix shows BAM+KAN as a passing (non-highlighted) cell.

**[MINOR-2] Placeholder citations ref3, ref17, ref21, ref22 — author-only resolution.**
File: `paper.bib` (four entries still tagged `[PLACEHOLDER]` / `CITATION PENDING`).
The manuscript text has been made *safe* relative to all four (ref21 no longer asserts "oral," ref17 carries no number, MODC-SET/ref22 carries the not-a-controlled-comparison caveat in both II-D and IX-B). But the bib records are unverified. **Fix (author):** confirm each against the master reference list. For ref22 specifically — if MODC-SET cannot be confirmed as a real citable paper, delete the MODC-SET sentence in II-D (l.115–127) and re-anchor IX-B on Rashid et al. (ref19); the revision has deliberately structured the text so this deletion costs nothing. This is the standard author-action carve-out, not a science defect.

**[MINOR-3] Binary-head evaluation scope — residual seam between IV-G and VI.**
Files: `04_methods.tex` IV-G l.366–369 vs `06_baseline_results.tex` l.121, l.227–230.
IV-G's evaluation-protocol bullet still says the binary head is evaluated on "DS1 together with the DS2 malignant subset," but Section VI was revised to state the binary metrics and confusion matrix are computed "over the 1,646-image DS2 test set." These two descriptions of the binary test set are not the same set. The 16-error / 1,646 arithmetic in VI is internally consistent, so VI is the intended description. **Fix:** reconcile IV-G to match VI — change the IV-G bullet to "for both the binary head and the subtype head, evaluated on the 1,646-image DS2 test set," or, if DS1-test images genuinely also feed the binary metric, give that combined count explicitly and make VI's "1,646-image DS2 test set" denominator consistent with it. One sentence; pick whichever matches what the code actually does.

**[NIT-1] "MBConv" re-expanded at first use in Section VIII.**
File: `08_explainability.tex` VIII-A l.35–37.
"MBConv" is already expanded in IV-C (l.114–115). VIII-A re-expands it as "Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation (MBConv+SE)." Drop the re-expansion in VIII-A; use the bare "MBConv+SE block." Cosmetic.

**[NIT-2] `fig:dataset` / `fig:xai_proposed` asset layout not verifiable from text.**
`fig:dataset` caption states a 9×4 grid; `fig:xai_proposed` caption states a 3×2 grid of curated cases. Text is internally consistent; the underlying PDFs were not pixel-inspected in this re-review. Author should glance once to confirm the rendered grids match the captions (Round-2 F4). Trivial.

**[NIT-3] Stale root-level `main.log` (27 pages).**
The root `paper/main.log` is from an earlier 21:23 build and reports 27 pages; the authoritative output `paper/build/main.pdf` (21:58) and `build3.log` both report **24 pages**. Not a manuscript defect — just stale build detritus. Optionally clean the root-level `*.log`/`*.aux`/`build*.log` scratch files before packaging the submission.

---

## 4. Overall Verdict

**The manuscript is acceptable apart from author-only actions.** The single Round-2 blocker — Section VII encoding a factually wrong ablation result (`bam_kan`/`bam_triplet` swapped, "two failing v1 pairs") — is **fully and correctly resolved**: Table 4 now matches the released `classification_metrics.json` ground truth cell-by-cell, and the corrected spatial-spatial-conflict story is consistent across the Abstract, Introduction contribution 2, all of Section VII (Table 4, the three Findings, the principle, VII-B bullets, VII-C), Section IX-A, and the Conclusion. The four parallel section-group revisions did not introduce contradictions; the only seam is the minor IV-G-vs-VI binary-test-set description (MINOR-3). All other Round-2 MAJORs (training-time anomaly, dual-head isolation, XAI calibration, the 5.8× over-claim, the `none`-cell envelope contradiction, the III-B taxonomy) are resolved at the prescribed level — F1 and D2/D3 via the "honest downgrade" / "make-the-text-safe" route the Round-2 prescriptions explicitly permitted.

Nothing in the science blocks the paper. The corrected ablation is, as predicted in Round 2, **cleaner** than the original: one v1 spatial-spatial failure (`bam_triplet`), one independent confirmation (Triplet+EMA), and `bam_kan` as a positive control that the principle correctly predicts. The remaining work is author-only: (1) confirm or delete the four placeholder citations (ref3/ref17/ref21/ref22), (2) supply real author names and affiliations (the `\author` block in `main.tex` is still `First~Author, Second~Author, Third~Author` with `[Department]/[Institution]` placeholders), and (3) confirm in the revision letter that the three ablation figure assets were re-rendered from corrected data (MINOR-1). The three small text fixes in §3 (MINOR-3, NIT-1) are one-line edits.

**Recommendation: Accept with minor revision.**

---

### Summary for the editor

- **RESOLVED: 22 · PARTIAL: 5 · NOT RESOLVED: 0** (of the 27 tracked BLOCKER/MAJOR items; all reviewed MINOR/NIT items resolved except H4, which is PARTIAL).
- **All 6 BLOCKERs resolved.** No BLOCKER or MAJOR item remains open.
- The 5 PARTIAL items are not science defects: **D2, D3** = author-only placeholder-citation confirmation (text already made safe); **F1** = quantitative XAI metric not added but language honestly downgraded as the Round-2 fix permitted; **E3** = resolved in substance, one residual cross-section wording seam (MINOR-3); **H4** = one cosmetic acronym re-expansion (NIT-1).
- **No remaining BLOCKER or MAJOR.** Outstanding items before camera-ready: confirm/delete ref3·ref17·ref21·ref22; insert author names/affiliations; confirm the three ablation figures (`fig05`, `fig05b`, `fig05d`) were re-rendered from corrected JSON; apply the two one-line text fixes (MINOR-3 IV-G binary-test-set wording, NIT-1 MBConv).
