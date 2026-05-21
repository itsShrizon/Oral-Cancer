# Referee Review — Round 2

**Manuscript:** "A Parameter-Efficient Dual-Head Oral Disease Classifier with an Ablation-Driven Triplet→SE Attention Cascade"
**Format:** IEEE journal (IEEEtran, two-column), 27 pp.
**Reviewer role:** Senior researcher / journal referee, medical-imaging deep learning.
**Recommendation:** Major revision. The contribution is genuine and the paper is well written, but Section VII currently encodes a factually wrong ablation result that contradicts the manuscript's own Abstract and the released JSON ground truth. That error propagates into the discussion and must be fixed before the paper can be accepted. Several over-claims, citation-accuracy issues, and consistency defects also need attention.

Severity tags: **[BLOCKER]** wrong/contradictory science or unsupported core claim · **[MAJOR]** weak argument, missing support, over-claim, structural problem · **[MINOR]** polish · **[NIT]** trivial.

---

## A. Section VII — Ablation Study (`sections/07_ablation.tex`)

### A1. [BLOCKER] Table 4 (`tab:ablation`) reports wrong accuracies for `bam_kan` and a wrong binary accuracy for `bam_triplet`

**Location:** `sections/07_ablation.tex`, Table `tab:ablation`, rows `bam_kan` and `bam_triplet`.

**Problem.** The released ground-truth `classification_metrics.json` files give:

| cell | binary | subtype |
|---|---|---|
| `bam_triplet` (`results/custom_efficientnet_v2_ablation_bam_triplet`) | **0.9825** | **0.9836** |
| `bam_kan` (`results/custom_efficientnet_v2_ablation_bam_kan`) | **0.9906** | **0.9915** |

The manuscript's Table 4 currently lists `bam_kan = 0.9825 / 0.9836` and `bam_triplet = 0.9842 / 0.9836`. Both rows are wrong: `bam_kan`'s accuracies belong to `bam_triplet`, `bam_kan`'s true result (0.9906 / 0.9915) is a *working* configuration that is absent from the table, and `bam_triplet`'s binary value of 0.9842 matches no JSON file at all. Only `bam_triplet` actually hits the 98.36 % ceiling; `bam_kan` does not fail. This is the single most important defect in the paper — every downstream claim about "two failing pairs" rests on it.

**Prescribed fix.** Replace the two rows with the verified values. The corrected table (still sorted by ascending binary accuracy) is:

```
\texttt{bam\_triplet}  & 0.9825 & 0.9836 & 4.790 & 0.493 & sp+ch $\cup$ cd-sp \\
\texttt{kan}           & 0.9842 & 0.9897 & 4.777 & 0.490 & channel \\
\texttt{none}          & 0.9860 & 0.9885 & 5.700 & 0.636 & MBConv+SE \\
\texttt{triplet}       & 0.9895 & 0.9939 & 4.778 & 0.491 & cd-spatial \\
\texttt{bam}           & 0.9906 & 0.9909 & 4.779 & 0.491 & spatial+ch \\
\texttt{bam\_kan}      & 0.9906 & 0.9915 & 4.789 & 0.493 & sp+ch $\cup$ ch \\
\texttt{full}          & 0.9906 & 0.9921 & 4.800 & 0.495 & all three \\
\texttt{triplet\_kan}  & 0.9912 & 0.9945 & 4.788 & 0.493 & cd-sp + ch \\
```

(Note `bam`, `bam_kan`, and `full` now tie at 0.9906 binary; under "ascending binary accuracy" any stable order among them is acceptable — placing `bam_kan` between `bam` and `full` keeps the subtype column monotone, which reads cleanly.) The params/GFLOPs columns for both touched rows are already correct and verified against the JSONs (`bam_triplet` 4.790/0.493; `bam_kan` 4.789/0.493).

Also update the table caption: the present caption is generic and survives the fix unchanged, but if you mention the ceiling in the caption, ensure it refers to one v1 pair, not two (see A4).

### A2. [BLOCKER] Section VII-A "Finding 3" claims two v1 pairs collapse; only one does

**Location:** `sections/07_ablation.tex`, "Finding 3" paragraph (begins "Finding 3: two pairs collapse...").

**Problem.** The text states "Both `bam_kan` and `bam_triplet` reach *exactly* 98.36 % subtype accuracy." With the corrected data, `bam_kan` = 99.15 % subtype — *above* the no-attention control (98.85 %) and a working configuration. Only `bam_triplet` lands on the ceiling. The "identical value to four significant figures across two distinct pairings" argument is false as written.

**Prescribed fix.** Rewrite Finding 3 so the within-v1 evidence is one cell (`bam_triplet`), and the *replication* of the ceiling comes from the v2-EMA negative result (Section VII-C), which is exactly how the Abstract already frames it. Suggested replacement text:

> **Finding 3: pairing two spatial-role modules collapses subtype accuracy to a sub-baseline ceiling.** The `bam_triplet` cell reaches only 98.36 % subtype accuracy — well below the no-attention control (98.85 %) and well below either constituent module evaluated singly (`bam`: 99.09 %; `triplet`: 99.39 %). Both BAM and Triplet exercise a spatial role (Section VII-B), and their combination regresses *beneath* the no-attention baseline rather than merely failing to help. By contrast, `bam_kan` — which pairs BAM with the purely channel-wise KAN — reaches 99.15 % subtype, comfortably above the control. The collapse is therefore specific to the spatial-role collision, not a generic consequence of adding a second module. Section VII-C shows that this same 98.36 % ceiling reappears in an independent configuration (Triplet+EMA), confirming that the value is the signature of a spatial-role conflict rather than a `bam_triplet`-specific artefact.

This keeps the strongest part of the original Finding 3 (the sub-baseline collapse, the "actively degrades" reading) while making it true.

### A3. [BLOCKER] Section VII-B — the role-complementarity principle and its bullet evidence misclassify the failure

**Location:** `sections/07_ablation.tex`, Section VII-B, the principle blockquote and the three-bullet "empirically supported, cell by cell" list.

**Problem.** Two errors:

1. The bullet `bam_kan — BAM ... and KAN (channel) both touch the channel role. Subtype accuracy = 98.36 % (×)` is factually wrong twice over: `bam_kan` scores 99.15 %, *not* 98.36 %, and it does *not* fail. Presenting it as a same-role-collision failure directly contradicts the corrected data and the Abstract.
2. The principle as stated ("If both modules contend for the same role — typically the spatial role —") is built to cover a channel-channel collision that did not happen. The real, empirically supported principle is narrower and cleaner: **spatial-spatial** collision is what regresses; a channel-side overlap (BAM+KAN) does *not* regress.

**Prescribed fix.**

Restate the principle so it is about spatial-role conflict, which is what the data actually show:

> **Role-complementarity principle.** *Two attention modules combined within a single hub should occupy disjoint functional roles. When both modules exercise a spatial role, the hub regresses to a fixed sub-baseline accuracy ceiling on this dataset; when one module is spatial and the other purely channel-wise, the pair is complementary and improves over either module alone.*

Note this is exactly the role-complementarity wording used in PAPER.md's Abstract ("pairing Triplet ... with any module that also exercises a spatial role — BAM or EMA — regresses ...").

Then replace the three-bullet evidence list with the *correct* cell-by-cell mapping:

> - `bam_triplet` — BAM (mixed spatial+channel) and Triplet (cross-dimensional spatial) **both exercise a spatial role**. Subtype = 98.36 % (×).
> - `bam_kan` — BAM (mixed spatial+channel) and KAN (purely channel) **do not collide on the spatial axis**; only one module is spatial. Subtype = 99.15 % (✓), above the no-attention control.
> - `triplet_kan` — Triplet (cross-dimensional spatial) and KAN (purely channel) cover **disjoint roles**. Subtype = 99.45 % (✓).

If a third confirmatory row for the failure is wanted inside VII-B, cite the v2-EMA result here as the second spatial-spatial collision (Triplet+EMA, 98.36 %) and forward-reference Section VII-C — do *not* invent a second v1 pair.

Also fix the sentence "The identical 98.36 % across the two failing pairs is the empirical signature of two modules fighting for the same role" — there is one failing v1 pair; the *replication* comes from v2-EMA. Reword: "The 98.36 % ceiling recurs across `bam_triplet` and the independent Triplet+EMA configuration of Section VII-C — two spatial-role collisions reached from different module sets — which is the empirical signature of spatial-role conflict rather than of a single bad cell."

### A4. [BLOCKER] Figure 5 (`fig:ablbars`) caption asserts two v1 pairs land on the ceiling

**Location:** `sections/07_ablation.tex`, caption of `fig:ablbars` ("...the two failing v1 pairs, `bam_triplet` and `bam_kan`, land exactly on it...").

**Problem.** Same factual error as A1–A3. Also a figure-versus-text consistency problem: if the underlying figure `fig05_ablation_bars_with_ceiling.pdf` was rendered from the wrong numbers, the bar for `bam_kan` is drawn at 98.36 % and will not match the corrected Table 4.

**Prescribed fix.** (a) Regenerate `fig05_ablation_bars_with_ceiling.pdf` from the corrected JSON values so the `bam_kan` bar sits at 99.15 % binary 99.06 % / subtype 99.15 %. (b) Rewrite the caption:

> Binary and subtype accuracy across the eight AttentionHub-v1 ablation cells and the proposed v2 cascade. The dashed line marks the 98.36 % role-conflict ceiling: the spatial-spatial pair `bam_triplet` lands exactly on it, while every role-complementary configuration — including `bam_kan` — sits above it.

Confirm in the revision letter that the figure was re-rendered, not just re-captioned.

### A5. [MAJOR] Figure 12 (`fig:rolematrix`) caption and the figure itself reference two highlighted role-conflict pairs

**Location:** `sections/07_ablation.tex`, caption of `fig:rolematrix` ("The two role-conflict pairs --- BAM with Triplet, and BAM with KAN --- are highlighted; both regress to the identical 98.36 % ceiling...").

**Problem.** The role-complementarity matrix figure is built around two highlighted failing off-diagonal cells. With the correction, BAM+KAN is a *success* cell (99.15 %) and must move out of the "conflict" highlight. The figure asset `fig05b_role_complementarity_matrix.pdf` very likely needs to be re-rendered.

**Prescribed fix.** Re-render the matrix so the BAM+Triplet cell is the only highlighted v1 conflict cell and BAM+KAN is shown as a non-conflict (passing) cell. If the matrix has room, add the Triplet+EMA off-grid cell as the second conflict exemplar. Rewrite the caption:

> Role-complementarity matrix. The diagonal reports single-module subtype accuracy and the off-diagonal reports module-pair subtype accuracy. The spatial-spatial pair BAM+Triplet is highlighted as the role-conflict cell, regressing to the 98.36 % ceiling; the disjoint-role pair Triplet+KAN reaches 99.45 %, and the mixed pair BAM+KAN — only one spatial module — reaches 99.15 %, above the no-attention control.

### A6. [MAJOR] Section VII-C v2-EMA negative result — wording must shift from "corroborates one of two" to "the second instance of the ceiling"

**Location:** `sections/07_ablation.tex`, Section VII-C, especially the final sentence ("The identical 98.36 % ceiling across `bam_triplet`, `bam_kan`, and v2-EMA is the empirical signature of role conflict.").

**Problem.** This sentence lists three configurations on the ceiling; only `bam_triplet` and v2-EMA actually are. After the correction, the v2-EMA result is *more* important, not less: it becomes the only independent replication of the ceiling, so the section carries more weight and the wording should reflect that.

**Confirmation requested in the review brief — addressed:** The v2-EMA negative result *still stands*. Triplet+EMA = 98.60 binary / 98.36 subtype is unaffected by the Table 4 correction; EMA is a spatial-role module, so a Triplet+EMA collision is a genuine spatial-spatial pair and the principle (as corrected in A3) predicts exactly this regression. The negative result is consistent and should be retained.

**Prescribed fix.** Replace the final sentence with:

> The 98.36 % ceiling is thus reached twice, from two disjoint module sets — BAM+Triplet within the v1 grid and Triplet+EMA outside it — and in both cases by a pair of spatial-role modules. This convergence is the empirical signature of spatial-role conflict, and it rules out the competing explanation that any second module degrades the hub: `bam_kan` and `triplet_kan`, whose second module is purely channel-wise, both improve on the no-attention control.

Also adjust the opening sentence of VII-C ("A regularity observed within a single ablation grid invites the alternative explanation that any second module degrades the hub once added") — with the correction, the within-grid evidence is a *single* cell, so the motivation for the EMA control is even stronger; reword to "A single failing cell within the v1 grid (`bam_triplet`) invites two competing explanations: that any second module degrades the hub, or that the failure is specific to this one module pair. To separate them we performed a non-ablation control...".

### A7. [MINOR] Section VII-A Finding 1 — verify the "every Triplet-containing variant matches or exceeds" claim survives the correction

**Location:** `sections/07_ablation.tex`, Finding 1 ("Every Triplet-containing variant matches or exceeds this figure with the single exception of `bam_triplet`.").

**Problem.** This sentence happens to remain true after the fix (Triplet-containing cells: `triplet` 99.39, `triplet_kan` 99.45, `full` 99.21, `bam_triplet` 98.36 — only `bam_triplet` is below 98.85). No change needed to the claim itself, but the author should re-verify it explicitly during the revision since the surrounding numbers moved. Flagged so it is not overlooked.

### A8. [MINOR] Section VII-B role-taxonomy bullet list duplicates Section III almost verbatim

**Location:** `sections/07_ablation.tex`, Section VII-B, the three-item "We classify each attention module by its operational role" list; compare `sections/03_preliminaries.tex` Section III-B.

**Problem.** Section III-B already defines spatial / channel / mixed roles with examples. VII-B restates the same taxonomy with a near-identical bullet list (and a slightly different example set — VII-B adds "CA, polarized self-attention" and "NAM" that never recur). In a 27-page paper this is avoidable redundancy and the diverging example lists invite reader confusion.

**Prescribed fix.** Delete the VII-B bullet list and replace with one sentence: "Recall the role taxonomy of Section III-B: BAM is a mixed spatial-and-channel module, Triplet a cross-dimensional spatial module, KAN a purely channel-wise module, and EMA a multi-scale spatial-and-channel module." Move any module not already in III-B (none are essential) out, or add them once to III-B if you want them.

### A9. [NIT] "sub+ch" / "cd-sp" abbreviations in Table 4 role column are never expanded

**Location:** `sections/07_ablation.tex`, Table `tab:ablation`, "Role(s)" column; the caption defines BAM/Triplet/KAN roles in prose but not the symbols.

**Problem.** The column uses `sp+ch`, `cd-sp`, `ch`, and the set-union symbol `∪` with no key. A reader meets `sp+ch $\cup$ cd-sp` before any expansion.

**Prescribed fix.** Add to the caption: "Role abbreviations: sp = spatial, ch = channel, cd-sp = cross-dimensional spatial; `$\cup$` denotes the union of roles the active branches collectively exercise." Or spell the roles out in the column, which `tabularx` has room for in a single-column table only if you shorten elsewhere — the caption key is simpler.

---

## B. Section IX — Discussion (`sections/09_discussion.tex`)

### B1. [BLOCKER] Section IX-A claims "two independent failure pairs (`bam_triplet` and `bam_kan`)"

**Location:** `sections/09_discussion.tex`, Section IX-A "What the Ablation Actually Proves", first paragraph.

**Problem.** Current text: "Two independent failure pairs (`bam_triplet` and `bam_kan`) and one independent confirmatory negative result (`v2-EMA`) all converge on the same 98.36 % subtype ceiling..." This is the same wrong claim as A2/A3. `bam_kan` is not a failure pair. The probability argument ("unlikely that three independently trained configurations would land on the same accuracy to four significant figures") is then built on three configurations when only two qualify.

**Prescribed fix.** Rewrite the paragraph so the convergence is *two* configurations — `bam_triplet` and `v2-EMA` — both spatial-spatial pairs:

> The headline subtype number — 99.51 % — is the easy story. The harder story, and the one we believe matters more for the field, is the **role-complementarity principle**. The `bam_triplet` cell and the independent v2-EMA negative result — two pairings of spatial-role attention modules reached from disjoint module sets — converge on the same 98.36 % subtype ceiling, each falling below the no-attention control. It is unlikely that two independently trained spatial-spatial configurations would land on the same accuracy to four significant figures by chance; the convergence is better read as the signature of a systematic spatial-role conflict than as a coincidence. The contrast with `bam_kan` (BAM paired with the purely channel-wise KAN, 99.15 %) and `triplet_kan` (99.45 %) sharpens the point: a non-spatial second module is complementary, a spatial one collides.

Note: with two data points rather than three the "probability under a noise null is small" claim is weaker. Be honest about this — two configurations converging is suggestive, not conclusive, and the multi-seed follow-up in IX-F item 3 is the proper remedy. Consider softening "is small" to "is low" and leaning on the *mechanistic* argument (the value sits below the no-attention control, which a harmless redundant module would not produce) rather than the pure coincidence-probability argument. See also C2.

### B2. [MAJOR] Section IX-A "design rule" statement should be re-derived from the corrected principle

**Location:** `sections/09_discussion.tex`, Section IX-A second paragraph ("the modules should be paired so that they cover disjoint functional roles; if a candidate second module duplicates a role already occupied, the more reliable choice is to replace it with a no-attention pass").

**Problem.** The phrase "duplicates a role already occupied" is the over-general version that the (wrong) BAM+KAN channel collision was meant to support. The corrected evidence only supports the *spatial*-collision claim. As written, the design rule over-reaches beyond what the ablation shows.

**Prescribed fix.** Tighten to: "...the modules should be paired so that they do not both exercise the spatial role; pairing a spatial module with a purely channel-wise module is complementary, whereas pairing two spatial modules regresses the hub below the no-attention control. When a candidate second module would duplicate the spatial role, the more reliable choice is to drop it." This matches the corrected VII-B principle in A3.

### B3. [MINOR] Section IX-C "narrow 99.0–99.4 % corridor" excludes models in the table

**Location:** `sections/09_discussion.tex`, Section IX-C, first sentence.

**Problem.** "The binary head sits within a narrow 99.0–99.4 % corridor across the strong baselines." Swin-T (98.19), EfficientNetV2-S (98.66), EfficientNetV2-B3 (98.89), DenseNet121 (98.89), EfficientNet-B0 (98.83) all sit below 99.0. The corridor only holds if "strong baselines" is read very narrowly (ResNet50, Inception V3, EffV2-B2, and the proposed model). The phrasing invites a reader to check Table 1 and find counterexamples.

**Prescribed fix.** Either widen the stated range to "a 98.2–99.4 % corridor across all baselines" or qualify explicitly: "across the strongest baselines (ResNet50, Inception V3, EfficientNetV2-B2) the binary head sits within a narrow 99.1–99.4 % corridor". The second is more accurate and still supports the plateau argument.

### B4. [MINOR] Section IX-B repeats the MODC-SET comparison already made in Section II-D

**Location:** `sections/09_discussion.tex`, Section IX-B; compare `sections/02_related_work.tex` Section II-D.

**Problem.** II-D already states MODC-SET = ensemble of MobileNetV2 + InceptionResNetV2 + ResNet50 + XGBoost at 99.32 %, with the "no ablation justifying the ensemble" critique. IX-B restates the same composition, the same number, and the same critique. Some recall is fine in a Discussion, but this is near-verbatim.

**Prescribed fix.** In IX-B, drop the re-description of MODC-SET's composition (the reader has Section II) and keep only the *new* content: the head-to-head ("the proposed model exceeds 99.32 % at a single-stage 4.79 M architecture") and the methodological contrast (ablation-driven vs ensemble-scale). One sentence of recall is enough: "MODC-SET (Section II-D), the closest seven-class result, reaches 99.32 % via a three-backbone XGBoost ensemble."

### B5. [MAJOR] Section IX-B comparison to MODC-SET should carry the over-claiming caveat

**Location:** `sections/09_discussion.tex`, Section IX-B ("The proposed model exceeds this number with a single-stage 4.79 M parameter architecture.").

**Problem.** MODC-SET is evaluated on a *different* dataset ("a newly curated dataset of seven oral disease categories", per II-D). "The proposed model exceeds this number" reads as a direct head-to-head win, but 99.51 % vs 99.32 % on two different datasets is not a controlled comparison — and the manuscript is otherwise scrupulous about not over-claiming. A demanding referee will flag this as exactly the kind of cross-dataset accuracy comparison the paper criticizes others for.

**Prescribed fix.** Add an explicit caveat: "Because MODC-SET is evaluated on a different seven-class collection, this is not a controlled head-to-head comparison; we report it to situate the magnitude of the result, not to claim a win on a shared benchmark. The substantive contrast is methodological — single principled model versus unablated three-backbone ensemble — rather than a 0.19-point accuracy delta." This also keeps IX-B consistent with the honest-comparison posture of the rest of the paper.

---

## C. Abstract (`sections/00_abstract.tex`) and Introduction (`sections/01_introduction.tex`)

### C1. [MINOR] Abstract is already correct — confirm it is *not* changed to match the wrong body

**Location:** `sections/00_abstract.tex`.

**Observation, not a defect.** The Abstract correctly states the failure mode as "pairing Triplet ... with any module that also exercises a spatial role, namely BAM or EMA". This matches the JSON ground truth and the corrected Section VII. The Abstract is the *reference* for the fix, not a target of it. Flagged explicitly so a revision pass does not "harmonize" the Abstract down to the incorrect body text. After the Section VII fixes (A1–A6, B1–B2), Abstract and body will agree.

### C2. [MAJOR] Abstract over-states the ceiling evidence given single-run, two-point data

**Location:** `sections/00_abstract.tex`, "regresses subtype accuracy to a fixed 98.36 % ceiling".

**Problem.** "Fixed ceiling" is strong language for a value observed in exactly two single-run configurations (`bam_triplet`, `v2-EMA`). The manuscript elsewhere is careful to hedge single-run results ("within single-run noise"). The Abstract should carry a comparable, if compact, hedge so it does not promise more rigor than the single-seed protocol delivers.

**Prescribed fix.** Soften minimally: "...regresses subtype accuracy to a recurring 98.36 % sub-baseline ceiling..." ("recurring" is accurate — it recurs across the two spatial-spatial configurations — and drops the implication of an established invariant). The full hedge belongs in the body (IX-A, IX-D-a); the Abstract just needs to not say "fixed".

### C3. [MAJOR] Introduction contribution 2 says "explains why certain attention pairings systematically regress" — must be delivered in the corrected, narrower form

**Location:** `sections/01_introduction.tex`, numbered contribution 2.

**Problem.** Contribution 2 promises a principle "that explains why certain attention pairings systematically regress to a fixed accuracy ceiling on this dataset." The body must deliver exactly this. After the A3 correction the body explains *spatial-spatial* regression, supported by two configurations — narrower than "certain attention pairings" implies, and "systematically" is strong for n=2. The contribution list and the body must match (review brief item 4: contributions delivered by the body).

**Prescribed fix.** Reword contribution 2: "...we derive a **role-complementarity principle**: pairing two spatial-role attention modules regresses the hub below its no-attention control, whereas pairing a spatial module with a purely channel-wise one is complementary. The principle is supported by the v1 ablation grid and an independent negative-result run." This is precise, delivered by the corrected body, and avoids "systematically" / "fixed".

### C4. [MINOR] Roadmap paragraph — "no extra parameters or compute" is slightly loose

**Location:** `sections/01_introduction.tex`, contribution 3 ("AttentionHub-v2 attains the highest subtype accuracy in the entire study (99.51 %) at no additional parameters or compute"); also Abstract ("at no extra parameters or compute" — actually the Abstract does not say this, the Introduction does).

**Problem.** v2 = 4.789 M / 0.493 GFLOPs; v1 `full` = 4.800 M / 0.495 GFLOPs. v2 is marginally *smaller* than v1, and both are far smaller than the donor `none` cell (5.700 M). "At no additional parameters or compute" is true relative to v1 but reads ambiguously — additional relative to what? A referee will ask "no additional vs which baseline".

**Prescribed fix.** State the anchor: "...at a parameter and compute budget (4.79 M / 0.493 GFLOPs) essentially identical to the v1 hub and well below every baseline." Minor, but removes an ambiguity in a contribution bullet.

### C5. [NIT] Abstract sentence length

**Location:** `sections/00_abstract.tex`.

**Problem.** The Abstract is a single paragraph of very long sentences; the third sentence (the role-complementarity sentence) runs ~90 words with three em-dash clauses. IEEE abstracts are typically 150–250 words of readable sentences; this one is dense.

**Prescribed fix.** Split the role-complementarity sentence at "whereas": one sentence for the failure mode, one for the success mode. No content change, improves readability. Optional but recommended.

---

## D. Related Work (`sections/02_related_work.tex`)

The manuscript has correctly applied most of the `CITATIONS_TODO.md` corrections (CLASEG → EfficientNet-B3 + Mask R-CNN, 74.49 % / AP50 72.18; SE-MobileViT → above 98 %; Jubair → 85 % with 81–90 % CI; Tiwari → 23 studies; HF-UNet → proposed model, Dice ≈ 0.80; ref18 → DETR+SAM+ViT comparison study, ≈0.93; Rashid et al. → InceptionResNetV2 single model). Good. Remaining issues:

### D1. [MAJOR] Section II-D conflates Rashid et al. (ref19) with MODC-SET (ref22) — and ref19's accuracy is silently dropped

**Location:** `sections/02_related_work.tex`, Section II-D, paragraphs on ref19 and ref22.

**Problem.** `CITATIONS_TODO.md` (ref22 note) records that the MOD seven-class dataset is used by **Rashid et al. (ref19)**, who report **99.51 %** with InceptionResNetV2 alone — and that "MODC-SET ... at 99.32 %" could not be identified as a real paper at all (ref22 is an unconfirmed placeholder). The manuscript's II-D currently (a) describes ref19 without its headline number, and (b) presents MODC-SET/ref22 as a solid 99.32 % comparator. This is risky on two fronts: ref22 may not exist in citable form, and — more seriously — if Rashid et al. genuinely reach 99.51 % on a seven-class oral dataset, then the manuscript's repeated claim that the proposed model has "the highest subtype score" needs the "in this benchmark / among these nine from-scratch baselines" qualifier to be airtight (it generally is qualified — see G1 — but II-D is where a reader would catch a literature result that ties the proposed 99.51 %).

**Prescribed fix.**
1. Resolve ref22 before submission (it is the most critical unresolved reference per `CITATIONS_TODO.md`). If MODC-SET cannot be confirmed as a real, citable paper, **remove the MODC-SET paragraph and the IX-B comparison entirely** — the paper cannot anchor a Discussion comparison on an unverifiable reference.
2. Add Rashid et al.'s reported accuracy to the ref19 description so the literature picture is complete and honest: "...classifying a seven-class oral disease dataset of intraoral images with an InceptionResNetV2 network, reporting approximately 99.5 % accuracy under a unified preprocessing protocol."
3. If ref19's 99.51 % is on the *same* MOD dataset family the present study uses, state that explicitly and make sure the "highest in this benchmark" claims everywhere are read as "among the nine from-scratch baselines trained here" — never as a literature record. (See G1.)

### D2. [MAJOR] ref22 / MODC-SET is an unconfirmed placeholder and is cited as load-bearing in two sections

**Location:** `sections/02_related_work.tex` II-D and `sections/09_discussion.tex` IX-B; `paper.bib` ref22 (marked `[PLACEHOLDER — unconfirmed]`).

**Problem.** `CITATIONS_TODO.md`: "could not identify the paper with acronym MODC-SET anywhere in indexed literature." A paper should not build a Discussion subsection (IX-B, "Comparison to Prior Multi-Class Systems") around a reference that may not exist. This is a submission blocker if unresolved, but I tag it MAJOR because it is fixable by either confirming or removing the reference.

**Prescribed fix.** Author action: confirm ref22 against the master reference list / Zotero. If confirmed, update `paper.bib` with the verified record and remove the `[PLACEHOLDER]` marker. If it cannot be confirmed, delete all MODC-SET content (II-D paragraph 2, IX-B paragraph 1, the abstract has no MODC-SET mention so it is safe) and re-anchor IX-B on Rashid et al. (ref19) as the closest verified seven-class result.

### D3. [MAJOR] ref3, ref17, ref21 are unconfirmed placeholders; ref21 is described as oral disease when the only located paper is dental

**Location:** `sections/02_related_work.tex` II-B (ref3), II-A (ref17), II-E (ref21); `paper.bib` (all three `[PLACEHOLDER]`).

**Problem.** Per `CITATIONS_TODO.md`: ref3 (RAU transfer-learning CNN), ref17 (LBP + deep CNN ensemble, ~90 %), and ref21 (DeiT+CoAtNet) are all unconfirmed. ref21 is the worst case — the only paper found is Elazab et al., which is *dental radiograph* classification, not oral mucosal disease, and post-dates PAPER.md. The manuscript's II-E states ref21 "combined DeiT and CoAtNet for oral disease classification" — if the only citable paper is the dental one, that sentence is factually wrong.

**Prescribed fix.** Author must resolve all three from the master list before submission. Specifically: (a) ref3 — confirm whether it is Zhou et al. 2024 (if so, note it is 3-class, not binary, and adjust "distinguish ulcer and non-ulcer classes" accordingly); (b) ref17 — locate the actual LBP+CNN paper or remove the claim; (c) ref21 — if no genuine oral-mucosal DeiT+CoAtNet paper exists, either remove the ref21 sentence from II-E or, if the dental Elazab paper is the intended citation, rewrite the sentence to say "dental disease classification on radiographs" and reconsider whether it belongs in an oral-mucosal related-work section at all. Until resolved, II-E contains an unsupported claim.

### D4. [MINOR] Section II-A says SE-MobileViT achieves "above 98 %" — `CITATIONS_TODO.md` gives the exact 98.39 %

**Location:** `sections/02_related_work.tex`, II-A, SE-MobileViT paragraph.

**Problem.** `CITATIONS_TODO.md` ref16 note recommends reporting 98.39 % (the verified figure). "Above 98 %" is correct but vague; a precise figure is better and is available.

**Prescribed fix.** Change "achieving above 98 % binary accuracy" to "achieving 98.39 % binary accuracy". Trivial, improves precision.

### D5. [MINOR] Section II-A ref17 still reports "approximately 90 %" — `CITATIONS_TODO.md` flags this as unverified

**Location:** `sections/02_related_work.tex`, II-A, LBP+CNN paragraph.

**Problem.** `CITATIONS_TODO.md`: "No specific paper combining LBP + deep CNN ensemble at approximately 90 % ... was identified." The "approximately 90 % ... improved robustness to illumination and texture variation" description is unverified and quite specific. If ref17 is resolved (D3), the description must be checked against the real paper; if it cannot be resolved, the specific quantitative and robustness claims should be removed.

**Prescribed fix.** Tie to D3 resolution. If ref17 stays, replace specifics with whatever the real paper reports. If it cannot be confirmed, soften to a generic "LBP-plus-CNN feature fusion has also been explored for oral cancer detection" with no number and no robustness claim, or drop the paragraph.

### D6. [MINOR] Section II opening lists "YOLO variants" as a strand but no YOLO paper is cited

**Location:** `sections/02_related_work.tex`, Section II opening paragraph ("Object-detection frameworks such as YOLO variants have been adapted for lesion localization...").

**Problem.** YOLO is named as a research strand but no citation is attached and no subsection discusses it. An IEEE referee will ask for a citation or removal — an uncited named method in Related Work reads as a gap.

**Prescribed fix.** Either cite a specific oral-lesion YOLO paper, or generalize to "Object-detection frameworks have been adapted for lesion localization but produce only bounding boxes..." without naming YOLO. Given the paper does not engage detection further, generalizing is the lighter fix.

### D7. [NIT] ref24 ChatGPT-5 "approximately 85 %" — `CITATIONS_TODO.md` says no single 85 % figure was confirmed

**Location:** `sections/02_related_work.tex`, II-E, ChatGPT-5 sentence.

**Problem.** `CITATIONS_TODO.md` ref24 note: the paper reports Top-1/Top-3/Top-5 ranked accuracies and "no single summary 85 % accuracy figure was confirmed." The manuscript's "reaching approximately 85 % accuracy" may misrepresent which metric.

**Prescribed fix.** Verify against the paper and specify the metric, e.g. "reaching approximately 85 % Top-3 differential accuracy" — or whichever is correct. If unverifiable, soften to "reaching competitive but sub-expert accuracy, lagging human experts at Top-1" (the one fact `CITATIONS_TODO.md` does confirm).

---

## E. Methods, Setup, and Baseline Results (`sections/03`–`06`)

### E1. [MAJOR] Section III-B example list cites modules (CA, polarized self-attention, ECA, NAM) that never reappear

**Location:** `sections/03_preliminaries.tex`, III-B role-taxonomy bullets; also `sections/07_ablation.tex` VII-B (see A8).

**Problem.** III-B lists ECA as a channel example and (via VII-B) CA, polarized self-attention, NAM. None of these are used in the study. Listing four never-used module names in a taxonomy meant to be "the technical foundation of the role-complementarity principle" dilutes the taxonomy and invites "why are these mentioned" questions.

**Prescribed fix.** Trim the examples to the modules actually used: spatial role — "the spatial gate of BAM, the cross-dimensional gates of Triplet, the multi-scale spatial branch of EMA"; channel role — "SE and KAN-Attention, and the channel gate of BAM"; mixed role — "BAM, EMA". Drop ECA, CA, NAM, polarized self-attention, CBAM-sequential unless one is genuinely needed as a familiar anchor (SE alone suffices).

### E2. [MAJOR] The `none` ablation cell's 5.70 M / 0.636 GFLOPs contradicts the stated 4.79–4.80 M / 0.493–0.495 backbone envelope

**Location:** `sections/04_methods.tex` IV-C ("between 4.79 M and 4.80 M parameters and ... 0.493 and 0.495 GFLOPs; the small range spans the various ablation variants"); `sections/07_ablation.tex` Table 4 `none` row (5.700 M / 0.636 GFLOPs).

**Problem.** IV-C asserts the ablation variants span only 4.79–4.80 M / 0.493–0.495 GFLOPs. But the `none` cell in Table 4 is 5.700 M / 0.636 GFLOPs — far outside that envelope. The `none` cell *is* an ablation variant (the no-attention control), so IV-C's "the small range spans the various ablation variants" is false: the no-attention control is markedly heavier (it restores the donor's MBConv+SE Block-4). This is a genuine internal inconsistency a referee will catch by reading IV-C against Table 4.

**Prescribed fix.** Correct IV-C to exclude the `none` control from the "small range" statement: "...contains between 4.79 M and 4.80 M parameters and requires between 0.493 and 0.495 GFLOPs across the attention-bearing ablation variants; the no-attention control, which restores the donor's heavier MBConv+SE Block-4, is the exception at 5.70 M parameters and 0.636 GFLOPs (Table IV)." This also strengthens Finding 1 (the `triplet` cell is cheaper than `none`), so it is worth stating clearly.

### E3. [MAJOR] Confusion-matrix counts in Section VI-B are not internally consistent with the binary test-set size, which is never stated

**Location:** `sections/06_baseline_results.tex`, VI-B ("the binary matrix shows 11 benign images predicted malignant and 5 malignant images predicted benign"); `sections/04_methods.tex` IV-A and IV-G.

**Problem.** The paper states the DS2 subtype test set is 1,646 images with per-class supports, but the *binary* test-set size is never given. VI-A refers to "approximately 1,646 DS2 samples together with the DS1 binary set" — vague. The binary confusion matrix has 16 errors (11+5); 16 errors at 99.06 % accuracy implies a binary test set of ≈1,700 images, but the reader cannot check this because the binary test count is absent. For a paper that prides itself on reproducibility and exact numbers, the binary test-set support is a missing primitive.

**Prescribed fix.** State the binary test-set size explicitly in IV-A (DS1 80/10/10 split → give the DS1 test count; and state how many DS2 images feed the binary head). Then VI-A and VI-B can give exact denominators. If the binary head is evaluated on DS1-test + DS2-test-malignant-subset, say so and give the total. The "roughly five images" gap claim in VI-A also needs this denominator to be verifiable.

### E4. [MINOR] Section IV-A subtype class glosses are clinically dubious and unsourced

**Location:** `sections/04_methods.tex`, IV-A ("CaS (carcinoma in situ), CoS (condyloma-like / oral squamous), ... MC (oral mucositis carcinoma), OC (oral candidiasis / oral cancer) ...").

**Problem.** Several expansions are non-standard and internally hedged with slashes, which a clinical referee will challenge. "MC (oral mucositis carcinoma)" is not a recognized diagnostic entity; "OC (oral candidiasis / oral cancer)" pairs a fungal infection with a malignancy under one label, which is incoherent if OC is also counted among `{MC, OC, CaS}` malignant subtypes. The dataset class names appear to be folder abbreviations from a public dataset (CaS/CoS/Gum/MC/OC/OLP/OT), and guessing their clinical expansion risks factual error.

**Prescribed fix.** Either (a) cite the source dataset and use its documented class definitions verbatim, or (b) if the abbreviations are not documented upstream, present them as dataset-provided labels without inventing clinical expansions: "the seven DS2 classes (CaS, CoS, Gum, MC, OC, OLP, OT) as labelled in the source dataset." Critically, resolve the malignant-subset incoherence: `{MC, OC, CaS}` is defined as the malignant set, so the gloss for OC cannot include "oral candidiasis". Make the malignant-set definition and the class glosses consistent. This is MINOR only because it does not affect the numbers, but a clinical co-reviewer would likely raise it to MAJOR.

### E5. [MINOR] Section V-B claims a `classification_metrics.json` per model, but baseline results use values not traceable in the repo description

**Location:** `sections/05_experimental_setup.tex` V-B item 4; `sections/06_baseline_results.tex` Table 1.

**Problem.** V-B promises every model emits `classification_metrics.json`. The proposed/ablation cells do (verified). The reproducibility claim is only as strong as the baselines also having committed JSONs — the manuscript should confirm the nine baseline backbones each have the committed artifacts, or the "every reported number can be independently regenerated" claim in IV-G is overstated for the baseline rows.

**Prescribed fix.** Confirm and, if true, state that all ten models (nine baselines + proposed) have committed `classification_metrics.json` / `performance_metrics.json`. If the baseline JSONs are partial, soften IV-G's "every reported number can be independently regenerated" accordingly.

### E6. [MINOR] Section VI-C — "Train (min)" column: v2 trains in 30.80 min vs v1 at 131.70 min; this 4.3× gap deserves a sanity check, not just an explanation

**Location:** `sections/06_baseline_results.tex`, Table 3 and the paragraph after it; `sections/04_methods.tex` IV-F (recipe).

**Problem.** v1 `full` and v2 use the *identical* recipe and nearly identical FLOPs (0.495 vs 0.493) and params (4.80 vs 4.79 M). Yet v1 trains 131.70 min over 172 epochs and v2 trains 30.80 min over 141 epochs. That is ~0.77 min/epoch for v1 vs ~0.22 min/epoch for v2 — a 3.5× per-epoch speed difference between two models that differ only in hub topology (parallel-3-branch vs sequential-2-module). The text attributes it to "the lighter, sequential topology", but a 3.5× per-epoch wall-clock gap from a Stage-4 hub swap, when total FLOPs differ by 0.4 %, is implausible on its face and looks like a measurement artifact (e.g., different machine load, background processes, or a logging bug). A referee will not accept "lighter topology" for a 3.5× per-epoch gap that the FLOPs do not support.

**Prescribed fix.** Investigate. If the 30.80 min figure is correct, explain the per-epoch discrepancy with an actual mechanism (data-loading bottleneck? the parallel branches serializing on a single GPU stream? mixed precision on one run only?). If it is a measurement artifact, re-measure under controlled conditions or report training time with an explicit caveat that wall-clock training time is not controlled across runs. As written, this row undercuts the paper's credibility on the efficiency claims. At minimum, the claim "the cheapest to train" should be removed or heavily caveated until the discrepancy is explained.

### E7. [MINOR] Section VI-A "Pareto-optimal" wording

**Location:** `sections/06_baseline_results.tex`, VI-A ("among the nine baselines trained from scratch under the matched recipe, it is the Pareto-optimal point of the benchmark on the subtype task").

**Problem.** "Pareto-optimal point ... on the subtype task" — Pareto optimality is a property in a multi-objective space (here accuracy vs params). The phrasing "Pareto-optimal on the subtype task" is loose; it should be "Pareto-optimal in the subtype-accuracy / parameter-count plane". Also: the proposed model has both the highest subtype accuracy *and* the lowest params, so it is not merely Pareto-optimal — it strictly dominates on these two axes. "Pareto-optimal" understates it; "dominates the frontier" is accurate.

**Prescribed fix.** "...it strictly dominates every baseline in the subtype-accuracy versus parameter-count plane — simultaneously the most accurate and the smallest." Then Fig. `fig:pareto` is correctly described too.

### E8. [NIT] Section V-A says "AMD Ryzen 7 CPU"; Section V-A elsewhere and the CLAUDE-level config do not pin the exact CPU model

**Location:** `sections/05_experimental_setup.tex`, V-A.

**Problem.** "an AMD Ryzen 7 central processing unit" — Ryzen 7 is a product line, not a model. For a reproducibility-focused setup section the exact SKU (e.g., Ryzen 7 5800X) is expected, especially since CPU governs data-loading throughput and therefore the training-time numbers in Table 3.

**Prescribed fix.** Add the exact CPU model, or if it is genuinely not recorded, note that latency/GPU-memory are GPU-bound and CPU-insensitive (but training time, per E6, is not — so the CPU SKU matters there).

---

## F. Explainability (`sections/08`) and figures/tables globally

### F1. [MAJOR] Section VIII explainability claims are entirely qualitative and self-assessed; no quantitative interpretability metric

**Location:** `sections/08_explainability.tex`, all subsections; especially VIII-C "three-tier ordering".

**Problem.** Every interpretability claim ("attends to lesion tissue", "tight focal spot", "the most clinically aligned tier") is the authors' own visual judgment of the authors' own figures. There is no quantitative metric (e.g., pointing-game accuracy against lesion masks, IoU of Grad-CAM peak with annotated lesion region, deletion/insertion AUC, or even an inter-rater agreement among clinicians). The "three-tier ordering" of models in VIII-C is presented as a finding but is unfalsifiable as written. For an IEEE journal, an explainability section that is purely qualitative self-report is a known weakness; a referee will ask for at least one quantitative anchor.

**Prescribed fix.** Add at least one quantitative interpretability measure. The dataset has no pixel masks (the paper says so), which rules out IoU/pointing-game — but deletion/insertion AUC (Petsiuk et al.) needs no masks and can be computed from the existing models. Alternatively, have ≥2 people independently rank the panels and report agreement. If no quantitative measure can be added in this revision, explicitly downgrade the language: VIII-C should say "we offer a qualitative ordering" and the Abstract/Conclusion claims that XAI "confirms" lesion focus should soften to "is consistent with" — "confirm" implies a verification that a qualitative eyeball of one's own figures does not provide. (The Abstract currently says "confirm that the network attends to lesion tissue" — see also G2.)

### F2. [MAJOR] Section VIII-A target-layer choice for the proposed model may make Grad-CAM++ blind to the AttentionHub

**Location:** `sections/08_explainability.tex`, VIII-A ("For the proposed Custom EfficientNet V2 the target is Stage 5 — the MBConv+SE block immediately downstream of the AttentionHub").

**Problem.** The paper's central contribution is the Stage-4 AttentionHub. But Grad-CAM++ is computed at **Stage 5**, downstream of the hub. The subsequent claim (VIII-A final paragraph) that the saliency patterns reflect "Triplet preserves spatial coverage while SE concentrates the channel response" is then an inference about a module the saliency map does not actually probe. A referee will note that to make a claim about what the *hub* does, the natural target layer is the hub output (Stage 4), or both Stage 4 and Stage 5 should be shown. As written, the explainability section attributes Stage-5 saliency behavior to Stage-4 module roles without evidence connecting them.

**Prescribed fix.** Either (a) add Grad-CAM++ at the AttentionHub output and compare it to the Stage-5 map, which would let you actually show the hub's spatial behavior; or (b) drop the causal attribution in VIII-A's final paragraph ("precisely the behaviour expected of an attention hub that couples a spatial branch with a channel-sharpening branch ... Triplet preserves spatial coverage while SE ...") and state only what is supported: the Stage-5 representation that the heads consume is lesion-focused. Option (a) is much stronger and ties the XAI section to the paper's thesis; option (b) is the minimum honest fix.

### F3. [MAJOR] Several figures are referenced once with thin interpretation; figures must be *read* into the argument

**Location:** `fig:radar` (VI-A), `fig:heatmap` (VI-B), `fig:xai_cross` (VIII-C), `fig:progression` (VII-D).

**Problem.** Per the review brief (item 6, each figure interpreted in text). Spot-check:
- `fig:radar` — referenced and described, adequate.
- `fig:heatmap` — referenced; the text says the MC/OC columns are "the coolest band" and the proposed row is "most uniformly saturated". Adequate but the heatmap adds little beyond Table 2a (same numbers). Consider whether it earns a full-width `figure*`; if kept, the text should say what the heatmap shows that the table does not (visual gestalt of the MC/OC band across all models at once).
- `fig:xai_cross` — referenced in VIII-C; the caption is good. OK.
- `fig:progression` (`fig05d`) — referenced; text says "subtype accuracy rises monotonically along the path". Verify monotonicity holds with corrected numbers: none 98.85 → full 99.21 → triplet_kan 99.45 → v2 99.51. Yes, monotone. OK, but ensure the figure was rendered from correct numbers.

**Prescribed fix.** For `fig:heatmap`, add one sentence stating its unique contribution over Table 2a or consider demoting it. For all ablation/progression figures (`fig05`, `fig05b`, `fig05d`), confirm in the revision letter they were re-rendered from corrected JSON (this overlaps A4, A5). No figure should silently encode the wrong `bam_kan` number.

### F4. [MINOR] `fig:dataset` caption and IV-A — sample grid says "four randomly sampled examples per class" but figure layout

**Location:** `sections/04_methods.tex`, `fig:dataset` caption.

**Problem.** Caption says "four randomly sampled examples per class (seed = 42)" and "top two rows show the DS1 binary categories ... lower seven rows show the DS2 subtype classes." Two binary classes + seven subtype classes = nine classes; "four examples per class" with one class per row needs nine rows of four columns. The caption says "top two rows" for two classes (consistent: one row per class) — fine — but then the grid is 9 rows × 4 columns. Confirm the figure matches; "four randomly sampled examples" should be visible as four columns.

**Prescribed fix.** Verify the figure is 9×4 and the caption's row accounting is exact. Minor, but caption/figure mismatch is an easy referee catch.

### F5. [MINOR] Table 5 (`tab:proposed_vs_base`) duplicates rows already in Tables 1 and 3

**Location:** `sections/07_ablation.tex`, Table `tab:proposed_vs_base`.

**Problem.** Every value in Table 5 (binary acc, subtype acc, params, GFLOPs, size, GPU peak for EffV2-B2 / Inception V3 / proposed) already appears in Table 1 and Table 3. Table 5 is a re-extraction. It is defensible as a focused summary, but a referee may flag redundancy in a 27-page paper.

**Prescribed fix.** Keep Table 5 only if VII-E genuinely needs a compact side-by-side (it arguably does, as the chapter's payoff). If kept, the caption should signal it is a digest: "Digest of Tables I and III for the proposed model and the two strongest baselines." Otherwise, replace the table with an in-text sentence and a pointer to Tables I/III.

### F6. [NIT] Inconsistent acronym: "GradCAM++" vs "Grad-CAM++"

**Location:** Throughout. `main.tex` keywords say "Grad-CAM++"; `OUTLINE.md` and some prose say "GradCAM++"; CLAUDE.md uses "GradCAM++". The section files mostly use "Grad-CAM++". PAPER.md uses "GradCAM++".

**Problem.** The manuscript itself appears to have standardized on "Grad-CAM++" (with hyphen) in the section files and keywords — good — but verify no "GradCAM++" survives in the .tex files.

**Prescribed fix.** Grep the `sections/` files for "GradCAM" without the hyphen and fix any to "Grad-CAM++". The canonical spelling (Chattopadhyay et al.) is "Grad-CAM++". Purely cosmetic but IEEE copyeditors enforce it.

### F7. [NIT] "GFLOPs" vs "GFLOPS" and the MACs-vs-FLOPs ambiguity

**Location:** `sections/04_methods.tex` IV-G ("GFLOPs computed with `thop`"); Tables 1, 4, 5.

**Problem.** `thop` reports multiply-accumulate operations (MACs), not FLOPs; 1 MAC ≈ 2 FLOPs. VI-A's table header and prose say "GFLOPs" and IV-A/IV-G say "multiply-accumulate cost in GFLOPs" (VI-A intro actually says "multiply-accumulate cost in GFLOPs" — internally contradictory: MACs and FLOPs differ by 2×). The user's own memory note (`project_flops_estimator.md`) records "All perf JSONs now use thop (MACs)". So the numbers are MACs.

**Prescribed fix.** Decide on one convention and apply it everywhere. Cleanest: label the column "GMACs" and state once "computed with `thop`, which counts multiply-accumulate operations". If you keep "GFLOPs", then either multiply the `thop` output by 2 (and re-verify every table) or add an explicit footnote "we follow the common convention of reporting `thop` MAC counts as GFLOPs". Pick one; right now IV-G's "multiply-accumulate cost in GFLOPs" is a contradiction in terms.

---

## G. Over-claiming audit (review brief item 3)

### G1. [MAJOR] "Highest subtype accuracy" claims — mostly well-qualified, but two spots need the qualifier added

**Location:** Abstract ("the highest subtype score in the benchmark" — OK); `sections/01_introduction.tex` contribution 3 ("the highest subtype accuracy in the entire study" — OK); `sections/06_baseline_results.tex` VI-A ("the highest subtype accuracy in this benchmark" — OK); `sections/08_explainability.tex` VIII-C ("the highest subtype accuracy in the benchmark" — OK); **`sections/10_conclusion.tex`** ("the highest subtype score in a benchmark of nine standard CNN and Transformer baselines" — OK).

**Assessment.** The manuscript is, to its credit, consistent about scoping the claim to "this benchmark / this study / nine baselines". This is the correct framing per the review brief (highest among nine from-scratch baselines, not vs all literature).

**The two remaining risks:**
1. If Rashid et al. (ref19) genuinely report ≈99.51 % on a comparable seven-class oral dataset (see D1), then a reader who reaches Section II will see a literature result equal to the proposed model's headline. Nothing in the paper *claims* a literature record, but the proximity should be acknowledged. Add to IX-B or VI-A: "A single-model literature result (Rashid et al.) reports a comparable subtype accuracy on a related dataset; our claim is restricted to the controlled nine-baseline benchmark trained here under one matched recipe."
2. `sections/07_ablation.tex` VII-E and IX intro use "delivering the highest subtype accuracy" without the "in this benchmark" tag in a couple of running sentences. Sweep VII-E and IX for any bare "highest subtype accuracy" and append "in this benchmark" / "among the nine baselines".

**Prescribed fix.** As above — add the literature-proximity acknowledgement once, and ensure every instance of "highest subtype accuracy" carries the benchmark scope.

### G2. [MAJOR] "Confirm" is too strong for the explainability evidence

**Location:** `sections/00_abstract.tex` ("Grad-CAM++ ... and LIME ... panels confirm that the network attends to lesion tissue"); `sections/10_conclusion.tex` ("Grad-CAM++ and LIME panels confirm that the model attends to lesion tissue"); `sections/08_explainability.tex` VIII-C ("strengthens the claim").

**Problem.** As established in F1, the XAI evidence is qualitative self-assessment of the authors' own figures. "Confirm" asserts verification. This is an over-claim of the same kind the paper is otherwise careful to avoid.

**Prescribed fix.** Replace "confirm" with "indicate" or "are consistent with" in the Abstract and Conclusion. VIII-C's "strengthens the claim" is acceptable. If F1's quantitative measure is added, "confirm" can stay.

### G3. [BLOCKER → resolved by Section VII fix] "regresses ... to a fixed 98.36 % ceiling" / "two failing pairs"

Covered in A1–A6, B1–B2, C2–C3. Listed here for completeness of the over-claiming audit: the "fixed ceiling" and "two failing pairs" framing is the principal over-claim, and it is also factually wrong (hence BLOCKER, not just over-claim). The fixes above resolve it. The corrected claim — one failing v1 pair plus one independent negative-result confirmation, both spatial-spatial — is accurate and adequately supported.

### G4. [MINOR] "the v2 ... 0.06 pp gain ... within single-run noise" — correctly hedged, verify it stays

**Location:** `sections/07_ablation.tex` VII-D ("The 0.06 percentage-point gain ... is within single-run noise; we therefore present v2 as the principled design, not as a statistically distinguishable improvement"); `sections/09_discussion.tex` IX-D-a (same hedge).

**Assessment.** This is correctly and explicitly hedged in both places — exactly as the review brief requires. No over-claim here. Flagged only to confirm: do not let a revision pass strengthen this. The "ten-image correction on a 1,646-sample test set" framing is good and quantitative. Keep it.

### G5. [MINOR] "2.1× and 5.8× smaller than the strongest baselines" — verify the 5.8× endpoint

**Location:** Abstract, `sections/01_introduction.tex`, `sections/10_conclusion.tex`.

**Problem.** 2.1× = EffV2-B2 (10.00 / 4.79 = 2.09). 5.8× = ? ConvNeXt-Tiny 28.59 / 4.79 = 5.97; Swin-T 28.29 / 4.79 = 5.91; ResNet50 25.61 / 4.79 = 5.35; Inception V3 23.85 / 4.79 = 4.98. None equals 5.8 exactly. The closest is Swin-T at 5.9. Also, ConvNeXt-Tiny and Swin-T are not "the strongest baselines" — they are middling/weak performers — so "5.8× smaller than the strongest baselines" is doubly off: the number does not match and the comparator is not a "strongest" baseline.

**Prescribed fix.** Either (a) state the range honestly against the *strongest* baselines only: "2.1× smaller than EfficientNetV2-B2 and 5.0× smaller than Inception V3, the two strongest baselines"; or (b) if you want the full spread, say "between 2.1× (vs EfficientNetV2-B2) and 6.0× (vs ConvNeXt-Tiny) smaller than the other architectures in the benchmark" — but then drop "strongest". The current "2.1×–5.8× smaller than the strongest baselines" is not supported by the param table. This appears in three places (Abstract, Intro, Conclusion); fix all three identically.

---

## H. IEEE conventions, structure, and writing

### H1. [MAJOR] Introduction roadmap in PAPER.md is duplicated/garbled — confirm the .tex version is the clean one

**Location:** `sections/01_introduction.tex` final paragraph vs `docs/PAPER.md` §1 roadmap.

**Problem.** PAPER.md's roadmap paragraph is visibly broken: "Section 3 describes the dataset, the backbone, the AttentionHub variants, and the training protocol. Section 3 fixes notation..." (Section 3 mentioned twice, section numbers off by one — PAPER.md is ACM-numbered). The manuscript's `01_introduction.tex` roadmap is correct (Sections II–X with proper `\ref`). Good — the .tex fixed it. Flagged so a reviewer cross-checking against PAPER.md does not "restore" the broken version. No fix needed in the .tex; this is a note that PAPER.md itself is unreliable as a structural reference (it is, per OUTLINE.md, only the *content/number* source).

### H2. [MAJOR] Title says "Dual-Head" and the dual-head multi-task design is real, but the paper's evidence centers almost entirely on the subtype head

**Location:** Title; `sections/04_methods.tex` IV-B, IV-E; whole results/ablation.

**Problem.** The title and Abstract foreground the dual-head multi-task classifier. But: (a) the binary task is explicitly a tie "we do not claim a binary improvement"; (b) the ablation is evaluated on subtype accuracy; (c) there is no experiment isolating the *value of multi-tasking itself* — e.g., subtype-only training vs joint training, to show the binary head helps (or does not hurt) the subtype head. The dual-head design is a described mechanism but not an *evaluated contribution*. A referee will ask: does joint training with DS1 actually improve the subtype head, or is it neutral? Without that experiment, "Dual-Head" in the title promises an evaluated contribution the body does not deliver.

**Prescribed fix.** Best: add a small experiment — subtype-head-only training (no DS1, no binary head) vs the joint dual-head model — and report whether multi-tasking helps the subtype head. This would materially strengthen the paper and justify the title. If that experiment cannot be run, the paper must explicitly state that the dual-head design is a *training-efficiency / data-utilization* mechanism (it lets DS1 contribute) and is not claimed to improve subtype accuracy over single-task training — and consider whether "Dual-Head" deserves title billing if its value is not measured. At minimum add a sentence to IX acknowledging "we do not isolate the effect of joint multi-task training on the subtype head; this is left to future work" (and add it to the Limitations list, making it five).

### H3. [MINOR] Limitations list (IX-D) should include the explainability-is-qualitative and dual-head-not-isolated gaps

**Location:** `sections/09_discussion.tex` IX-D.

**Problem.** IX-D lists four limitations (single-seed, dataset scale, image-level-only, training-batch-timing). Two further genuine limitations surfaced in this review are absent: (1) interpretability claims are qualitative only (F1/G2), (2) the dual-head multi-task design's contribution is not isolated (H2). An honest limitations section should name them.

**Prescribed fix.** Add limitation (e): "Interpretability evidence is qualitative. We present Grad-CAM++ and LIME panels and a visual alignment ranking, but report no quantitative interpretability metric (e.g., deletion/insertion AUC); the interpretability ordering should therefore be read as indicative." And either add (f) on the dual-head isolation or fold it into IX as H2 prescribes.

### H4. [MINOR] Acronym definitions — spot-check first-use

**Location:** Throughout.

**Findings:**
- "CNN" — defined in Abstract and again in Introduction and again in IV-C ("five-stage convolutional neural network (CNN)"). Define once (Abstract or Intro), then reuse. IV-C redefining CNN is redundant.
- "MBConv" — IV-C expands "Mobile Inverted Bottleneck Convolution (MBConv)"; VIII-A re-expands it ("Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation (MBConv+SE)"). Define once.
- "RAU" — II-B defines "recurrent aphthous ulcer (RAU)". Introduction §I uses "recurrent aphthous ulcer" unabbreviated earlier — fine, but ensure RAU is defined at first *abbreviated* use (it is, in II-B).
- "DS1/DS2" — defined in IV-A. But VI-A uses "DS2 samples" and the Abstract/Intro do not need them — OK.
- "TTA" — CLAUDE.md mentions TTA for the custom model; verify the manuscript does not use "TTA" undefined (it appears not to — the matched recipe section does not mention TTA; good, since TTA would break the "single matched recipe" fairness claim — confirm no TTA leaked into the proposed-model description).
- "P50/P90/P95/P99" — IV-G lists all four; Table 3 reports only P50/P95. Fine, but IV-G could say "we tabulate P50 and P95" to match.

**Prescribed fix.** Remove the redundant CNN and MBConv re-definitions; otherwise acronym hygiene is acceptable.

### H5. [MINOR] `\IEEEPARstart` used once (good); ensure no other section starts with a drop-cap and verify IEEE float placement

**Location:** `main.tex`, all `figure*`/`table*`.

**Problem.** Minor IEEE-style check: the manuscript uses `[!t]` placement for all floats (per OUTLINE). With 15 figures and 8 tables in a 27-page two-column paper, `[!t]`-only placement can cause float pileups and large gaps. A spot-check of `build/main.pdf` is warranted to confirm no page is mostly white space and no float drifts more than a page from its first reference.

**Prescribed fix.** Compile and visually scan the PDF. If floats pile up, allow `[tb]` or `[!htbp]` selectively. Not a science issue, but IEEE camera-ready review will catch egregious float drift.

### H6. [NIT] "Section" vs "Sec." and figure/table abbreviation consistency

**Location:** Throughout.

**Problem.** The manuscript writes "Section~\ref{...}" in full and "Fig.~\ref{...}" / "Table~\ref{...}" abbreviated. IEEE convention: "Fig." and "Table" abbreviated (correct here), "Section" usually spelled out (also acceptable; IEEE allows "Section"). Consistent within the manuscript — OK. Just confirm no stray "Sec." appears.

### H7. [NIT] Em-dash density

**Location:** Throughout, especially Abstract, II opening, IV intro.

**Problem.** Heavy use of `---` (em-dash) parenthetical asides. Several sentences have two or three em-dash clauses. It is a house style and not wrong, but the density occasionally hurts readability (e.g., II opening paragraph, IV-C). 

**Prescribed fix.** Optional: in the densest paragraphs, convert one em-dash aside per sentence to a subordinate clause or a separate sentence. Low priority.

### H8. [MINOR] Back matter — Conflict-of-Interest placement

**Location:** `sections/11_backmatter.tex`.

**Problem.** The COI statement is currently a bare paragraph appended to "Data and Code Availability". OUTLINE.md says COI "may fold into a `\thanks` or a short paragraph". IEEE journals typically want COI as its own labeled statement or in the author `\thanks`. Appending it to the Data/Code section is untidy.

**Prescribed fix.** Give COI its own `\section*{Conflict of Interest}` or move it to a `\thanks` in `main.tex`. Minor formatting.

---

## I. Internal-consistency cross-checks (all numbers verified against released JSON)

The following were checked and are **correct** — listed so the author knows they were audited and need no change:
- Table 1 / `tab:benchmark` — all ten rows match PAPER.md §6.1.
- Table 4 params/GFLOPs columns — all eight rows match the `performance_metrics.json` files (`none` 5.700/0.636; `bam` 4.779/0.491; `triplet` 4.778/0.491; `kan` 4.777/0.490; `bam_kan` 4.789/0.493; `bam_triplet` 4.790/0.493; `triplet_kan` 4.788/0.493; `full` 4.800/0.495).
- Table 4 accuracy columns — **`bam` (0.9906/0.9909), `kan` (0.9842/0.9897), `none` (0.9860/0.9885), `triplet` (0.9895/0.9939), `triplet_kan` (0.9912/0.9945), `full` (0.9906/0.9921) all match the JSONs.** Only `bam_kan` and `bam_triplet` are wrong (see A1).
- Proposed v2 = 0.9906 binary / 0.9951 subtype — matches `custom_efficientnet_v2_hub_v2/classification_metrics.json`.
- v1 `full` = 0.9906 / 0.9921 — matches `custom_efficientnet_v2_baseline_recipe/classification_metrics.json`.
- Per-class supports (CaS 256, CoS 239, Gum 192, MC 288, OC 173, OLP 288, OT 210; sum = 1646) — internally consistent.
- Table 3 efficiency numbers — match PAPER.md §6.3.
- v2-EMA = 98.60 binary / 98.36 subtype — matches PAPER.md §7.3; unaffected by the Table 4 correction; the negative result stands.

One discrepancy worth noting (not a manuscript error, an awareness item): the repository also contains `results/custom_efficientnet_v2/classification_metrics.json` reporting **0.9947 / 0.9970** — a *different, higher* result for the custom model with no `recipe` field. This appears to be the pre-fair-comparison run (the original proposal, before the matched recipe and from-scratch protocol). The manuscript correctly uses the `baseline_recipe` (0.9906/0.9921) and `hub_v2` (0.9906/0.9951) numbers, not these. Confirm this stale directory is not referenced anywhere and consider removing it from the release to avoid a reader finding a 99.70 % number that contradicts the paper.

---

## Overall assessment

The paper makes a real and worthwhile contribution: a genuinely parameter-efficient backbone, a disciplined ablation, and an interpretable design rule (the role-complementarity principle) that pushes back against the "stack attention and report the best" habit. The writing is clean, the fair-comparison protocol is principled, the limitations and clinical-significance sections are unusually honest, and most numbers are faithfully transcribed and verifiable against the released JSON.

However, the manuscript currently **cannot be accepted** because Section VII encodes a factually wrong ablation result. Table 4 misreports `bam_kan` and `bam_triplet`; Finding 3, the role-complementarity principle, its bullet evidence, two figure captions, and the Discussion (IX-A) all claim "two failing v1 pairs" when the released ground truth shows only one (`bam_triplet`). The Abstract already states the correct version ("BAM or EMA"), so the manuscript contradicts itself. The corrected science is actually cleaner — a **spatial–spatial conflict** principle, with `bam_triplet` as the one v1 failure and the Triplet+EMA run as an independent confirmation, while `bam_kan` (99.15 %) is a working configuration that *supports* the principle by showing a non-spatial second module is harmless. Fixing this (items A1–A6, B1–B2, C2–C3) is mandatory and mostly mechanical, but the ablation figures must be re-rendered, not just re-captioned.

Beyond the Section VII correction, the major work for the next round is: (1) resolve the unconfirmed/placeholder references — especially MODC-SET (ref22), which a Discussion subsection depends on, and ref21, currently describing a dental paper as oral (D1–D3); (2) add at least one quantitative interpretability measure or honestly downgrade the XAI language from "confirm" to "indicate" (F1, G2); (3) explain or re-measure the implausible 3.5× per-epoch training-time gap between v1 and v2 (E6); (4) either evaluate the dual-head multi-task design's actual contribution or stop letting the title promise it (H2); (5) fix the over-claims — the "2.1×–5.8× smaller than the strongest baselines" figure is unsupported (G5), and the MODC-SET cross-dataset comparison needs a caveat (B5). The single-run / "within noise" hedging on the 0.06 pp v2 gain is already correct and should be preserved.

With the Section VII correction made faithfully, the reference issues resolved, the explainability claims calibrated, and the training-time anomaly explained, this is a publishable paper. **Recommendation: Major revision.**

Count of items: 11 BLOCKER/MAJOR-tagged blockers and majors requiring substantive work; the remainder are minor/nit polish. Specifically — BLOCKER: A1, A2, A3, A4, B1, G3 (=Section VII cluster). MAJOR: A5, A6, B2, B5, C2, C3, D1, D2, D3, E1, E2, E3, E6, F1, F2, F3, G1, G5, H1, H2.
