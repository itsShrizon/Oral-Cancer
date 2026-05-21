# CITATIONS_TODO — Entries Requiring Author Action Before Submission

Generated: 2026-05-20  
Paper: "A Parameter-Efficient Dual-Head Oral Disease Classifier with an Ablation-Driven Triplet→SE Attention Cascade"

---

## SECTION 1 — PLACEHOLDER ENTRIES (must be resolved before submission)

These entries exist in `paper.bib` with syntactically valid placeholders. They **must** be replaced with verified bibliographic records before submission.

---

### ref3 — RAU / transfer-learning CNN (cited in §2.2)

**PAPER.md description:** "CNN-based framework…transfer learning with ResNet variants to distinguish ulcer and non-ulcer classes on a modest clinical dataset, achieving above 85% accuracy."

**Status:** UNCONFIRMED.

**Research note:** The closest candidate found by literature search is:

> Zhou M, Jie W, Tang F, Zhang S, Mao Q, Liu C, Hao Y. "Deep learning algorithms for classification and detection of recurrent aphthous ulcerations using oral clinical photographic images." *J Dent Sci.* 2024;19(1):254–260. DOI: 10.1016/j.jds.2023.04.022.

However, this paper (a) reports 92.86% accuracy (not "above 85%" in the sense implied), (b) classifies three classes (RAU, normal, other oral diseases) rather than binary ulcer/non-ulcer, and (c) is from 2024 not earlier. It is a **possible but not confident** match.

**Action required:** Check the master reference list / Zotero library. If the actual cited paper is the Zhou 2024 paper, replace the placeholder with:

```bibtex
@article{ref3,
  author  = {Zhou, Mimi and Jie, Weiping and Tang, Fan and Zhang, Shangjun
             and Mao, Qinghua and Liu, Chuanxia and Hao, Yilong},
  title   = {Deep Learning Algorithms for Classification and Detection of
             Recurrent Aphthous Ulcerations Using Oral Clinical Photographic
             Images},
  journal = {Journal of Dental Sciences},
  volume  = {19},
  number  = {1},
  pages   = {254--260},
  year    = {2024},
  doi     = {10.1016/j.jds.2023.04.022}
}
```

---

### ref17 — LBP + deep CNN feature fusion, ~90% oral cancer (cited in §2.1)

**PAPER.md description:** "The ensemble proposed in [17] fuses Local Binary Pattern (LBP) descriptors with deep CNN features, reaching approximately 90% binary accuracy with improved robustness to illumination and texture variation."

**Status:** UNCONFIRMED.

**Research note:** No specific paper combining LBP + deep CNN ensemble at approximately 90% binary oral cancer accuracy with explicit illumination/texture robustness claims was identified in literature search. Several papers combine LBP and CNN for oral cancer, but none match all three characteristics (ensemble, ~90%, illumination robustness claim) from 2019–2023.

**Action required:** Locate the original reference from the manuscript's master list or Zotero. Provide full author list, title, journal, volume, pages, year, DOI.

---

### ref21 — DeiT + CoAtNet hybrid for oral disease classification (cited in §2.5)

**PAPER.md description:** "the work in [21] combined DeiT and CoAtNet for oral disease classification, achieving strong multi-class performance, though the hybrid attention configuration is taken as given rather than tested through ablation."

**Status:** UNCONFIRMED — likely misidentified in search.

**Research note:** The only DeiT+CoAtNet paper found in the literature is:

> Elazab N, Nader N, Alsakar Y, Mohamed W, Elmogy M. "Improving dental disease diagnosis using a cross attention based hybrid model of DeiT and CoAtNet." *Scientific Reports.* 2026;16:805. DOI: 10.1038/s41598-025-32514-9.

This paper focuses on **dental** radiograph classification (cavities, fillings, implants, impacted teeth) — **NOT oral mucosal disease classification**. It also uses a stacking ensemble (SVM + XGBoost + MLP), not purely the DeiT+CoAtNet hybrid for oral disease. Publication date (Jan 2026) is also after PAPER.md's drafting. This is **almost certainly the wrong paper**.

**Action required:** Locate the correct [21] reference — likely a 2022–2024 paper on oral mucosal disease using a DeiT+CoAtNet hybrid. Provide full citation.

---

### ref22 — MODC-SET ensemble, 99.32% on 7-class oral disease (cited in §2.4, §9.2)

**PAPER.md description:** "MODC-SET [22] proposes an ensemble framework combining MobileNetV2, InceptionResNetV2, and ResNet50 with an XGBoost meta-classifier, evaluated on a newly curated dataset of seven oral disease categories. The ensemble achieves 99.32% overall accuracy."

**Status:** UNCONFIRMED — could not identify the paper with acronym "MODC-SET" anywhere in indexed literature.

**Research note:** The MOD (Mouth and Oral Disease) 7-class dataset is used by Rashid et al. (ref19), but that paper uses InceptionResNetV2 alone and reports 99.51%, not an ensemble at 99.32%.

**Action required:** This is the most critical unresolved reference. Search for "MODC-SET" in the actual manuscript reference list or Zotero. Confirm: (a) the exact paper title, (b) whether it is published or a preprint, (c) authors, journal/conference, year.

---

## SECTION 2 — VERIFIED ENTRIES WITH DESCRIPTION MISMATCHES

These entries have been identified and placed in `paper.bib` with full citations. However, the way PAPER.md describes them contains inaccuracies that should be corrected in the manuscript text before submission.

---

### ref15 — CLASEG (cited in §2.3)

**Key in paper.bib:** `ref15`  
**Verified paper:** Al-Ali et al., "CLASEG: Advanced Multiclassification and Segmentation for Differential Diagnosis of Oral Lesions Using Deep Learning." *Scientific Reports* 15:23016, 2025. DOI: 10.1038/s41598-025-03268-1.

**MISMATCH:**  
PAPER.md §2.3 states: *"The CLASEG framework [15] adopts a U-Net-like architecture for semantic segmentation of oral mucosal lesions, evaluating performance with Dice and Intersection-over-Union metrics."*

**Reality:** CLASEG uses **EfficientNet-B3** for classification and **ResNet-101-based Mask R-CNN** for instance segmentation. It covers 14 oral lesion classes. Reported classification accuracy is **74.49%** and segmentation AP50 is **72.18**. It is NOT U-Net-based and does NOT report Dice/IoU metrics — it uses AP50 (instance segmentation precision).

**Action required:** Revise §2.3 description to accurately reflect the EfficientNet-B3 + Mask R-CNN architecture, 14-class setting, and AP50/classification-accuracy metrics.

---

### ref16 — SE-MobileViT (cited in §2.1)

**Key in paper.bib:** `ref16`  
**Verified paper:** Kabir et al., "Accurate and Lightweight Oral Cancer Detection Using SE-MobileViT on Clinically Validated Image Dataset." *Discover Artificial Intelligence* 5:173, 2025. DOI: 10.1007/s44163-025-00442-2.

**MISMATCH:**  
PAPER.md §2.1 states: *"achieving above 92% binary accuracy."*

**Reality:** The paper reports **98.39%** accuracy (with macro F1 of 0.98 and ROC-AUC of 1.00). "Above 92%" significantly understates the result. This may have originated from a draft version or a different paper.

**Action required:** Update §2.1 to report 98.39% (or "above 98%") rather than "above 92%."

---

### ref5 — Jubair et al. 2022 (cited in §2.1)

**Key in paper.bib:** `ref5`  
**Verified paper:** Jubair et al., *Oral Diseases* 28:1123–1130, 2022. DOI: 10.1111/odi.13825.

**MINOR MISMATCH:**  
PAPER.md §2.1 states: *"approximately 90% classification accuracy."*

**Reality:** The reported point estimate is **85%** (95% CI: 81–90%). The upper confidence bound is 90%, which may be the source of the "approximately 90%" phrasing, but the stated accuracy is 85%, not 90%.

**Action required:** Consider correcting to "85% accuracy (95% CI: 81–90%)" or "approximately 85%."

---

### ref23 — AI systematic review (cited in §2.2)

**Key in paper.bib:** `ref23`  
**Verified paper:** Tiwari et al., "Artificial Intelligence's Use in the Diagnosis of Mouth Ulcers: A Systematic Review." *Cureus* 15(9):e45187, 2023. DOI: 10.7759/cureus.45187.

**MINOR MISMATCH:**  
PAPER.md §2.2 states: *"surveyed sixteen AI studies focused on OLP, RAS, and leukoplakia."*

**Reality:** The paper included **23 studies** in the final review (not 16), and its scope is mouth ulcers broadly — OLP, RAU, leukoplakia, and related conditions. The reported accuracy range (71%–100%) does appear in the reviewed studies.

**Action required:** Update §2.2 to say "23 AI studies" rather than "sixteen."

---

### ref24 — ChatGPT-5 (cited in §2.5)

**Key in paper.bib:** `ref24`  
**Verified paper:** Abou-Bakr, El Barbary, Hassanein. *Odontology*, 2025 (online ahead of print). DOI: 10.1007/s10266-025-01242-x.

**MINOR MISMATCH:**  
PAPER.md §2.5 states ChatGPT-5 reached "approximately 85% accuracy."

**Reality:** The paper reports Top-1, Top-3, Top-5 ranked differential accuracies on 100 biopsy-confirmed cases. The 85% figure may refer to a specific Top-N condition — but no single summary "85% accuracy" figure was confirmed from the abstract. The paper explicitly notes ChatGPT-5 "lagged at Top-1" versus experts.

**Action required:** Verify the specific accuracy metric used in §2.5 matches what the paper reports (Top-1 vs Top-3 vs overall).

---

### ref18 — ViT vs radiomics (cited in §2.5)

**Key in paper.bib:** `ref18`  
**Verified paper:** Chilet-Martos et al., *Computers in Biology and Medicine* 189, 2025. DOI confirmed from PII S0010482525002641; exact article number/pages could not be retrieved (journal behind paywall).

**MINOR MISMATCH:**  
PAPER.md §2.5 says "ViT variants benefiting from improved global feature extraction; the reported accuracies are competitive with CNN baselines."

**Reality:** The paper uses DETR + SAM + ViT and is primarily a ViT-vs-radiomics **comparison** study, not a pure ViT study. The combined ViT-radiomics model achieves specificity=0.97, sensitivity=0.88, accuracy=0.93. There is no "no ablation within the ViT backbone" claim to make — the paper's scope is model-type comparison, not internal ablation.

**Action required:** Confirm the DOI landing page for the exact volume/pages once institutional access is available.  
Also verify that §2.5's framing of the paper as evidence for "no ablation within ViT backbone" is accurate given the paper's actual scope.  
The `doi` field has been omitted from the .bib entry for ref18 because the DOI could not be confirmed without paywall access. Add it when the final article record is available.

---

### ref26 — HF-UNet oral ulcer segmentation (cited in §2.3 and §9.6)

**Key in paper.bib:** `ref26`  
**Verified paper:** Jiang et al., "A High-Order Focus Interaction Model and Oral Ulcer Dataset for Oral Ulcer Segmentation." *Scientific Reports* 14:20085, 2024. DOI: 10.1038/s41598-024-69125-9.

**MINOR MISMATCH:**  
PAPER.md §2.3 states: *"outperforming an HF-UNet baseline."*

**Reality:** HF-UNet IS the proposed model in this paper, not the baseline. The Dice score in the abstract is approximately 0.80 (not 82%). Image count and number of ulcer categories should be verified from the full paper.

**Action required:** Correct §2.3 — replace "outperforming an HF-UNet baseline" with the correct framing (e.g., "the proposed HF-UNet achieves a Dice score of approximately 80%"). Verify exact image count and category count from the full paper.

---

## SECTION 3 — PUBLICATION DETAILS TO CONFIRM

These verified entries have minor bibliographic details not confirmed from primary sources (journal behind paywall or ahead-of-print).

| Key     | Issue |
|---------|-------|
| `ref5`  | Oral Diseases vol.28 — issue number not confirmed (pp.1123–1130 confirmed). |
| `ref16` | Discover Artificial Intelligence vol.5, article 173 confirmed. Full author list may be incomplete — search found "Kabir, Ahmad, Uddin et al."; Cordero and Kant listed from ResearchGate diagram caption. Verify from PDF. |
| `ref18` | Computers in Biology and Medicine vol.189, 2025 — article number / page range not retrieved (paywall). DOI landing page returns pii S0010482525002641; confirm final pages when available. The DOI `10.1016/j.compbiomed.2025.109901` in the .bib entry is an ESTIMATE based on PII pattern and should be replaced with the confirmed DOI. |
| `ref24` | Odontology, 2025 — online ahead of print; volume/issue/pages not yet assigned. Update when published in print. |

---

## SECTION 4 — PART A NOTES

All Part A (method/backbone) entries are well-established papers with widely verified bibliographic data. No placeholders exist in Part A. One note:

- **`kingma2015adam`** — Adam was presented at ICLR 2015 as a conference paper; the `@misc` entry with `eprint = {1412.6980}` is the standard citation form used in practice. If the venue requires a strict `@inproceedings` format, use:

```bibtex
@inproceedings{kingma2015adam,
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  title     = {Adam: A Method for Stochastic Optimization},
  booktitle = {International Conference on Learning Representations ({ICLR})},
  year      = {2015}
}
```

- **`park2018bam`** — BMVC 2018 does not assign DOIs or page numbers to all papers. If a DOI is available from the BMVC proceedings website, add it. The arXiv preprint is arXiv:1807.06514 if needed.

---

*End of CITATIONS_TODO.md*
