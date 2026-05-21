# Oral Disease Classifier — IEEE LaTeX Manuscript

LaTeX source for the paper *"A Parameter-Efficient Dual-Head Oral Disease
Classifier with an Ablation-Driven Triplet→SE Attention Cascade."*

The compiled paper is `build/main.pdf` (also copied to `PAPER.pdf`), 24 pages,
two-column IEEE journal format.

## Build

Requires a TeX distribution with `pdflatex` and `bibtex`. Compile **from this
`paper/` directory** — figure and section paths are relative to it:

```
pdflatex -interaction=nonstopmode -output-directory=build main.tex
bibtex   build/main
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex
```

Output: `build/main.pdf`. No `latexmk` needed. Two `pdflatex` passes after
`bibtex` resolve the citations and two-column float numbering.

## Files

```
main.tex            preamble + \input of all sections; the compile target
IEEEtran.cls        IEEE journal class  (bundled — see note)
IEEEtran.bst        IEEE bibliography style  (bundled)
paper.bib           bibliography (29 entries)
sections/           one .tex per section (00_abstract … 11_backmatter)
figures/            15 figure PDFs
build/main.pdf      the compiled paper
OUTLINE.md          section / figure / table plan
REVIEW_round2.md    referee review (round 2)
REVIEW_round3.md    verification re-review (round 3)
CITATIONS_TODO.md   references the author must confirm before submission
```

## Bundled `IEEEtran` + font note

`IEEEtran.cls` / `IEEEtran.bst` are bundled here (not installed system-wide).
`IEEEtran.cls` carries one **local patch** near line 494: the machine this was
built on runs a minimal TeX Live that lacks the Times/Helvetica/Courier
(`ptm/phv/pcr`) font metrics, so the class is patched to fall back to Computer
Modern. For a submission build on a full TeX installation or on Overleaf,
restore the three original font lines (the patch comment lists them:
`phv / ptm / pcr`) to get the standard IEEE Times look — or just delete the
bundled `IEEEtran.cls`/`.bst` and let the system copies be used.

## Before submission — author actions

- **Authors:** fill the names and affiliations in `main.tex` (`\author{…}`).
- **Citations:** resolve the placeholder references flagged in
  `CITATIONS_TODO.md` (`ref3`, `ref17`, `ref21`, `ref22`) and the listed
  description mismatches before submitting.
- **PDF size:** `build/main.pdf` is ≈22 MB, dominated by three high-resolution
  raster figures (`fig02_dataset_sample_grid`, `fig06_gradcam_cross_model_composite`,
  `fig06c_proposed_explain_panel_landscape`). To shrink it for a submission
  portal, run Ghostscript on the output:
  ```
  gs -sDEVICE=pdfwrite -dPDFSETTINGS=/printer -dCompatibilityLevel=1.5 \
     -o PAPER_small.pdf build/main.pdf
  ```
  or regenerate those three figures at a lower DPI.

## Provenance

Drafted from `../docs/PAPER.md` and the figures in `../figures/`, then taken
through two referee-review/revision rounds and a LaTeX-typesetting pass.
Section VII's ablation numbers were corrected against the released
`results/*/classification_metrics.json` files, which superseded an error in
`PAPER.md`'s Table 4 (see `REVIEW_round2.md`, item A1).
