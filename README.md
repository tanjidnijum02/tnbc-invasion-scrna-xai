# Invasive vs non-invasive TNBC cells: single-cell transcriptomics and explainable machine learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23494607.svg)](https://doi.org/10.5281/zenodo.23494607)

Code and results for the manuscript:

> **Integrative Single-Cell Transcriptomics and Explainable AI for Disease Mechanism and Biomarker Discovery in Triple-Negative Breast Cancer**
> Tanjidul Huda and Sriporna Biswas.

> For research use only. The models and gene lists here are computational hypotheses from one cell line, not clinical biomarkers.

## Data

Public dataset **GEO [GSE329526](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE329526)**, file `GSE329526_MB231.h5ad`
(ClonMapper-barcoded MDA-MB-231 cells, transwell invasion assay; 15,294 cells × 36,601 genes; 11 samples). The file is not
redistributed here; see [`data/README.md`](data/README.md).

| Group | Samples | Cells |
|---|---|---|
| Invasive | C3, C5, C6, C7, C8 | 6,289 |
| NonInvasive | B3, B5, B6, B7, B8 | 5,983 |
| Parental (labelled `Control`) | PT | 3,022 |

## What is in this repository

```
notebooks/original/TNCB.ipynb                          original Colab analysis, outputs kept (cell-level)
notebooks/TNBC_corrected_sample_aware_analysis.ipynb   sample-aware re-analysis (Colab version; downloads data from GEO)
scripts/tnbc_corrected_local.py                        same re-analysis as a script (version used for the reported results)
results/original/                                      CSV outputs of the original notebook
results/sample_aware/                                  outputs of the sample-aware re-analysis + corrected_results.json
figures/                                               key figures
```

## Analyses

**Original (cell level)** — Scanpy Wilcoxon DE (Invasive vs NonInvasive; High-Invasive lineage vs rest), XGBoost 3-class model
on all genes with a random 80/20 cell split, SHAP (TreeExplainer), Enrichr GO/KEGG.

**Sample-aware re-analysis** (the basis of the manuscript's conclusions):

1. Sample-level DE: mean expression per sample, 5 invasive vs 5 non-invasive samples, Welch t-test, Benjamini–Hochberg FDR.
2. XGBoost (Invasive vs NonInvasive, 2,491 HVGs) with 5-fold cross-validation grouped by well number, so whole samples are held out.
3. Controls: model without ribosomal/mitochondrial genes; model on 4 QC metrics only; random cell split for comparison.
4. SHAP on held-out samples (seed 42).
5. GSEA prerank on the sample-level t statistic (Hallmark, GO BP, KEGG).

## Key results (sample-aware)

| Quantity | Value |
|---|---|
| Genes tested / FDR < 0.05 (sample level) | 11,727 / 6,651 (1,313 up, 5,338 down in invasive) |
| XGBoost, grouped CV, HVGs: pooled AUC (fold range) | 0.919 (0.887–0.945) |
| Same, without ribosomal/mitochondrial genes | 0.919 |
| QC-metrics-only baseline | 0.630 |
| Random cell split (for comparison) | 0.931 |
| Top held-out SHAP genes | CCL2, LGALS12, DUSP1, MTRNR2L8, PRSS2, CXCL3, KYNU, CXCL2, DDIT4 |
| GSEA Hallmark | KRAS signalling up (NES 2.30, FDR 0.001), E2F targets (1.62, 0.038); interferon-α (−1.97, 0.001) and -γ (−1.65, 0.019) down |

Important caveat: according to GEO, invasive, non-invasive and parental cells were sequenced in separate libraries, so
library effects cannot be fully separated from biology even with sample-grouped validation.

## How to reproduce

**Colab:** open `notebooks/TNBC_corrected_sample_aware_analysis.ipynb` in Google Colab and choose *Runtime → Run all*. It installs
the packages and downloads the h5ad from GEO (~1.5 GB; High-RAM runtime recommended).

**Local:** put `GSE329526_MB231.h5ad` in `~/Downloads/` and run

```bash
uv run --python 3.12 --with scanpy==1.12.1 --with gseapy==1.2.1 --with xgboost==3.4.1 --with shap==0.54.1 --with matplotlib \
    python scripts/tnbc_corrected_local.py
```

Outputs are written to `~/Downloads/TNBC_corrected_results/`; the input file is only read. The reported results were produced
with Python 3.12.15, Scanpy 1.12.1, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.54.1, GSEApy 1.2.1
(see `requirements.txt`). On macOS, XGBoost needs the OpenMP runtime (`libomp`). The GSEA/Enrichr steps need internet access.

## Notes on the original notebook

`notebooks/original/TNCB.ipynb` is kept unchanged as a record. Known limitations, addressed by the re-analysis: cell-level tests
treat cells as independent; the random split puts cells of every sample in training and test sets; SHAP cells were sampled
without a seed; STRING hub-gene degrees were entered manually and the STRING settings were not recorded.

## Citation

See [`CITATION.cff`](CITATION.cff):

> Huda T, Biswas S. Code for: Integrative Single-Cell Transcriptomics and Explainable AI for Disease Mechanism and Biomarker
> Discovery in Triple-Negative Breast Cancer (v1.0.0). Zenodo; 2026. https://doi.org/10.5281/zenodo.23494607

 Please also cite the data generators (GSE329526) and the ClonMapper method
(Gutierrez et al., Nat Cancer 2021, doi:10.1038/s43018-021-00222-8).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Result tables are derived from public GEO data.

## Contact

Tanjidul Huda, Department of Biomedical Science, BCAS, University of Delhi — tanjidnijum@gmail.com
