"""Corrected, sample-aware re-analysis of GSE329526 (MDA-MB-231) for the TNBC manuscript.
Run on the Mac with:
  uv run --python 3.12 --with scanpy==1.12.1 --with gseapy==1.2.1 --with xgboost --with shap --with matplotlib python tnbc_corrected_local.py
Reads ~/Downloads/GSE329526_MB231.h5ad without modifying it; writes everything to ~/Downloads/TNBC_corrected_results/.
"""
# # Corrected, sample-aware re-analysis of GSE329526 (MDA-MB-231)
#
# This notebook re-runs the TNBC analysis with the corrections identified in the audit. It does **not**
# replace the original notebook (`TNCB.ipynb`), which stays as the record of the original analysis.
#
# What it changes, and why:
#
# 1. **Data check** – confirms what `adata.X` contains (the original logFC values assume log1p-normalised data).
# 2. **Sample-level (pseudobulk) differential expression** – the original Wilcoxon test treated ~12,000 cells
#    as independent replicates. Here each of the 10 transwell samples (5 invasive, 5 non-invasive) is one replicate.
# 3. **Sample-aware machine-learning validation** – the original random cell-level split put cells from the same
#    sample in both training and test sets. Here whole samples are held out (grouped cross-validation).
# 4. **Technical-confounding check** – a model trained only on QC metrics (mitochondrial %, ribosomal %, counts)
#    shows how much of the class separation could be technical.
# 5. **Reproducible SHAP** – fixed random seed, held-out cells, per-class plots with feature-value colouring.
# 6. **Enrichment with a defined background** and rank-based GSEA (no arbitrary top-N cut-off).
#
# How to run: open this file in Google Colab and choose Runtime → Run all. Nothing else is needed: the notebook installs the packages
# and downloads GSE329526_MB231.h5ad from GEO by itself. If Colab runs out of memory, choose Runtime → Change runtime type → High-RAM.
# The final cell writes `corrected_results.json` and a ZIP of all outputs — send those back for the manuscript.


import os, json, zipfile, warnings
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
H5AD = os.path.expanduser("~/Downloads/GSE329526_MB231.h5ad")   # original file: opened read-only, never modified
OUT_ROOT = os.path.expanduser("~/Downloads/TNBC_corrected_results")    # all outputs go into this NEW folder
os.makedirs(OUT_ROOT, exist_ok=True); os.chdir(OUT_ROOT)
OUT = "corrected_outputs"; os.makedirs(OUT, exist_ok=True)
RES = {}
import scanpy as sc
adata = sc.read_h5ad(H5AD)   # read only
print(adata)

# ## 1. What does the object contain?

def dense_row_sample(M, n=2000, seed=SEED):
    idx = np.random.default_rng(seed).choice(M.shape[0], min(n, M.shape[0]), replace=False)
    S = M[idx]
    return S.toarray() if sp.issparse(S) else np.asarray(S)

X_s = dense_row_sample(adata.X)
info = {
    "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
    "X_min": float(X_s.min()), "X_max": float(X_s.max()),
    "X_is_integer": bool(np.allclose(X_s, np.round(X_s))),
    "has_raw": adata.raw is not None,
    "layers": list(adata.layers.keys()),
    "uns_log1p": str(adata.uns.get("log1p", None)),
}
if "log1p_norm" in adata.layers:
    L_s = dense_row_sample(adata.layers["log1p_norm"])
    info["log1p_norm_min"], info["log1p_norm_max"] = float(L_s.min()), float(L_s.max())
    info["X_equals_log1p_norm"] = bool(np.allclose(X_s, L_s))
# example genes: in log1p(CP10k)-type data, highly expressed genes such as VIM or RPL3 are typically ~3-6
for g in ["VIM", "RPL3", "MT-ATP6", "GAPDH"]:
    if g in adata.var_names:
        v = adata[:, g].X; v = v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()
        info[f"mean_X_{g}"] = float(v.mean())
info["outlier_true"] = int(adata.obs["outlier"].astype(bool).sum()) if "outlier" in adata.obs else None
info["mt_outlier_true"] = int(adata.obs["mt_outlier"].astype(bool).sum()) if "mt_outlier" in adata.obs else None
RES["data_check"] = info
print(json.dumps(info, indent=2))

design = pd.crosstab(adata.obs["sample"], adata.obs["label"])
design.to_csv(f"{OUT}/cells_per_sample_and_label.csv"); print(design)
RES["cells_per_label"] = adata.obs["label"].value_counts().to_dict()
RES["cells_per_sample"] = adata.obs["sample"].value_counts().to_dict()
qc_cols = [c for c in ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"] if c in adata.obs]
qc_by_sample = adata.obs.groupby("sample", observed=True)[qc_cols].median()
qc_by_sample.to_csv(f"{OUT}/qc_median_by_sample.csv"); print(qc_by_sample)
RES["qc_median_by_label"] = adata.obs.groupby("label", observed=True)[qc_cols].median().round(3).to_dict()

# ## 2. Expression matrix used downstream
# Use the log-normalised layer if it exists (standard scale for log fold changes); otherwise `X`.

EXPR_KEY = "log1p_norm" if "log1p_norm" in adata.layers else None
E = adata.layers[EXPR_KEY] if EXPR_KEY else adata.X
E = E.tocsr() if sp.issparse(E) else sp.csr_matrix(E)
RES["expression_matrix_used"] = EXPR_KEY or "X"
genes = np.asarray(adata.var_names)
obs = adata.obs[["sample", "label"]].copy()
obs["label"] = obs["label"].astype(str); obs["sample"] = obs["sample"].astype(str)

# ## 3. Sample-level (pseudobulk) differential expression: Invasive vs NonInvasive
# One value per sample = mean log-normalised expression across that sample's cells.
# Genes are kept if detected in >=10% of cells in at least one of the two groups.
# Test: Welch t-test across the 5 vs 5 samples, Benjamini–Hochberg FDR.
# (If raw integer counts are available, the optional PyDESeq2 cell below is preferable.)

def pseudobulk_means(E, obs, groups_col="sample"):
    samples = sorted(obs[groups_col].unique())
    M = np.vstack([np.asarray(E[np.where(obs[groups_col].values == s)[0]].mean(axis=0)).ravel() for s in samples])
    return pd.DataFrame(M, index=samples)

def detection_rate(E, mask):
    sub = E[np.where(mask)[0]]
    return np.asarray((sub > 0).mean(axis=0)).ravel()

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out

inv = obs["label"].eq("Invasive").values; non = obs["label"].eq("NonInvasive").values
keep = np.maximum(detection_rate(E, inv), detection_rate(E, non)) >= 0.10
PB = pseudobulk_means(E[:, keep], obs.assign(sample=obs["sample"]))
PB.columns = genes[keep]
samp_label = obs.drop_duplicates("sample").set_index("sample")["label"]
inv_s = [s for s in PB.index if samp_label[s] == "Invasive"]
non_s = [s for s in PB.index if samp_label[s] == "NonInvasive"]
t, p = stats.ttest_ind(PB.loc[inv_s], PB.loc[non_s], equal_var=False)
pbde = pd.DataFrame({"gene": PB.columns, "mean_invasive": PB.loc[inv_s].mean().values,
                     "mean_noninvasive": PB.loc[non_s].mean().values, "t": t, "p": p})
pbde["diff_log_expr"] = pbde.mean_invasive - pbde.mean_noninvasive
pbde["fdr"] = bh(pbde.p.fillna(1))
pbde = pbde.sort_values("p"); pbde.to_csv(f"{OUT}/pseudobulk_DE_invasive_vs_noninvasive.csv", index=False)
RES["pseudobulk_DE"] = {
    "n_samples": {"Invasive": len(inv_s), "NonInvasive": len(non_s)}, "genes_tested": int(keep.sum()),
    "fdr_lt_0.05": int((pbde.fdr < 0.05).sum()), "fdr_lt_0.10": int((pbde.fdr < 0.10).sum()),
    "fdr_lt_0.05_up": int(((pbde.fdr < 0.05) & (pbde.diff_log_expr > 0)).sum()),
    "fdr_lt_0.05_down": int(((pbde.fdr < 0.05) & (pbde.diff_log_expr < 0)).sum()),
    "top20": pbde.head(20)[["gene", "diff_log_expr", "t", "p", "fdr"]].round(6).to_dict("records")}
print(json.dumps(RES["pseudobulk_DE"], indent=1)[:3000])

# How do the original cell-level top genes behave at sample level?
orig_top = ["PHLDA2","ANTXR2","VIM","HMGA1","CALM1","MANCR","ARHGDIB","FOXP1","ATP5MC2","RPSA"]
RES["original_top10_at_sample_level"] = pbde.set_index("gene").reindex(orig_top)[["diff_log_expr","p","fdr"]].round(6).reset_index().to_dict("records")

# OPTIONAL: PyDESeq2 on summed raw counts (only if integer counts exist in adata.raw or a 'counts' layer)
COUNTS = None
if adata.raw is not None and np.allclose(dense_row_sample(adata.raw.X, 200), np.round(dense_row_sample(adata.raw.X, 200))):
    COUNTS = adata.raw.X
elif "counts" in adata.layers:
    COUNTS = adata.layers["counts"]
RES["raw_counts_available"] = COUNTS is not None
print("Raw counts available:", COUNTS is not None)
# If True: !pip -q install pydeseq2, sum COUNTS per sample, and run DeseqDataSet(design="~label") on the 10 samples.

# ## 4. Machine learning with sample-aware validation
# Task A (main): Invasive vs NonInvasive, 10 samples. Grouped 5-fold CV where each fold holds out one
# well number (e.g. B3 and C3 together), so no sample contributes cells to both training and test data.
# Task B (reproduction): the original 3-class model with a random cell-level split, for comparison only.
# The Control class is a single sample (PT) sequenced in its own library, so it cannot be validated across samples.

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

XGB_PARAMS = dict(n_estimators=50, max_depth=4, random_state=SEED, n_jobs=-1, tree_method="hist")
hvg = adata.var["highly_variable"].values if "highly_variable" in adata.var else np.ones(adata.n_vars, bool)
feat_all = np.where(hvg)[0]
is_ribo_mt = pd.Series(genes).str.match(r"^(RPL|RPS|MRPL|MRPS|MT-)").values
feat_noRM = np.array([i for i in feat_all if not is_ribo_mt[i]])
RES["n_features"] = {"hvg": int(len(feat_all)), "hvg_without_ribo_mito": int(len(feat_noRM))}

idx2 = np.where(inv | non)[0]
y2 = inv[idx2].astype(int)
samples2 = obs["sample"].values[idx2]
well = pd.Series(samples2).str.extract(r"(\d+)")[0].values      # B3/C3 -> "3"
groups = well if len(set(well)) == 5 else samples2

def grouped_cv(Xm, y, groups, model_fn, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    prob = np.zeros(len(y)); fold_rows = []
    for k, (tr, te) in enumerate(gkf.split(Xm, y, groups)):
        m = model_fn(); m.fit(Xm[tr], y[tr]); prob[te] = m.predict_proba(Xm[te])[:, 1]
        fold_rows.append({"fold": k, "held_out_groups": sorted(set(groups[te])),
                          "auc": float(roc_auc_score(y[te], prob[te])) if len(set(y[te])) > 1 else None,
                          "bal_acc": float(balanced_accuracy_score(y[te], prob[te] > 0.5))})
    return prob, fold_rows

def summarise(name, y, prob, folds, samples):
    per_sample = pd.DataFrame({"sample": samples, "y": y, "p": prob}).groupby("sample").agg(
        label=("y", "first"), mean_p_invasive=("p", "mean"), frac_called_invasive=("p", lambda v: float((v > 0.5).mean())))
    per_sample.to_csv(f"{OUT}/{name}_per_sample.csv")
    aucs = [f["auc"] for f in folds if f["auc"] is not None]
    return {"pooled_auc": float(roc_auc_score(y, prob)), "pooled_bal_acc": float(balanced_accuracy_score(y, prob > 0.5)),
            "fold_auc_mean": float(np.mean(aucs)), "fold_auc_min": float(np.min(aucs)), "fold_auc_max": float(np.max(aucs)),
            "folds": folds, "per_sample": per_sample.round(4).reset_index().to_dict("records")}

X_hvg = E[idx2][:, feat_all]; X_noRM = E[idx2][:, feat_noRM]
X_qc = adata.obs.iloc[idx2][qc_cols].values.astype(float)
mk = lambda: XGBClassifier(objective="binary:logistic", **XGB_PARAMS)

RES["ml_grouped"] = {}
for name, Xm in [("xgb_hvg", X_hvg), ("xgb_hvg_no_ribo_mito", X_noRM), ("xgb_qc_metrics_only", X_qc)]:
    prob, folds = grouped_cv(Xm, y2, groups, mk)
    RES["ml_grouped"][name] = summarise(name, y2, prob, folds, samples2)
    print(name, {k: v for k, v in RES["ml_grouped"][name].items() if k in ("pooled_auc", "pooled_bal_acc", "fold_auc_min", "fold_auc_max")})

# Same binary task with a random cell-level split (what the original analysis did) for direct comparison
tr, te = train_test_split(np.arange(len(y2)), test_size=0.2, random_state=SEED, stratify=y2)
m = mk(); m.fit(X_hvg[tr], y2[tr]); pr = m.predict_proba(X_hvg[te])[:, 1]
RES["ml_random_cell_split_binary"] = {"auc": float(roc_auc_score(y2[te], pr)), "bal_acc": float(balanced_accuracy_score(y2[te], pr > 0.5))}
print("random cell split:", RES["ml_random_cell_split_binary"])

# Reproduction of the original 3-class model (all genes, random cell split, seed 42) — for transparency only
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); y3 = le.fit_transform(obs["label"].values)
Xtr, Xte, ytr, yte = train_test_split(adata.X, y3, test_size=0.2, random_state=SEED)
m3 = XGBClassifier(objective="multi:softprob", num_class=3, **XGB_PARAMS); m3.fit(Xtr, ytr)
p3 = m3.predict(Xte)
RES["ml_original_3class_reproduction"] = {"classes": list(le.classes_), "accuracy": float(accuracy_score(yte, p3)),
    "confusion_matrix": confusion_matrix(yte, p3).tolist(), "report": classification_report(yte, p3, target_names=le.classes_, output_dict=True)}
print(RES["ml_original_3class_reproduction"]["accuracy"])

# ## 5. SHAP on held-out cells (Invasive vs NonInvasive, grouped split)

import shap
gkf = GroupKFold(n_splits=5); tr, te = next(gkf.split(X_hvg, y2, groups))
mS = mk(); mS.fit(X_hvg[tr], y2[tr])
rng = np.random.default_rng(SEED); te_s = rng.choice(te, min(1000, len(te)), replace=False)
Xs = X_hvg[te_s].toarray()
sv = shap.TreeExplainer(mS).shap_values(Xs)
fn = genes[feat_all]
shap_rank = pd.DataFrame({"gene": fn, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
shap_rank.to_csv(f"{OUT}/shap_rank_invasive_vs_noninvasive_heldout.csv", index=False)
RES["shap_heldout"] = {"held_out_groups": sorted(set(groups[te])), "n_cells": int(len(te_s)),
                       "top20": shap_rank.head(20).round(6).to_dict("records"),
                       "ribo_mito_in_top20": int(pd.Series(shap_rank.head(20).gene).str.match(r"^(RPL|RPS|MRPL|MRPS|MT-)").sum())}
shap.summary_plot(sv, Xs, feature_names=fn, max_display=20, show=False)
plt.title("SHAP, Invasive vs NonInvasive (held-out samples)"); plt.savefig(f"{OUT}/shap_summary_heldout.png", dpi=300, bbox_inches="tight"); plt.close()

# ## 6. Enrichment
# (a) GSEA prerank on the sample-level t statistic (all tested genes, no cut-off).
# (b) Enrichr over-representation for FDR<0.05 sample-level genes (if any), with the tested genes as background.

import gseapy as gp
rnk = pbde[["gene", "t"]].dropna().sort_values("t", ascending=False)
RES["gsea"] = {}
for lib in ["KEGG_2021_Human", "GO_Biological_Process_2023", "MSigDB_Hallmark_2020"]:
    try:
        pre = gp.prerank(rnk=rnk, gene_sets=lib, permutation_num=1000, seed=SEED, min_size=15, max_size=500, threads=2, outdir=None)
        r = pre.res2d.sort_values("FDR q-val"); r.to_csv(f"{OUT}/gsea_prerank_{lib}.csv", index=False)
        RES["gsea"][lib] = {"fdr_lt_0.25": int((r["FDR q-val"].astype(float) < 0.25).sum()),
                            "top10": r.head(10)[["Term", "NES", "FDR q-val"]].astype(str).to_dict("records")}
    except Exception as e:
        RES["gsea"][lib] = f"failed: {e}"
sig = pbde.loc[pbde.fdr < 0.05, "gene"].tolist()
RES["enrichr_sample_level"] = {"n_input": len(sig)}
if len(sig) >= 10:
    enr = gp.enrichr(gene_list=sig, gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2023"],
                     background=pbde.gene.tolist(), outdir=None)
    enr.results.to_csv(f"{OUT}/enrichr_sample_level_DE.csv", index=False)
    RES["enrichr_sample_level"]["top10"] = enr.results.sort_values("Adjusted P-value").head(10)[[c for c in ["Gene_set", "Term", "Overlap", "Odds Ratio", "Adjusted P-value"] if c in enr.results.columns]].astype(str).to_dict("records")

# ## 7. Software versions and export

import sklearn, xgboost, platform
RES["versions"] = {"python": platform.python_version(), "scanpy": sc.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__, "gseapy": gp.__version__}
json.dump(RES, open("corrected_results.json", "w"), indent=1, default=str)
with zipfile.ZipFile("corrected_outputs.zip", "w") as z:
    z.write("corrected_results.json")
    for f in os.listdir(OUT): z.write(os.path.join(OUT, f))
print("Saved corrected_results.json and corrected_outputs.zip")
print("Results folder:", OUT_ROOT)
