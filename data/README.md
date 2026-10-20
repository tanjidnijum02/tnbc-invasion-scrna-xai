# Input data

Download `GSE329526_MB231.h5ad` (~1.5 GB) from GEO series
[GSE329526](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE329526) (Supplementary files). Raw reads: SRA BioProject PRJNA1459420.

The file is the processed AnnData deposited by the data generators (De Santiago C, Gardner AL, Parker P, Thomas JR, Callahan G,
Ghosh M, Baker AB, Brock A). It contains log-normalised values only (`X` = layer `log1p_norm`, range 0–1.16), no raw counts,
plus QC metrics, 2,491 highly variable genes, PCA/UMAP/t-SNE, Leiden clusters and lineage annotations
(`collapsedLineages`, `AbunInvLineage`). It is not redistributed in this repository.
