# Classification with Minimal-Distance Methods — Deliverables

This bundle contains everything you need for the two assignment exercises
(*Classification — minimal distance methods*).

## Files

| File | Purpose |
|------|---------|
| **classification_documentation.tex** | The full LaTeX source. Compile with `pdflatex classification_documentation.tex` (run twice for ToC/cross-refs). Covers theory, hand calculations, results, and Python tutorial. |
| **classification_documentation.pdf** | Pre-built PDF of the documentation (16 pages). |
| **knn_full.py** | Self-contained Python script. Runs both exercises end-to-end and produces all figures, tables, and CSV. |
| **ex1_dane1_scatter.png** | Exercise 1 – scatter plot of the Dane1 data with the query point and its `k=5` neighbour circle. |
| **ex2_quality_vs_k.png** | Exercise 2 Part II – quality (accuracy / sensitivity / specificity / F1) vs. `k` for each evaluation method. |
| **ex2_confusion_best.png** | Exercise 2 – confusion matrix of the best generalisation result. |
| **ex2_roc.png** | Exercise 2 – ROC curves at `k=5` for all four evaluation methods. |
| **ex2_results.csv** | Exercise 2 – full numeric results table (16 rows, 7 columns). |

## Quick start (Python)

```bash
# 1. Install dependencies (one-time)
pip install numpy pandas matplotlib scikit-learn scipy

# 2. Run everything
python knn_full.py
```

The script prints the Exercise 1 distance table, the predictions for every
(metric, k) combination, the full Exercise 2 results table, and saves all
figures to the working directory.



A pre-built PDF is included so you do not have to compile if you only want
to read.

## Re-using the script on your own data

Open `knn_full.py` and locate the user-parameter block near the top:

```python
NEW_OBJ = np.array([5.0, 5.0])      # query point for Exercise 1
EX1_KS = [1, 3, 5]                  # k values for Exercise 1
EX2_KS = [1, 3, 5, 7]               # k values for Exercise 2
EX2_KS_CURVE = list(range(1, 22, 2))   # 1, 3, ..., 21 for the curves
RNG = 42                            # random seed everywhere
```

Change the values, re-run, and the report will be regenerated against
your new settings. To plug in your own CSV, replace the
`load_breast_cancer()` call with `pandas.read_csv(...)` and adjust the
column that holds the class label — Section 8 of the PDF walks through this.

## What each exercise covers

**Exercise 1** — Classify the point `(5, 5)` using the 10 Dane1 points,
with k ∈ {1, 3, 5} and four distance metrics (Euclidean, Manhattan,
Chebyshev, Mahalanobis). Eleven of the twelve (metric, k) combinations
predict class **1**; only Manhattan with k=3 dissents. Full hand
calculations and interpretation are in §5 of the PDF.

**Exercise 2** — Run k-NN on the 569-sample, 30-feature Wisconsin Breast
Cancer Diagnostic data set (loaded from scikit-learn). Compare four
evaluation methods × four k values × five quality measures. The best
honest accuracy is **0.9708 at k=5 (70/30 split)**, with k=5 LOO and
10-fold CV both at 0.9701 — practically tied. The confusion matrix and
ROC curves are in §7 of the PDF.

## Relation to descriptive statistics

The PDF contains a dedicated short section (§4) explaining how every
piece of the classifier — the Mahalanobis covariance Σ, the
standardisation step, the centroid of the Nearest-Mean classifier, and
the confusion matrix itself — is built from descriptive-statistics
primitives (mean, variance, covariance, frequency tables).
