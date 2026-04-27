# Classification with k-NN — Project Files

A complete bundle for the assignment **Classification — minimal distance methods**.

---

## The exercises

### Exercise 1 — Dane1 (10 points, 2 features, 2 classes)

Apply the **k-NN method** to classify one new object. Use **three values of k** and **four distance metrics**:

- Euclidean
- Manhattan
- Chebyshev
- Mahalanobis

**Data 1**

| i | x1  | x2  | class |
|---|-----|-----|-------|
| 1 | 1.0 | 5.3 | 2 |
| 2 | 2.8 | 7.6 | 1 |
| 3 | 4.2 | 9.3 | 2 |
| 4 | 1.5 | 3.1 | 1 |
| 5 | 9.8 | 7.5 | 2 |
| 6 | 6.1 | 0.5 | 2 |
| 7 | 4.7 | 8.9 | 2 |
| 8 | 1.2 | 8.0 | 1 |
| 9 | 8.2 | 3.3 | 1 |
| 10| 6.4 | 5.5 | 1 |

The new object chosen for classification is **(5.0, 5.0)**.

### Exercise 2 — A larger real dataset

Apply **1-NN** and **k-NN** (three different values of k) to a chosen dataset. Repeat the calculations for **four evaluation methods**:

- Resubstitution
- Train/test split
- Cross-validation
- Leave-one-out

**Part I — Report (in tables):**
- accuracy
- sensitivity
- specificity
- F-measure
- AUC
- confusion matrix (for the best result)

**Part II:** plot classification quality vs. number of neighbours.

Dataset used: **Wisconsin Breast Cancer Diagnostic** (569 patients, 30 features, 2 classes).

---

## Files in this folder

| File | What it is |
|------|-----------|
| `classification_documentation.pdf` | The full 16-page report — theory, hand calculations, results, Python tutorial. **Read this first.** |
| `knn_full.py` | One Python script that computes everything and produces all figures. |
| `ex1_dane1_scatter.png` | Exercise 1 — scatter plot with the query point and its neighbours. |
| `ex2_quality_vs_k.png` | Exercise 2 Part II — quality vs. k for each evaluation method. |
| `ex2_confusion_best.png` | Exercise 2 — confusion matrix of the best result. |
| `ex2_roc.png` | Exercise 2 — ROC curves at k = 5. |
| `ex2_results.csv` | Exercise 2 — full numeric results table. |

---

## How to run the Python script

```bash
# 1. Install the libraries (one time only)
pip install numpy pandas matplotlib scikit-learn scipy

# 2. Run everything
python knn_full.py
```

The script prints both result tables to the screen and saves all figures next to itself.

---

## Trying your own values

Open `knn_full.py` and edit the block at the top:

```python
NEW_OBJ = np.array([5.0, 5.0])         # query point for Exercise 1
EX1_KS  = [1, 3, 5]                    # k values for Exercise 1
EX2_KS  = [1, 3, 5, 7]                 # k values for Exercise 2 table
EX2_KS_CURVE = list(range(1, 22, 2))   # k values for the curves
RNG     = 42                           # random seed
```

Change anything, re-run, and all results regenerate.

To use your own CSV instead of breast cancer, replace `load_breast_cancer()` with `pandas.read_csv("yourfile.csv")` and point the script to your label column. Section 8 of the PDF walks through this step by step.

---

## Main results (quick summary)

**Exercise 1.** The point (5.0, 5.0) is classified as **class 1** by 11 of the 12 (metric, k) combinations. Only Manhattan with k = 3 says class 2.

**Exercise 2.** Best honest accuracy is **0.9708 at k = 5** with the 70/30 split. Cross-validation and leave-one-out at k = 5 both give 0.9701 — basically tied, which confirms the result is real and not a lucky split. The k = 1 resubstitution score of 1.000 is the well-known overfitting trap and is **not** a useful number.

Full interpretation, formulas, and step-by-step explanations are in the PDF.
