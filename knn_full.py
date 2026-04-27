"""
==================================================================
  knn_full.py
  Complete companion script for the document
    "Classification with Minimal-Distance Methods"
==================================================================

WHAT THIS SCRIPT DOES
---------------------
1.  Exercise 1
    - Loads the small Dane1 dataset (10 points, 2 features, 2 classes).
    - Computes Euclidean, Manhattan, Chebyshev, Mahalanobis distances
      from a chosen query point to every training point.
    - For k = 1, 3, 5 votes the majority class with each metric.
    - Prints the distance table and the predictions.
    - Saves a scatter plot 'ex1_dane1_scatter.png'.

2.  Exercise 2
    - Loads the Wisconsin Breast Cancer Diagnostic data set.
    - Standardises the 30 features.
    - Runs k-NN for k = 1, 3, 5, 7 with four evaluation methods:
        - Resubstitution
        - 70/30 train/test split
        - 10-fold stratified cross-validation
        - Leave-one-out
    - For every (k, method) pair computes accuracy, sensitivity,
      specificity, F1, AUC and the confusion matrix.
    - Prints a single combined results table.
    - Saves:
        ex2_quality_vs_k.png      (quality curves vs k)
        ex2_confusion_best.png    (confusion matrix of best result)
        ex2_roc.png               (ROC curves for k=5)

HOW TO RUN
----------
    pip install numpy pandas matplotlib scikit-learn scipy
    python knn_full.py

EVERY MAGIC NUMBER (k values, query point, dataset choice) IS
DEFINED AT THE TOP OF THE SCRIPT. CHANGE THEM AND RE-RUN.
==================================================================
"""

# ----------------------------- imports ----------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     LeaveOneOut, cross_val_predict)
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

# ------------------------- user parameters ------------------------- #
NEW_OBJ = np.array([5.0, 5.0])      # query point for Exercise 1
EX1_KS = [1, 3, 5]                  # k values for Exercise 1
EX2_KS = [1, 3, 5, 7]               # k values for Exercise 2 (table)
EX2_KS_CURVE = list(range(1, 22, 2))   # 1, 3, ..., 21 for the curves
RNG = 42                            # random seed everywhere


# ====================================================================
# EXERCISE 1 -- hand-calculated distances on Dane1
# ====================================================================
def exercise_one():
    print("\n" + "=" * 60)
    print("EXERCISE 1 -- Dane1 (small 2-D dataset)")
    print("=" * 60)

    # 10 points: [x1, x2, class]
    data = np.array([
        [1.0, 5.3, 2], [2.8, 7.6, 1], [4.2, 9.3, 2], [1.5, 3.1, 1],
        [9.8, 7.5, 2], [6.1, 0.5, 2], [4.7, 8.9, 2], [1.2, 8.0, 1],
        [8.2, 3.3, 1], [6.4, 5.5, 1],
    ])
    X = data[:, :2]
    y = data[:, 2].astype(int)

    # ----- distance functions -----
    def d_euclid(a, b): return float(np.sqrt(np.sum((a - b) ** 2)))
    def d_manhat(a, b): return float(np.sum(np.abs(a - b)))
    def d_cheby(a, b):  return float(np.max(np.abs(a - b)))

    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    def d_maha(a, b):   return float(mahalanobis(a, b, inv_cov))

    print("\nCovariance matrix Sigma =")
    print(np.round(cov, 4))
    print("Sigma^-1 =")
    print(np.round(inv_cov, 4))

    # ----- distance table -----
    rows = []
    for i, (xi, lab) in enumerate(zip(X, y), start=1):
        rows.append({
            "i": i,
            "x1": xi[0], "x2": xi[1], "class": lab,
            "Euclid":      round(d_euclid(NEW_OBJ, xi), 4),
            "Manhattan":   round(d_manhat(NEW_OBJ, xi), 4),
            "Chebyshev":   round(d_cheby(NEW_OBJ, xi), 4),
            "Mahalanobis": round(d_maha(NEW_OBJ, xi), 4),
        })
    df = pd.DataFrame(rows)
    print(f"\nQuery point = {tuple(NEW_OBJ)}\n")
    print("Distance table:")
    print(df.to_string(index=False))

    # ----- predictions for each metric and k -----
    print("\nPredictions (majority vote of k nearest):")
    metrics = {"Euclid":      d_euclid,
               "Manhattan":   d_manhat,
               "Chebyshev":   d_cheby,
               "Mahalanobis": d_maha}
    pred_table = []
    for name, fn in metrics.items():
        d = sorted([(fn(NEW_OBJ, xi), int(lab)) for xi, lab in zip(X, y)])
        row = {"metric": name}
        for k in EX1_KS:
            nbrs = [lab for _, lab in d[:k]]
            vals, cnt = np.unique(nbrs, return_counts=True)
            pred = int(vals[np.argmax(cnt)])
            row[f"k={k}"] = f"{pred}  (nbrs: {nbrs})"
        pred_table.append(row)
    print(pd.DataFrame(pred_table).to_string(index=False))

    # ----- scatter plot -----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cls, color, marker in [(1, "#1f77b4", "o"), (2, "#d62728", "s")]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=color, marker=marker, s=80,
                   edgecolors="k", label=f"class {cls}")
    # query point
    ax.scatter(NEW_OBJ[0], NEW_OBJ[1], c="black", marker="*", s=260,
               edgecolors="white", linewidths=1.4, label="query (5,5)",
               zorder=5)
    # neighbours circle for k=5 Euclidean
    eu = sorted([(d_euclid(NEW_OBJ, xi), i) for i, xi in enumerate(X)])
    radius = eu[4][0]
    circ = plt.Circle(NEW_OBJ, radius, fill=False, color="gray",
                      linestyle="--", label=f"k=5 (Euclidean) r={radius:.2f}")
    ax.add_patch(circ)
    # annotate points
    for i, (xi, lab) in enumerate(zip(X, y), start=1):
        ax.annotate(str(i), xy=(xi[0] + 0.15, xi[1] + 0.15), fontsize=9)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Exercise 1 -- Dane1 dataset, query point and k=5 neighbours")
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1, 11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("Figures/ex1_dane1_scatter.png", dpi=150)
    plt.close(fig)
    print("\nSaved figure: ex1_dane1_scatter.png")


# ====================================================================
# EXERCISE 2 -- evaluation pipeline on the breast cancer dataset
# ====================================================================
def evaluate(y_true, y_pred, y_proba):
    """Return the five quality measures as a dict."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "f1":          f1_score(y_true, y_pred, pos_label=1),
        "auc":         roc_auc_score(y_true, y_proba),
    }


def run_evaluation(X, y, k, method):
    """Run a single (k, method) experiment and return predictions + metrics."""
    if method == "resub":
        clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        y_pred  = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]
        y_true  = y

    elif method == "split":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=RNG, stratify=y)
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        y_true  = y_te

    elif method == "cv10":
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RNG)
        y_pred  = cross_val_predict(KNeighborsClassifier(k), X, y, cv=skf)
        y_proba = cross_val_predict(KNeighborsClassifier(k), X, y, cv=skf,
                                    method="predict_proba")[:, 1]
        y_true  = y

    elif method == "loo":
        loo = LeaveOneOut()
        y_pred  = cross_val_predict(KNeighborsClassifier(k), X, y, cv=loo)
        y_proba = cross_val_predict(KNeighborsClassifier(k), X, y, cv=loo,
                                    method="predict_proba")[:, 1]
        y_true  = y
    else:
        raise ValueError(method)

    metrics = evaluate(y_true, y_pred, y_proba)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # benign first
    return y_true, y_pred, y_proba, metrics, cm


def exercise_two():
    print("\n" + "=" * 60)
    print("EXERCISE 2 -- Breast Cancer Wisconsin Diagnostic")
    print("=" * 60)

    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X = StandardScaler().fit_transform(X)
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, "
          f"benign(1): {(y==1).sum()}, malignant(0): {(y==0).sum()}")

    # -- Big results table --
    method_labels = {
        "resub": "resubstitution",
        "split": "70/30 split",
        "cv10":  "10-fold CV",
        "loo":   "leave-one-out",
    }

    rows = []
    cms = {}                 # store every confusion matrix for later
    for k in EX2_KS:
        for m_key, m_name in method_labels.items():
            _, _, _, metrics, cm = run_evaluation(X, y, k, m_key)
            cms[(k, m_key)] = cm
            rows.append({
                "k": k, "evaluation": m_name,
                **{kk: round(vv, 4) for kk, vv in metrics.items()},
            })

    table = pd.DataFrame(rows)
    print("\nFull results table (Exercise 2 Part I):\n")
    print(table.to_string(index=False))

    # -- Best generalisation result (excluding resubstitution) --
    nonresub = [r for r in rows if r["evaluation"] != "resubstitution"]
    best = max(nonresub, key=lambda r: r["accuracy"])
    print(f"\nBest *generalisation* result: k={best['k']}, "
          f"{best['evaluation']}, accuracy={best['accuracy']}")

    # ------------------- Plot 1: confusion matrix --------------------
    method_key_lookup = {v: k for k, v in method_labels.items()}
    best_cm = cms[(best["k"], method_key_lookup[best["evaluation"]])]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(best_cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(best_cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > best_cm.max() / 2 else "black",
                fontsize=18, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred. benign", "pred. malignant"])
    ax.set_yticklabels(["true benign", "true malignant"])
    ax.set_title(f"Confusion matrix -- best result\n"
                 f"k={best['k']}, {best['evaluation']}, "
                 f"acc={best['accuracy']:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("Figures/ex2_confusion_best.png", dpi=150)
    plt.close(fig)
    print("Saved figure: ex2_confusion_best.png")

    # --------------- Plot 2: quality vs k for each method ------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for idx, (mkey, mlabel) in enumerate(method_labels.items()):
        ax = axes[idx]
        accs, senss, specs, f1s = [], [], [], []
        for k in EX2_KS_CURVE:
            _, _, _, mm, _ = run_evaluation(X, y, k, mkey)
            accs.append(mm["accuracy"])
            senss.append(mm["sensitivity"])
            specs.append(mm["specificity"])
            f1s.append(mm["f1"])
        ax.plot(EX2_KS_CURVE, accs,  marker="o", label="accuracy")
        ax.plot(EX2_KS_CURVE, senss, marker="s", label="sensitivity")
        ax.plot(EX2_KS_CURVE, specs, marker="^", label="specificity")
        ax.plot(EX2_KS_CURVE, f1s,   marker="D", label="F1")
        ax.set_title(mlabel)
        ax.set_xlabel("k")
        ax.set_ylabel("quality")
        ax.set_ylim(0.85, 1.005)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Exercise 2 Part II -- Quality vs k", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("Figures/ex2_quality_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure: ex2_quality_vs_k.png")

    # ------------------- Plot 3: ROC curves for k=5 ------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for mkey, mlabel in method_labels.items():
        y_true, _, y_proba, mm, _ = run_evaluation(X, y, 5, mkey)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{mlabel} (AUC={mm['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate (1 - specificity)")
    ax.set_ylabel("True Positive Rate (sensitivity)")
    ax.set_title("ROC curves -- k=5, all evaluation methods")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("Figures/ex2_roc.png", dpi=150)
    plt.close(fig)
    print("Saved figure: ex2_roc.png")

    # ----- save the table to a CSV/Markdown for the report -----
    table.to_csv("ex2_results.csv", index=False)
    print("Saved table:  ex2_results.csv")


# ============================== main =============================== #
if __name__ == "__main__":
    exercise_one()
    exercise_two()
    print("\nAll done.")