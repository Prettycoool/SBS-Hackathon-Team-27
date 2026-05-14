"""
10-Fold Stratified Cross-Validation & Multi-Classifier Comparison
=================================================================
Part 1 — XGBoost 10-fold CV: AUC, Sensitivity, Specificity, F1
           (threshold 0.46 applied to sensitivity/specificity/F1)
Part 2 — AUC comparison: XGBoost vs Random Forest vs
           Logistic Regression vs SVM (same 10-fold splits)

Outputs
-------
  plots/cv_xgboost_10fold.png   per-fold metric strip + mean/std
  plots/model_comparison_auc.png  AUC bar chart with fold-level dots
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

PCOS_THRESHOLD = 0.46   # optimal threshold from sensitivity analysis

# ── Colours ───────────────────────────────────────────────────────────────────
COL = {
    "auc":   "#8E44AD",   # purple
    "sens":  "#C0392B",   # red
    "spec":  "#2471A3",   # blue
    "f1":    "#17A589",   # teal
    "xgb":   "#E67E22",   # amber   — XGBoost
    "rf":    "#27AE60",   # green   — Random Forest
    "lr":    "#2980B9",   # blue    — Logistic Regression
    "svm":   "#8E44AD",   # purple  — SVM
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_excel(
        "data/(Main_Dataset)_PCOS_data_without_infertility.xlsx",
        sheet_name="Full_new",
    )
    df.columns = df.columns.str.strip()
    df.drop(columns=["Sl. No", "Patient File No."], inplace=True)
    df["Cycle(R/I)"] = df["Cycle(R/I)"].map(lambda v: 0 if v == 2 else 1)
    for col in ["AMH(ng/mL)", "II    beta-HCG(mIU/mL)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df.pop("PCOS (Y/N)")
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_models(neg, pos):
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=2000, random_state=42,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=42,
            )),
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — XGBOOST 10-FOLD CV  (full metric set per fold)
# ══════════════════════════════════════════════════════════════════════════════

def xgboost_cv_metrics(X, y, cv, model):
    """Return dict of per-fold arrays: auc, sens, spec, f1."""
    folds = {"auc": [], "sens": [], "spec": [], "f1": []}

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_te)[:, 1]
        y_pred  = (y_proba >= PCOS_THRESHOLD).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

        folds["auc"].append(roc_auc_score(y_te, y_proba))
        folds["sens"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        folds["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        folds["f1"].append(f1_score(y_te, y_pred, zero_division=0))

    return {k: np.array(v) for k, v in folds.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — MULTI-CLASSIFIER AUC  (cross_val_score per model)
# ══════════════════════════════════════════════════════════════════════════════

def all_models_auc(X, y, cv, models):
    """Return dict of model_name → per-fold AUC array."""
    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = scores
        print(f"  {name:<22}  AUC {scores.mean():.4f} ± {scores.std():.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — XGBoost detailed metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_xgboost_metrics(folds: dict):
    metrics = [
        ("auc",  "ROC-AUC",     COL["auc"]),
        ("sens", "Sensitivity", COL["sens"]),
        ("spec", "Specificity", COL["spec"]),
        ("f1",   "F1 Score",    COL["f1"]),
    ]
    n_folds = len(next(iter(folds.values())))

    fig, axes = plt.subplots(1, 4, figsize=(13, 6), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"XGBoost  ·  {n_folds}-Fold Stratified Cross-Validation\n"
        f"(threshold = {PCOS_THRESHOLD}, n = 541 patients)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, (key, label, color) in zip(axes, metrics):
        vals = folds[key]
        mu, sd = vals.mean(), vals.std()

        # ── shaded ±1 std band ────────────────────────────────────────────────
        ax.axhspan(mu - sd, mu + sd, alpha=0.12, color=color, zorder=1)
        # ── mean line ─────────────────────────────────────────────────────────
        ax.axhline(mu, color=color, linewidth=2.2, zorder=3)

        # ── individual fold dots (jittered x) ─────────────────────────────────
        rng   = np.random.default_rng(0)
        jitter = rng.uniform(-0.18, 0.18, size=n_folds)
        ax.scatter(
            jitter, vals,
            s=55, color=color, alpha=0.80,
            edgecolors="white", linewidth=0.8, zorder=4,
        )

        # ── annotation: mean ± std ────────────────────────────────────────────
        ax.text(
            0, mu + sd + 0.007,
            f"{mu:.3f} ± {sd:.3f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color, zorder=5,
        )

        # ── min / max fold labels ─────────────────────────────────────────────
        ax.text(0, vals.min() - 0.010, f"min {vals.min():.3f}",
                ha="center", va="top", fontsize=8, color="#7F8C8D")
        ax.text(0, vals.max() + 0.002, f"max {vals.max():.3f}",
                ha="center", va="bottom", fontsize=8, color="#7F8C8D")

        ax.set_xlim(-0.6, 0.6)
        lo = max(0, vals.min() - 0.06)
        hi = min(1.02, vals.max() + 0.06)
        ax.set_ylim(lo, hi)
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=12, fontweight="bold", labelpad=8, color=color)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#DDE3EA", zorder=0)

    # ── shared y-label on leftmost axis ──────────────────────────────────────
    axes[0].set_ylabel("Score", fontsize=11)

    fig.tight_layout()
    out = "plots/cv_xgboost_10fold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Multi-classifier AUC comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(auc_results: dict):
    model_colors = {
        "XGBoost":            COL["xgb"],
        "Random Forest":      COL["rf"],
        "Logistic Regression":COL["lr"],
        "SVM":                COL["svm"],
    }
    # Sort by mean AUC descending
    order = sorted(auc_results, key=lambda m: auc_results[m].mean(), reverse=True)
    means = [auc_results[m].mean() for m in order]
    stds  = [auc_results[m].std()  for m in order]
    colors = [model_colors[m] for m in order]
    n_folds = len(next(iter(auc_results.values())))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    x = np.arange(len(order))
    bar_w = 0.52

    # ── bars ──────────────────────────────────────────────────────────────────
    bars = ax.bar(
        x, means, width=bar_w,
        color=colors, alpha=0.82, zorder=2,
        error_kw=dict(elinewidth=1.8, capsize=5, capthick=1.8, ecolor="#2C3E50"),
        yerr=stds,
    )

    # ── individual fold AUC dots (jittered) ───────────────────────────────────
    rng = np.random.default_rng(1)
    for i, model in enumerate(order):
        fold_aucs = auc_results[model]
        jitter    = rng.uniform(-bar_w * 0.36, bar_w * 0.36, size=n_folds)
        ax.scatter(
            x[i] + jitter, fold_aucs,
            s=28, color="white", zorder=4,
            edgecolors=colors[i], linewidth=1.0, alpha=0.9,
        )

    # ── mean value labels above bars ──────────────────────────────────────────
    for i, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(
            x[i], mu + sd + 0.004,
            f"{mu:.4f}\n±{sd:.4f}",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=colors[i], zorder=5,
        )

    # ── rank labels (text, not emoji — emoji renders as boxes in matplotlib) ──
    for rank, i in enumerate(range(len(order))):
        label = ["#1", "#2", "#3", "#4"][rank]
        ax.text(
            x[i], means[i] * 0.5, label,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white",
            zorder=5, alpha=0.55,
        )

    # ── reference lines ───────────────────────────────────────────────────────
    for ref in [0.90, 0.95]:
        ax.axhline(ref, color="#BDC3C7", linewidth=1.0,
                   linestyle="--", zorder=1, alpha=0.8)
        ax.text(len(order) - 0.3, ref + 0.001,
                f"{ref:.2f}", fontsize=8, color="#95A5A6", va="bottom")

    # ── legend: fold dots ─────────────────────────────────────────────────────
    fold_patch = mpatches.Patch(
        facecolor="white", edgecolor="#2C3E50",
        label=f"Individual fold AUCs ({n_folds} folds)",
    )
    ax.legend(handles=[fold_patch], fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#DDE3EA")

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=12, labelpad=8)
    ax.set_ylim(
        max(0, min(m - 3 * s for m, s in zip(means, stds)) - 0.05),
        min(1.0, max(m + 3 * s for m, s in zip(means, stds)) + 0.06),
    )
    ax.set_title(
        f"Model Comparison — ROC-AUC  ·  {n_folds}-Fold Stratified CV\n"
        "(class-imbalance correction applied to all models,  n = 541 patients)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#DDE3EA", zorder=0)

    fig.tight_layout()
    out = "plots/model_comparison_auc.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("10-FOLD STRATIFIED CV  ·  PCOS CLASSIFIER EVALUATION")
    print("=" * 60)

    X, y = load_data()
    neg, pos = (y == 0).sum(), (y == 1).sum()
    print(f"\nDataset: {len(y)} patients  |  PCOS: {pos}  Not-PCOS: {neg}")

    cv     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = make_models(neg, pos)

    # ── Part 1: XGBoost detailed per-fold metrics ─────────────────────────────
    print(f"\n{'─'*60}")
    print("Part 1  — XGBoost 10-fold CV (threshold = 0.46)")
    print(f"{'─'*60}")

    xgb_folds = xgboost_cv_metrics(X, y, cv, models["XGBoost"])

    print(f"\n{'Metric':<16}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("─" * 56)
    labels = [("auc","ROC-AUC"), ("sens","Sensitivity"),
               ("spec","Specificity"), ("f1","F1 Score")]
    for key, label in labels:
        v = xgb_folds[key]
        print(f"  {label:<14}  {v.mean():>8.4f}  {v.std():>8.4f}"
              f"  {v.min():>8.4f}  {v.max():>8.4f}")

    print("\nPer-fold AUC values:")
    print("  " + "  ".join(f"Fold {i+1}: {v:.4f}"
                            for i, v in enumerate(xgb_folds["auc"])))

    # ── Part 2: all-model AUC comparison ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Part 2  — Multi-classifier AUC comparison (10-fold CV)")
    print(f"{'─'*60}")
    auc_results = all_models_auc(X, y, cv, models)

    print(f"\n{'Rank':<6}  {'Model':<24}  {'Mean AUC':>10}  {'Std':>8}")
    print("─" * 56)
    ranked = sorted(auc_results.items(), key=lambda x: x[1].mean(), reverse=True)
    for rank, (name, scores) in enumerate(ranked, 1):
        print(f"  {rank:<4}  {name:<24}  {scores.mean():>10.4f}  {scores.std():>8.4f}")

    best_name, best_scores = ranked[0]
    print(f"\n  Best model: {best_name}  ({best_scores.mean():.4f} ± {best_scores.std():.4f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Generating plots …")
    plot_xgboost_metrics(xgb_folds)
    plot_model_comparison(auc_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
