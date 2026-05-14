"""
Threshold Sensitivity Analysis — PCOS XGBoost Classifier
=========================================================
Computes sensitivity, specificity and F1 score across classification
thresholds 0.10 – 0.90 and identifies the optimal F1 threshold.

Method: 5-fold stratified cross-validated predicted probabilities on the
full 541-patient dataset.  Using out-of-fold predictions avoids train-set
optimism while using every labelled example for evaluation.

Output: plots/threshold_sensitivity.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import xgboost as xgb

warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)

# ── Colour palette (matches DIANA frontend) ────────────────────────────────────
C_SENS  = "#C0392B"   # red    — sensitivity / recall
C_SPEC  = "#2471A3"   # blue   — specificity
C_F1    = "#17A589"   # teal   — F1 score
C_OPT   = "#E67E22"   # amber  — optimal threshold marker
C_DEF   = "#7F8C8D"   # grey   — default 0.50 threshold marker
C_BG    = "#F8F9FA"
C_GRID  = "#DDE3EA"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING  (identical preprocessing to stage1_pcos_classifier.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess():
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
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATED PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_cv_probabilities(X, y):
    """Return out-of-fold PCOS probabilities for all 541 patients."""
    neg, pos = (y == 0).sum(), (y == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]  # probability of PCOS


# ══════════════════════════════════════════════════════════════════════════════
# 3. THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def sweep_thresholds(y_true, y_proba, thresholds):
    """Return arrays of sensitivity, specificity, precision, F1 at each threshold."""
    sens_arr, spec_arr, prec_arr, f1_arr = [], [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        sens_arr.append(sens)
        spec_arr.append(spec)
        prec_arr.append(prec)
        f1_arr.append(f1)

    return (
        np.array(sens_arr),
        np.array(spec_arr),
        np.array(prec_arr),
        np.array(f1_arr),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_analysis(thresholds, sens, spec, f1, opt_idx):
    opt_t    = thresholds[opt_idx]
    opt_f1   = f1[opt_idx]
    opt_sens = sens[opt_idx]
    opt_spec = spec[opt_idx]

    # ── value at default 0.50 threshold ───────────────────────────────────────
    def_idx  = np.argmin(np.abs(thresholds - 0.50))
    def_f1   = f1[def_idx]
    def_sens = sens[def_idx]
    def_spec = spec[def_idx]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(C_BG)

    # ── subtle shaded band between default and optimal thresholds ─────────────
    lo, hi = min(opt_t, 0.50), max(opt_t, 0.50)
    ax.axvspan(lo, hi, alpha=0.08, color=C_OPT, zorder=1)

    # ── metric lines ──────────────────────────────────────────────────────────
    ax.plot(thresholds, sens, color=C_SENS, linewidth=2.5,
            label="Sensitivity (Recall)", zorder=3)
    ax.plot(thresholds, spec, color=C_SPEC, linewidth=2.5,
            label="Specificity",          zorder=3)
    ax.plot(thresholds, f1,   color=C_F1,   linewidth=2.8,
            label="F1 Score",             zorder=4)

    # ── default 0.50 threshold ────────────────────────────────────────────────
    ax.axvline(0.50, color=C_DEF, linewidth=1.6, linestyle="--",
               label="Default threshold (0.50)", zorder=2)

    # ── optimal F1 threshold ──────────────────────────────────────────────────
    ax.axvline(opt_t, color=C_OPT, linewidth=2.2, linestyle="-",
               label=f"Optimal threshold ({opt_t:.2f})", zorder=5)

    # Dot markers at optimal threshold
    for val, col in [(opt_sens, C_SENS), (opt_spec, C_SPEC), (opt_f1, C_F1)]:
        ax.scatter(opt_t, val, s=70, color=col, zorder=6,
                   edgecolors="white", linewidth=1.2)

    # ── annotation box — optimal values ───────────────────────────────────────
    box_x = opt_t + 0.02 if opt_t < 0.72 else opt_t - 0.19
    box_txt = (
        f"Optimal threshold: {opt_t:.2f}\n"
        f"  F1 Score:    {opt_f1:.3f}\n"
        f"  Sensitivity: {opt_sens:.3f}\n"
        f"  Specificity: {opt_spec:.3f}"
    )
    ax.text(
        box_x, 0.20, box_txt,
        fontsize=9.5, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=C_OPT, linewidth=1.5, alpha=0.95),
        color="#1A252F", zorder=7,
    )

    # ── annotation for F1 gain vs default ────────────────────────────────────
    f1_gain = opt_f1 - def_f1
    if abs(f1_gain) > 0.001:
        gain_txt = f"F1 gain vs default: +{f1_gain:.3f}"
        ax.annotate(
            gain_txt,
            xy=(opt_t, opt_f1),
            xytext=(opt_t + (0.04 if opt_t < 0.72 else -0.04), opt_f1 + 0.04),
            fontsize=8.5, color=C_F1,
            arrowprops=dict(arrowstyle="->", color=C_F1, lw=1.2),
            zorder=7,
        )

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(0.5, color="#BDC3C7", linewidth=0.8, linestyle=":",
               zorder=1, alpha=0.8)
    ax.axhline(0.9, color="#BDC3C7", linewidth=0.8, linestyle=":",
               zorder=1, alpha=0.6)

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Classification Threshold", fontsize=12, labelpad=8)
    ax.set_ylabel("Score", fontsize=12, labelpad=8)
    ax.set_title(
        "Threshold Sensitivity Analysis — PCOS XGBoost Classifier\n"
        "(5-fold cross-validated, n = 541 patients)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(np.arange(0.1, 0.91, 0.1))
    ax.set_xticklabels([f"{t:.1f}" for t in np.arange(0.1, 0.91, 0.1)])
    ax.grid(axis="both", linestyle=":", linewidth=0.7,
            color=C_GRID, zorder=0)

    # ── legend ────────────────────────────────────────────────────────────────
    ax.legend(
        fontsize=9.5, loc="lower left", framealpha=0.92,
        edgecolor="#DDE3EA", ncol=1,
    )

    # ── table of values at key thresholds (bottom inset) ──────────────────────
    row_labels = ["0.30", "0.40", f"{opt_t:.2f} ★", "0.50", "0.60", "0.70"]
    key_ts     = [0.30, 0.40, opt_t, 0.50, 0.60, 0.70]
    col_labels = ["Threshold", "Sensitivity", "Specificity", "F1 Score"]
    table_data = []
    for t in key_ts:
        i   = np.argmin(np.abs(thresholds - t))
        row = [
            f"{t:.2f}{'  ★' if t == opt_t else ''}",
            f"{sens[i]:.3f}",
            f"{spec[i]:.3f}",
            f"{f1[i]:.3f}",
        ]
        table_data.append(row)

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        bbox=[0.59, -0.38, 0.41, 0.32],  # [left, bottom, width, height] in axes coords
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDE3EA")
        if r == 0:
            cell.set_facecolor("#1A5276")
            cell.set_text_props(color="white", fontweight="bold")
        elif "★" in str(cell.get_text().get_text()):
            cell.set_facecolor("#FEF9E7")
            cell.set_text_props(color=C_OPT, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F8F9FA")
        else:
            cell.set_facecolor("white")

    fig.subplots_adjust(bottom=0.30)
    fig.text(
        0.5, 0.01,
        "★ Optimal threshold identified by maximum F1 score · "
        "Shaded region shows difference between default (0.50) and optimal thresholds",
        ha="center", fontsize=8, color="#7F8C8D",
    )

    out = "plots/threshold_sensitivity.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\nLoading and preprocessing data …")
    X, y = load_and_preprocess()
    print(f"  {len(y)} patients  |  PCOS: {y.sum()}  Not-PCOS: {(y==0).sum()}")

    print("\nGenerating 5-fold cross-validated probabilities …")
    y_proba = get_cv_probabilities(X, y)
    print(f"  Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

    thresholds = np.arange(0.05, 0.96, 0.01)
    print(f"\nSweeping {len(thresholds)} thresholds (0.05 – 0.95) …")
    sens, spec, prec, f1 = sweep_thresholds(y, y_proba, thresholds)

    opt_idx = int(np.argmax(f1))
    opt_t   = thresholds[opt_idx]

    print(f"\n{'─'*40}")
    print(f"Optimal threshold (max F1): {opt_t:.2f}")
    print(f"  F1 Score:    {f1[opt_idx]:.4f}")
    print(f"  Sensitivity: {sens[opt_idx]:.4f}")
    print(f"  Specificity: {spec[opt_idx]:.4f}")
    print(f"  Precision:   {prec[opt_idx]:.4f}")

    def_idx = np.argmin(np.abs(thresholds - 0.50))
    print(f"\nAt default threshold (0.50):")
    print(f"  F1 Score:    {f1[def_idx]:.4f}")
    print(f"  Sensitivity: {sens[def_idx]:.4f}")
    print(f"  Specificity: {spec[def_idx]:.4f}")
    print(f"  F1 gain from optimal:  +{f1[opt_idx]-f1[def_idx]:.4f}")
    print(f"  Sens gain from optimal: {sens[opt_idx]-sens[def_idx]:+.4f}")
    print(f"  Spec change:            {spec[opt_idx]-spec[def_idx]:+.4f}")

    print(f"\n{'─'*40}")
    print("Key threshold table:")
    print(f"{'Threshold':>10}  {'Sensitivity':>12}  {'Specificity':>12}  {'F1 Score':>10}")
    for t in [0.20, 0.30, 0.40, opt_t, 0.50, 0.60, 0.70, 0.80]:
        i = np.argmin(np.abs(thresholds - t))
        marker = " ★ OPTIMAL" if t == opt_t else ""
        print(f"  {t:>8.2f}  {sens[i]:>12.4f}  {spec[i]:>12.4f}  {f1[i]:>10.4f}{marker}")

    print("\nGenerating plot …")
    plot_sensitivity_analysis(thresholds, sens, spec, f1, opt_idx)
    print("\nDone.")


if __name__ == "__main__":
    main()
