"""
Presentation Plots Generator
=============================
Produces four publication-ready PNG files in plots/:

  1. roc_curve_stage1.png        — ROC curve for the Stage 1 PCOS classifier
  2. shap_summary_stage1.png     — SHAP beeswarm aligned with Rotterdam criteria
  3. confusion_matrix_stage1.png — Annotated confusion matrix (held-out test set)
  4. patient_funnel.png          — Flow diagram: 541 patients → Stage 1 → Stage 2

Requires that stage1_pcos_classifier.py and stage2_differential_diagnosis.py
have already been run (models/ must contain the saved .json models).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, ConfusionMatrixDisplay,
                             confusion_matrix)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
PCOS_DATA_PATH  = "data/(Main_Dataset)_PCOS_data_without_infertility.xlsx"
PCOS_MODEL_PATH = "models/pcos_classifier.json"
MEDIANS_PATH    = "models/pcos_feature_medians.csv"
RESULTS_PATH    = "models/pipeline_results.csv"
PLOTS_DIR       = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── shared palette ─────────────────────────────────────────────────────────────
C_PCOS     = "#C0392B"   # red   — PCOS positive
C_NOT_PCOS = "#2980B9"   # blue  — Not-PCOS
C_ENDO     = "#27AE60"   # green — endometriosis
C_THYROID  = "#8E44AD"   # purple — thyroid
C_PRL      = "#E67E22"   # orange — hyperprolactinemia
C_NONE     = "#95A5A6"   # grey  — no flags
C_DIAG     = "#1A252F"   # near-black for diagnostics box
FONT_TITLE = 15
FONT_LABEL = 12
FONT_TICK  = 10


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PREPROCESSING  (mirrors stage1_pcos_classifier.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def load_pcos(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(path, sheet_name="Full_new")
    df.columns = df.columns.str.strip()
    df.drop(columns=["Sl. No", "Patient File No."], inplace=True)
    df["Cycle(R/I)"] = df["Cycle(R/I)"].map(lambda v: 0 if v == 2 else 1)
    for col in ["AMH(ng/mL)", "II    beta-HCG(mIU/mL)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df.pop("PCOS (Y/N)")
    return df, y


def impute(X: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    imp = SimpleImputer(strategy="median")
    return pd.DataFrame(imp.fit_transform(X), columns=X.columns), imp


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROC CURVE  (Stage 1 — held-out test set, same split as training)
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curve() -> None:
    print("  Generating ROC curve …")
    X, y = load_pcos(PCOS_DATA_PATH)
    X_imp, _ = impute(X)

    # Reproduce the exact train/test split used in stage1 training
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor("white")

    # Random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="#BDC3C7",
            linewidth=1.5, label="Random classifier (AUC = 0.50)")

    # ROC curve with gradient-like fill
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_PCOS)
    ax.plot(fpr, tpr, color=C_PCOS, linewidth=2.5,
            label=f"PCOS classifier  (AUC = {roc_auc:.3f})")

    # Optimal threshold point (closest to top-left)
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    opt_idx   = np.argmin(distances)
    ax.scatter(fpr[opt_idx], tpr[opt_idx], s=90, zorder=5,
               color=C_PCOS, edgecolors="white", linewidth=1.5,
               label=f"Optimal threshold  (FPR={fpr[opt_idx]:.2f}, TPR={tpr[opt_idx]:.2f})")

    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=FONT_LABEL)
    ax.set_title("Stage 1 — PCOS Classifier ROC Curve\n(held-out 20% test set, n=109)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=FONT_TICK, loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.6)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "roc_curve_stage1.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAP SUMMARY  (Rotterdam-annotated beeswarm)
# ══════════════════════════════════════════════════════════════════════════════

# Map each feature to its Rotterdam criterion (or None for non-Rotterdam)
ROTTERDAM_MAP = {
    "Follicle No. (R)":     "Polycystic ovaries",
    "Follicle No. (L)":     "Polycystic ovaries",
    "Avg. F size (R) (mm)": "Polycystic ovaries",
    "Avg. F size (L) (mm)": "Polycystic ovaries",
    "hair growth(Y/N)":     "Hyperandrogenism",
    "Skin darkening (Y/N)": "Hyperandrogenism",
    "AMH(ng/mL)":           "Hyperandrogenism",
    "Pimples(Y/N)":         "Hyperandrogenism",
    "FSH/LH":               "Hyperandrogenism",
    "LH(mIU/mL)":           "Hyperandrogenism",
    "Cycle(R/I)":           "Anovulation",
    "Cycle length(days)":   "Anovulation",
}
CRITERION_COLOR = {
    "Polycystic ovaries": "#2980B9",
    "Hyperandrogenism":   "#C0392B",
    "Anovulation":        "#27AE60",
}


def plot_shap_summary() -> None:
    print("  Generating SHAP summary …")
    X, y = load_pcos(PCOS_DATA_PATH)
    X_imp, _ = impute(X)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_imp, y)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    # Top 15 features by mean |SHAP|
    mean_abs    = np.abs(shap_values).mean(axis=0)
    top15_idx   = np.argsort(mean_abs)[-15:]
    feat_names  = X.columns.tolist()
    top15_names = [feat_names[i] for i in top15_idx]

    # Colour each label by Rotterdam criterion
    label_colors = [
        CRITERION_COLOR.get(ROTTERDAM_MAP.get(n, ""), "#7F8C8D")
        for n in top15_names
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    shap.summary_plot(
        shap_values[:, top15_idx],
        X_imp.iloc[:, top15_idx],
        feature_names=top15_names,
        show=False,
        max_display=15,
        plot_size=None,
    )

    # Re-colour y-tick labels by criterion
    for tick, color in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(color)
        tick.set_fontweight("bold" if color != "#7F8C8D" else "normal")
        tick.set_fontsize(10)

    ax.set_title(
        "Stage 1 — SHAP Feature Importance\n(colour = Rotterdam criterion)",
        fontsize=FONT_TITLE, fontweight="bold", pad=14
    )
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend patches for Rotterdam criteria
    legend_handles = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in CRITERION_COLOR.items()
    ] + [mpatches.Patch(color="#7F8C8D", label="Other feature")]
    ax.legend(handles=legend_handles, fontsize=9,
              loc="lower right", framealpha=0.85, title="Rotterdam criterion")

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "shap_summary_stage1.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONFUSION MATRIX  (held-out test set)
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix() -> None:
    print("  Generating confusion matrix …")
    X, y = load_pcos(PCOS_DATA_PATH)
    X_imp, _ = impute(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("white")

    # Custom colour map: correct = strong, wrong = light red
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not-PCOS", "PCOS"]
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)

    # Annotate cells with extra stats
    total = len(y_test)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i - 0.25, f"{val / total:.0%} of test set",
                ha="center", va="center", fontsize=8.5, color="grey")

    # Metric annotations below the matrix
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics_txt = (f"Sensitivity (recall): {recall:.2%}   "
                   f"Specificity: {specificity:.2%}   "
                   f"Precision: {precision:.2%}")
    fig.text(0.5, -0.02, metrics_txt, ha="center", fontsize=9, color="#555555")

    ax.set_title(
        "Stage 1 — PCOS Classifier Confusion Matrix\n(held-out 20% test set, n=109)",
        fontsize=FONT_TITLE, fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted label", fontsize=FONT_LABEL)
    ax.set_ylabel("True label", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_LABEL)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "confusion_matrix_stage1.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PATIENT FUNNEL DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def _box(ax, x, y, w, h, color, text, subtext=None,
         text_color="white", radius=0.03, fontsize=12):
    """Draw a rounded rectangle node with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3
    )
    ax.add_patch(box)
    # Shift title up and subtext down proportionally to box height so both
    # stay comfortably inside the box regardless of its size.
    ax.text(x, y + (h * 0.18 if subtext else 0), text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4)
    if subtext:
        ax.text(x, y - h * 0.28, subtext,
                ha="center", va="center", fontsize=fontsize - 2,
                color=text_color, alpha=0.88, zorder=4, linespacing=1.4)


def _arrow(ax, x1, y1, x2, y2, color="#555555", label=None, lx=None, ly=None):
    """Draw a downward arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=2.0, mutation_scale=18),
        zorder=2
    )
    if label:
        lx = lx if lx is not None else (x1 + x2) / 2 + 0.03
        ly = ly if ly is not None else (y1 + y2) / 2
        ax.text(lx, ly, label, fontsize=9, color=color,
                ha="left", va="center", style="italic")


def plot_patient_funnel() -> None:
    print("  Generating patient funnel diagram …")

    # ── patient counts from pipeline results ──────────────────────────────────
    results   = pd.read_csv(RESULTS_PATH)
    n_total   = len(results)
    n_pcos    = int(results["pcos_prediction"].sum())
    n_not     = n_total - n_pcos
    not_pcos  = results[results["pcos_prediction"] == 0]

    n_hypo    = int(not_pcos["flag_hypothyroidism"].sum())
    n_prl     = int(not_pcos["flag_hyperprolactinemia"].sum())
    n_sbp     = int(not_pcos["flag_cushings_hypertension"].sum())
    n_endo    = int((not_pcos["endo_prediction"] == 1).sum())

    any_flag  = (
        not_pcos["flag_hypothyroidism"].astype(bool)  |
        not_pcos["flag_hyperprolactinemia"].astype(bool) |
        not_pcos["flag_cushings_hypertension"].astype(bool) |
        (not_pcos["endo_prediction"] == 1)
    )
    n_none    = int((~any_flag).sum())
    n_overlap = int(any_flag.sum()) - (n_hypo + n_prl + n_sbp + n_endo -
                    int((
                        not_pcos["flag_hypothyroidism"].astype(bool) &
                        not_pcos["flag_hyperprolactinemia"].astype(bool)
                    ).sum()))

    # ── canvas — taller figure for generous vertical breathing room ────────────
    fig, ax = plt.subplots(figsize=(13, 13))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ══ Y-COORDINATE MAP (top → bottom) ══════════════════════════════════════
    # Title            0.975
    # Subtitle         0.952
    # Stage 1 strip    0.915   (height 0.026)
    # All patients     0.866   (height 0.058)
    # XGBoost          0.762   (height 0.052)
    # PCOS / Not-PCOS  0.648   (height 0.056)
    # Stage 2 strip    0.498   (height 0.026)
    # Stage 2 box      0.458   (height 0.052)
    # Dist. line       0.365
    # Bottom nodes     0.228   (height 0.092)
    # Footnote         0.108
    # ═════════════════════════════════════════════════════════════════════════

    # ── title & subtitle ──────────────────────────────────────────────────────
    ax.text(0.5, 0.975, "PCOS Diagnostic Pipeline — Patient Flow",
            ha="center", va="center", fontsize=17,
            fontweight="bold", color=C_DIAG)
    ax.text(0.5, 0.952, f"Total cohort: {n_total} patients",
            ha="center", va="center", fontsize=12, color="#555555")

    # ── stage label strips ────────────────────────────────────────────────────
    SH = 0.026   # strip height
    for label, sy in [
        ("STAGE 1  ·  PCOS Classifier (XGBoost)", 0.915),
        ("STAGE 2  ·  Differential Diagnosis",     0.498),
    ]:
        rect = FancyBboxPatch((0.01, sy - SH / 2), 0.98, SH,
                              boxstyle="round,pad=0.005",
                              facecolor="#ECF0F1", edgecolor="#BDC3C7",
                              linewidth=1, zorder=1)
        ax.add_patch(rect)
        ax.text(0.5, sy, label, ha="center", va="center",
                fontsize=10, color="#2C3E50", fontweight="bold")

    # ── node: all patients ────────────────────────────────────────────────────
    N1H = 0.058
    _box(ax, 0.5, 0.866, 0.30, N1H, C_DIAG,
         f"{n_total} Patients", "Input cohort", fontsize=13)

    # arrow: all patients → XGBoost
    _arrow(ax, 0.5, 0.866 - N1H / 2, 0.5, 0.762 + 0.052 / 2 + 0.010)

    # ── node: XGBoost classifier ──────────────────────────────────────────────
    N2H = 0.052
    _box(ax, 0.5, 0.762, 0.34, N2H, "#2C3E50",
         "XGBoost PCOS Classifier", "ROC-AUC 0.954  ·  Accuracy 91%",
         fontsize=11)

    # branch arrows: XGBoost → PCOS (left) and Not-PCOS (right)
    xgb_bot = 0.762 - N2H / 2
    _arrow(ax, 0.36, xgb_bot, 0.22, 0.648 + 0.056 / 2 + 0.008,
           color=C_PCOS,     label="PCOS",     lx=0.155, ly=0.712)
    _arrow(ax, 0.64, xgb_bot, 0.78, 0.648 + 0.056 / 2 + 0.008,
           color=C_NOT_PCOS, label="Not-PCOS", lx=0.800, ly=0.712)

    # ── outcome nodes: PCOS and Not-PCOS ─────────────────────────────────────
    N3H = 0.056
    _box(ax, 0.22, 0.648, 0.26, N3H, C_PCOS,
         f"{n_pcos} PCOS", f"{n_pcos/n_total:.0%} of cohort", fontsize=12)
    _box(ax, 0.78, 0.648, 0.28, N3H, C_NOT_PCOS,
         f"{n_not} Not-PCOS", f"{n_not/n_total:.0%} of cohort", fontsize=12)

    # PCOS terminal note
    ax.text(0.22, 0.648 - N3H / 2 - 0.016,
            "→ Refer to Endocrinologist",
            ha="center", va="top", fontsize=9, color=C_PCOS,
            style="italic", alpha=0.85)

    # long arrow: Not-PCOS node → Stage 2 box (crosses the Stage 2 strip)
    not_pcos_bot = 0.648 - N3H / 2
    stage2_top   = 0.458 + 0.052 / 2
    _arrow(ax, 0.78, not_pcos_bot, 0.78, stage2_top + 0.010,
           color=C_NOT_PCOS)
    ax.text(0.815, (not_pcos_bot + stage2_top) / 2,
            f"All {n_not}\nNot-PCOS\npatients",
            ha="left", va="center", fontsize=9, color=C_NOT_PCOS,
            style="italic", linespacing=1.5)

    # ── Stage 2 box ───────────────────────────────────────────────────────────
    N4H = 0.052
    _box(ax, 0.78, 0.458, 0.32, N4H, "#2C3E50",
         "Stage 2: Differential Diagnosis",
         "Rule-based flags  +  Endo XGBoost", fontsize=10)

    # connector: Stage 2 box bottom → distributor line
    dist_y    = 0.365
    stage2_bot = 0.458 - N4H / 2
    ax.plot([0.78, 0.78], [stage2_bot, dist_y],
            color="#95A5A6", linewidth=1.5, zorder=1)

    # horizontal distributor line
    NODE_XS = [0.13, 0.38, 0.63, 0.88]
    ax.plot([NODE_XS[0], NODE_XS[-1]], [dist_y, dist_y],
            color="#95A5A6", linewidth=1.5, linestyle="--", zorder=1)

    # ── four outcome nodes ────────────────────────────────────────────────────
    nodes = [
        (NODE_XS[0], C_THYROID, f"{n_hypo} Flagged",  "Hypothyroidism",     "TSH > 4.0 mIU/L"),
        (NODE_XS[1], C_PRL,     f"{n_prl} Flagged",   "Hyperprolactinemia", "PRL > 25 ng/mL"),
        (NODE_XS[2], C_ENDO,    f"{n_endo} Flagged",  "Endo Risk",          "XGBoost ≥ 50%"),
        (NODE_XS[3], C_NONE,    f"{n_none} Patients", "No Flags Raised",    "Routine follow-up"),
    ]
    N5H   = 0.092
    drop_tip = 0.228 + N5H / 2 + 0.012

    for nx, nc, ntitle, nlabel, nsub in nodes:
        # vertical drop from distributor line to node top
        ax.plot([nx, nx], [dist_y, drop_tip], color=nc, linewidth=1.8, zorder=1)
        ax.annotate("", xy=(nx, drop_tip), xytext=(nx, dist_y),
                    arrowprops=dict(arrowstyle="-|>", color=nc,
                                   lw=1.8, mutation_scale=14), zorder=2)
        _box(ax, nx, 0.228, 0.215, N5H, nc,
             ntitle, f"{nlabel}\n{nsub}", fontsize=10)

    # ── footnote ──────────────────────────────────────────────────────────────
    ax.text(0.5, 0.108,
            "Note: flags are non-exclusive — a patient may carry multiple flags simultaneously.",
            ha="center", va="center", fontsize=8.5, color="#7F8C8D",
            style="italic")

    out = os.path.join(PLOTS_DIR, "patient_funnel.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("GENERATING PRESENTATION PLOTS")
    print("=" * 60)
    plot_roc_curve()
    plot_confusion_matrix()
    plot_shap_summary()
    plot_patient_funnel()
    print()
    print("All plots saved to plots/")
    print()
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(PLOTS_DIR, f)) // 1024
            print(f"  {f:<42} {size_kb:>4} KB")


if __name__ == "__main__":
    main()
