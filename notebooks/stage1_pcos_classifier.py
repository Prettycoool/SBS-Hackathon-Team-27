"""
Stage 1: PCOS Binary Classifier (XGBoost + SHAP)
=================================================
Trains an XGBoost model to classify PCOS vs Not-PCOS on the main dataset,
then explains predictions using SHAP values aligned with Rotterdam criteria.

Rotterdam criteria (2+ of 3 required for PCOS diagnosis):
  1. Oligo/anovulation       -> Cycle(R/I), Cycle length(days)
  2. Clinical/biochemical    -> hair growth, Skin darkening, Pimples,
     hyperandrogenism           LH(mIU/mL), FSH/LH, AMH(ng/mL)
  3. Polycystic ovaries      -> Follicle No. (L/R), Avg. F size (L/R)

Output:
  - models/pcos_classifier.json  (saved XGBoost model)
  - prints classification report, ROC-AUC, and SHAP summary
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/(Main_Dataset)_PCOS_data_without_infertility.xlsx"
SHEET_NAME  = "Full_new"
MODEL_PATH  = "models/pcos_classifier.json"
PLOT_DIR    = "models"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Rotterdam-aligned feature groups (for SHAP annotation) ─────────────────────
ROTTERDAM_FEATURES = {
    "Anovulation":        ["Cycle(R/I)", "Cycle length(days)"],
    "Hyperandrogenism":   ["hair growth(Y/N)", "Skin darkening (Y/N)",
                           "Pimples(Y/N)", "LH(mIU/mL)", "FSH/LH", "AMH(ng/mL)"],
    "Polycystic ovaries": ["Follicle No. (L)", "Follicle No. (R)",
                           "Avg. F size (L) (mm)", "Avg. F size (R) (mm)"],
}
# flat list used to highlight SHAP bars
ROTTERDAM_FLAT = [f for group in ROTTERDAM_FEATURES.values() for f in group]

# ── medical thresholds used in Stage 2 (kept here for reference) ───────────────
THYROID_TSH_HIGH   = 4.0    # mIU/L  -> hypothyroidism flag
HYPERPRL_PRL_HIGH  = 25.0   # ng/mL  -> hyperprolactinemia flag
CUSHINGS_SBP_HIGH  = 140    # mmHg   -> Cushing's / hypertension flag


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path: str, sheet: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y after cleaning."""
    df = pd.read_excel(path, sheet_name=sheet)

    # ── drop pure identifiers ─────────────────────────────────────────────────
    df.drop(columns=["Sl. No", "Patient File No."], inplace=True)

    # ── Cycle(R/I): Regular=2 → 0, Irregular=4 → 1, anything else → 1 ────────
    # Value 5 appears once (likely data-entry error); treat as irregular.
    df["Cycle(R/I)"] = df["Cycle(R/I)"].map(lambda v: 0 if v == 2 else 1)

    # ── coerce object columns that contain stray non-numeric strings ──────────
    # AMH(ng/mL) has one 'a' entry; II beta-HCG has one '1.99.' entry.
    for col in ["AMH(ng/mL)", "II    beta-HCG(mIU/mL)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── strip leading/trailing spaces from column names ───────────────────────
    df.columns = df.columns.str.strip()

    # ── separate target ───────────────────────────────────────────────────────
    y = df.pop("PCOS (Y/N)")
    X = df

    print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Class distribution — PCOS: {y.sum()} | Not-PCOS: {(y==0).sum()}")
    print(f"Missing values per column (non-zero):\n"
          f"{X.isnull().sum()[X.isnull().sum() > 0]}\n")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def build_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Train XGBoost with median imputation; return fitted model."""
    # Class imbalance: ~2:1 Not-PCOS to PCOS → boost minority weight
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,   # compensates class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Impute missing values with column median before training
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ── 5-fold cross-validation for unbiased AUC estimate ────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X_imp, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"5-fold CV ROC-AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # ── held-out test split for final evaluation ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nHeld-out Test Set Results")
    print("─" * 40)
    print(classification_report(y_test, y_pred, target_names=["Not-PCOS", "PCOS"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion matrix saved to disk
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Not-PCOS", "PCOS"], ax=ax, cmap="Blues"
    )
    ax.set_title("Stage 1 – PCOS Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "pcos_confusion_matrix.png"), dpi=150)
    plt.close()

    # Refit on full dataset before saving (maximises training data)
    model.fit(X_imp, y)
    return model, imputer


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHAP EXPLAINABILITY (Rotterdam-aligned)
# ══════════════════════════════════════════════════════════════════════════════

def explain_with_shap(model: xgb.XGBClassifier,
                      X_imp: pd.DataFrame,
                      feature_names: list[str]) -> None:
    """Compute SHAP values and produce a summary plot highlighting Rotterdam features."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    # ── global feature importance plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_imp, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Summary – Stage 1 PCOS Classifier", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "pcos_shap_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved → {PLOT_DIR}/pcos_shap_summary.png")

    # ── Rotterdam feature SHAP importance (mean |SHAP|) ───────────────────────
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    print("\nRotterdam Criteria – Mean |SHAP| Importance")
    print("─" * 50)
    for criterion, feats in ROTTERDAM_FEATURES.items():
        available = [f for f in feats if f in mean_shap.index]
        if not available:
            continue
        print(f"\n  [{criterion}]")
        for f in sorted(available, key=lambda x: mean_shap.get(x, 0), reverse=True):
            bar = "█" * int(mean_shap.get(f, 0) * 50)
            print(f"    {f:<35} {mean_shap.get(f, 0):.4f}  {bar}")

    # ── bar chart for Rotterdam features only ─────────────────────────────────
    rotterdam_imp = mean_shap[[f for f in ROTTERDAM_FLAT
                                if f in mean_shap.index]].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e07070" if v > rotterdam_imp.median() else "#7090d0"
              for v in rotterdam_imp.values]
    rotterdam_imp.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Rotterdam Criteria Feature Importance (SHAP)")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "pcos_rotterdam_shap.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Rotterdam SHAP bar chart saved → {PLOT_DIR}/pcos_rotterdam_shap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 1 – PCOS BINARY CLASSIFIER")
    print("=" * 60)

    X, y = load_and_preprocess(DATA_PATH, SHEET_NAME)
    model, imputer = build_model(X, y)

    # Impute full dataset for SHAP (model was refit on full data)
    X_imp = pd.DataFrame(
        imputer.transform(X), columns=X.columns
    )

    explain_with_shap(model, X_imp, X.columns.tolist())

    # ── save model ────────────────────────────────────────────────────────────
    model.save_model(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── also export imputer medians so Stage 2 / inference can reuse them ─────
    medians = pd.Series(imputer.statistics_, index=X.columns)
    medians.to_csv("models/pcos_feature_medians.csv", header=["median"])
    print("Feature medians saved → models/pcos_feature_medians.csv")

    print("\nStage 1 complete.\n")


if __name__ == "__main__":
    main()
