"""
Stage 2: Differential Diagnosis for Not-PCOS Cases
====================================================
Applied only to patients who score "Not-PCOS" in Stage 1.

Two components run in parallel:

A) Rule-based flags using published clinical thresholds:
   - TSH > 4.0 mIU/L          → hypothyroidism flag
   - PRL > 25 ng/mL            → hyperprolactinemia flag
   - BP_Systolic > 140 mmHg    → Cushing's / hypertension flag

B) XGBoost endometriosis risk classifier trained on the supplementary
   dataset (10 000 synthetic patients, 6 clinical features).

Output:
  - models/endo_classifier.json           (saved XGBoost model)
  - models/stage2_flag_thresholds.csv     (thresholds reference file)
  - prints classification report and ROC-AUC for endometriosis model
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ENDO_DATA_PATH = "data/(Supplementary_Dataset)_structured_endometriosis_data.csv"
ENDO_MODEL_PATH = "models/endo_classifier.json"
PLOT_DIR = "models"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── clinical thresholds for rule-based flagging ───────────────────────────────
# Sources: ATA guidelines (TSH), Endocrine Society (PRL), JNC-8 (BP)
THRESHOLDS = {
    "hypothyroidism":       {"column": "TSH (mIU/L)",        "operator": ">", "value": 4.0},
    "hyperprolactinemia":   {"column": "PRL(ng/mL)",          "operator": ">", "value": 25.0},
    "cushings_hypertension":{"column": "BP _Systolic (mmHg)", "operator": ">", "value": 140},
}


# ══════════════════════════════════════════════════════════════════════════════
# A. RULE-BASED FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def apply_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flag columns to a PCOS-dataset-formatted DataFrame.
    Each flag is 1 if the patient exceeds the clinical threshold, else 0.
    Input df must contain the original PCOS dataset columns.
    """
    flagged = df.copy()

    flagged["flag_hypothyroidism"] = (
        flagged["TSH (mIU/L)"].fillna(0) > THRESHOLDS["hypothyroidism"]["value"]
    ).astype(int)

    flagged["flag_hyperprolactinemia"] = (
        flagged["PRL(ng/mL)"].fillna(0) > THRESHOLDS["hyperprolactinemia"]["value"]
    ).astype(int)

    # Systolic > 140 alone is not diagnostic of Cushing's, but it is a red-flag
    # that warrants further workup (cortisol, 24-h UFC) when co-presented.
    flagged["flag_cushings_hypertension"] = (
        flagged["BP _Systolic (mmHg)"].fillna(0)
        > THRESHOLDS["cushings_hypertension"]["value"]
    ).astype(int)

    return flagged


def summarise_flags(flagged_df: pd.DataFrame) -> None:
    """Print flag prevalence for a cohort."""
    flag_cols = [c for c in flagged_df.columns if c.startswith("flag_")]
    print("\nRule-based Flag Prevalence (among Not-PCOS subset)")
    print("─" * 50)
    for col in flag_cols:
        n = flagged_df[col].sum()
        pct = 100 * n / len(flagged_df)
        label = col.replace("flag_", "").replace("_", " ").title()
        print(f"  {label:<35} {n:>4} patients  ({pct:.1f}%)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# B. ENDOMETRIOSIS RISK CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def load_endo_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the endometriosis dataset."""
    df = pd.read_csv(path)

    assert df.isnull().sum().sum() == 0, "Unexpected nulls in endo dataset"

    y = df["Diagnosis"]   # 0 = no endometriosis, 1 = endometriosis
    X = df.drop(columns=["Diagnosis"])

    print(f"Endometriosis dataset: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"  Features: {X.columns.tolist()}")
    neg, pos = (y == 0).sum(), (y == 1).sum()
    print(f"  Class distribution — No Endo: {neg} | Endo: {pos}  "
          f"(ratio {neg/pos:.2f}:1)\n")
    return X, y


def train_endo_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Train and evaluate XGBoost endometriosis classifier."""
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale = neg / pos   # mild imbalance; weighting still helps

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"5-fold CV ROC-AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # ── held-out evaluation ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nHeld-out Test Set Results")
    print("─" * 40)
    print(classification_report(y_test, y_pred,
                                target_names=["No Endometriosis", "Endometriosis"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Endo", "Endo"],
        ax=ax, cmap="Greens"
    )
    ax.set_title("Stage 2 – Endometriosis Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "endo_confusion_matrix.png"), dpi=150)
    plt.close()

    # Refit on full data before saving
    model.fit(X, y)

    # ── SHAP feature importance ───────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(7, 4))
    shap.summary_plot(shap_values, X, feature_names=X.columns.tolist(),
                      show=False, max_display=10, plot_type="bar")
    plt.title("SHAP Feature Importance – Endometriosis Classifier")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "endo_shap_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Endo SHAP bar chart saved → {PLOT_DIR}/endo_shap_bar.png")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("STAGE 2 – DIFFERENTIAL DIAGNOSIS (Not-PCOS Patients)")
    print("=" * 60)

    # ── A. save thresholds reference ──────────────────────────────────────────
    thresh_records = [
        {"condition": k, **v} for k, v in THRESHOLDS.items()
    ]
    pd.DataFrame(thresh_records).to_csv(
        "models/stage2_flag_thresholds.csv", index=False
    )
    print("Rule thresholds saved → models/stage2_flag_thresholds.csv")

    # Demonstrate flag logic on the PCOS dataset's Not-PCOS subset
    print("\n[A] Rule-based flags (demo on PCOS dataset Not-PCOS subset)")
    pcos_df = pd.read_excel(
        "data/(Main_Dataset)_PCOS_data_without_infertility.xlsx",
        sheet_name="Full_new"
    )
    pcos_df.columns = pcos_df.columns.str.strip()
    not_pcos = pcos_df[pcos_df["PCOS (Y/N)"] == 0].copy()
    flagged   = apply_rule_flags(not_pcos)
    summarise_flags(flagged)

    # ── B. train endometriosis classifier ─────────────────────────────────────
    print("[B] Training Endometriosis Risk Classifier")
    print("─" * 60)
    X_endo, y_endo = load_endo_data(ENDO_DATA_PATH)
    endo_model     = train_endo_model(X_endo, y_endo)

    endo_model.save_model(ENDO_MODEL_PATH)
    print(f"\nEndometriosis model saved → {ENDO_MODEL_PATH}")

    # Save expected feature names for inference alignment
    pd.Series(X_endo.columns.tolist()).to_csv(
        "models/endo_feature_names.csv", index=False, header=["feature"]
    )
    print("Endo feature names saved → models/endo_feature_names.csv")
    print("\nStage 2 complete.\n")


if __name__ == "__main__":
    main()
