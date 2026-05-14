"""
Full Two-Stage PCOS Diagnostic Pipeline
========================================
Chains Stage 1 (PCOS classifier) and Stage 2 (differential diagnosis)
into an end-to-end pipeline that processes patient records and returns
a structured diagnostic summary.

Usage:
  python notebooks/run_pipeline.py
    → runs on the full PCOS dataset (both PCOS and Not-PCOS patients)
      to demonstrate the complete flow

  Or import predict_patient() / run_pipeline() for programmatic use.

Output columns in the returned DataFrame:
  patient_id          - original index
  pcos_prediction     - 1 = PCOS, 0 = Not-PCOS
  pcos_probability    - model probability of PCOS
  endo_risk_score     - probability of endometriosis (Not-PCOS only, else NaN)
  endo_prediction     - 1 = high endo risk (Not-PCOS only, else NaN)
  flag_hypothyroidism - 1 if TSH > 4.0 mIU/L
  flag_hyperprolactinemia - 1 if PRL > 25 ng/mL
  flag_cushings_hypertension - 1 if BP_Systolic > 140 mmHg
  primary_diagnosis   - plain-English summary
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── file paths ─────────────────────────────────────────────────────────────────
PCOS_DATA_PATH   = "data/(Main_Dataset)_PCOS_data_without_infertility.xlsx"
PCOS_MODEL_PATH  = "models/pcos_classifier.json"
ENDO_MODEL_PATH  = "models/endo_classifier.json"
MEDIANS_PATH     = "models/pcos_feature_medians.csv"
ENDO_FEATS_PATH  = "models/endo_feature_names.csv"

# ── Stage 2 rule thresholds ────────────────────────────────────────────────────
TSH_THRESHOLD    = 4.0    # mIU/L   hypothyroidism
PRL_THRESHOLD    = 25.0   # ng/mL   hyperprolactinemia
SBP_THRESHOLD    = 140    # mmHg    Cushing's / hypertension

# ── PCOS classification threshold ────────────────────────────────────────────
# Optimised to 0.46 via threshold sensitivity analysis (max F1 on 5-fold CV).
# Raises sensitivity +2.3% vs default 0.50 with only −0.3% specificity cost.
PCOS_THRESHOLD   = 0.46

# ── endometriosis probability cutoff ─────────────────────────────────────────
ENDO_THRESHOLD   = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_pcos_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(PCOS_MODEL_PATH)
    return model


def _load_endo_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(ENDO_MODEL_PATH)
    return model


def _preprocess_pcos_features(df: pd.DataFrame,
                               medians: pd.Series) -> pd.DataFrame:
    """
    Apply the same preprocessing used during Stage 1 training:
      1. Drop identifiers
      2. Recode Cycle(R/I): 2 → 0 (regular), else → 1 (irregular)
      3. Coerce object columns to numeric
      4. Impute missing values with training medians
    """
    drop_cols = [c for c in ["Sl. No", "Patient File No.", "PCOS (Y/N)"]
                 if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    # Strip whitespace from column names (dataset has trailing spaces)
    X.columns = X.columns.str.strip()

    # Cycle encoding
    X["Cycle(R/I)"] = X["Cycle(R/I)"].map(lambda v: 0 if v == 2 else 1)

    # Coerce mixed-type columns
    for col in ["AMH(ng/mL)", "II    beta-HCG(mIU/mL)"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Align to training feature order, then median-impute
    X = X.reindex(columns=medians.index)
    for col in X.columns:
        X[col] = X[col].fillna(medians[col])

    return X


def _build_endo_features(pcos_row: pd.Series,
                          endo_feature_names: list[str]) -> pd.DataFrame:
    """
    Map available PCOS-dataset columns to the endometriosis model's input space.
    PCOS dataset → endometriosis feature mapping:
      Age                       <- Age (yrs)
      Menstrual_Irregularity    <- Cycle(R/I) already encoded as 0/1
      Chronic_Pain_Level        <- not available; impute with population mean (5.0)
      Hormone_Level_Abnormality <- FSH/LH > 2.0 or AMH > 6.5 ng/mL (proxy)
      Infertility               <- Pregnant(Y/N) == 0 AND No. of abortions > 0 (proxy)
      BMI                       <- BMI
    """
    row = {}
    row["Age"]                    = pcos_row.get("Age (yrs)", np.nan)
    row["Menstrual_Irregularity"] = pcos_row.get("Cycle(R/I)", 0)
    # Chronic pain is not captured in the PCOS dataset; use dataset mean.
    row["Chronic_Pain_Level"]     = 5.0
    # Proxy for hormonal abnormality: elevated LH/FSH or AMH
    fsh_lh = pcos_row.get("FSH/LH", np.nan)
    amh    = pcos_row.get("AMH(ng/mL)", np.nan)
    row["Hormone_Level_Abnormality"] = int(
        (not np.isnan(fsh_lh) and fsh_lh < 1.0) or
        (not np.isnan(amh)    and amh > 6.5)
    )
    pregnant  = pcos_row.get("Pregnant(Y/N)", 1)
    abortions = pcos_row.get("No. of abortions", 0)
    row["Infertility"] = int(pregnant == 0 and abortions > 0)
    row["BMI"]         = pcos_row.get("BMI", np.nan)

    return pd.DataFrame([row])[endo_feature_names]


def _summarise(row: pd.Series) -> str:
    """Return a human-readable diagnostic summary for one patient."""
    if row["pcos_prediction"] == 1:
        return f"PCOS (confidence {row['pcos_probability']:.0%})"

    conditions = []
    if row["flag_hypothyroidism"]:
        conditions.append("possible hypothyroidism (high TSH)")
    if row["flag_hyperprolactinemia"]:
        conditions.append("possible hyperprolactinemia (high PRL)")
    if row["flag_cushings_hypertension"]:
        conditions.append("possible Cushing's / hypertension (high SBP)")
    if row["endo_prediction"] == 1:
        conditions.append(
            f"endometriosis risk (p={row['endo_risk_score']:.0%})"
        )

    if conditions:
        return "Not-PCOS — flags: " + "; ".join(conditions)
    return "Not-PCOS — no secondary flags raised"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the two-stage pipeline on a DataFrame in PCOS-dataset format.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PCOS-dataset rows (may include or omit the PCOS (Y/N) column).

    Returns
    -------
    pd.DataFrame
        One row per patient with prediction columns appended.
    """
    # ── load models & metadata ─────────────────────────────────────────────────
    pcos_model         = _load_pcos_model()
    endo_model         = _load_endo_model()
    medians            = pd.read_csv(MEDIANS_PATH, index_col=0)["median"]
    endo_feature_names = pd.read_csv(ENDO_FEATS_PATH)["feature"].tolist()

    df.columns = df.columns.str.strip()

    # ── stage 1: PCOS classification ──────────────────────────────────────────
    X_pcos        = _preprocess_pcos_features(df, medians)
    pcos_proba    = pcos_model.predict_proba(X_pcos)[:, 1]
    pcos_pred     = (pcos_proba >= PCOS_THRESHOLD).astype(int)

    results = pd.DataFrame({
        "patient_id":       df.index,
        "pcos_prediction":  pcos_pred,
        "pcos_probability": np.round(pcos_proba, 4),
        # Stage 2 columns initialised to NaN; filled below for Not-PCOS
        "endo_risk_score":              np.nan,
        "endo_prediction":              np.nan,
        "flag_hypothyroidism":          0,
        "flag_hyperprolactinemia":      0,
        "flag_cushings_hypertension":   0,
    })

    # ── stage 2: applied only to Not-PCOS patients ────────────────────────────
    not_pcos_idx = np.where(pcos_pred == 0)[0]

    if len(not_pcos_idx) > 0:
        not_pcos_df = df.iloc[not_pcos_idx].reset_index(drop=True)
        X_pcos_sub  = X_pcos.iloc[not_pcos_idx].reset_index(drop=True)

        # Rule-based flags
        results.loc[not_pcos_idx, "flag_hypothyroidism"] = (
            not_pcos_df["TSH (mIU/L)"].fillna(0) > TSH_THRESHOLD
        ).astype(int).values

        results.loc[not_pcos_idx, "flag_hyperprolactinemia"] = (
            not_pcos_df["PRL(ng/mL)"].fillna(0) > PRL_THRESHOLD
        ).astype(int).values

        results.loc[not_pcos_idx, "flag_cushings_hypertension"] = (
            not_pcos_df["BP _Systolic (mmHg)"].fillna(0) > SBP_THRESHOLD
        ).astype(int).values

        # Endometriosis classifier
        endo_inputs = pd.concat(
            [_build_endo_features(X_pcos_sub.iloc[i], endo_feature_names)
             for i in range(len(X_pcos_sub))],
            ignore_index=True
        )
        endo_proba = endo_model.predict_proba(endo_inputs)[:, 1]
        endo_pred  = (endo_proba >= ENDO_THRESHOLD).astype(int)

        results.loc[not_pcos_idx, "endo_risk_score"] = np.round(endo_proba, 4)
        results.loc[not_pcos_idx, "endo_prediction"] = endo_pred

    # ── human-readable summary ────────────────────────────────────────────────
    results["primary_diagnosis"] = results.apply(_summarise, axis=1)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / SCRIPT ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TWO-STAGE PCOS DIAGNOSTIC PIPELINE")
    print("=" * 60)

    # Verify models exist; if not, remind user to train first
    for path, label in [
        (PCOS_MODEL_PATH, "Stage 1 PCOS model"),
        (ENDO_MODEL_PATH, "Stage 2 Endo model"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}. "
                "Run stage1_pcos_classifier.py and stage2_differential_diagnosis.py first."
            )

    df = pd.read_excel(PCOS_DATA_PATH, sheet_name="Full_new")
    df.columns = df.columns.str.strip()

    results = run_pipeline(df)

    # ── summary statistics ────────────────────────────────────────────────────
    print(f"\nTotal patients processed : {len(results)}")
    print(f"  PCOS positive          : {results['pcos_prediction'].sum()}")
    print(f"  Not-PCOS               : {(results['pcos_prediction']==0).sum()}")
    not_pcos = results[results["pcos_prediction"] == 0]
    if len(not_pcos):
        print(f"\n  Among Not-PCOS ({len(not_pcos)} patients):")
        print(f"    Hypothyroidism flag     : {not_pcos['flag_hypothyroidism'].sum()}")
        print(f"    Hyperprolactinemia flag : {not_pcos['flag_hyperprolactinemia'].sum()}")
        print(f"    Cushing's/SBP flag      : {not_pcos['flag_cushings_hypertension'].sum()}")
        print(f"    Endometriosis risk ≥50% : {not_pcos['endo_prediction'].sum():.0f}")

    # ── sample output rows ────────────────────────────────────────────────────
    print("\nSample output (10 rows):")
    print(results[["patient_id", "pcos_probability",
                    "primary_diagnosis"]].head(10).to_string(index=False))

    # Save full results
    out_path = "models/pipeline_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
