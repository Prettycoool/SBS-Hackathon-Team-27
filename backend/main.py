"""
PCOS Diagnostic API — FastAPI Backend
=======================================
Exposes POST /diagnose which runs the two-stage pipeline:
  Stage 1 — XGBoost PCOS classifier + SHAP top-3 explanations
  Stage 2 — Rule-based flags + endometriosis classifier (Not-PCOS only)

Run from the project root:
  uvicorn backend.main:app --reload --port 8000
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

warnings.filterwarnings("ignore")

# ── Paths (resolved relative to this file so CWD doesn't matter) ──────────────
ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

# ── Globals — populated once at startup ───────────────────────────────────────
pcos_model:      xgb.XGBClassifier   = None
endo_model:      xgb.XGBClassifier   = None
pcos_explainer:  shap.TreeExplainer  = None
medians:         pd.Series           = None
endo_feat_names: list[str]           = None

# ── Column mapping: Pydantic field → stripped training column name ─────────────
FIELD_TO_COL: dict[str, str] = {
    "age":               "Age (yrs)",
    "weight_kg":         "Weight (Kg)",
    "height_cm":         "Height(Cm)",
    "bmi":               "BMI",
    "cycle_length_days": "Cycle length(days)",
    "fsh_miu_ml":        "FSH(mIU/mL)",
    "lh_miu_ml":         "LH(mIU/mL)",
    "fsh_lh_ratio":      "FSH/LH",
    "tsh_miu_l":         "TSH (mIU/L)",
    "amh_ng_ml":         "AMH(ng/mL)",
    "prl_ng_ml":         "PRL(ng/mL)",
    "weight_gain":       "Weight gain(Y/N)",
    "hair_growth":       "hair growth(Y/N)",
    "skin_darkening":    "Skin darkening (Y/N)",
    "hair_loss":         "Hair loss(Y/N)",
    "pimples":           "Pimples(Y/N)",
    "bp_systolic":       "BP _Systolic (mmHg)",
    "bp_diastolic":      "BP _Diastolic (mmHg)",
    "follicle_no_l":     "Follicle No. (L)",
    "follicle_no_r":     "Follicle No. (R)",
    "avg_f_size_l_mm":   "Avg. F size (L) (mm)",
    "avg_f_size_r_mm":   "Avg. F size (R) (mm)",
    "progesterone_ng_ml": "PRG(ng/mL)",
}

DISPLAY_NAMES: dict[str, str] = {
    "Age (yrs)":             "Age",
    "Weight (Kg)":           "Weight",
    "Height(Cm)":            "Height",
    "BMI":                   "BMI",
    "Blood Group":           "Blood Group",
    "Pulse rate(bpm)":       "Pulse Rate",
    "RR (breaths/min)":      "Respiratory Rate",
    "Hb(g/dl)":              "Hemoglobin",
    "Cycle(R/I)":            "Cycle Regularity",
    "Cycle length(days)":    "Cycle Length",
    "Marraige Status (Yrs)": "Years Married",
    "Pregnant(Y/N)":         "Pregnant",
    "No. of abortions":      "No. of Abortions",
    "I   beta-HCG(mIU/mL)": "β-HCG (Test 1)",
    "II    beta-HCG(mIU/mL)":"β-HCG (Test 2)",
    "FSH(mIU/mL)":           "FSH",
    "LH(mIU/mL)":            "LH",
    "FSH/LH":                "FSH / LH Ratio",
    "Hip(inch)":             "Hip Circumference",
    "Waist(inch)":           "Waist Circumference",
    "Waist:Hip Ratio":       "Waist:Hip Ratio",
    "TSH (mIU/L)":           "TSH",
    "AMH(ng/mL)":            "AMH",
    "PRL(ng/mL)":            "Prolactin (PRL)",
    "Vit D3 (ng/mL)":        "Vitamin D3",
    "PRG(ng/mL)":            "Progesterone",
    "RBS(mg/dl)":            "Random Blood Sugar",
    "Weight gain(Y/N)":      "Weight Gain",
    "hair growth(Y/N)":      "Excessive Hair Growth",
    "Skin darkening (Y/N)":  "Skin Darkening",
    "Hair loss(Y/N)":        "Hair Loss",
    "Pimples(Y/N)":          "Acne / Pimples",
    "Fast food (Y/N)":       "Fast Food Consumption",
    "Reg.Exercise(Y/N)":     "Regular Exercise",
    "BP _Systolic (mmHg)":   "Systolic BP",
    "BP _Diastolic (mmHg)":  "Diastolic BP",
    "Follicle No. (L)":      "Follicles — Left Ovary",
    "Follicle No. (R)":      "Follicles — Right Ovary",
    "Avg. F size (L) (mm)":  "Avg. Follicle Size (L)",
    "Avg. F size (R) (mm)":  "Avg. Follicle Size (R)",
    "Endometrium (mm)":      "Endometrial Thickness",
}

# ── PCOS classification threshold ────────────────────────────────────────────
# Optimised to 0.46 via threshold sensitivity analysis (max F1 on 5-fold CV).
# Raises sensitivity +2.3% vs default 0.50 with only −0.3% specificity cost.
PCOS_THRESHOLD = 0.46

# ── Clinical thresholds (Stage 2) ────────────────────────────────────────────
TSH_THRESHOLD  = 4.0    # mIU/L  — hypothyroidism
PRL_THRESHOLD  = 25.0   # ng/mL  — hyperprolactinemia
SBP_THRESHOLD  = 140    # mmHg   — Cushing's / hypertension
ENDO_THRESHOLD = 0.50   # XGBoost probability cutoff


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PatientData(BaseModel):
    # Demographics
    age:              Optional[float] = Field(None, ge=10, le=80)
    weight_kg:        Optional[float] = Field(None, ge=25, le=180)
    height_cm:        Optional[float] = Field(None, ge=120, le=210)
    bmi:              Optional[float] = Field(None, ge=10, le=55)
    # Menstrual
    cycle_regular:    Optional[bool]  = Field(None, description="True = Regular, False = Irregular")
    cycle_length_days:Optional[int]   = Field(None, ge=1, le=30)
    # Hormonal
    fsh_miu_ml:       Optional[float] = Field(None, ge=0, le=60)
    lh_miu_ml:        Optional[float] = Field(None, ge=0, le=60)
    fsh_lh_ratio:     Optional[float] = Field(None, ge=0, le=25)
    amh_ng_ml:        Optional[float] = Field(None, ge=0, le=20)
    tsh_miu_l:        Optional[float] = Field(None, ge=0, le=20)
    prl_ng_ml:           Optional[float] = Field(None, ge=0, le=200)
    progesterone_ng_ml:  Optional[float] = Field(None, ge=0, le=40)
    # Ultrasound
    follicle_no_l:    Optional[int]   = Field(None, ge=0, le=30)
    follicle_no_r:    Optional[int]   = Field(None, ge=0, le=30)
    avg_f_size_l_mm:  Optional[float] = Field(None, ge=0, le=40)
    avg_f_size_r_mm:  Optional[float] = Field(None, ge=0, le=40)
    # Symptoms
    weight_gain:      Optional[bool]  = None
    hair_growth:      Optional[bool]  = None
    skin_darkening:   Optional[bool]  = None
    hair_loss:        Optional[bool]  = None
    pimples:             Optional[bool]  = None
    galactorrhea:        Optional[bool]  = None
    chronic_pelvic_pain: Optional[int]   = Field(None, ge=0, le=10)
    # Vitals
    bp_systolic:      Optional[int]   = Field(None, ge=60, le=220)
    bp_diastolic:     Optional[int]   = Field(None, ge=40, le=140)


class ShapEntry(BaseModel):
    feature_name:  str
    display_name:  str
    feature_value: float
    shap_value:    float
    direction:     str   # "increases_risk" | "decreases_risk"


class DifferentialFlags(BaseModel):
    hypothyroidism:            bool
    hyperprolactinemia:        bool
    cushings_hypertension:     bool
    endometriosis_risk:        bool
    endometriosis_probability: Optional[float]
    tsh_value:                 float
    prl_value:                 float
    sbp_value:                 float
    galactorrhea_strengthened: bool = False
    pelvic_pain_note:          str  = ""


class PhenotypeInfo(BaseModel):
    code:        str   # "A", "B", "C", "D", or "Atypical"
    label:       str   # e.g. "Phenotype A"
    description: str   # one-line clinical description


class DiagnosisResponse(BaseModel):
    pcos_probability:    float
    pcos_prediction:     bool
    primary_diagnosis:   str
    pcos_phenotype:      Optional[PhenotypeInfo] = None
    rotterdam_gap_flag:  bool = False
    age_flag:            str = ""   # "peak_gap" | "adolescent" | ""
    differential_flags:  Optional[DifferentialFlags] = None
    shap_explanations:   list[ShapEntry]


# ══════════════════════════════════════════════════════════════════════════════
# APP + STARTUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="PCOS Diagnostic API",
    description="Two-stage PCOS pipeline: XGBoost classifier + differential diagnosis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_models() -> None:
    global pcos_model, endo_model, pcos_explainer, medians, endo_feat_names

    pcos_model = xgb.XGBClassifier()
    pcos_model.load_model(str(MODELS_DIR / "pcos_classifier.json"))

    endo_model = xgb.XGBClassifier()
    endo_model.load_model(str(MODELS_DIR / "endo_classifier.json"))

    # TreeExplainer is expensive; cache it globally so each request is fast
    pcos_explainer = shap.TreeExplainer(pcos_model)

    medians = pd.read_csv(
        str(MODELS_DIR / "pcos_feature_medians.csv"), index_col=0
    )["median"]
    endo_feat_names = pd.read_csv(
        str(MODELS_DIR / "endo_feature_names.csv")
    )["feature"].tolist()

    print(f"Models loaded — {len(medians)} PCOS features, {len(endo_feat_names)} endo features.")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(patient: PatientData) -> pd.DataFrame:
    """Merge user-supplied fields with training medians for any missing columns."""
    row = medians.to_dict()

    # Map named API fields to their training column names
    for api_field, col in FIELD_TO_COL.items():
        val = getattr(patient, api_field, None)
        if val is not None:
            row[col] = int(val) if isinstance(val, bool) else float(val)

    # Cycle encoding: Regular→0, Irregular→1 (matches training preprocessing)
    if patient.cycle_regular is not None:
        row["Cycle(R/I)"] = 0 if patient.cycle_regular else 1

    # Auto-compute FSH/LH ratio when both hormones are provided
    if (patient.fsh_lh_ratio is None
            and patient.fsh_miu_ml is not None
            and patient.lh_miu_ml is not None
            and patient.lh_miu_ml > 0):
        row["FSH/LH"] = patient.fsh_miu_ml / patient.lh_miu_ml

    # Auto-compute BMI from weight/height if BMI not explicitly supplied
    if (patient.bmi is None
            and patient.weight_kg is not None
            and patient.height_cm is not None
            and patient.height_cm > 0):
        row["BMI"] = patient.weight_kg / (patient.height_cm / 100) ** 2

    # Return a single-row DataFrame in the exact column order used during training
    return pd.DataFrame([row])[medians.index]


def get_shap_top3(X_row: pd.DataFrame) -> list[ShapEntry]:
    """Return the 3 features with the largest absolute SHAP contribution."""
    shap_vals = pcos_explainer.shap_values(X_row)[0]
    top3_idx  = np.argsort(np.abs(shap_vals))[-3:][::-1]

    return [
        ShapEntry(
            feature_name  = medians.index[i],
            display_name  = DISPLAY_NAMES.get(medians.index[i], medians.index[i]),
            feature_value = round(float(X_row.iloc[0, i]), 3),
            shap_value    = round(float(shap_vals[i]), 4),
            direction     = "increases_risk" if shap_vals[i] > 0 else "decreases_risk",
        )
        for i in top3_idx
    ]


def build_endo_features(X: pd.DataFrame) -> pd.DataFrame:
    """Map PCOS feature vector to the endometriosis model's 6-feature input space."""
    row = {
        "Age":                       float(X["Age (yrs)"].iloc[0]),
        "Menstrual_Irregularity":    int(X["Cycle(R/I)"].iloc[0]),
        "Chronic_Pain_Level":        5.0,   # not available in PCOS dataset; use mean
        "Hormone_Level_Abnormality": int(
            float(X["FSH/LH"].iloc[0]) < 1.0
            or float(X["AMH(ng/mL)"].iloc[0]) > 6.5
        ),
        "Infertility": int(
            float(X["Pregnant(Y/N)"].iloc[0]) == 0
            and float(X["No. of abortions"].iloc[0]) > 0
        ),
        "BMI": float(X["BMI"].iloc[0]),
    }
    return pd.DataFrame([row])[endo_feat_names]


def classify_phenotype(
    irregular_cycle: bool,
    hyperandrogenism: bool,
    polycystic_ovaries: bool,
) -> PhenotypeInfo:
    """
    Rotterdam 2003 phenotype classification.
    Requires ≥2 of 3 criteria for PCOS; returns which combination is present.
    """
    if irregular_cycle and hyperandrogenism and polycystic_ovaries:
        return PhenotypeInfo(
            code="A", label="Phenotype A",
            description="Irregular cycle + Hyperandrogenism + Polycystic ovaries",
        )
    if irregular_cycle and hyperandrogenism:
        return PhenotypeInfo(
            code="B", label="Phenotype B",
            description="Irregular cycle + Hyperandrogenism (no polycystic ovaries)",
        )
    if hyperandrogenism and polycystic_ovaries:
        return PhenotypeInfo(
            code="C", label="Phenotype C",
            description="Hyperandrogenism + Polycystic ovaries (regular cycle)",
        )
    if irregular_cycle and polycystic_ovaries:
        return PhenotypeInfo(
            code="D", label="Phenotype D",
            description="Irregular cycle + Polycystic ovaries (no hyperandrogenism)",
        )
    return PhenotypeInfo(
        code="Atypical", label="Atypical Presentation",
        description="Fewer than 2 Rotterdam criteria met — ML-flagged presentation",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": pcos_model is not None}


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(patient: PatientData):
    if pcos_model is None:
        raise HTTPException(503, detail="Models not yet loaded — wait a moment and retry.")

    X = build_feature_vector(patient)

    # ── Stage 1: PCOS prediction ──────────────────────────────────────────────
    pcos_proba = float(pcos_model.predict_proba(X)[0, 1])
    pcos_pred  = pcos_proba >= PCOS_THRESHOLD

    primary = (
        f"PCOS detected — confidence {pcos_proba:.0%}"
        if pcos_pred
        else f"PCOS unlikely — confidence {(1 - pcos_proba):.0%}"
    )

    shap_top3 = get_shap_top3(X)

    # ── Rotterdam criteria (used for both phenotype and gap flag) ─────────────
    irregular_cycle    = patient.cycle_regular is False
    hyperandrogenism   = bool(patient.hair_growth or patient.skin_darkening or patient.pimples)
    polycystic_ovaries = bool(
        (patient.follicle_no_l or 0) > 12 or
        (patient.follicle_no_r or 0) > 12
    )
    rotterdam_met = sum([irregular_cycle, hyperandrogenism, polycystic_ovaries])

    # ── Phenotype subtyping (PCOS-positive only) ──────────────────────────────
    phenotype = classify_phenotype(irregular_cycle, hyperandrogenism, polycystic_ovaries) \
        if pcos_pred else None

    # ── Rotterdam gap flag: Not-PCOS with exactly 1 criterion met ─────────────
    rotterdam_gap = (not pcos_pred) and (rotterdam_met == 1)

    # ── Age-aware diagnostic flag ─────────────────────────────────────────────
    age = patient.age or 0
    if 26 <= age <= 30:
        age_flag = "peak_gap"
    elif age < 18:
        age_flag = "adolescent"
    else:
        age_flag = ""

    # ── Stage 2: comorbidity / differential flags (ALL patients) ────────────────
    tsh = float(X["TSH (mIU/L)"].iloc[0])
    prl = float(X["PRL(ng/mL)"].iloc[0])
    sbp = float(X["BP _Systolic (mmHg)"].iloc[0])

    X_endo = build_endo_features(X)
    if patient.chronic_pelvic_pain is not None:
        X_endo["Chronic_Pain_Level"] = float(patient.chronic_pelvic_pain)
    endo_prob = float(endo_model.predict_proba(X_endo)[0, 1])

    # Galactorrhea strengthens hyperprolactinemia flag
    galactorrhea_strengthened = bool(patient.galactorrhea) and prl > PRL_THRESHOLD

    # Pelvic pain elevates endometriosis suspicion
    pelvic_pain_note = (
        "Chronic pelvic pain reported — endometriosis risk elevated."
        if (patient.chronic_pelvic_pain or 0) >= 5
        else ""
    )

    # Annotate primary_diagnosis string only for Not-PCOS cases
    if not pcos_pred:
        flags_raised = []
        if tsh > TSH_THRESHOLD:         flags_raised.append("hypothyroidism")
        if prl > PRL_THRESHOLD:         flags_raised.append("hyperprolactinemia")
        if sbp > SBP_THRESHOLD:         flags_raised.append("hypertension / Cushing's")
        if endo_prob >= ENDO_THRESHOLD: flags_raised.append(f"endometriosis risk ({endo_prob:.0%})")
        if flags_raised:
            primary += " — flags: " + ", ".join(flags_raised)

    diff_flags = DifferentialFlags(
        hypothyroidism            = tsh > TSH_THRESHOLD,
        hyperprolactinemia        = prl > PRL_THRESHOLD or galactorrhea_strengthened,
        cushings_hypertension     = sbp > SBP_THRESHOLD,
        endometriosis_risk        = endo_prob >= ENDO_THRESHOLD,
        endometriosis_probability = round(endo_prob, 3),
        tsh_value                 = round(tsh, 2),
        prl_value                 = round(prl, 2),
        sbp_value                 = round(sbp, 1),
        galactorrhea_strengthened = galactorrhea_strengthened,
        pelvic_pain_note          = pelvic_pain_note,
    )

    return DiagnosisResponse(
        pcos_probability   = round(pcos_proba, 4),
        pcos_prediction    = pcos_pred,
        primary_diagnosis  = primary,
        pcos_phenotype     = phenotype,
        rotterdam_gap_flag = rotterdam_gap,
        age_flag           = age_flag,
        differential_flags = diff_flags,
        shap_explanations  = shap_top3,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOCTOR'S NOTE OCR  —  /parse-note
# ══════════════════════════════════════════════════════════════════════════════

_NOTE_SYSTEM_PROMPT = """\
You are a clinical data extractor. Read the doctor's note and return ONLY a \
valid JSON object with these exact keys (use null for any field not found):

{
  "age": null, "weight": null, "height": null, "bmi": null,
  "bp_systolic": null, "bp_diastolic": null,
  "fsh": null, "lh": null, "amh": null, "tsh": null,
  "prolactin": null, "progesterone": null,
  "cycle_regular": null, "cycle_length": null,
  "follicle_no_l": null, "follicle_no_r": null,
  "avg_f_size_l": null, "avg_f_size_r": null,
  "endometrium": null,
  "weight_gain": null, "hair_growth": null, "skin_darkening": null,
  "hair_loss": null, "pimples": null, "galactorrhea": null,
  "chronic_pelvic_pain": null,
  "rbs": null, "waist_hip_ratio": null, "hb": null
}

Rules:
- cycle_regular: true if described as regular, false if irregular, null if not mentioned
- Symptom booleans (weight_gain, hair_growth, skin_darkening, hair_loss, pimples, galactorrhea): \
true if present/reported, false if explicitly denied, null if not mentioned
- chronic_pelvic_pain: integer 0-10; estimate from words if no score given \
(none=0, mild=2, moderate=5, severe=8), null if not mentioned
- All numeric values must be numbers, not strings
- Return ONLY the JSON object with no extra text or markdown\
"""


class NoteRequest(BaseModel):
    note: str


@app.post("/parse-note")
async def parse_note(req: NoteRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, detail="GROQ_API_KEY not set in backend/.env")

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": _NOTE_SYSTEM_PROMPT},
                {"role": "user",   "content": req.note},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(502, detail=f"Groq API error: {e}")

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: pull the first {...} block from the response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise HTTPException(500, detail="Model returned a non-JSON response.")

    return result
