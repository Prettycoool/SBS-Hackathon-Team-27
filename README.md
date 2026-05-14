# PCOS Diagnostic ML Pipeline

A two-stage machine learning pipeline for PCOS (Polycystic Ovary Syndrome) diagnosis and differential diagnosis of related conditions.

---

## The Problem

PCOS is one of the most common hormonal disorders in women of reproductive age, affecting 8–13% of the population globally — yet it takes an average of two years and multiple specialist visits to diagnose. Misdiagnosis is frequent because PCOS symptoms overlap heavily with thyroid disorders, hyperprolactinemia, Cushing's syndrome, and endometriosis.

This pipeline provides:

1. A fast, explainable PCOS classifier grounded in Rotterdam diagnostic criteria
2. Automatic differential flagging for the four most commonly confused conditions when PCOS is ruled out

---

## How It Works

### Stage 1 — PCOS Binary Classifier

An XGBoost model trained on 541 patient records classifies each patient as **PCOS** or **Not-PCOS**.

Preprocessing:
- `Cycle(R/I)` recoded to binary (Regular → 0, Irregular → 1)
- Object-typed columns (`AMH`, `beta-HCG`) coerced to numeric; corrupted entries median-imputed
- Class imbalance (~2:1) handled via `scale_pos_weight`

Explainability is provided by SHAP values, with feature importance grouped by the three Rotterdam criteria:

| Rotterdam Criterion | Key Features | Top SHAP Score |
|---|---|---|
| Polycystic ovaries | Follicle No. (L/R), Avg. F size | 1.86 |
| Hyperandrogenism | Hair growth, Skin darkening, AMH, FSH/LH | 0.74 |
| Anovulation | Cycle(R/I), Cycle length | 0.41 |

### Stage 2 — Differential Diagnosis (Not-PCOS only)

Patients who clear Stage 1 are assessed by two parallel mechanisms:

**A) Rule-based clinical flags**

| Condition | Biomarker | Threshold |
|---|---|---|
| Hypothyroidism | TSH | > 4.0 mIU/L |
| Hyperprolactinemia | PRL | > 25 ng/mL |
| Cushing's / Hypertension | BP Systolic | > 140 mmHg |

**B) Endometriosis risk classifier**

A second XGBoost model trained on 10,000 patient records estimates endometriosis probability using age, BMI, menstrual irregularity, hormone abnormality, infertility history, and chronic pain level.

---

## Model Performance

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy |
|---|---|---|---|
| PCOS Classifier (Stage 1) | **0.962 ± 0.012** | **0.954** | **91%** |
| Endometriosis Classifier (Stage 2) | 0.640 ± 0.014 | 0.641 | 60% |

The endometriosis model's moderate AUC reflects the limited feature set (6 clinical variables) in the supplementary dataset; it is intended as a screening risk flag rather than a standalone diagnostic.

---

## Datasets

| Dataset | File | Source | Rows | Target |
|---|---|---|---|---|
| PCOS (main) | `data/(Main_Dataset)_PCOS_data_without_infertility.xlsx` | Kaggle PCOS Dataset | 541 | `PCOS (Y/N)` |
| Endometriosis | `data/(Supplementary_Dataset)_structured_endometriosis_data.csv` | Structured synthetic | 10,000 | `Diagnosis` |

---

## Installation

**Requirements:** Python 3.10+, Homebrew (macOS)

```bash
# macOS: XGBoost requires OpenMP
brew install libomp

# Python dependencies
pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn openpyxl

# Windows:
pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn openpyxl
```

---

## Running the Pipeline

Train both models first, then run the combined pipeline:

```bash
# Step 1: Train Stage 1 PCOS classifier and generate SHAP plots
python notebooks/stage1_pcos_classifier.py

# Step 2: Train Stage 2 endometriosis classifier and verify rule flags
python notebooks/stage2_differential_diagnosis.py

# Step 3: Run the full two-stage pipeline on the PCOS dataset
python notebooks/run_pipeline.py
```

Output files are written to `models/`:

| File | Description |
|---|---|
| `pcos_classifier.json` | Trained Stage 1 XGBoost model |
| `endo_classifier.json` | Trained Stage 2 endometriosis model |
| `pcos_feature_medians.csv` | Training medians used for imputation at inference |
| `pcos_shap_summary.png` | SHAP beeswarm plot (all features) |
| `pcos_rotterdam_shap.png` | SHAP bar chart grouped by Rotterdam criterion |
| `endo_shap_bar.png` | SHAP feature importance for endometriosis model |
| `pipeline_results.csv` | Per-patient predictions and diagnostic flags |

---

## Running the Web App (DIANA)

DIANA is a full-stack clinical decision support interface built with React + FastAPI.

### Backend (FastAPI)
```bash
# From project root
pip install fastapi uvicorn
python backend/main.py
# API runs at http://localhost:8000
# Docs available at http://localhost:8000/docs
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
# App runs at http://localhost:5173
```

Open http://localhost:5173 in your browser. Enter patient clinical values and click "Run Diagnostic Pipeline" to get a real-time PCOS diagnosis with differential flags and SHAP explanations.

---

## Tech Stack

- **ML / Backend:** Python, XGBoost, SHAP, scikit-learn, pandas, FastAPI, uvicorn
- **Frontend:** React, Vite, Tailwind CSS
- **Data:** Trained on 541 real patient records from 10 hospitals in Kerala, India

---

## Project Structure

```
.
├── data/
│   ├── (Main_Dataset)_PCOS_data_without_infertility.xlsx
│   └── (Supplementary_Dataset)_structured_endometriosis_data.csv
├── models/                         # generated — see .gitignore
├── notebooks/
│   ├── stage1_pcos_classifier.py   # Stage 1: train + explain
│   ├── stage2_differential_diagnosis.py  # Stage 2: rules + endo model
│   └── run_pipeline.py             # end-to-end pipeline runner
└── README.md
```

---

## Programmatic Use

`run_pipeline.py` exposes `run_pipeline(df)` for use in other code:

```python
import pandas as pd
from notebooks.run_pipeline import run_pipeline

df = pd.read_excel("data/(Main_Dataset)_PCOS_data_without_infertility.xlsx",
                   sheet_name="Full_new")
results = run_pipeline(df)
print(results[["patient_id", "pcos_probability", "primary_diagnosis"]])
```

The returned DataFrame contains one row per patient with columns for PCOS probability, endometriosis risk score, individual condition flags, and a plain-English `primary_diagnosis` string.
