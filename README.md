
---

# SEWA — Sepsis Early Warning Agent

**SEWA** is a hybrid-intelligence clinical decision support system for **early sepsis detection**.
It combines **statistical trend analysis**, **machine learning**, **rule-based safety logic**, and **LLM-generated explanations** to deliver **interpretable, safety-aware alerts** in near real time.

> ⚠️ **Research / Educational Use Only**
> SEWA is **not a diagnostic device** and is **not FDA-approved**. All outputs must be reviewed by qualified clinicians.

---

## 🎯 Problem Addressed

Sepsis evolves dynamically and is often detected **too late** due to:

* Fragmented vitals interpretation
* Overreliance on static thresholds
* Black-box ML alerts without clinical context

**SEWA solves this by detecting temporal deterioration patterns early and explaining *why* an alert was raised.**

---

## ✨ Key Capabilities

* **Real-time monitoring** (15-minute intervals)
* **5-class risk stratification**
  `NO_RISK → WATCH → MODERATE → HIGH → CRITICAL`
* **60–70 explainable trend features per patient**
* **Hard safety rules that override ML predictions**
* **Natural-language clinical narratives**
* **Production-oriented modular architecture**

---

## 🏗️ System Architecture

```
Patient Vitals
      ↓
Trend Recognition Engine
(statistical temporal features)
      ↓
ML Risk Model
(probabilistic fusion)
      ↓
Clinical Rule Engine
(safety overrides)
      ↓
LLM Explanation Layer
(natural language)
      ↓
Structured Alert + Action
```

---

## 🧩 Core Components

### 1️⃣ Trend Recognition Engine (`trend_engine.py`)

* Multi-window exponential moving averages
* Short / medium / long-term slopes
* Volatility & instability metrics
* Acceleration and directionality detection

### 2️⃣ Synthetic Data Generator (`data_generator.py`)

* Realistic patient deterioration trajectories
* 5 distinct sepsis risk progressions
* Noise, missing data, artifacts
* Timestamped physiological evolution

### 3️⃣ ML Risk Scoring Pipeline (`ml_pipeline.py`)

* Logistic Regression (baseline)
* Gradient Boosting (optional)
* SHAP-based interpretability
* Calibration and validation checks

### 4️⃣ Core SEWA System (`core_system.py`)

* End-to-end orchestration
* Rule-based clinical overrides
* Alert generation & routing
* LLM-based explanation synthesis

---

## 🚀 Quick Start

### 1. Installation

```bash
cd sewa
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Generate Synthetic Training Data

```bash
python scripts/generate_data.py \
  --patients-per-class 200 \
  --output data/synthetic_patients.csv
```

**Output:**
~48,000 timestamped measurements from 1,000 simulated patients.

---

### 3. Train the ML Model

```bash
python scripts/train_model.py \
  --data data/synthetic_patients.csv \
  --model-type logistic \
  --output models/sewa_risk_model.pkl
```

**Expected Performance**

* Accuracy: **87–90%**
* ROC-AUC: **~0.95**

---

### 4. (Optional) Configure LLM Provider

```bash
cp .env.example .env
```

Supported providers:

* **Together AI** (recommended)
* **Groq**
* **OpenRouter**

Only **de-identified metadata** is sent to LLM APIs.

---

### 5. Run End-to-End Test

```bash
python scripts/test_system.py
```

Simulates a deteriorating patient over **6 hours**.

---

## 💻 Usage Examples

### Process a Single Measurement

```python
from sewa.core_system import SEWASystem, PatientState
from datetime import datetime

alert = sewa.process_measurement(
    PatientState(
        timestamp=datetime.now(),
        lactate=2.8,
        map=66,
        hr=108,
        temp=38.5,
        rr=24,
        spo2=94,
        infection_suspected=True
    )
)

if alert:
    print(alert.final_risk_level)
    print(alert.recommended_action)
    print(alert.clinical_narrative)
```

---

### Batch Process a Patient Timeline

```python
for _, row in patient_df.iterrows():
    alert = sewa.process_measurement(PatientState(**row))
    if alert:
        print(alert.to_dict())
```

---

## 📊 Model Performance

### Overall

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 87.5% |
| ROC-AUC (macro) | 0.952 |
| Macro F1        | 0.884 |

### Per-Class Performance

| Class    | Precision | Recall | F1   |
| -------- | --------- | ------ | ---- |
| NO_RISK  | 0.92      | 0.95   | 0.93 |
| WATCH    | 0.85      | 0.80   | 0.82 |
| MODERATE | 0.88      | 0.87   | 0.88 |
| HIGH     | 0.87      | 0.89   | 0.88 |
| CRITICAL | 0.90      | 0.92   | 0.91 |

### Top Predictive Features

* `lactate_slope_short`
* `map_slope_short`
* `lactate_acceleration`
* `hr_volatility_medium`
* `lactate_slope_medium`

---

## 🛡️ Clinical Safety Rules (Overrides ML)

1. MAP < 55 **and** Lactate > 2.0 → **CRITICAL**
2. Lactate > 4.0 → minimum **HIGH**
3. On vasopressors + ML ≥ MODERATE → **HIGH**
4. Lactate slope > 0.8 mmol/L/hr → **MODERATE**
5. Data stale > 2 hours → max **WATCH**
6. Multi-organ dysfunction → **MODERATE**
7. SIRS + infection → **WATCH**

---

## 🧪 Testing & Quality

```bash
pytest tests/
pytest --cov=sewa tests/
```

---

## 📦 Alert Output (JSON)

```json
{
  "patient_id": "PT-001",
  "final_risk_level": "HIGH",
  "risk_score": 0.58,
  "rules_triggered": ["MULTI_ORGAN_DYSFUNCTION"],
  "recommended_action": "RAPID_RESPONSE",
  "clinical_narrative": "High sepsis risk due to rising lactate and hypotension."
}
```

---

## ⚠️ Limitations

* Trained on **synthetic data**
* Requires validation on real patient cohorts
* Conservative missing-data handling
* Performance may vary by population

---

## 🛣️ Roadmap

### Phase 1 (Completed)

* Core system
* Synthetic data
* Baseline ML
* Safety rules

### Phase 2

* Real patient validation
* Advanced ML models
* Multimodal inputs

### Phase 3

* EHR integration
* Real-time dashboards
* Regulatory preparation

---


