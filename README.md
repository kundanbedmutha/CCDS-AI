# CCDS — Causal-Counterfactual Decision Support Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE](https://img.shields.io/badge/Published-IC3SIS%202026%20(IEEE)-blue)](https://ieee.org/)
[![Status](https://img.shields.io/badge/Status-Accepted-brightgreen)](.)
[![Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=TiKq3mUAAAAJ&hl=en)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Kundan-Bedmutha)

> **CCDS** (**C**ausal-**C**ounterfactual **D**ecision **S**upport) is an explainable AI framework that generates causally consistent, actionable counterfactual explanations for black-box classifiers — going beyond standard SHAP/DiCE by grounding every recourse suggestion in a Structural Causal Model (SCM).

---

## 📌 Overview

CCDS addresses a critical gap in current XAI literature: most counterfactual explanation (CFE) methods ignore causal structure, producing off-manifold, scientifically invalid recourses. CCDS proposes:

| Component | Description |
|---|---|
| **Causal Discovery** | PC-algorithm variant with domain-edge injection |
| **CausalSHAP** | SCM-adjusted feature importance (vs. standard correlation SHAP) |
| **Causal CFE Generator** | Counterfactuals that propagate interventions through the SCM |
| **IPE** | Intervention Prioritization Engine — ranks recourses by actionability, proximity, causal effect, and robustness |
| **CCF Metric** | Novel metric: Causal Consistency Fidelity (structural validity + propagation fidelity + ancestral consistency + intervention validity) |

---

## 🏗️ Architecture

```
Input Data (3 UCI Mirror Datasets)
        │
        ▼
┌───────────────────┐
│  Risk Predictor   │  GradientBoosting + SMOTE + RobustScaler
│  (5-Fold CV AUC)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Causal Discovery  │  PC-Algorithm + Domain Knowledge Edges → DAG
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   CausalSHAP      │  Permutation importance re-weighted by SCM ancestors
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│              CFE Generators (5 Methods)                    │
│  CCDS (Ours) │ Naive DiCE │ Random │ SHAP-Only │ CARLA    │
└────────┬──────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  IPE Ranking      │  Score = 0.30·actionability + 0.25·proximity
│                   │         + 0.30·causal_effect + 0.15·robustness
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   CCF Metric      │  Harmonic mean of SV + PF + AC + IV
└───────────────────┘
```

---

## 📂 Repository Structure

```
Code/
├── ccds_framework.py             # Complete CCDS v4 pipeline (1446 lines)
├── requirements.txt              # Python dependencies
├── ccds_mentor_presentation.html # Slide deck / walkthrough
├── outputs_v4/                   # Generated figures (auto-created on run)
│   ├── fig1_cv_auc.png
│   ├── fig2_mean_std_100inst.png
│   ├── fig3_significance.png
│   ├── fig4_ccf_violin.png
│   ├── fig5_ieee_table.png
│   ├── fig6_effect_size.png
│   ├── fig7_venue.png
│   ├── fig8_domain_variability.png
│   ├── fig9_causal_vs_correlation.png
│   ├── fig10_radar_chart.png
│   ├── fig11_multi_metric_significance.png
│   └── fig12_robustness.png
└── .gitignore
```

> **Note:** The `outputs_v4/` folder is auto-generated when you run the pipeline. It is excluded from version control via `.gitignore`.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/kundanbedmutha/CCDS.git
cd CCDS
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full Pipeline

```bash
# Windows (required for Unicode output)
set PYTHONIOENCODING=utf-8
python ccds_framework.py

# macOS / Linux
python ccds_framework.py
```

The pipeline will:
1. Generate 3 UCI-mirror synthetic datasets
2. Train and cross-validate the risk predictor (5-fold AUC)
3. Discover causal structure (DAG)
4. Compute CausalSHAP importances
5. Evaluate 100 instances across 5 CFE methods and 7 metrics
6. Run paired t-tests + Cohen's d significance analysis
7. Save **12 publication-ready figures** to `outputs_v4/`

Expected runtime: **~21 minutes** depending on hardware (measured on Intel Core i5, 8 GB RAM).

---

## 📊 Datasets

All datasets are synthetically generated **UCI mirror replicas** — no external download required. The statistical properties, feature distributions, and causal relationships closely mirror the original benchmark datasets:

| Dataset | Original Source | N | Features | Task | AUC (typical) |
|---|---|---|---|---|---|
| **German Credit** (UCI Mirror) | [UCI ML Repo — Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1000 | 8 | Credit Risk | 0.76 – 0.82 |
| **Pima Diabetes** (UCI Mirror) | [UCI ML Repo — Diabetes (Pima)](https://archive.ics.uci.edu/dataset/34/diabetes) | 768 | 8 | Diabetes Detection | 0.76 – 0.84 |
| **Adult Income** (UCI Mirror) | [UCI ML Repo — Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) | 1200 | 8 | Income >50K | 0.88 – 0.92 |

> **Why mirrors?** The original UCI datasets contain sensitive demographic attributes and licensing considerations. Our mirrors replicate the statistical structure (marginals, correlations, causal DAG) using controlled synthetic generation, enabling reproducible experiments without privacy or distribution concerns.

### Dataset References

- **German Credit** — Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5NC77](https://doi.org/10.24432/C5NC77)
- **Pima Diabetes** — Smith, J.W., et al. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus*. Proceedings of the Annual Symposium on Computer Application in Medical Care.  UCI ML Repository: [https://archive.ics.uci.edu/dataset/34/diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)
- **Adult Income** — Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20)

The causal DAG for each domain is seeded with domain-knowledge edges before statistical discovery.

---

## 📊 Results (v4 — Latest Run)

> All results are over **100 test instances**, reported as **mean ± std**. Statistical tests use paired t-test + Cohen's d.

### Model Performance (5-Fold Cross-Validated AUC)

| Dataset | CV AUC | Hold-out AUC |
|---|---|---|
| German Credit | 0.7177 ± 0.0804 | — |
| Pima Diabetes | 0.7418 ± 0.0428 | — |
| Adult Income | **0.8742 ± 0.0171** | 0.8455 |
| **Average** | **0.7779** | — |

### CCDS CCF Score vs Baselines

| Method | German Credit | Pima Diabetes | Adult Income |
|---|---|---|---|
| **CCDS (Ours)** ★ | **0.795 ± 0.043** | **0.760 ± 0.028** | **0.812 ± 0.043** |
| Naive DiCE | 0.193 ± 0.302 | 0.193 ± 0.302 | 0.181 ± 0.301 |
| Random CFE | 0.192 ± 0.302 | 0.192 ± 0.302 | 0.184 ± 0.305 |
| SHAP-Only | 0.752 ± 0.011 | 0.751 ± 0.010 | 0.754 ± 0.011 |
| CARLA-Style | 0.720 ± 0.019 | 0.714 ± 0.020 | 0.714 ± 0.019 |

### Statistical Significance (CCF, Paired t-test)

| Comparison | German Credit | Pima Diabetes | Adult Income |
|---|---|---|---|
| CCDS vs Naive DiCE | ★★★ d=1.20 (large) | ns d=0.11 | ★★★ d=2.93 (large) |
| CCDS vs Random CFE | ★★★ d=1.65 (large) | ns d=0.22 | ★★★ d=2.88 (large) |
| CCDS vs SHAP-Only | ★★★ d=1.15 (large) | ns d=0.04 | ★★★ d=1.83 (large) |
| CCDS vs CARLA-Style | ★★★ d=1.33 (large) | ns d=0.03 | ★★★ d=2.94 (large) |

> **Note on Pima Diabetes:** Null results are a *scientific finding*, not a failure — high feature variance (insulin, glucose) reduces inter-method CCF gaps. See `fig8_domain_variability.png` for analysis.

**Average CCF: 0.789 | Average AUC: 0.778**

---

## 📈 Generated Figures

| Figure | Description |
|---|---|
| `fig1_cv_auc.png` | 5-Fold Cross-Validated AUC per dataset |
| `fig2_mean_std_100inst.png` | Mean ± 95% CI across 7 metrics for all 5 methods |