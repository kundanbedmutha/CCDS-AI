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