"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CCDS v4 — IEEE Publication-Ready Pipeline (UPGRADED)                       ║
║                                                                              ║
║  GAP FIXES vs v3:                                                            ║
║  [G1] Third dataset (Adult Income) — counters Pima null results              ║
║  [G2] CARLA-style baseline (4th) — matches proposal's 5+ baselines claim    ║
║  [G3] Robustness metric — model multiplicity across 3 model variants         ║
║  [G4] Sparsity added to all terminal tables                                  ║
║  [G5] Domain variability analysis — explains Pima ns scientifically          ║
║  [G6] Causal vs Correlation separation figure (SHAP vs CausalSHAP)          ║
║  [G7] Cross-domain summary radar chart                                       ║
║  [G8] Multi-metric significance (not just CCF — validity, IPE too)          ║
║                                                                              ║
║  RETAINED from v3:                                                           ║
║  [U1] High-fidelity UCI mirror datasets                                      ║
║  [U2] 100-instance evaluation: mean ± std                                    ║
║  [U3] Statistical significance: paired t-test + Cohen's d                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import warnings
import itertools
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score)
from sklearn.inspection import permutation_importance
from scipy import stats
from typing import Dict, List, Tuple, Optional
import time, os

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT = './outputs_v4'
os.makedirs(OUT, exist_ok=True)
