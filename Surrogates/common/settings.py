"""
Every constant the pipeline's analysis and evaluation depend on, in one place.
"""

import os

import numpy as np

SMOKE = os.environ.get("SMOKE", "0") == "1"

PT_MIN = 50.0
PT_MAX = 1000.0
N_PT_BINS = 14
PT_EDGES = np.logspace(np.log10(PT_MIN), np.log10(PT_MAX), N_PT_BINS + 1)
K_LADDER = (5, 10, 20, 50, 100)
CONES = (0.20, 0.15, 0.10)
SEEDS = (1, 2, 3)
DATA_PATH = os.environ.get("DATA", "Data/dataset/PJ_dataset_qcd_flat_15to7000.parquet")

CRPS_EVAL_EVENTS = 50_000  # main.py: multi-sample CRPS on the first N test events
FLOOR_ESTIMATOR = "gini_exact_all_pairs"

if SMOKE:
    SCORING_EVENTS = 2_000
    MAX_TRAIN_PER_FLAVOUR = 20_000
    MIN_COUNT = 20
    OVERLAP_ROWS = 500
    DUMP_M = 20
    DUMP_SUBSET = 500
    TRAIN_OVERRIDES = {
        "epochs": 1,
        "n_eval_samples": 5,
        "val_crps_events": 500,
        "n_hdr_events": 200,
    }
else:
    SCORING_EVENTS = 200_000
    MAX_TRAIN_PER_FLAVOUR = 400_000
    MIN_COUNT = 200
    OVERLAP_ROWS = 20_000
    DUMP_M = 200
    DUMP_SUBSET = 100_000
    TRAIN_OVERRIDES = {}
