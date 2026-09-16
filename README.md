# Estimating Spectrum-Free Information Bounds for Generative Surrogates Beyond Conditional Collapse

This repository contains the code, data, and configurations to reproduce the paper submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026), **"Estimating Spectrum-Free Information Bounds for Generative Surrogates Beyond Conditional Collapse"**.

**Abstract**:
Conditional generative surrogates in high-energy physics, such as those predicting reconstructed jet 4-vectors from underlying parton kinematics are routinely evaluated using 1D marginal divergences. We demonstrate that this paradigm actively rewards conditional collapse, failing to reliably distinguish highly expressive Conditional Flow Matching (CFM) models and Mixture Density Networks (MDN) from zero-information marginal baselines. To resolve this, we establish a spectrum-free Continuous Ranked Probability Skill Score (CRPSS) normalized against an exact marginal floor. Furthermore, we introduce a data-driven k-Nearest-Neighbor (k-NN) extrapolation to estimate the Bayes-optimal performance ceiling. Evaluated against this k-NN bound, we reveal that current neural architectures have already saturated the extractable physical signal, establishing data-driven ceilings as a mandatory diagnostic.

## Steps to Reproduce

### 1. Set Up the Environment
Create and activate the dedicated Conda environment. This setup installs all required pip dependencies (including PyTorch, Polars, CERN Open Data Client, Uproot, Vector, and Awkward), and is optimized for Blackwell cards:
```bash
bash Scripts/setup_env.sh
conda activate q2j_estimating_information_bounds
```

### 2. Download the Raw Data
The data uses the CMS QCD Simulation dataset. The `.root` files can be downloaded directly from the CERN Open Data portal. 
```bash
bash Data/Root_files/download_root.sh
```

### 3. Parse ROOT Files to Parquet
Extract the relevant kinematics into a flat structured dataset.
```bash
python Data/root2par_dataset_gen_flat.py
```

### 4. Reproduce the Training and Evaluation Pipeline
The complete experimental pipeline (training, testing, metric extraction, and figure generation) is automated in the reproduction shell script. It processes the dataset, trains the CFM and MDN models across different configurations, and evaluates the test distributions.
```bash
bash Scripts/reproduce.sh
```
*Tip: For a quick test using generated "smoke" data, the pipeline can be run with the environment variable set: `SMOKE=1 bash Scripts/reproduce.sh`.*

## Repository Guide

```text
.
├── Data/
│   ├── Root_files/
│   │   └── download_root.sh             # Fetches CMS datasets from CERN Open Data
│   └── root2par_dataset_gen_flat.py     # Parses raw ROOT files to Parquet
├── Experiments/
│   ├── configs/                         # Model parameters, ablations, and test regimes
│   ├── run_experiment.py                # Main experiment runner logic
│   └── ...                              # Auxiliary scripts (baselines, plots, etc.)
├── Results/                             # (Generated) Evaluation metrics and artifact outputs
├── Scripts/
│   ├── setup_env.sh                     # Environment build sequence
│   ├── reproduce.sh                     # Automated runner for reproducing paper figures
│   ├── record_environment.py            # Hardware/library environment check
│   └── make_smoke_data.py               # Generates minimal fake data for pipeline testing
├── Surrogates/
│   ├── CFM/                             # Conditional Flow Matching implementation
│   ├── MDN/                             # Mixture Density Network implementation
│   └── common/                          # Datasets, utilities, metrics, and plotting code
├── environment.yml                      # Conda package dependencies configuration
└── README.md                            # Repository documentation
```

# Estimating-Spectrum-Free-Information-Bounds-for-Generative-Surrogates-Beyond-Conditional-Collapse
