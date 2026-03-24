# Three-Phase Seizure Segmentation in Stereotactic EEG Using Envelope-Based Multivariate Changepoint Analysis

**Himanshu Kumar, Guhan Seshadri N P, David Martinez, Imad Najm, Andreas Alexopoulos, Juan C Bulacio, Demitre Serletis, Balu Krishnan**

*Epilepsy Centre, Neurological Institute, Cleveland Clinic, Cleveland, OH, USA*

> Corresponding author: Balu Krishnan

---

## Overview

This repository contains the analysis code for our paper on automated three-phase seizure segmentation in stereoelectroencephalography (SEEG) recordings.

**Key contributions:**
- A semi-supervised framework that jointly detects **seizure onset**, **intra-ictal transition**, and **seizure termination** in a single pipeline.
- Seven complementary **envelope-based features**: RMS amplitude, relative bandpower in θ (4–8 Hz), α (8–13 Hz), β (13–30 Hz), γ (30–80 Hz) bands, line length, and spectral entropy.
- **PELT** (Pruned Exact Linear Time) changepoint detection with phase-specific feature weighting and penalty parameters.
- Parameters optimised via **nested leave-one-subject-out cross-validation** (LOSO-CV) with Optuna.
- **Length-invariant** validation through random pre- and post-ictal window extension.

**Dataset:** 32 seizures from 10 patients with drug-resistant focal epilepsy undergoing presurgical SEEG evaluation at Cleveland Clinic. 179 seizure-onset zone bipolar channels analysed.

**Results (mean ± SD absolute error):**

| Phase | MAE (s) | Acc ±5 s |
|---|---|---|
| Seizure onset | 4.19 ± 2.69 | 71.6 % |
| Intra-ictal transition | 6.93 ± 5.75 | 60.0 % |
| Seizure termination | 3.82 ± 4.24 | 75.0 % |

---

## Repository Structure

```
├── src/
│   ├── features.py
│   ├── detection.py
│   └── metrics.py
│
├── scripts/
│   ├── 01_feature_extraction/
│   │   └── plot_segmentation_example.py
│   │
│   ├── 02_evaluation/
│   │   ├── run_loso_cv.py
│   │   └── evaluate_final_model.py
│   │
│   ├── 03_analysis/
│   │   ├── generate_ablation_tables_and_plots.py
│   │   ├── analyze_feature_importance.py
│   │   └── extract_optimized_parameters.py
│   │
│   └── 04_visualization/
│       ├── aggregate_epoched_features.py
│       ├── plot_event_aligned_features.py
│       ├── plot_feature_importance_radar.py
│       └── plot_feature_importance_evolution.py
│
├── figures/
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/<your-username>/seeg-changepoint-seizure-segmentation.git
cd seeg-changepoint-seizure-segmentation
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Signal processing & feature extraction |
| `ruptures` | PELT changepoint detection |
| `optuna` | Bayesian hyperparameter optimisation |
| `pandas` | Results management |
| `matplotlib` | Visualisation |
| `scikit-learn` | Preprocessing utilities |
| `statsmodels` | Statistical testing |

---

## Usage

### Feature extraction

```python
import scipy.io as sio
from src.features import highpass_filter, extract_all_features, stack_features

# Load your SEEG data (expected format: .mat file with bipolar montage)
data = sio.loadmat("path/to/seizure.mat", simplify_cells=True)
signal = data["filtered_signals"][channel_idx]   # 1-D array, 1000 Hz

fs = 1000  # Hz

# Extract all seven features (onset-phase parameters as example)
features = extract_all_features(signal, fs, window_size=1000, step=150)

# Stack into a feature matrix (normalised, equal weights)
feature_names = ["rms", "theta", "alpha", "beta", "gamma", "ll", "se"]
X = stack_features(features, feature_names)
print(X.shape)  # (n_windows, 7)
```

### Changepoint detection

```python
from src.detection import detect_onset, detect_termination, detect_transition, DEFAULT_PARAMS

# Onset detection (uses first changepoint)
onset_idx = detect_onset(X, DEFAULT_PARAMS["onset"])
onset_time = features["time_indices"][onset_idx] / fs  # convert samples → seconds
print(f"Detected onset: {onset_time:.2f} s")
```

### Running a full example

```bash
python scripts/01_feature_extraction/plot_segmentation_example.py
```

This will generate a multi-panel segmentation plot for the best-centred seizure case in your dataset.

---

## Data Availability

SEEG recordings contain protected health information and **cannot be publicly shared**. All procedures were approved by the Cleveland Clinic Institutional Review Board (IRB) and conducted in accordance with the Declaration of Helsinki.

Researchers interested in collaboration or data access may contact the corresponding author.

---

## Citation

If you use this code in your research, please cite:

```
Kumar H, Seshadri NP G, Martinez D, Najm I, Alexopoulos A, Bulacio JC,
Serletis D, Krishnan B. Three-Phase Seizure Segmentation in Stereotactic EEG
Using Envelope-Based Multivariate Changepoint Analysis.
Annals of Biomedical Engineering, 2026.
```

---

## License

This code is released for academic and research purposes. Please contact the authors before use in commercial applications.
