"""
Evaluation metrics for three-phase seizure segmentation.

Reference:
    Kumar et al., "Three-Phase Seizure Segmentation in Stereotactic EEG
    Using Envelope-Based Multivariate Changepoint Analysis", Ann. Biomed. Eng.
"""

import numpy as np


def absolute_error(predicted, ground_truth):
    """Absolute error between a predicted and ground-truth time (seconds)."""
    if predicted is None or np.isnan(predicted) or np.isnan(ground_truth):
        return np.nan
    return abs(predicted - ground_truth)


def accuracy_within_tolerance(errors, tolerance=5.0):
    """Fraction of detections within *tolerance* seconds of ground truth (%)."""
    errors = np.asarray(errors, dtype=float)
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return np.nan
    return np.mean(valid <= tolerance) * 100.0


def mae(errors):
    """Mean absolute error, ignoring NaNs."""
    errors = np.asarray(errors, dtype=float)
    valid = errors[~np.isnan(errors)]
    return np.mean(valid) if len(valid) > 0 else np.nan


def rmse(errors):
    """Root-mean-squared error, ignoring NaNs."""
    errors = np.asarray(errors, dtype=float)
    valid = errors[~np.isnan(errors)]
    return np.sqrt(np.mean(valid ** 2)) if len(valid) > 0 else np.nan


def summarise(errors, tolerance=5.0, label=""):
    """Print and return a summary dict for a list of absolute errors."""
    errors = np.asarray(errors, dtype=float)
    valid = errors[~np.isnan(errors)]
    summary = {
        "label":      label,
        "n":          len(valid),
        "mae":        mae(errors),
        "rmse":       rmse(errors),
        "median":     np.median(valid) if len(valid) > 0 else np.nan,
        "iqr":        (np.percentile(valid, 75) - np.percentile(valid, 25))
                      if len(valid) > 0 else np.nan,
        f"acc_{int(tolerance)}s": accuracy_within_tolerance(errors, tolerance),
    }
    if label:
        print(f"{label}  |  MAE={summary['mae']:.2f}s  "
              f"RMSE={summary['rmse']:.2f}s  "
              f"Acc(±{tolerance:.0f}s)={summary[f'acc_{int(tolerance)}s']:.1f}%  "
              f"N={summary['n']}")
    return summary
