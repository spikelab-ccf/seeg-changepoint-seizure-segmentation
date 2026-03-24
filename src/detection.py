"""
PELT-based changepoint detection for three-phase seizure segmentation.

Wraps the `ruptures` library's PELT algorithm and applies phase-specific
strategies for onset, intra-ictal transition, and termination detection.

Reference:
    Kumar et al., "Three-Phase Seizure Segmentation in Stereotactic EEG
    Using Envelope-Based Multivariate Changepoint Analysis", Ann. Biomed. Eng.
"""

import numpy as np

try:
    import ruptures as rpt
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False
    print("Warning: `ruptures` not installed. PELT detection unavailable. "
          "Install with: pip install ruptures")


# ---------------------------------------------------------------------------
# Default (optimised) phase parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "onset": {
        "window_size": 1000,   # ms (= samples at 1 kHz)
        "step":         150,
        "penalty":       11,
        "features": ["rms", "theta", "alpha", "beta", "gamma", "ll", "se"],
    },
    "transition": {
        "window_size":  700,
        "step":         130,
        "penalty":        7,
        "features": ["rms", "theta", "alpha", "beta", "gamma", "ll", "se"],
    },
    "termination": {
        "window_size": 1000,
        "step":         200,
        "penalty":       10,
        "features": ["rms", "theta", "alpha", "beta", "gamma", "ll", "se"],
    },
}


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_changepoints_pelt(X, penalty, model="rbf", min_size=2, jump=1):
    """Run PELT on a feature matrix X and return changepoint indices.

    Parameters
    ----------
    X : ndarray, shape (n_windows, n_features)
    penalty : float
    model : str  – cost function ('rbf', 'l2', 'normal', …)

    Returns
    -------
    changepoints : list of int  (indices into the feature time axis)
    """
    if not _RUPTURES_AVAILABLE:
        raise ImportError("Install ruptures: pip install ruptures")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(X)
    result = algo.predict(pen=penalty)
    # ruptures returns the *end* of each segment; drop the last (= len(X))
    return result[:-1]


# ---------------------------------------------------------------------------
# Phase-specific strategies
# ---------------------------------------------------------------------------

def detect_onset(X, params):
    """Return the **first** changepoint (earliest transition)."""
    cps = detect_changepoints_pelt(X, params["penalty"])
    return cps[0] if cps else None


def detect_termination(X, params):
    """Return the **last** changepoint (latest transition)."""
    cps = detect_changepoints_pelt(X, params["penalty"])
    return cps[-1] if cps else None


def detect_transition(X, params):
    """Return the changepoint **closest to the midpoint** of the signal window.

    This identifies the intra-ictal transition within an already-segmented
    onset-to-termination window.
    """
    cps = detect_changepoints_pelt(X, params["penalty"])
    if not cps:
        return None
    mid = len(X) / 2.0
    return min(cps, key=lambda cp: abs(cp - mid))


# ---------------------------------------------------------------------------
# Convenience: run all three phases
# ---------------------------------------------------------------------------

def run_three_phase_detection(feature_dicts, params=None, time_indices=None):
    """Detect all three changepoints from precomputed feature dictionaries.

    Parameters
    ----------
    feature_dicts : dict with keys 'onset', 'transition', 'termination',
                    each being the output of src.features.stack_features()
    params : dict or None  – phase params (default: DEFAULT_PARAMS)
    time_indices : dict or None  – sample-index arrays for each phase

    Returns
    -------
    dict with keys 'onset', 'transition', 'termination', values in samples
    (or None if not detected)
    """
    if params is None:
        params = DEFAULT_PARAMS

    from src.features import stack_features  # avoid circular at module level

    results = {}
    for phase, fn in [("onset", detect_onset),
                      ("transition", detect_transition),
                      ("termination", detect_termination)]:
        X = feature_dicts[phase]
        cp_idx = fn(X, params[phase])
        if cp_idx is not None and time_indices is not None:
            t = time_indices[phase]
            results[phase] = t[cp_idx] if cp_idx < len(t) else None
        else:
            results[phase] = cp_idx

    return results
