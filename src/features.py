"""
Feature extraction functions for SEEG seizure segmentation.

Extracts seven multivariate envelope-based features from SEEG signals:
  - RMS envelope
  - Relative bandpower: theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-80 Hz)
  - Line length
  - Spectral entropy

Reference:
    Kumar et al., "Three-Phase Seizure Segmentation in Stereotactic EEG
    Using Envelope-Based Multivariate Changepoint Analysis", Ann. Biomed. Eng.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.integrate import trapezoid
from scipy.stats import entropy

# Frequency band definitions (Hz)
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 80),
}
TOTAL_BAND = (0.5, 150)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def highpass_filter(signal, fs=1000, cutoff=0.5, order=3):
    """Apply a Butterworth high-pass filter to remove DC and low-frequency drift."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="highpass")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def rms_envelope(signal, window_size, step):
    """Root-mean-square amplitude envelope.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
    window_size : int  (samples)
    step : int  (samples)

    Returns
    -------
    envelope : ndarray
    time_indices : ndarray  (sample index at window centre)
    """
    signal = np.asarray(signal)
    if len(signal) < window_size:
        return np.array([]), np.array([])
    n_windows = (len(signal) - window_size) // step + 1
    envelope = np.array([
        np.sqrt(np.mean(np.square(signal[i * step: i * step + window_size])))
        for i in range(n_windows)
    ])
    time_indices = np.arange(n_windows) * step + window_size // 2
    return envelope, time_indices


def line_length(signal, window_size, step):
    """Sum of absolute first differences within each window (line-length feature)."""
    signal = np.asarray(signal)
    if len(signal) < window_size:
        return np.array([])
    return np.array([
        np.sum(np.abs(np.diff(signal[s: s + window_size])))
        for s in range(0, len(signal) - window_size + 1, step)
    ])


def spectral_entropy(signal, fs, window_size, step):
    """Shannon entropy of the normalised Welch power spectral density."""
    signal = np.asarray(signal)
    if len(signal) < window_size:
        return np.array([])
    se_values = []
    for s in range(0, len(signal) - window_size + 1, step):
        seg = signal[s: s + window_size]
        _, Pxx = welch(seg, fs=fs, nperseg=min(window_size, 256))
        Pxx_norm = Pxx / (Pxx.sum() + 1e-12)
        se_values.append(entropy(Pxx_norm))
    return np.array(se_values)


def relative_bandpower_envelope(signal, fs, band, total_band, window_size, step):
    """Relative bandpower envelope for a specified frequency band.

    Parameters
    ----------
    band : tuple (f_low, f_high)  Hz
    total_band : tuple (f_low, f_high)  Hz  – denominator band
    """
    signal = np.asarray(signal)
    if len(signal) < window_size:
        return np.array([]), np.array([])
    n_windows = (len(signal) - window_size) // step + 1
    relative_power = np.zeros(n_windows)
    time_indices = np.arange(n_windows) * step + window_size // 2
    for i in range(n_windows):
        seg = signal[i * step: i * step + window_size]
        f, Pxx = welch(seg, fs=fs, nperseg=min(window_size, 256))
        total_mask = (f >= total_band[0]) & (f <= total_band[1])
        band_mask  = (f >= band[0])       & (f <= band[1])
        total_power = trapezoid(Pxx[total_mask], f[total_mask]) + 1e-12
        band_power  = trapezoid(Pxx[band_mask],  f[band_mask])
        relative_power[i] = band_power / total_power
    return relative_power, time_indices


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_all_features(signal, fs, window_size, step):
    """Extract all seven features and return a dictionary.

    Returns
    -------
    dict with keys: 'rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se',
                    'time_indices'
    """
    rms, t = rms_envelope(signal, window_size, step)
    ll = line_length(signal, window_size, step)
    se = spectral_entropy(signal, fs, window_size, step)
    theta, _ = relative_bandpower_envelope(signal, fs, BANDS["theta"], TOTAL_BAND, window_size, step)
    alpha, _ = relative_bandpower_envelope(signal, fs, BANDS["alpha"], TOTAL_BAND, window_size, step)
    beta,  _ = relative_bandpower_envelope(signal, fs, BANDS["beta"],  TOTAL_BAND, window_size, step)
    gamma, _ = relative_bandpower_envelope(signal, fs, BANDS["gamma"], TOTAL_BAND, window_size, step)
    return {
        "rms": rms, "theta": theta, "alpha": alpha,
        "beta": beta, "gamma": gamma, "ll": ll, "se": se,
        "time_indices": t,
    }


def minmax_normalize(feat):
    """Min-Max normalization to [0, 1] (Section 2.6, Eq. 7)."""
    f_min, f_max = feat.min(), feat.max()
    denom = f_max - f_min
    return (feat - f_min) / (denom if denom > 1e-12 else 1.0)


def exponential_smooth(feat, alpha=0.1):
    """Exponential weighted averaging to reduce envelope noise (Section 2.6, Eq. 8).

    x_smooth(t) = alpha * x(t) + (1 - alpha) * x_smooth(t-1),  alpha = 0.1
    """
    smoothed = np.zeros_like(feat)
    smoothed[0] = feat[0]
    for t in range(1, len(feat)):
        smoothed[t] = alpha * feat[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def stack_features(feature_dict, feature_names, weights=None, alpha=0.1):
    """Min-Max normalise, exponentially smooth, optionally weight, and
    column-stack features into a 2-D array (Section 2.6).

    Parameters
    ----------
    feature_dict : dict  (output of extract_all_features)
    feature_names : list of str
    weights : dict or None  – per-feature scalar weights (default: equal = 1.0)
    alpha : float  – smoothing parameter (default: 0.1 as per paper)

    Returns
    -------
    X : ndarray, shape (n_windows, n_features)
    """
    if weights is None:
        weights = {f: 1.0 for f in feature_names}
    columns = []
    for name in feature_names:
        feat = feature_dict[name].astype(float)
        feat_norm = minmax_normalize(feat)
        feat_smooth = exponential_smooth(feat_norm, alpha=alpha)
        columns.append(feat_smooth * weights.get(name, 1.0))
    n = min(len(c) for c in columns)
    return np.column_stack([c[:n] for c in columns])
