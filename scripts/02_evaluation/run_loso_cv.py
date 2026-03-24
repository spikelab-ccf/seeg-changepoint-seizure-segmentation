# run_cv_fold.py

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import ruptures as rpt
from tabulate import tabulate
import optuna
import random
import traceback
import json
import argparse
from collections import defaultdict
import warnings
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- HELPER FUNCTIONS (Unchanged) ---
def highpass_filter2(signal, fs=1000, cutoff=0.5, order=3):
    try:
        from scipy.signal import butter, filtfilt
        wn = cutoff / (fs / 2)
        b, a = butter(order, wn, btype='highpass')
        return filtfilt(b, a, signal)
    except Exception:
        return signal
def exp_weighted_avg(data, alpha=0.1):
    smoothed = np.zeros(len(data))
    if len(data) > 0:
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed
def rms_envelope(signal, window_size, step):
    if len(signal) < window_size: return np.array([]), np.array([])
    try:
        n_windows = (len(signal) - window_size) // step + 1
        envelope = np.array([np.sqrt(np.mean(np.square(signal[i*step:i*step+window_size]))) for i in range(n_windows)])
        time_indices = np.arange(0, n_windows * step, step) + window_size // 2
        return envelope, time_indices
    except Exception: return np.array([]), np.array([])
def line_length(signal, window_size, step):
    if len(signal) < window_size: return np.array([])
    try:
        return np.array([np.sum(np.abs(np.diff(signal[start:start+window_size]))) for start in range(0, len(signal) - window_size + 1, step)])
    except Exception: return np.array([])
def spectral_entropy(signal, fs, window_size, step):
    try:
        from scipy.signal import welch
        from scipy.stats import entropy
        if len(signal) < window_size: return np.array([])
        se_values = []
        for start in range(0, len(signal) - window_size + 1, step):
            try:
                f, Pxx = welch(signal[start:start+window_size], fs=fs, nperseg=window_size)
                Pxx_norm = Pxx / (np.sum(Pxx) + 1e-12)
                se_values.append(entropy(Pxx_norm))
            except Exception: se_values.append(0.0)
        return np.array(se_values)
    except Exception: return np.array([])
def relative_bandpower_envelope(signal, fs, band, total_band, window_size, step):
    try:
        from scipy.signal import welch
        from scipy.integrate import trapezoid
        if len(signal) < window_size: return np.array([]), np.array([])
        n_windows = (len(signal) - window_size) // step + 1
        relative_power = np.zeros(n_windows)
        time_indices = np.arange(0, n_windows * step, step) + window_size // 2
        for i in range(n_windows):
            try:
                f, Pxx = welch(signal[i*step:i*step+window_size], fs=fs, nperseg=window_size)
                total_mask = (f >= total_band[0]) & (f <= total_band[1])
                total_power = trapezoid(Pxx[total_mask], f[total_mask]) + 1e-12
                band_mask = (f >= band[0]) & (f <= band[1])
                band_power = trapezoid(Pxx[band_mask], f[band_mask])
                relative_power[i] = band_power / total_power
            except Exception: relative_power[i] = 0.0
        return relative_power, time_indices
    except Exception: return np.array([]), np.array([])
def extract_label(x):
    try:
        if isinstance(x, str): return x
        elif isinstance(x, bytes): return x.decode()
        elif np.isscalar(x) or (isinstance(x, (np.ndarray, list)) and np.array(x).size == 1): return extract_label(np.array(x).flatten()[0])
        else: return str(x)
    except Exception: return ""
def extract_time_array(x):
    try:
        if np.isscalar(x): return [float(x)]
        return list(np.array(x).flatten())
    except Exception: return []
def safe_changepoint_detection(data, penalty=10):
    try:
        if len(data) < 10: return []
        if data.ndim == 1: data = data.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf", min_size=5)
        changepoints = algo.fit(data).predict(pen=penalty)
        return [cp for cp in changepoints if 0 <= cp < len(data)]
    except Exception: return []

def run_analysis(seizure_paths, contacts_map, 
                 onset_weights, term_weights, trans_weights, 
                 penalty_onset, penalty_term, penalty_trans,
                 onset_window_size, onset_step,
                 term_window_size, term_step,
                 trans_window_size, trans_step,
                 ablate_feature=None, debug=False):
    
    all_seizure_performances = []
    all_possible_features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    
    for seizure_path in seizure_paths:
        this_seizure_onset_errors, this_seizure_mid_errors, this_seizure_end_errors = [], [], []
        
        try:
            patient_folder = os.path.basename(os.path.dirname(os.path.dirname(seizure_path)))
            contacts = contacts_map.get(patient_folder)
            if not contacts: continue
            
            channel_file = seizure_path.replace('data_block001_montage.mat', 'channel.mat')
            if not os.path.exists(channel_file): continue
            
            channel_data = sio.loadmat(channel_file)
            channel_labels = [ch[0] for ch in channel_data['Channel']['Name'][0]]
            channel_index = [i for i, ch in enumerate(channel_labels) if ch in contacts]
            if not channel_index: continue

            data = sio.loadmat(seizure_path)
            event_labels_raw, event_times_raw = data["Events"]["label"], data["Events"]["times"]
            event_labels = [extract_label(lbl) for lbl in event_labels_raw.flatten()]
            event_times = [extract_time_array(t) for t in event_times_raw.flatten()]
            dm_cpt_list = [event_times[i] for i, label in enumerate(event_labels) if 'dm_cpt' in label.lower()]
            
            if not dm_cpt_list or len(dm_cpt_list[0]) < 2: continue
            dm_cpt = dm_cpt_list[0]
            
            time_array = data["Time"][0]
            t0_index, tend_index = np.argmin(np.abs(time_array - dm_cpt[0])), np.argmin(np.abs(time_array - dm_cpt[-1]))
            if t0_index >= tend_index: continue
            
            eeg, gt_t0, gt_tend = data['F'][channel_index, :], dm_cpt[0], dm_cpt[-1]
            gt_tmid = dm_cpt[1] if len(dm_cpt) > 2 else np.nan
            seizure_has_valid_mid_gt = not np.isnan(gt_tmid)
            
        except Exception: 
            continue

        for j in range(len(eeg)):
            if not debug and j > 2: break
            try:
                fs = 1000
                bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
                
                start_shift = np.random.randint(5 * fs, min(30 * fs + 1, t0_index))
                end_shift = np.random.randint(5 * fs, min(30 * fs + 1, eeg.shape[1] - tend_index))
                start_idx, end_idx = max(0, t0_index - start_shift), min(eeg.shape[1], tend_index + end_shift)
                
                signal_segment = eeg[j, start_idx:end_idx]
                if len(signal_segment) < max(onset_window_size, term_window_size) * 3: continue
                
                filtered_signal = highpass_filter2(signal_segment)
                scaler = MinMaxScaler(feature_range=(0.1, 1))

                # --- STAGE 1: ONSET DETECTION ---
                onset_raw_features = {
                    'rms': rms_envelope(filtered_signal, onset_window_size, onset_step)[0],
                    'theta': relative_bandpower_envelope(filtered_signal, fs, bands['theta'], (0.5, 150), onset_window_size, onset_step)[0],
                    'alpha': relative_bandpower_envelope(filtered_signal, fs, bands['alpha'], (0.5, 150), onset_window_size, onset_step)[0],
                    'beta': relative_bandpower_envelope(filtered_signal, fs, bands['beta'], (0.5, 150), onset_window_size, onset_step)[0],
                    'gamma': relative_bandpower_envelope(filtered_signal, fs, bands['gamma'], (0.5, 150), onset_window_size, onset_step)[0],
                    'll': line_length(filtered_signal, onset_window_size, onset_step),
                    'se': spectral_entropy(filtered_signal, fs, onset_window_size, onset_step)
                }
                onset_time_indices = rms_envelope(filtered_signal, onset_window_size, onset_step)[1]
                if any(f.size == 0 for f in onset_raw_features.values()) or onset_time_indices.size == 0: continue
                
                min_len_onset = min(len(f) for f in onset_raw_features.values() if f.size > 0)
                if min_len_onset < 10: continue

                s_features_onset = {name: exp_weighted_avg(scaler.fit_transform(onset_raw_features[name][:min_len_onset].reshape(-1, 1)).flatten()) for name in onset_weights.keys()}
                ONSET_MATRIX = np.vstack([s_features_onset[name] * onset_weights[name] for name in onset_weights.keys()]).T
                cps_onset = safe_changepoint_detection(ONSET_MATRIX, penalty_onset)
                if not cps_onset: continue
                t0_glob = start_idx + int(onset_time_indices[cps_onset[0]])

                # --- STAGE 2: TERMINATION DETECTION ---
                term_raw_features = {
                    'rms': rms_envelope(filtered_signal, term_window_size, term_step)[0],
                    'theta': relative_bandpower_envelope(filtered_signal, fs, bands['theta'], (0.5, 150), term_window_size, term_step)[0],
                    'alpha': relative_bandpower_envelope(filtered_signal, fs, bands['alpha'], (0.5, 150), term_window_size, term_step)[0],
                    'beta': relative_bandpower_envelope(filtered_signal, fs, bands['beta'], (0.5, 150), term_window_size, term_step)[0],
                    'gamma': relative_bandpower_envelope(filtered_signal, fs, bands['gamma'], (0.5, 150), term_window_size, term_step)[0],
                    'll': line_length(filtered_signal, term_window_size, term_step),
                    'se': spectral_entropy(filtered_signal, fs, term_window_size, term_step)
                }
                term_time_indices = rms_envelope(filtered_signal, term_window_size, term_step)[1]
                if any(f.size == 0 for f in term_raw_features.values()) or term_time_indices.size == 0: continue

                min_len_term = min(len(f) for f in term_raw_features.values() if f.size > 0)
                if min_len_term < 10: continue

                s_features_term = {name: exp_weighted_avg(scaler.fit_transform(term_raw_features[name][:min_len_term].reshape(-1, 1)).flatten()) for name in term_weights.keys()}
                TERM_MATRIX = np.vstack([s_features_term[name] * term_weights[name] for name in term_weights.keys()]).T
                cps_term = safe_changepoint_detection(TERM_MATRIX, penalty_term)
                if not cps_term: continue
                tend_glob = start_idx + int(term_time_indices[cps_term[-1]])

                if tend_glob <= t0_glob: continue

                # --- STAGE 3: TRANSITION (MID) DETECTION ---
                tmid_time = np.nan
                if seizure_has_valid_mid_gt:
                    mid_segment = eeg[j, t0_glob:tend_glob]
                    if len(mid_segment) >= trans_window_size:
                        mid_filt = highpass_filter2(mid_segment)
                        trans_raw_features = {
                            'rms': rms_envelope(mid_filt, trans_window_size, trans_step)[0],
                            'theta': relative_bandpower_envelope(mid_filt, fs, bands['theta'], (0.5, 150), trans_window_size, trans_step)[0],
                            'alpha': relative_bandpower_envelope(mid_filt, fs, bands['alpha'], (0.5, 150), trans_window_size, trans_step)[0],
                            'beta': relative_bandpower_envelope(mid_filt, fs, bands['beta'], (0.5, 150), trans_window_size, trans_step)[0],
                            'gamma': relative_bandpower_envelope(mid_filt, fs, bands['gamma'], (0.5, 150), trans_window_size, trans_step)[0],
                            'll': line_length(mid_filt, trans_window_size, trans_step),
                            'se': spectral_entropy(mid_filt, fs, trans_window_size, trans_step)
                        }
                        trans_time_indices = rms_envelope(mid_filt, trans_window_size, trans_step)[1]
                        
                        if not any(f.size == 0 for f in trans_raw_features.values()) and trans_time_indices.size > 5:
                            min_len_trans = min(len(f) for f in trans_raw_features.values() if f.size > 0)
                            s_features_trans = {name: exp_weighted_avg(scaler.fit_transform(trans_raw_features[name][:min_len_trans].reshape(-1, 1)).flatten()) for name in trans_weights.keys()}
                            TRANS_MATRIX = np.vstack([s_features_trans[name] * trans_weights[name] for name in trans_weights.keys()]).T
                            cps_trans = safe_changepoint_detection(TRANS_MATRIX, penalty_trans)
                            if cps_trans and cps_trans[0] < len(trans_time_indices):
                                tmid_time = time_array[t0_glob + trans_time_indices[cps_trans[0]]]

                this_seizure_onset_errors.append(time_array[t0_glob] - gt_t0)
                this_seizure_end_errors.append(time_array[tend_glob] - gt_tend)
                
                if seizure_has_valid_mid_gt and not np.isnan(tmid_time):
                    this_seizure_mid_errors.append(tmid_time - gt_tmid)

            except Exception:
                continue
        
        if this_seizure_onset_errors or this_seizure_end_errors:
            def compute_mae(errors):
                valid = np.array([e for e in errors if not np.isnan(e)])
                return np.mean(np.abs(valid)) if len(valid) > 0 else np.nan
            
            seizure_performance = {
                'seizure': os.path.basename(os.path.dirname(seizure_path)),
                'onset_mae': compute_mae(this_seizure_onset_errors),
                'end_mae': compute_mae(this_seizure_end_errors)
            }
            
            if seizure_has_valid_mid_gt:
                seizure_performance['mid_mae'] = compute_mae(this_seizure_mid_errors)
            else:
                seizure_performance['mid_mae'] = np.nan
            
            all_seizure_performances.append(seizure_performance)

    return all_seizure_performances

def objective(study, seizure_subset, contacts_map, ablate_feature=None):
    trial = study.ask()
    
    # Define the full set of possible features and active features
    all_feature_names = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    active_features = [name for name in all_feature_names if name != ablate_feature]
    
    # --- NEW: Define search space for windowing and penalties ---
    params = {
        'onset_window_size': trial.suggest_int('onset_window_size', 200, 2000, step=100),
        'onset_step': trial.suggest_int('onset_step', 50, 500, step=50),
        'penalty_onset': trial.suggest_int('penalty_onset', 4, 15),
        
        'term_window_size': trial.suggest_int('term_window_size', 200, 2000, step=100),
        'term_step': trial.suggest_int('term_step', 50, 500, step=50),
        'penalty_term': trial.suggest_int('penalty_term', 4, 15),

        'trans_window_size': trial.suggest_int('trans_window_size', 100, 1000, step=50),
        'trans_step': trial.suggest_int('trans_step', 20, 200, step=20),
        'penalty_trans': trial.suggest_int('penalty_trans', 2, 10)
    }

    # Add constraints to ensure step is not larger than window size
    if params['onset_step'] > params['onset_window_size']:
        params['onset_step'] = params['onset_window_size']
    if params['term_step'] > params['term_window_size']:
        params['term_step'] = params['term_window_size']
    if params['trans_step'] > params['trans_window_size']:
        params['trans_step'] = params['trans_window_size']

    # --- NEW: Optimizing feature weights ---
    onset_weights = {name: trial.suggest_float(f'onset_weight_{name}', 0.0, 1.0) for name in active_features}
    term_weights = {name: trial.suggest_float(f'term_weight_{name}', 0.0, 1.0) for name in active_features}
    trans_weights = {name: trial.suggest_float(f'trans_weight_{name}', 0.0, 1.0) for name in active_features}
    
    # Add weights to the params dict to be passed to run_analysis
    params['onset_weights'] = onset_weights
    params['term_weights'] = term_weights
    params['trans_weights'] = trans_weights

    try:
        list_of_seizure_perfs = run_analysis(seizure_subset, contacts_map, **params, ablate_feature=ablate_feature)
        
        all_maes_for_trial = []
        if not list_of_seizure_perfs:
             avg_mae = 999.0
        else:
            for perf_dict in list_of_seizure_perfs:
                for key, mae in perf_dict.items():
                    if key.endswith('_mae') and isinstance(mae, (int, float)) and not np.isnan(mae):
                        all_maes_for_trial.append(mae)
            
            avg_mae = np.mean(all_maes_for_trial) if all_maes_for_trial else 999.0

        study.tell(trial, avg_mae)
        
    except Exception as e:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(f"Trial failed with exception: {e}")

if __name__ == "__main__":
    # This parser defines all the command-line arguments the script accepts
    parser = argparse.ArgumentParser(description="Run a single fold of LOSO-CV for ablation study.")
    parser.add_argument("--fold_index", type=int, required=True, help="The index of the patient to hold out.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for Optuna trials.")
    parser.add_argument("--ablate", type=str, default=None, help="Name of the feature to ablate (e.g., 'alpha', 'll').")
    args = parser.parse_args()
    
    N_TRIALS_PER_FOLD = 50

    # Get the pre-validated list of patients passed by the sbatch script
    patient_list_str = os.getenv('PATIENT_LIST')
    if not patient_list_str:
        print("FATAL: PATIENT_LIST environment variable not set.")
        exit(1)
    valid_patients = patient_list_str.split()

    # Load contacts map (still needed by run_analysis)
    CONTACTS_FILE = "/mnt/beegfs/kumarh4/pname_contacts.mat"
    mat_data = sio.loadmat(CONTACTS_FILE, struct_as_record=True, squeeze_me=True)
    contacts_map = {rec["name"]: list(rec["contacts"]) for rec in mat_data["pname_contacts"]}

    # Load all seizure paths (still needed)
    BASE_INPUT_FOLDER = '/mnt/beegfs/kumarh4/SEEG_2023_new/'
    seizures_by_patient = defaultdict(list)
    for pname in valid_patients:
        patient_folder = os.path.join(BASE_INPUT_FOLDER, pname)
        if not os.path.isdir(patient_folder): continue
        for f in os.listdir(patient_folder):
            if f.startswith('SZ') and 'bipolar_2' in f:
                montage_file = os.path.join(patient_folder, f, 'data_block001_montage.mat')
                if os.path.exists(montage_file):
                    seizures_by_patient[pname].append(montage_file)

    fold_idx = args.fold_index
    patient_to_leave_out = valid_patients[fold_idx]

    # --- The rest of the script is unchanged ---
    
    ablation_info = f"ablating '{args.ablate}'" if args.ablate else "using full feature set"
    if args.ablate:
        RESULTS_DIR = f"cv_results_no_{args.ablate}"
    else:
        RESULTS_DIR = "cv_results_full_model"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"EXECUTING FOLD {fold_idx + 1}/{len(valid_patients)} ({ablation_info}): Holding out <{patient_to_leave_out}> with {args.n_jobs} workers")

    test_seizure_paths = seizures_by_patient[patient_to_leave_out]
    training_seizure_paths = []
    for p_train in valid_patients:
        if p_train != patient_to_leave_out:
            training_seizure_paths.extend(seizures_by_patient[p_train])

    study_db_path = f"sqlite:///{RESULTS_DIR}/fold_{fold_idx}_study.db?timeout=60"
    study_name = f"fold-{fold_idx}"
    study = optuna.create_study(direction='minimize', storage=study_db_path, study_name=study_name, load_if_exists=True)
    
    Parallel(n_jobs=args.n_jobs)(
        delayed(objective)(study, training_seizure_paths, contacts_map, ablate_feature=args.ablate) for _ in range(N_TRIALS_PER_FOLD)
    )

    best_trial = study.best_trial
    
    # Get best hyperparams from the study
    best_params_flat = best_trial.params

    # Reconstruct the nested weight dictionaries from the flat best_params
    all_feature_names = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    active_features = [name for name in all_feature_names if name != args.ablate]
    
    best_params = {}
    
    # Extract weights into nested dictionaries
    onset_weights = {name: best_params_flat.get(f'onset_weight_{name}') for name in active_features}
    term_weights = {name: best_params_flat.get(f'term_weight_{name}') for name in active_features}
    trans_weights = {name: best_params_flat.get(f'trans_weight_{name}') for name in active_features}
    
    best_params['onset_weights'] = onset_weights
    best_params['term_weights'] = term_weights
    best_params['trans_weights'] = trans_weights

    # Add the other parameters, filtering out the flat weight entries
    for key, value in best_params_flat.items():
        if not key.startswith(('onset_weight_', 'term_weight_', 'trans_weight_')):
            best_params[key] = value
    
    test_results = run_analysis(test_seizure_paths, contacts_map, **best_params, ablate_feature=args.ablate)
    
    final_fold_result = {
        "fold_index": fold_idx, "held_out_subject": patient_to_leave_out,
        "seizure_performances": test_results, # MODIFIED: save the list of seizure results
        "best_params": best_trial.params
    }
    
    result_file = os.path.join(RESULTS_DIR, f"fold_{fold_idx}_results.json")
    with open(result_file, 'w') as f:
        json.dump(final_fold_result, f, indent=4)
        
    print(f"Fold {fold_idx + 1} complete. Results saved to {result_file}")
    if os.path.exists(study_db_path.replace("sqlite:///", "")):
        os.remove(study_db_path.replace("sqlite:///", ""))