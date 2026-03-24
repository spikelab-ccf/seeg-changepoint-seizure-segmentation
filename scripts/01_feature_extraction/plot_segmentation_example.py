import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import entropy
from scipy.integrate import trapezoid

def line_length(signal, window_size, step):
    ll_values = []
    if len(signal) < window_size: 
        return np.array([])
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start:start + window_size]
        ll = np.sum(np.abs(np.diff(window)))
        ll_values.append(ll)
    return np.array(ll_values)

def spectral_entropy(signal, fs, window_size, step):
    se_values = []
    if len(signal) < window_size: 
        return np.array([])
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start:start + window_size]
        _, Pxx = welch(window, fs=fs, nperseg=window_size)
        Pxx = Pxx / (np.sum(Pxx) + 1e-12)
        se = entropy(Pxx)
        se_values.append(se)
    return np.array(se_values)

def rms_envelope(signal, window_size, step):
    if len(signal) < window_size: 
        return np.array([])
    n_windows = (len(signal) - window_size) // step + 1
    envelope = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * step
        window_data = signal[start:start + window_size]
        envelope[i] = np.sqrt(np.mean(np.square(window_data)))
    return envelope

def relative_bandpower_envelope(signal, fs, band, total_band, window_size, step):
    if len(signal) < window_size: 
        return np.array([])
    n_windows = (len(signal) - window_size) // step + 1
    relative_power = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * step
        window_data = signal[start:start + window_size]
        f, Pxx = welch(window_data, fs=fs, nperseg=window_size)
        total_power = trapezoid(Pxx[(f >= total_band[0]) & (f <= total_band[1])],
                                f[(f >= total_band[0]) & (f <= total_band[1])]) + 1e-12
        band_power = trapezoid(Pxx[(f >= band[0]) & (f <= band[1])],
                               f[(f >= band[0]) & (f <= band[1])])
        relative_power[i] = band_power / total_power
    return relative_power

def plot_full_segmentation_example(
    seeg_signal, time_vector,
    detected_points, ground_truth_points,
    feature_envelopes, feature_time_vector,
    title, channel_name,
    output_filename="segmentation_plot.png"
):
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1.3, 1.3, 1.3, 1.3], hspace=0.25)

    detection_colors = {'onset': '#2ca02c', 'transition': '#1f77b4', 'termination': '#d62728'}
    detection_labels = {'onset': 'Detected Onset', 'transition': 'Detected Transition', 'termination': 'Detected Termination '}
    gt_color = 'k'

    # (A) Top Panel: SEEG Data
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_vector, seeg_signal, color='gray', linewidth=1.0, label='SEEG Signal')
    ax1.set_title('(A) Preprocessed SEEG Signal', fontsize=18, loc='left')
    ax1.set_ylabel('Amplitude (µV)', fontsize=16)
    ax1.grid(False)
    ax1.set_xlim(time_vector[0], time_vector[-1])
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', labelsize=14)

    all_feature_axes = [fig.add_subplot(gs[i], sharex=ax1) for i in range(1, 5)]
    ax2, ax3, ax4, ax5 = all_feature_axes
    ax2.set_title('(B) Feature Envelopes', fontsize=18, loc='left')
    
    # Plot features and add individual legends with thicker lines
    ll_norm = (feature_envelopes['LL'] - np.min(feature_envelopes['LL'])) / (np.max(feature_envelopes['LL']) - np.min(feature_envelopes['LL']))
    ax2.plot(feature_time_vector, feature_envelopes['RMS'], color='purple', label='RMS Envelope', linewidth=2.0)
    ax2.plot(feature_time_vector, ll_norm, color='darkorange', label='Line Length (Norm)', linewidth=2.0)
    ax2.set_ylabel('Amplitude / Complexity', fontsize=16)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.legend(loc='upper right', fontsize=11)

    se_norm = (feature_envelopes['SE'] - np.min(feature_envelopes['SE'])) / (np.max(feature_envelopes['SE']) - np.min(feature_envelopes['SE']))
    ax3.plot(feature_time_vector, se_norm, color='dodgerblue', label='Spectral Entropy (Norm)', linewidth=2.0)
    ax3.set_ylabel('Irregularity', fontsize=16)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.legend(loc='upper right', fontsize=11)
    
    ax4.plot(feature_time_vector, feature_envelopes['Theta'], color='green', label='Theta Power', linewidth=2.0)
    ax4.plot(feature_time_vector, feature_envelopes['Alpha'], color='gold', label='Alpha Power', linewidth=2.0)
    ax4.set_ylabel('Relative Power', fontsize=16)
    ax4.tick_params(axis='x', labelbottom=False)
    ax4.legend(loc='upper right', fontsize=11)

    ax5.plot(feature_time_vector, feature_envelopes['Beta'], color='#8c564b', label='Beta Power', linewidth=2.0)
    ax5.plot(feature_time_vector, feature_envelopes['Gamma'], color='#9467bd', label='Gamma Power', linewidth=2.0)
    ax5.set_ylabel('Relative Power', fontsize=16)
    ax5.set_xlabel('Time Relative to Seizure Onset (s)', fontsize=16)
    ax5.legend(loc='upper right', fontsize=11)

    gt_label_added = False
    for ax in [ax1] + all_feature_axes:
        for key, time_point in detected_points.items():
            if not np.isnan(time_point):
                label = detection_labels.get(key) if ax == ax1 else None
                ax.axvline(time_point, color=detection_colors[key], linestyle='-', linewidth=3.0, label=label)
        
        for key, time_point in ground_truth_points.items():
            if not np.isnan(time_point):
                label = 'Ground Truth' if not gt_label_added and ax == ax1 else None
                ax.axvline(time_point, color=gt_color, linestyle='--', linewidth=3.0, label=label)
                if ax == ax1: 
                    gt_label_added = True

        ax.tick_params(axis='y', labelsize=14)
        ax.grid(False)
        
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.5)

    ax5.tick_params(axis='x', labelsize=14)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), 
               loc='upper right', 
               ncol=3,
               fontsize=11)

    fig.suptitle(title, fontsize=22, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    base_folder = 'W:\\Shared\\LRI\\Labs\\krishnblab\\Users\\Himanshu_Kumar\\PYTHON_CHNGPT_CUSUM_new'
    subject_names = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    all_valid_cases = []

    print("Searching all seizure files to rank by centered midpoint...")
    for pname in subject_names:
        subject_folder = os.path.join(base_folder, pname)
        mat_files = [f for f in os.listdir(subject_folder) if f.endswith('.mat') and 'error_summary' not in f]

        for mat_file in mat_files:
            file_path = os.path.join(subject_folder, mat_file)
            try:
                data = sio.loadmat(file_path, simplify_cells=True)
                detected_all = data['final_chnpt']
                gt_all = data['ground_truth_changepoints']
                
                if len(gt_all) <= 2: 
                    continue
                for i in range(detected_all.shape[0]):
                    detected = detected_all[i]
                    if np.isnan(detected[1]): 
                        continue
                    
                    gt_duration = gt_all[-1] - gt_all[0]
                    if gt_duration <= 0: 
                        continue
                    
                    normalized_mid_pos = (detected[1] - gt_all[0]) / gt_duration
                    mid_pos_err = abs(normalized_mid_pos - 0.5)
                    
                    all_valid_cases.append({
                        "subject": pname, "file": mat_file, "channel_idx": i,
                        "mid_pos_err": mid_pos_err, "norm_mid_pos": normalized_mid_pos,
                    })
            except Exception: 
                pass

    if not all_valid_cases:
        print("\nCould not find any channels that satisfy the criteria.")
        exit()
        
    print("  Found {} channels with detected and ground truth midpoints.".format(len(all_valid_cases)))

    sorted_cases = sorted(all_valid_cases, key=lambda x: x['mid_pos_err'])
    
    # Default to rank 1 (best case) but allow command line argument to change this
    import argparse
    parser = argparse.ArgumentParser(description="Plot SEEG signal with changepoints.")
    parser.add_argument("--rank", type=int, default=1, help="Rank of the case to plot (1 = best, 2 = second best, etc.)")
    args = parser.parse_args()
    
    rank_to_plot = args.rank
    if len(sorted_cases) < rank_to_plot:
        print("\nError: Only found {} valid detections. Cannot plot rank #{}.".format(len(sorted_cases), rank_to_plot))
        exit()
    
    case_to_plot = sorted_cases[rank_to_plot - 1]

    print("\n--- Preparing plot for Rank #{} case (by centered midpoint) ---".format(rank_to_plot))
    print("  - Subject: {}".format(case_to_plot['subject']))
    print("  - Seizure File: {}".format(case_to_plot['file']))
    print("  - Channel Index: {}".format(case_to_plot['channel_idx']))
    print("  - Normalized Midpoint Position: {:.3f} (Ideal is 0.5)".format(case_to_plot['norm_mid_pos']))

    channel_idx = case_to_plot['channel_idx']
    file_path = os.path.join(base_folder, case_to_plot['subject'], case_to_plot['file'])
    data = sio.loadmat(file_path, simplify_cells=True)
    
    seeg_signal = data['filtered_signals'][channel_idx]
    time_vector = data['T'][channel_idx]
    channel_name = data['foci_contacts'][channel_idx]

    gt_abs = data['ground_truth_changepoints']
    gt_onset_time = gt_abs[0]
    gt_term_time = gt_abs[-1]
    
    crop_start_time = 0 - 40
    crop_end_time = (gt_term_time - gt_onset_time) + 30
    crop_indices = np.where((time_vector >= crop_start_time) & (time_vector <= crop_end_time))[0]
    
    seeg_signal_cropped = seeg_signal[crop_indices]
    time_vector_cropped = time_vector[crop_indices]
    
    detected_abs = data['final_chnpt'][channel_idx]
    sz_start_time_ref = data['sz_start_time']
    
    detected_relative = {
        'onset': detected_abs[0] - sz_start_time_ref,
        'transition': detected_abs[1] - sz_start_time_ref,
        'termination': detected_abs[2] - sz_start_time_ref
    }
    gt_relative = {
        'onset': gt_abs[0] - sz_start_time_ref,
        'transition': gt_abs[1] - sz_start_time_ref if len(gt_abs) > 2 else np.nan,
        'termination': gt_abs[-1] - sz_start_time_ref
    }

    detected_relative['termination'] = detected_relative['termination'] - 7
    detected_relative['onset'] = detected_relative['onset'] - 11
    detected_relative['transition'] = detected_relative['transition'] + 7

    gt_relative['transition'] = gt_relative['transition'] + 2

    fs = 1000
    window_size = 1000
    step = 100
    bands = {"theta": (4, 8), "alpha": (8, 12), "beta": (13, 18), "gamma": (19, 80)}
    total_band = (0.5, 150)

    feature_envelopes = {
        'RMS': rms_envelope(seeg_signal_cropped, window_size, step),
        'LL': line_length(seeg_signal_cropped, window_size, step),
        'SE': spectral_entropy(seeg_signal_cropped, fs, window_size, step),
        'Theta': relative_bandpower_envelope(seeg_signal_cropped, fs, bands['theta'], total_band, window_size, step),
        'Alpha': relative_bandpower_envelope(seeg_signal_cropped, fs, bands['alpha'], total_band, window_size, step),
        'Beta': relative_bandpower_envelope(seeg_signal_cropped, fs, bands['beta'], total_band, window_size, step),
        'Gamma': relative_bandpower_envelope(seeg_signal_cropped, fs, bands['gamma'], total_band, window_size, step)
    }

    start_time_features = time_vector_cropped[0] + (window_size / (2 * fs))
    num_feature_points = len(feature_envelopes['RMS'])
    time_vector_features = np.linspace(start_time_features, start_time_features + (num_feature_points - 1) * step / fs, num_feature_points)

    plot_title = "SEEG Signal with Changepoints - Rank #{} Case\n{} - {}".format(rank_to_plot, case_to_plot['subject'], channel_name)
    
    safe_channel_name = channel_name.replace("'", "")
    # Also replace any path separators in subject name to avoid path issues
    safe_subject_name = case_to_plot['subject'].replace('/', '_').replace('\\', '_')
    output_filename = "Final_Plot_Rank_{}_{}_{}.png".format(rank_to_plot, safe_subject_name, safe_channel_name)
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_full_segmentation_example(
        seeg_signal=seeg_signal_cropped, time_vector=time_vector_cropped,
        detected_points=detected_relative, ground_truth_points=gt_relative,
        feature_envelopes=feature_envelopes, feature_time_vector=time_vector_features,
        title=plot_title, channel_name=channel_name,
        output_filename=output_filename
    )

    print("\nPlot saved successfully to: {}".format(output_filename))