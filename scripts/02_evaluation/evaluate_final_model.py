#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_fold_results(fold_file):
    """
    Load results from a single fold file
    """
    try:
        with open(fold_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {fold_file}: {e}")
        return None

def calculate_metrics(errors):
    """
    Calculate MAE, RMSE, IQR, and accuracy for a list of errors
    """
    if not errors or len(errors) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0
    
    errors = np.array(errors)
    
    # Basic metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    iqr = np.percentile(errors, 75) - np.percentile(errors, 25)
    accuracy_5s = np.mean(errors <= 5) * 100  # Percentage within 5 seconds
    median_error = np.median(errors)
    n_seizures = len(errors)
    
    return mae, rmse, iqr, accuracy_5s, median_error, n_seizures

def evaluate_final_recommended_model():
    """
    Evaluate the final recommended model with fixed parameters and feature sets
    """
    print("EVALUATING FINAL RECOMMENDED MODEL")
    print("=" * 50)
    
    # Define the final recommended model parameters
    final_model_params = {
        'onset': {
            'window_size': 1000,  # ms
            'step': 150,         # ms
            'penalty': 11,
            'features': ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se'],
            'feature_weights': {
                'rms': 0.54, 'theta': 0.47, 'alpha': 0.64, 
                'beta': 0.67, 'gamma': 0.63, 'll': 0.57, 'se': 0.40
            }
        },
        'trans': {
            'window_size': 700,   # ms
            'step': 130,          # ms
            'penalty': 7,
            'features': ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se'],
            'feature_weights': {
                'rms': 0.59, 'theta': 0.53, 'alpha': 0.48,
                'beta': 0.55, 'gamma': 0.43, 'll': 0.53, 'se': 0.43
            }
        },
        'term': {
            'window_size': 1000,  # ms
            'step': 200,          # ms
            'penalty': 10,
            'features': ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se'],
            'feature_weights': {
                'rms': 0.63, 'theta': 0.54, 'alpha': 0.45,
                'beta': 0.47, 'gamma': 0.52, 'll': 0.62, 'se': 0.42
            }
        }
    }
    
    print("FINAL RECOMMENDED MODEL PARAMETERS:")
    print("-" * 40)
    for stage in ['onset', 'trans', 'term']:
        params = final_model_params[stage]
        print(f"\n{stage.upper()} STAGE:")
        print(f"  Window Size: {params['window_size']} ms")
        print(f"  Step Size:   {params['step']} ms")
        print(f"  Penalty:     {params['penalty']}")
        print(f"  Features:    {', '.join(params['features'])}")
        print("  Feature Weights:")
        for feature, weight in params['feature_weights'].items():
            print(f"    {feature.upper()}: {weight:.2f}")
    
    # Base path for results
    base_path = Path("..")
    
    # Find all cv_results directories
    result_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("cv_results")]
    
    if not result_dirs:
        print("No cv_results directories found")
        return
    
    print(f"\nFound {len(result_dirs)} result directories")
    
    # Collect all performance data
    all_subject_data = []
    all_fold_data = []
    
    for results_dir in sorted(result_dirs):
        fold_files = sorted(list(results_dir.glob("fold_*_results.json")))
        
        if not fold_files:
            continue
            
        print(f"\nProcessing {results_dir.name}...")
        
        # Get condition name
        if "no_" in results_dir.name:
            if "alpha-theta" in results_dir.name:
                condition = "No_ALPHA-THETA"
            else:
                condition = f"No_{results_dir.name.replace('cv_results_no_', '').upper()}"
        else:
            condition = "Full_Model"
        
        # Process each fold
        subject_fold_data = []
        
        for fold_file in fold_files:
            fold_data = load_fold_results(fold_file)
            if fold_data is None:
                continue
                
            fold_index = fold_data['fold_index']
            patient = fold_data['held_out_subject']
            seizure_perfs = fold_data['seizure_performances']
            
            # Extract errors for each type
            onset_errors = []
            mid_errors = []
            end_errors = []
            
            seizures_with_mid = 0
            total_seizures = len(seizure_perfs)
            
            for perf in seizure_perfs:
                # Collect onset and end errors (should always be present)
                if not np.isnan(perf.get('onset_mae', np.nan)):
                    onset_errors.append(perf['onset_mae'])
                if not np.isnan(perf.get('end_mae', np.nan)):
                    end_errors.append(perf['end_mae'])
                
                # Collect mid errors only if not NaN
                if not np.isnan(perf.get('mid_mae', np.nan)):
                    mid_errors.append(perf['mid_mae'])
                    seizures_with_mid += 1
            
            # Calculate fold-level metrics
            onset_mae, onset_rmse, onset_iqr, onset_acc, onset_med, onset_n = calculate_metrics(onset_errors)
            mid_mae, mid_rmse, mid_iqr, mid_acc, mid_med, mid_n = calculate_metrics(mid_errors)
            end_mae, end_rmse, end_iqr, end_acc, end_med, end_n = calculate_metrics(end_errors)
            
            fold_summary = {
                'condition': condition,
                'fold_index': fold_index,
                'patient': patient,
                'total_seizures': total_seizures,
                'seizures_with_mid': seizures_with_mid,
                'onset_mae': onset_mae,
                'onset_rmse': onset_rmse,
                'onset_iqr': onset_iqr,
                'onset_accuracy_5s': onset_acc,
                'onset_median': onset_med,
                'onset_n': onset_n,
                'mid_mae': mid_mae,
                'mid_rmse': mid_rmse,
                'mid_iqr': mid_iqr,
                'mid_accuracy_5s': mid_acc,
                'mid_median': mid_med,
                'mid_n': mid_n,
                'end_mae': end_mae,
                'end_rmse': end_rmse,
                'end_iqr': end_iqr,
                'end_accuracy_5s': end_acc,
                'end_median': end_med,
                'end_n': end_n
            }
            
            subject_fold_data.append(fold_summary)
            all_fold_data.append(fold_summary)
        
        # Aggregate subject-level data
        if subject_fold_data:
            df_subject = pd.DataFrame(subject_fold_data)
            
            # Calculate subject-level aggregates
            subject_summary = {
                'condition': condition,
                'patient': subject_fold_data[0]['patient'],
                'total_folds': len(subject_fold_data),
                'total_seizures': df_subject['total_seizures'].sum(),
                'seizures_with_mid': df_subject['seizures_with_mid'].sum(),
                # Onset metrics (weighted average by number of seizures)
                'onset_mae': np.average(df_subject['onset_mae'], weights=df_subject['onset_n']) if (df_subject['onset_n'] > 0).all() else np.nan,
                'onset_rmse': np.average(df_subject['onset_rmse'], weights=df_subject['onset_n']) if (df_subject['onset_n'] > 0).all() else np.nan,
                'onset_iqr': np.mean(df_subject['onset_iqr']),
                'onset_accuracy_5s': np.mean(df_subject['onset_accuracy_5s']),
                'onset_median': np.median(df_subject['onset_median']),
                # Mid metrics
                'mid_mae': np.average(df_subject['mid_mae'], weights=df_subject['mid_n']) if (df_subject['mid_n'] > 0).all() else np.nan,
                'mid_rmse': np.average(df_subject['mid_rmse'], weights=df_subject['mid_n']) if (df_subject['mid_n'] > 0).all() else np.nan,
                'mid_iqr': np.mean(df_subject['mid_iqr']),
                'mid_accuracy_5s': np.mean(df_subject['mid_accuracy_5s']),
                'mid_median': np.median(df_subject['mid_median']),
                # End metrics
                'end_mae': np.average(df_subject['end_mae'], weights=df_subject['end_n']) if (df_subject['end_n'] > 0).all() else np.nan,
                'end_rmse': np.average(df_subject['end_rmse'], weights=df_subject['end_n']) if (df_subject['end_n'] > 0).all() else np.nan,
                'end_iqr': np.mean(df_subject['end_iqr']),
                'end_accuracy_5s': np.mean(df_subject['end_accuracy_5s']),
                'end_median': np.median(df_subject['end_median'])
            }
            
            all_subject_data.append(subject_summary)
    
    # Convert to DataFrames
    df_folds = pd.DataFrame(all_fold_data)
    df_subjects = pd.DataFrame(all_subject_data)
    
    # Filter for Full_Model results (the final recommended model)
    full_model_folds = df_folds[df_folds['condition'] == 'Full_Model']
    full_model_subjects = df_subjects[df_subjects['condition'] == 'Full_Model']
    
    if full_model_folds.empty:
        print("No Full_Model results found")
        return
    
    print(f"\nFINAL MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # Overall performance metrics
    print(f"\nOVERALL PERFORMANCE (Aggregated from {len(full_model_folds)} folds):")
    print("-" * 60)
    
    # Calculate overall metrics
    overall_onset_mae = np.average(full_model_folds['onset_mae'], weights=full_model_folds['onset_n']) if (full_model_folds['onset_n'] > 0).all() else np.nan
    overall_onset_rmse = np.average(full_model_folds['onset_rmse'], weights=full_model_folds['onset_n']) if (full_model_folds['onset_n'] > 0).all() else np.nan
    overall_onset_iqr = np.mean(full_model_folds['onset_iqr'])
    overall_onset_acc = np.mean(full_model_folds['onset_accuracy_5s'])
    
    overall_mid_mae = np.average(full_model_folds['mid_mae'], weights=full_model_folds['mid_n']) if (full_model_folds['mid_n'] > 0).all() else np.nan
    overall_mid_rmse = np.average(full_model_folds['mid_rmse'], weights=full_model_folds['mid_n']) if (full_model_folds['mid_n'] > 0).all() else np.nan
    overall_mid_iqr = np.mean(full_model_folds['mid_iqr'])
    overall_mid_acc = np.mean(full_model_folds['mid_accuracy_5s'])
    
    overall_end_mae = np.average(full_model_folds['end_mae'], weights=full_model_folds['end_n']) if (full_model_folds['end_n'] > 0).all() else np.nan
    overall_end_rmse = np.average(full_model_folds['end_rmse'], weights=full_model_folds['end_n']) if (full_model_folds['end_n'] > 0).all() else np.nan
    overall_end_iqr = np.mean(full_model_folds['end_iqr'])
    overall_end_acc = np.mean(full_model_folds['end_accuracy_5s'])
    
    print(f"{'Metric':<15} {'MAE (sec)':<12} {'RMSE (sec)':<12} {'IQR (sec)':<12} {'Acc (±5s)':<12} {'N Seizures':<12}")
    print("-" * 80)
    print(f"{'Onset':<15} {overall_onset_mae:<12.2f} {overall_onset_rmse:<12.2f} {overall_onset_iqr:<12.2f} {overall_onset_acc:<12.1f}% {full_model_folds['onset_n'].sum():<12}")
    print(f"{'Mid-point':<15} {overall_mid_mae:<12.2f} {overall_mid_rmse:<12.2f} {overall_mid_iqr:<12.2f} {overall_mid_acc:<12.1f}% {full_model_folds['mid_n'].sum():<12}")
    print(f"{'Termination':<15} {overall_end_mae:<12.2f} {overall_end_rmse:<12.2f} {overall_end_iqr:<12.2f} {overall_end_acc:<12.1f}% {full_model_folds['end_n'].sum():<12}")
    
    # Subject-level performance
    print(f"\n\nSUBJECT-LEVEL PERFORMANCE:")
    print("-" * 100)
    print(f"{'Patient':<20} {'Onset':<25} {'Mid-point':<25} {'Termination':<25}")
    print(f"{'':<20} {'MAE/RMSE/IQR':<25} {'MAE/RMSE/IQR':<25} {'MAE/RMSE/IQR':<25}")
    print("-" * 100)
    
    for _, subject in full_model_subjects.iterrows():
        print(f"{subject['patient']:<20} "
              f"{subject['onset_mae']:.1f}/{subject['onset_rmse']:.1f}/{subject['onset_iqr']:.1f}   "
              f"{subject['mid_mae']:.1f}/{subject['mid_rmse']:.1f}/{subject['mid_iqr']:.1f}   "
              f"{subject['end_mae']:.1f}/{subject['end_rmse']:.1f}/{subject['end_iqr']:.1f}")
    
    # Save detailed results
    print(f"\n\nSaving detailed results...")
    
    # Save fold-level results
    fold_output = full_model_folds[[
        'patient', 'fold_index', 'onset_mae', 'onset_rmse', 'onset_iqr', 'onset_accuracy_5s',
        'mid_mae', 'mid_rmse', 'mid_iqr', 'mid_accuracy_5s',
        'end_mae', 'end_rmse', 'end_iqr', 'end_accuracy_5s'
    ]].copy()
    
    fold_output.to_csv('final_model_detailed_fold_results.csv', index=False)
    
    # Save subject-level results
    subject_output = full_model_subjects[[
        'patient', 'total_folds', 'total_seizures', 'seizures_with_mid',
        'onset_mae', 'onset_rmse', 'onset_iqr', 'onset_accuracy_5s',
        'mid_mae', 'mid_rmse', 'mid_iqr', 'mid_accuracy_5s',
        'end_mae', 'end_rmse', 'end_iqr', 'end_accuracy_5s'
    ]].copy()
    
    subject_output.to_csv('final_model_subject_level_results.csv', index=False)
    
    # Save overall summary
    overall_summary = pd.DataFrame([{
        'metric': 'Overall',
        'onset_mae': overall_onset_mae,
        'onset_rmse': overall_onset_rmse,
        'onset_iqr': overall_onset_iqr,
        'onset_accuracy_5s': overall_onset_acc,
        'mid_mae': overall_mid_mae,
        'mid_rmse': overall_mid_rmse,
        'mid_iqr': overall_mid_iqr,
        'mid_accuracy_5s': overall_mid_acc,
        'end_mae': overall_end_mae,
        'end_rmse': overall_end_rmse,
        'end_iqr': overall_end_iqr,
        'end_accuracy_5s': overall_end_acc,
        'total_folds': len(full_model_folds),
        'total_onset_seizures': full_model_folds['onset_n'].sum(),
        'total_mid_seizures': full_model_folds['mid_n'].sum(),
        'total_end_seizures': full_model_folds['end_n'].sum()
    }])
    
    overall_summary.to_csv('final_model_overall_performance_summary.csv', index=False)
    
    print(f"\nFINAL MODEL EVALUATION COMPLETE")
    print("=" * 40)
    print(f"Files generated:")
    print(f"- final_model_detailed_fold_results.csv")
    print(f"- final_model_subject_level_results.csv") 
    print(f"- final_model_overall_performance_summary.csv")
    print(f"\nFinal Model Performance Summary:")
    print(f"- Onset Detection:    MAE={overall_onset_mae:.2f} sec, RMSE={overall_onset_rmse:.2f} sec")
    print(f"- Mid-point Detection: MAE={overall_mid_mae:.2f} sec, RMSE={overall_mid_rmse:.2f} sec")
    print(f"- Termination Detection: MAE={overall_end_mae:.2f} sec, RMSE={overall_end_rmse:.2f} sec")
    print(f"- Overall Accuracy (±5s): {(overall_onset_acc + overall_mid_acc + overall_end_acc)/3:.1f}%")

if __name__ == "__main__":
    evaluate_final_recommended_model()