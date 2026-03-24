#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

def load_ablation_data():
    """
    Load all ablation study results and prepare data for tables and plots
    """
    # Go up one directory to access the cv_results directories
    base_path = Path("..")
    result_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("cv_results")]
    
    if not result_dirs:
        print("No cv_results directories found")
        return None, None

    all_fold_data = {}
    all_fold_maes = {}

    for results_dir in sorted(result_dirs):
        if "no_" in results_dir.name:
            condition = f"No_{results_dir.name.replace('cv_results_no_', '').upper()}"
        else:
            condition = "Full_Model"
        
        fold_files = sorted(list(results_dir.glob("fold_*_results.json")))
        condition_fold_data = []
        condition_maes = []
        
        for fold_file in fold_files:
            with open(fold_file, 'r') as f:
                data = json.load(f)
            
            fold_idx = data['fold_index']
            patient = data['held_out_subject']
            seizure_perfs = data['seizure_performances']
            
            # Extract errors for this fold
            fold_onset_errors = []
            fold_mid_errors = []
            fold_end_errors = []
            
            seizures_with_mid = 0
            total_seizures = len(seizure_perfs)
            
            for perf in seizure_perfs:
                # Collect onset and end errors (should always be present)
                if not np.isnan(perf.get('onset_mae', np.nan)):
                    fold_onset_errors.append(perf['onset_mae'])
                if not np.isnan(perf.get('end_mae', np.nan)):
                    fold_end_errors.append(perf['end_mae'])
                
                # Collect mid errors only if not NaN
                if not np.isnan(perf.get('mid_mae', np.nan)):
                    fold_mid_errors.append(perf['mid_mae'])
                    seizures_with_mid += 1
            
            # Calculate fold-level MAEs
            fold_onset_mae = np.mean(fold_onset_errors) if fold_onset_errors else np.nan
            fold_mid_mae = np.mean(fold_mid_errors) if fold_mid_errors else np.nan
            fold_end_mae = np.mean(fold_end_errors) if fold_end_errors else np.nan
            
            condition_fold_data.append({
                'fold': fold_idx,
                'patient': patient,
                'total_seizures': total_seizures,
                'seizures_with_mid': seizures_with_mid,
                'onset_mae': fold_onset_mae,
                'mid_mae': fold_mid_mae,
                'end_mae': fold_end_mae
            })
            
            # For statistical analysis, we need the mean MAE for each fold
            if fold_mid_errors:
                fold_mae = np.mean(fold_mid_errors)
            else:
                fold_mae = np.nan
            condition_maes.append(fold_mae)
        
        all_fold_data[condition] = condition_fold_data
        all_fold_maes[condition] = condition_maes

    return all_fold_data, all_fold_maes

def generate_table_2(all_fold_data):
    """
    Generate Table 2: Ablation Study Comparison
    """
    print("Generating Table 2: Ablation Study Comparison")
    print("="*80)
    
    # Prepare data for comparison
    comparison_data = []
    
    for condition, fold_data in all_fold_data.items():
        # Calculate overall statistics
        onset_maes = [f['onset_mae'] for f in fold_data if not np.isnan(f['onset_mae'])]
        mid_maes = [f['mid_mae'] for f in fold_data if not np.isnan(f['mid_mae'])]
        end_maes = [f['end_mae'] for f in fold_data if not np.isnan(f['end_mae'])]
        
        onset_mean = np.mean(onset_maes) if onset_maes else np.nan
        onset_std = np.std(onset_maes) if onset_maes else np.nan
        mid_mean = np.mean(mid_maes) if mid_maes else np.nan
        mid_std = np.std(mid_maes) if mid_maes else np.nan
        end_mean = np.mean(end_maes) if end_maes else np.nan
        end_std = np.std(end_maes) if end_maes else np.nan
        
        valid_mid_folds = sum(1 for f in fold_data if not np.isnan(f['mid_mae']))
        mid_n_seizures = sum(f['seizures_with_mid'] for f in fold_data)
        
        comparison_data.append({
            'condition': condition,
            'onset_mae': onset_mean,
            'onset_std': onset_std,
            'mid_mae': mid_mean,
            'mid_std': mid_std,
            'end_mae': end_mean,
            'end_std': end_std,
            'valid_mid_folds': valid_mid_folds,
            'mid_n_seizures': mid_n_seizures
        })
    
    # Print table
    print("| Condition    | Onset MAE     | Mid MAE       | End MAE       | Mid Folds |")
    print("|--------------|---------------|---------------|---------------|-----------|")
    
    # Sort by mid MAE (ascending)
    sorted_data = sorted(comparison_data, key=lambda x: x['mid_mae'] if not np.isnan(x['mid_mae']) else 999)
    
    for data in sorted_data:
        onset_str = f"{data['onset_mae']:.2f}+/-{data['onset_std']:.2f}" if not np.isnan(data['onset_mae']) else "N/A"
        mid_str = f"{data['mid_mae']:.2f}+/-{data['mid_std']:.2f}" if not np.isnan(data['mid_mae']) else "N/A"
        end_str = f"{data['end_mae']:.2f}+/-{data['end_std']:.2f}" if not np.isnan(data['end_mae']) else "N/A"
        
        print(f"| {data['condition']:<12} | {onset_str:<13} | {mid_str:<13} | {end_str:<13} | {data['valid_mid_folds']}/10      |")
    
    print("="*80)
    
    # Save to CSV
    df = pd.DataFrame(sorted_data)
    df.to_csv("table_2_ablation_comparison.csv", index=False)
    print("Table 2 saved to 'table_2_ablation_comparison.csv'")
    
    return sorted_data

def generate_table_3(all_fold_maes):
    """
    Generate Table 3: Statistical Significance Analysis
    """
    print("\nGenerating Table 3: Statistical Significance Analysis")
    print("="*80)
    
    if 'Full_Model' not in all_fold_maes:
        print("Full_Model results not found. Cannot perform comparison.")
        return
    
    # Remove NaN values and align data
    df = pd.DataFrame(all_fold_maes)
    df = df.dropna() # Use only folds where all models produced a result

    print(f"Found {len(df)} common folds for statistical comparison.")

    # Perform pairwise comparisons against the full model
    baseline_data = df['Full_Model']
    ablation_models = [col for col in df.columns if col != 'Full_Model']
    
    comparisons = []
    p_values = []
    mean_differences = []

    for model_name in ablation_models:
        comparison_data = df[model_name]
        
        # Calculate mean difference
        mean_diff = np.mean(comparison_data) - np.mean(baseline_data)
        mean_differences.append(mean_diff)
        
        # Wilcoxon signed-rank test
        try:
            stat, p_val = wilcoxon(baseline_data, comparison_data)
            comparisons.append(model_name)
            p_values.append(p_val)
        except ValueError as e:
            print(f"Could not compare Full_Model with {model_name}: {e}")

    if not comparisons:
        print("No valid comparisons could be made.")
        return

    # Apply correction for multiple comparisons
    reject, p_vals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

    # Display results
    results_df = pd.DataFrame({
        'Comparison': [f'Full_Model vs. {name}' for name in comparisons],
        'Mean_Difference': mean_differences,
        'P-Value (raw)': p_values,
        'P-Value (Holm-corrected)': p_vals_corrected,
        'Significant (p < 0.05)': reject
    })

    # Sort by p-value
    results_df = results_df.sort_values('P-Value (raw)')
    
    print("| Comparison                        | Mean Diff | P-Value (raw) | P-Value (corrected) | Significant |")
    print("|----------------------------------|-----------|---------------|---------------------|-------------|")
    
    for _, row in results_df.iterrows():
        mean_diff_str = f"{row['Mean_Difference']:.3f}"
        p_raw_str = f"{row['P-Value (raw)']:.4f}"
        p_corr_str = f"{row['P-Value (Holm-corrected)']:.4f}"
        sig_str = "Yes" if row['Significant (p < 0.05)'] else "No"
        
        print(f"| {row['Comparison']:<32} | {mean_diff_str:>9} | {p_raw_str:>13} | {p_corr_str:>19} | {sig_str:>11} |")
    
    print("="*80)
    
    # Save to CSV
    results_df.to_csv("table_3_statistical_analysis.csv", index=False)
    print("Table 3 saved to 'table_3_statistical_analysis.csv'")
    
    return results_df

def generate_error_distribution_plot(all_fold_maes):
    """
    Generate error distribution plots
    """
    print("\nGenerating Error Distribution Plots")
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution of errors for each condition
    conditions = list(all_fold_maes.keys())
    data_for_boxplot = []
    labels = []
    
    for condition in conditions:
        errors = [e for e in all_fold_maes[condition] if not np.isnan(e)]
        if errors:
            data_for_boxplot.append(errors)
            labels.append(condition.replace('_', ' '))
    
    # Box plot
    axes[0].boxplot(data_for_boxplot, labels=labels)
    axes[0].set_title('Distribution of Mid-Point Detection Errors', fontsize=14, weight='bold')
    axes[0].set_ylabel('Mean Absolute Error (seconds)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Error vs Distribution (binned)
    if 'Full_Model' in all_fold_maes:
        full_model_errors = [e for e in all_fold_maes['Full_Model'] if not np.isnan(e)]
        if full_model_errors:
            # Create bins
            bins = np.linspace(0, max(full_model_errors) + 2, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_counts, _ = np.histogram(full_model_errors, bins=bins)
            
            axes[1].bar(bin_centers, bin_counts, width=bins[1]-bins[0], alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].set_title('Distribution of Full Model Errors (Binned)', fontsize=14, weight='bold')
            axes[1].set_xlabel('Mean Absolute Error (seconds)', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Error distribution plots saved to 'error_distribution.png'")

def generate_error_vs_distribution_binned_plot(all_fold_maes):
    """
    Generate Error vs Distribution binned plot
    """
    print("\nGenerating Error vs Distribution Binned Plot")
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Focus on the full model for this plot
    if 'Full_Model' in all_fold_maes:
        full_model_errors = [e for e in all_fold_maes['Full_Model'] if not np.isnan(e)]
        
        if full_model_errors:
            # Create bins for error ranges
            bins = np.linspace(0, max(full_model_errors) + 2, 11)
            bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
            bin_counts, _ = np.histogram(full_model_errors, bins=bins)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(bin_counts)), bin_counts, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Error Range (seconds)', fontsize=12)
            ax.set_ylabel('Number of Seizures', fontsize=12)
            ax.set_title('Distribution of Mid-Point Detection Errors (Binned)', fontsize=14, weight='bold')
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45)
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, bin_counts)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(count), ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('Error_vs_Distribution_binned_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Error vs Distribution binned plot saved to 'Error_vs_Distribution_binned_plot.png'")

def main():
    """
    Main function to generate all tables and plots
    """
    print("Loading ablation study data...")
    all_fold_data, all_fold_maes = load_ablation_data()
    
    if not all_fold_data or not all_fold_maes:
        print("Failed to load data. Exiting.")
        return
    
    # Generate Table 2
    table2_data = generate_table_2(all_fold_data)
    
    # Generate Table 3
    table3_data = generate_table_3(all_fold_maes)
    
    # Generate plots
    generate_error_distribution_plot(all_fold_maes)
    generate_error_vs_distribution_binned_plot(all_fold_maes)
    
    print("\nAll tables and plots have been generated successfully!")

if __name__ == "__main__":
    main()