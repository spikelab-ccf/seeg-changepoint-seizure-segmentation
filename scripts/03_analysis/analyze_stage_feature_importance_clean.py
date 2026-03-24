#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def analyze_individual_stage_feature_importance():
    """
    Analyze feature importance for each detection stage individually
    """
    print("INDIVIDUAL STAGE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Load the optimized hyperparameters
    try:
        df = pd.read_csv("../optimized_hyperparameters_summary.csv")
        print(f"Loaded {len(df)} fold result records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Features to analyze
    features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    stages = ['onset', 'term', 'trans']
    
    # Store results for each stage
    stage_results = {}
    
    # Analyze each stage
    for stage in stages:
        print(f"\n{stage.upper()} STAGE FEATURE IMPORTANCE:")
        print("-" * 40)
        
        stage_weights = {}
        for feature in features:
            column_name = f'{stage}_weight_{feature}'
            if column_name in df.columns:
                median_weight = df[column_name].median()
                mean_weight = df[column_name].mean()
                std_weight = df[column_name].std()
                min_weight = df[column_name].min()
                max_weight = df[column_name].max()
                
                stage_weights[feature] = {
                    'median': median_weight,
                    'mean': mean_weight,
                    'std': std_weight,
                    'min': min_weight,
                    'max': max_weight
                }
                
                print(f"{feature.upper():>6}: {median_weight:.3f} +/- {std_weight:.3f} (range: {min_weight:.3f}-{max_weight:.3f})")
        
        # Sort by median weight
        sorted_features = sorted(stage_weights.items(), key=lambda x: x[1]['median'], reverse=True)
        
        print(f"\n{stage.upper()} STAGE RANKING:")
        for i, (feature, stats) in enumerate(sorted_features, 1):
            print(f"  {i}. {feature.upper():>6}: {stats['median']:.3f}")
        
        stage_results[stage] = sorted_features
    
    return stage_results, df

def create_individual_stage_visualizations(stage_results):
    """
    Create individual visualizations for each stage
    """
    print("\n\nCREATING INDIVIDUAL STAGE VISUALIZATIONS")
    print("=" * 50)
    
    features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    stage_names = {'onset': 'Onset', 'trans': 'Transition', 'term': 'Termination'}
    
    # Create individual plots for each stage
    for stage_key, sorted_features in stage_results.items():
        # Extract feature names and weights
        feature_names = [item[0].upper() for item in sorted_features]
        weights = [item[1]['median'] for item in sorted_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(feature_names))
        colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
        
        bars = ax.barh(y_pos, weights, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, weights)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=12, weight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=14)
        ax.set_xlabel('Median Feature Weight', fontsize=14)
        ax.set_title(f'{stage_names[stage_key]} Detection Stage\nFeature Importance', fontsize=16, weight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at 0.5 for reference
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(0.51, 0.5, '0.5', transform=ax.get_yaxis_transform(), 
                ha='left', va='center', color='red', fontweight='bold')
        
        plt.tight_layout()
        filename = f'feature_importance_{stage_key}_stage.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {stage_names[stage_key]} stage feature importance plot to '{filename}'")
    
    # Create comparison plot showing all three stages
    create_stages_comparison_plot(stage_results)

def create_stages_comparison_plot(stage_results):
    """
    Create a comparison plot showing feature importance across all stages
    """
    print("\nCreating stages comparison plot...")
    
    features = ['RMS', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'LL', 'SE']
    
    # Prepare data for comparison
    onset_weights = [next((item[1]['median'] for item in stage_results['onset'] if item[0] == feature.lower()), 0) 
                     for feature in ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']]
    trans_weights = [next((item[1]['median'] for item in stage_results['trans'] if item[0] == feature.lower()), 0) 
                     for feature in ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']]
    term_weights = [next((item[1]['median'] for item in stage_results['term'] if item[0] == feature.lower()), 0) 
                     for feature in ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']]
    
    # Sort by average weight
    avg_weights = [(o + t + te) / 3 for o, t, te in zip(onset_weights, trans_weights, term_weights)]
    sorted_indices = np.argsort(avg_weights)[::-1]
    
    sorted_features = [features[i] for i in sorted_indices]
    sorted_onset = [onset_weights[i] for i in sorted_indices]
    sorted_trans = [trans_weights[i] for i in sorted_indices]
    sorted_term = [term_weights[i] for i in sorted_indices]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(sorted_features))
    width = 0.25
    
    bars1 = ax.bar(x - width, sorted_onset, width, 
                   label='Onset', color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, sorted_trans, width, 
                   label='Transition', color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, sorted_term, width, 
                   label='Termination', color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, sorted_onset)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    for i, (bar, value) in enumerate(zip(bars2, sorted_trans)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    for i, (bar, value) in enumerate(zip(bars3, sorted_term)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Features (Ranked by Average Importance)', fontsize=14)
    ax.set_ylabel('Feature Weight', fontsize=14)
    ax.set_title('Feature Importance Comparison Across Detection Stages', fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_features, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'feature_importance_all_stages_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved stages comparison plot to '{filename}'")

def identify_stage_specific_features(stage_results):
    """
    Identify features that are particularly important for specific stages
    """
    print("\n\nSTAGE-SPECIFIC FEATURE ANALYSIS")
    print("=" * 40)
    
    # Calculate coefficient of variation for each feature across stages
    features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    
    print("\nFeatures with high variation across stages (stage-specific):")
    print("-" * 60)
    
    stage_specific_features = []
    
    for feature in features:
        # Get weights for this feature across all stages
        onset_weight = next((item[1]['median'] for item in stage_results['onset'] if item[0] == feature), 0)
        trans_weight = next((item[1]['median'] for item in stage_results['trans'] if item[0] == feature), 0)
        term_weight = next((item[1]['median'] for item in stage_results['term'] if item[0] == feature), 0)
        
        weights = [onset_weight, trans_weight, term_weight]
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        
        # Coefficient of variation (std/mean)
        if mean_weight > 0:
            cv = std_weight / mean_weight
        else:
            cv = 0
            
        if cv > 0.3:  # High coefficient of variation indicates stage-specific importance
            stage_specific_features.append((feature, cv, weights))
            print(f"{feature.upper():>6}: CV={cv:.2f} (Onset:{onset_weight:.3f}, Trans:{trans_weight:.3f}, Term:{term_weight:.3f})")
    
    print(f"\nHighly stage-specific features: {[f[0].upper() for f in stage_specific_features]}")
    
    return stage_specific_features

def create_detailed_statistics_table(df):
    """
    Create detailed statistics for feature weights by stage
    """
    print("\n\nDETAILED STATISTICS TABLE")
    print("=" * 40)
    
    features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    stages = ['onset', 'trans', 'term']
    
    # Create detailed table
    print(f"{'Feature':<8} {'Stage':<6} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 65)
    
    for feature in features:
        for stage in stages:
            column_name = f'{stage}_weight_{feature}'
            if column_name in df.columns:
                mean_val = df[column_name].mean()
                median_val = df[column_name].median()
                std_val = df[column_name].std()
                min_val = df[column_name].min()
                max_val = df[column_name].max()
                
                print(f"{feature.upper():<8} {stage.upper():<6} {mean_val:<8.3f} {median_val:<8.3f} {std_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f}")
    
    # Save to CSV
    stats_data = []
    for feature in features:
        for stage in stages:
            column_name = f'{stage}_weight_{feature}'
            if column_name in df.columns:
                stats_data.append({
                    'Feature': feature.upper(),
                    'Stage': stage.upper(),
                    'Mean': df[column_name].mean(),
                    'Median': df[column_name].median(),
                    'Std': df[column_name].std(),
                    'Min': df[column_name].min(),
                    'Max': df[column_name].max()
                })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv('detailed_feature_weights_statistics.csv', index=False)
    print(f"\nDetailed statistics saved to 'detailed_feature_weights_statistics.csv'")

def main():
    """
    Main function to analyze individual stage feature importance
    """
    print("Analyzing individual stage feature importance...")
    
    # Analyze feature importance for each stage
    stage_results, df = analyze_individual_stage_feature_importance()
    
    # Create individual visualizations
    create_individual_stage_visualizations(stage_results)
    
    # Identify stage-specific features
    stage_specific_features = identify_stage_specific_features(stage_results)
    
    # Create detailed statistics table
    create_detailed_statistics_table(df)
    
    print(f"\nSUMMARY OF INDIVIDUAL STAGE ANALYSIS:")
    print("=" * 50)
    
    print(f"\nOnset Detection Key Features:")
    for i, (feature, stats) in enumerate(stage_results['onset'][:3], 1):
        print(f"  {i}. {feature.upper()}: {stats['median']:.3f}")
    
    print(f"\nTransition Detection Key Features:")
    for i, (feature, stats) in enumerate(stage_results['trans'][:3], 1):
        print(f"  {i}. {feature.upper()}: {stats['median']:.3f}")
    
    print(f"\nTermination Detection Key Features:")
    for i, (feature, stats) in enumerate(stage_results['term'][:3], 1):
        print(f"  {i}. {feature.upper()}: {stats['median']:.3f}")
    
    print(f"\nStage-specific features identified: {[f[0].upper() for f in stage_specific_features]}")
    
    print(f"\nAll individual stage feature importance visualizations have been generated!")
    print(f"\nGenerated files:")
    print(f"- feature_importance_onset_stage.png")
    print(f"- feature_importance_trans_stage.png") 
    print(f"- feature_importance_term_stage.png")
    print(f"- feature_importance_all_stages_comparison.png")
    print(f"- detailed_feature_weights_statistics.csv")

if __name__ == "__main__":
    main()