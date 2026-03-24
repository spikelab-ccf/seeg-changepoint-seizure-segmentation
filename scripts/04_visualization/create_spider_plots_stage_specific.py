#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def create_spider_plot_stage_specific_rankings():
    """
    Create a spider plot showing feature rankings for each detection stage
    """
    print("Creating spider plot for stage-specific feature rankings...")
    
    # Feature importance data (from our analysis)
    features = ['RMS', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'LL', 'SE']
    
    # Rankings for each stage (1 = highest importance, 7 = lowest importance)
    # Based on median weights from our previous analysis
    onset_rankings = [5, 6, 2, 1, 3, 4, 7]  # BETA highest, SE lowest
    trans_rankings = [1, 4, 6, 2, 7, 3, 5]  # RMS highest, GAMMA lowest
    term_rankings = [1, 3, 5, 4, 2, 2, 6]   # RMS highest, SE lowest
    
    # Normalize rankings to 0-1 scale for better visualization
    # Lower ranking number = higher importance = higher value on spider plot
    def normalize_rankings(rankings):
        max_rank = max(rankings)
        min_rank = min(rankings)
        return [(max_rank - r + min_rank) / (max_rank - min_rank) for r in rankings]
    
    onset_normalized = normalize_rankings(onset_rankings)
    trans_normalized = normalize_rankings(trans_rankings)
    term_normalized = normalize_rankings(term_rankings)
    
    # Create spider plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(features)
    
    # Compute angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Normalize data to 0-1 range for better visualization
    def normalize_to_01(data):
        min_val = min(data)
        max_val = max(data)
        return [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for val in data]
    
    onset_plot = onset_normalized + [onset_normalized[0]]
    trans_plot = trans_normalized + [trans_normalized[0]]
    term_plot = term_normalized + [term_normalized[0]]
    
    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=14)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data for each stage
    ax.plot(angles, onset_plot, linewidth=2, linestyle='solid', label='Onset', color='skyblue')
    ax.fill(angles, onset_plot, color='skyblue', alpha=0.25)
    
    ax.plot(angles, trans_plot, linewidth=2, linestyle='solid', label='Transition', color='lightcoral')
    ax.fill(angles, trans_plot, color='lightcoral', alpha=0.25)
    
    ax.plot(angles, term_plot, linewidth=2, linestyle='solid', label='Termination', color='lightgreen')
    ax.fill(angles, term_plot, color='lightgreen', alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    
    # Add title
    ax.set_title('Stage-Specific Feature Importance Rankings\n(Spider/Radar Plot)', 
                size=18, weight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig('spider_plot_stage_specific_feature_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Spider plot saved to 'spider_plot_stage_specific_feature_rankings.png'")
    
    return True

def create_normalized_spider_plot():
    """
    Create a normalized spider plot showing feature importance values
    """
    print("Creating normalized spider plot for feature importance values...")
    
    # Feature importance data (median weights from our analysis)
    features = ['RMS', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'LL', 'SE']
    
    # Median weights for each feature at each stage
    onset_weights = [0.540, 0.470, 0.639, 0.672, 0.630, 0.569, 0.400]
    trans_weights = [0.589, 0.528, 0.478, 0.545, 0.431, 0.535, 0.431]
    term_weights = [0.626, 0.540, 0.452, 0.473, 0.519, 0.617, 0.417]
    
    # Normalize data to 0-1 range for better visualization
    def normalize_data(data):
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        if range_val == 0:
            return [0.5] * len(data)
        return [(val - min_val) / range_val for val in data]
    
    onset_normalized = normalize_data(onset_weights)
    trans_normalized = normalize_data(trans_weights)
    term_normalized = normalize_data(term_weights)
    
    # Create spider plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(features)
    
    # Compute angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Prepare data for plotting
    onset_plot = onset_normalized + [onset_normalized[0]]
    trans_plot = trans_normalized + [trans_normalized[0]]
    term_plot = term_normalized + [term_normalized[0]]
    
    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=14)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data for each stage
    ax.plot(angles, onset_plot, linewidth=2, linestyle='solid', label='Onset', color='skyblue')
    ax.fill(angles, onset_plot, color='skyblue', alpha=0.25)
    
    ax.plot(angles, trans_plot, linewidth=2, linestyle='solid', label='Transition', color='lightcoral')
    ax.fill(angles, trans_plot, color='lightcoral', alpha=0.25)
    
    ax.plot(angles, term_plot, linewidth=2, linestyle='solid', label='Termination', color='lightgreen')
    ax.fill(angles, term_plot, color='lightgreen', alpha=0.25)
    
    # Add value labels at each point
    for i, (angle, onset_val, trans_val, term_val) in enumerate(zip(angles[:-1], onset_normalized, trans_normalized, term_normalized)):
        # Onset values
        ax.text(angle, onset_val + 0.05, f'{onset_weights[i]:.2f}', 
                ha='center', va='center', fontsize=10, weight='bold', color='skyblue')
        
        # Transition values
        ax.text(angle, trans_val + 0.10, f'{trans_weights[i]:.2f}', 
                ha='center', va='center', fontsize=10, weight='bold', color='lightcoral')
        
        # Termination values
        ax.text(angle, term_val + 0.15, f'{term_weights[i]:.2f}', 
                ha='center', va='center', fontsize=10, weight='bold', color='lightgreen')
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    
    # Add title
    ax.set_title('Normalized Feature Importance Across Detection Stages\n(Spider/Radar Plot with Actual Values)', 
                size=18, weight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig('normalized_spider_plot_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Normalized spider plot saved to 'normalized_spider_plot_feature_importance.png'")
    
    return True

def create_feature_importance_comparison_spider():
    """
    Create a comprehensive spider plot comparing feature importance across stages
    """
    print("Creating comprehensive feature importance comparison spider plot...")
    
    # Feature importance data
    features = ['RMS', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'LL', 'SE']
    
    # Median weights for each feature at each stage
    onset_weights = [0.540, 0.470, 0.639, 0.672, 0.630, 0.569, 0.400]
    trans_weights = [0.589, 0.528, 0.478, 0.545, 0.431, 0.535, 0.431]
    term_weights = [0.626, 0.540, 0.452, 0.473, 0.519, 0.617, 0.417]
    
    # Normalize each stage independently to 0-1 range
    def normalize_stage(data):
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        if range_val == 0:
            return [0.5] * len(data)
        return [(val - min_val) / range_val for val in data]
    
    onset_normalized = normalize_stage(onset_weights)
    trans_normalized = normalize_stage(trans_weights)
    term_normalized = normalize_stage(term_weights)
    
    # Create spider plot
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of variables
    N = len(features)
    
    # Compute angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Prepare data for plotting
    onset_plot = onset_normalized + [onset_normalized[0]]
    trans_plot = trans_normalized + [trans_normalized[0]]
    term_plot = term_normalized + [term_normalized[0]]
    
    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=16, weight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=12)
    plt.ylim(0, 1)
    
    # Plot data for each stage with different line styles
    ax.plot(angles, onset_plot, linewidth=3, linestyle='-', label='Onset Detection', 
            color='forestgreen', marker='o', markersize=8)
    ax.fill(angles, onset_plot, color='forestgreen', alpha=0.15)
    
    ax.plot(angles, trans_plot, linewidth=3, linestyle='-', label='Transition Detection', 
            color='royalblue', marker='s', markersize=8)
    ax.fill(angles, trans_plot, color='royalblue', alpha=0.15)
    
    ax.plot(angles, term_plot, linewidth=3, linestyle='-', label='Termination Detection', 
            color='firebrick', marker='^', markersize=8)
    ax.fill(angles, term_plot, color='firebrick', alpha=0.15)
    
    # Add value labels at each point
    for i, (angle, onset_val, trans_val, term_val) in enumerate(zip(angles[:-1], onset_normalized, trans_normalized, term_normalized)):
        # Onset values
        ax.text(angle, onset_val + 0.08, f'{onset_weights[i]:.2f}', 
                ha='center', va='center', fontsize=11, weight='bold', color='forestgreen')
        
        # Transition values
        ax.text(angle, trans_val + 0.12, f'{trans_weights[i]:.2f}', 
                ha='center', va='center', fontsize=11, weight='bold', color='royalblue')
        
        # Termination values
        ax.text(angle, term_val + 0.16, f'{term_weights[i]:.2f}', 
                ha='center', va='center', fontsize=11, weight='bold', color='firebrick')
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    
    # Add title
    ax.set_title('Comprehensive Feature Importance Comparison\nAcross Seizure Detection Stages', 
                size=20, weight='bold', pad=40)
    
    # Add stage-specific subtitles
    ax.text(0.5, 0.95, 'Onset: Forest Green ○\nTransition: Royal Blue □\nTermination: Firebrick △', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_feature_importance_spider_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive feature importance spider plot saved to 'comprehensive_feature_importance_spider_plot.png'")
    
    return True

def main():
    """
    Main function to create all spider plot visualizations
    """
    print("Creating spider plot visualizations for stage-specific feature importance...")
    
    # Create spider plot with rankings
    create_spider_plot_stage_specific_rankings()
    
    # Create normalized spider plot with actual values
    create_normalized_spider_plot()
    
    # Create comprehensive feature importance comparison
    create_feature_importance_comparison_spider()
    
    print(f"\nAll spider plot visualizations have been generated successfully!")
    print(f"\nGenerated files:")
    print(f"- spider_plot_stage_specific_feature_rankings.png")
    print(f"- normalized_spider_plot_feature_importance.png")
    print(f"- comprehensive_feature_importance_spider_plot.png")

if __name__ == "__main__":
    main()