#!/usr/bin/env python3
"""
Plotting script that loads pre-processed data and generates visualizations.
This allows for rapid iteration on plot styles without re-processing the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_processed_data(filename='processed_epoched_data.pkl'):
    """Load the pre-processed data from file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Processed data file '{filename}' not found. Please run the main processing script first.")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data['all_epoched_data'], data['time_axis']

def plot_all_features_subplots(all_epoch_data, time_axis):
    """
    Plots the event-related averages for all features in a single column of subplots
    with a single, consolidated legend.
    """
    print("\n--- Generating Event-Related Average Subplots for All Features ---")
    
    feature_names = ['RMS', 'Line Length', 'Spectral Entropy', 'Theta Power', 'Alpha Power', 'Beta Power', 'Gamma Power']
    
    # Create multiple plot variations
    # 1. Main subplot figure
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(
        nrows=7, 
        ncols=1, 
        figsize=(10, 16),
        sharex=True, 
        dpi=300
    )
    
    #fig.suptitle("Average Feature Dynamics at Detected Changepoints", fontsize=18, weight='bold')

    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        epoch_data = all_epoch_data.get(feature_name, {})
        
        # Check if we have any data for this feature
        has_data = any(f'{align_type}_mean' in epoch_data and len(epoch_data[f'{align_type}_mean']) > 0 
                      for align_type in ['onset', 'transition', 'termination'])
        
        if has_data:
            # Plot Onset-aligned average
            if 'onset_mean' in epoch_data and len(epoch_data['onset_mean']) > 0:
                ax.plot(time_axis, epoch_data.get('onset_mean', []), color='forestgreen', linewidth=2.5, label='Onset')
                if 'onset_sem' in epoch_data and len(epoch_data['onset_sem']) > 0:
                    ax.fill_between(time_axis, epoch_data['onset_mean'] - epoch_data['onset_sem'], 
                                  epoch_data['onset_mean'] + epoch_data['onset_sem'], color='forestgreen', alpha=0.2)

            # Plot Transition-aligned average
            if 'transition_mean' in epoch_data and len(epoch_data['transition_mean']) > 0:
                ax.plot(time_axis, epoch_data.get('transition_mean', []), color='royalblue', linewidth=2.5, label='Transition')
                if 'transition_sem' in epoch_data and len(epoch_data['transition_sem']) > 0:
                    ax.fill_between(time_axis, epoch_data['transition_mean'] - epoch_data['transition_sem'], 
                                  epoch_data['transition_mean'] + epoch_data['transition_sem'], color='royalblue', alpha=0.2)

            # Plot Termination-aligned average
            if 'termination_mean' in epoch_data and len(epoch_data['termination_mean']) > 0:
                ax.plot(time_axis, epoch_data.get('termination_mean', []), color='firebrick', linewidth=2.5, label='Termination')
                if 'termination_sem' in epoch_data and len(epoch_data['termination_sem']) > 0:
                    ax.fill_between(time_axis, epoch_data['termination_mean'] - epoch_data['termination_sem'], 
                                  epoch_data['termination_mean'] + epoch_data['termination_sem'], color='firebrick', alpha=0.2)
        else:
            # If no data, show a message
            ax.text(0.5, 0.5, f'No data available for {feature_name}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
        
        # Formatting for each subplot
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.set_ylabel(f"Mean {feature_name}", fontsize=13)
        ax.grid(False)
        
        # Formatting
        if i < len(axes) - 1:
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', which='major', labelsize=12, width=1.2, length=5, direction='out')
        else:
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5, direction='out')
            
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        # Only add legend to the first subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
            labels.append('Detected Changepoint')
            ax.legend(handles, labels, loc='upper right', fontsize=12, ncol=2, bbox_to_anchor=(0.99, 0.92))


    # --- FINAL FORMATTING ---
    axes[-1].set_xlabel("Time Relative to Detected Changepoint (s)", fontsize=14)
    
    # Set x-axis ticks to show -5 to +5 seconds
    axes[-1].set_xlim(time_axis[0], time_axis[-1])
    axes[-1].set_xticks(np.arange(-5, 6, 1))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig('feature_dynamics_main.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure object to free up memory
    
    # 2. Create a heatmap-style plot showing all features for each alignment type
    create_heatmap_plot(all_epoch_data, time_axis, feature_names)
    
    # 3. Create a normalized comparison plot (single column)
    create_normalized_comparison_plot(all_epoch_data, time_axis, feature_names)
    
    print("All plots generated successfully!")

def create_heatmap_plot(all_epoch_data, time_axis, feature_names):
    """Create a heatmap-style visualization of the data in a single column"""
    # Create separate figures for each alignment type in a single column
    alignment_types = ['onset', 'transition', 'termination']
    alignment_labels = ['Onset', 'Transition', 'Termination']
    colors = ['forestgreen', 'royalblue', 'firebrick']
    
    # Make the figure taller and with larger fonts
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), dpi=300)
    
    for idx, (align_type, label, color) in enumerate(zip(alignment_types, alignment_labels, colors)):
        ax = axes[idx]
        
        # Filter features that have data for this alignment type
        data_matrix = []
        feature_labels = []
        
        for feature_name in feature_names:
            epoch_data = all_epoch_data.get(feature_name, {})
            if f'{align_type}_mean' in epoch_data and len(epoch_data[f'{align_type}_mean']) > 0:
                data_matrix.append(epoch_data[f'{align_type}_mean'])
                feature_labels.append(feature_name)
        
        if data_matrix and len(data_matrix) > 0:
            data_matrix = np.array(data_matrix)
            
            # Normalize each feature independently to 0-1 range for better visualization
            normalized_data = np.zeros_like(data_matrix)
            for i in range(data_matrix.shape[0]):
                feature_data = data_matrix[i, :]
                min_val = np.min(feature_data)
                max_val = np.max(feature_data)
                if max_val > min_val:
                    normalized_data[i, :] = (feature_data - min_val) / (max_val - min_val)
                else:
                    normalized_data[i, :] = feature_data  # All values are the same
            
            # Create heatmap with proper time axis
            im = ax.imshow(normalized_data, aspect='auto', cmap='viridis', origin='lower', 
                          extent=[time_axis[0], time_axis[-1], 0, len(feature_labels)])
            
            # Larger font sizes
            ax.set_title(f'{label}-Aligned Features', fontsize=24, weight='bold')
            ax.set_xlabel('Time Relative to Changepoint (s)', fontsize=22)
            ax.set_ylabel('Features', fontsize=22)
            ax.set_yticks(np.arange(len(feature_labels)) + 0.5)
            ax.set_yticklabels(feature_labels, fontsize=20)
            ax.axvline(0, color='red', linestyle='--', linewidth=2.5)
            
            # Larger tick label fonts
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Add colorbar with larger font
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Scaled Feature Value', fontsize=22)
            cbar.ax.tick_params(labelsize=20)
        else:
            # If no data, show a message with larger fonts
            ax.text(0.5, 0.5, f'No data available for {label}-Aligned features', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=22)
            ax.set_title(f'{label}-Aligned Features', fontsize=24, weight='bold')
            ax.set_xlabel('Time Relative to Changepoint (s)', fontsize=22)
            ax.set_ylabel('Features', fontsize=22)
    
    plt.tight_layout()
    plt.savefig('feature_dynamics_heatmap_single_column.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_normalized_comparison_plot(all_epoch_data, time_axis, feature_names):
    """Create a normalized comparison plot showing relative changes in a single column"""
    # Create a single column layout like the main plot
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(
        nrows=7, 
        ncols=1, 
        figsize=(10, 16),
        sharex=True, 
        dpi=300
    )
    
    # Use a consistent baseline window for normalization (first 25% of time window)
    baseline_window = max(1, len(time_axis) // 4)
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        epoch_data = all_epoch_data.get(feature_name, {})
        
        # Check if we have any data for this feature
        has_data = any(f'{align_type}_mean' in epoch_data and len(epoch_data[f'{align_type}_mean']) > 0 
                      for align_type, _, _ in [('onset', 'Onset', 'forestgreen'), 
                                              ('transition', 'Transition', 'royalblue'), 
                                              ('termination', 'Termination', 'firebrick')])
        
        if has_data:
            # Determine a common baseline across all alignment types for this feature
            all_data = []
            for align_type in ['onset', 'transition', 'termination']:
                if f'{align_type}_mean' in epoch_data and len(epoch_data[f'{align_type}_mean']) > 0:
                    all_data.extend(epoch_data[f'{align_type}_mean'][:baseline_window])
            
            common_baseline = np.mean(all_data) if all_data else 1.0
            # Ensure we don't divide by zero
            common_baseline = max(common_baseline, 1e-12)
            
            # Plot normalized data for each alignment type
            for align_type, label, color in [('onset', 'Onset', 'forestgreen'), 
                                            ('transition', 'Transition', 'royalblue'), 
                                            ('termination', 'Termination', 'firebrick')]:
                if f'{align_type}_mean' in epoch_data and len(epoch_data[f'{align_type}_mean']) > 0:
                    # Normalize to common baseline
                    normalized_data = epoch_data[f'{align_type}_mean'] / common_baseline
                    ax.plot(time_axis, normalized_data, color=color, linewidth=2.5, label=label)
            
            # Formatting for each subplot
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            ax.set_ylabel(f"{feature_name}\n(Relative to Baseline)", fontsize=13)
            ax.grid(False)
            
            # Only add legend to the first subplot
            if i == 0:
                ax.legend(fontsize=12, loc='upper right')
        else:
            # If no data, show a message
            ax.text(0.5, 0.5, f'No data available for {feature_name}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_ylabel(f"{feature_name}\n(Relative to Baseline)", fontsize=13)
        
        # Formatting
        if i < len(axes) - 1:
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', which='major', labelsize=12, width=1.2, length=5, direction='out')
        else:
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5, direction='out')
            
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    axes[-1].set_xlabel("Time Relative to Detected Changepoint (s)", fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('feature_dynamics_normalized_single_column.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to load data and generate plots"""
    try:
        # Load the processed data
        print("Loading processed data...")
        all_epoched_data, time_axis = load_processed_data()
        print("Data loaded successfully!")
        
        # Generate all plots
        plot_all_features_subplots(all_epoched_data, time_axis)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the main processing script first to generate the processed data file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()