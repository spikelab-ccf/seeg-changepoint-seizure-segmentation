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

def create_publication_ready_temporal_evolution_plot():
    """
    Create a publication-ready visualization of temporal feature evolution
    """
    # Feature importance data
    features = ['RMS', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'LL', 'SE']
    stages = ['Onset', 'Transition', 'Termination']
    
    # Median weights for each feature at each stage
    weights = {
        'RMS': [0.540, 0.589, 0.626],
        'THETA': [0.470, 0.528, 0.540],
        'ALPHA': [0.639, 0.478, 0.452],
        'BETA': [0.672, 0.545, 0.473],
        'GAMMA': [0.630, 0.431, 0.519],
        'LL': [0.569, 0.535, 0.617],
        'SE': [0.400, 0.431, 0.417]
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors and markers for each feature
    colors = plt.cm.tab10(np.linspace(0, 1, len(features)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Position data
    x_positions = np.arange(len(stages))
    
    # Plot each feature's evolution
    for i, (feature, color, marker) in enumerate(zip(features, colors, markers)):
        feature_weights = weights[feature]
        line = ax.plot(x_positions, feature_weights, 
                      marker=marker, linewidth=2.5, markersize=8,
                      label=feature, color=color, alpha=0.8,
                      markeredgecolor='black', markeredgewidth=1)
        
        # Add value labels at each point
        for j, (x, y) in enumerate(zip(x_positions, feature_weights)):
            ax.text(x, y + 0.02, f'{y:.2f}', 
                   ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Customize the plot
    ax.set_xlabel('Detection Stage', fontsize=14)
    ax.set_ylabel('Feature Weight', fontsize=14)
    ax.set_title('Temporal Evolution of Feature Importance Across Seizure Detection Stages', 
                fontsize=16, weight='bold', pad=20)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), 
                      fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(len(stages)-0.5, 0.52, '0.5', color='red', fontweight='bold', 
            ha='right', va='bottom', fontsize=12)
    
    # Customize axes
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(stages)-0.5)
    
    plt.tight_layout()
    plt.savefig('temporal_feature_evolution_publication_ready.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Publication-ready temporal evolution plot saved to 'temporal_feature_evolution_publication_ready.png'")

def create_stage_specific_feature_rankings():
    """
    Create bar charts showing feature rankings for each stage
    """
    # Feature importance data for each stage
    stage_data = {
        'Onset': {
            'features': ['BETA', 'ALPHA', 'GAMMA', 'LL', 'RMS', 'THETA', 'SE'],
            'weights': [0.672, 0.639, 0.630, 0.569, 0.540, 0.470, 0.400]
        },
        'Transition': {
            'features': ['RMS', 'BETA', 'LL', 'THETA', 'ALPHA', 'SE', 'GAMMA'],
            'weights': [0.589, 0.545, 0.535, 0.528, 0.478, 0.431, 0.431]
        },
        'Termination': {
            'features': ['RMS', 'LL', 'THETA', 'GAMMA', 'BETA', 'ALPHA', 'SE'],
            'weights': [0.626, 0.617, 0.540, 0.519, 0.473, 0.452, 0.417]
        }
    }
    
    # Create subplots for each stage
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    # Colors for bars
    colors = plt.cm.viridis(np.linspace(0, 1, 7))
    
    # Plot each stage
    for i, (stage, data) in enumerate(stage_data.items()):
        ax = axes[i]
        
        # Sort by weight (descending)
        features = data['features']
        weights = data['weights']
        
        y_pos = np.arange(len(features))
        
        # Create horizontal bar chart
        bars = ax.barh(y_pos, weights, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, weights)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', ha='left', va='center', 
                   fontsize=12, weight='bold')
        
        # Customize subplot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=13)
        ax.set_xlabel('Feature Weight', fontsize=14)
        ax.set_title(f'{stage} Detection', fontsize=16, weight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    # Set common y-label
    axes[0].set_ylabel('Features', fontsize=14)
    
    # Add overall title
    fig.suptitle('Stage-Specific Feature Importance Rankings', fontsize=18, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('stage_specific_feature_rankings_publication_ready.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Stage-specific feature rankings plot saved to 'stage_specific_feature_rankings_publication_ready.png'")

def create_feature_importance_comparison_matrix():
    """
    Create a matrix visualization comparing feature importance across stages
    """
    # Feature importance data
    features = ['RMS', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'LL', 'SE']
    stages = ['Onset', 'Transition', 'Termination']
    
    # Median weights for each feature at each stage
    weights_matrix = np.array([
        [0.540, 0.589, 0.626],  # RMS
        [0.470, 0.528, 0.540],  # THETA
        [0.639, 0.478, 0.452],  # ALPHA
        [0.672, 0.545, 0.473],  # BETA
        [0.630, 0.431, 0.519],  # GAMMA
        [0.569, 0.535, 0.617],  # LL
        [0.400, 0.431, 0.417]   # SE
    ])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with colorbar
    im = ax.imshow(weights_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(stages)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(stages, fontsize=14)
    ax.set_yticklabels(features, fontsize=14)
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(stages)):
            text = ax.text(j, i, f'{weights_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white", 
                          fontsize=12, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Feature Weight', fontsize=14)
    
    # Customize plot
    ax.set_title('Feature Importance Matrix Across Seizure Detection Stages', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Detection Stage', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('feature_importance_matrix_publication_ready.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Feature importance matrix plot saved to 'feature_importance_matrix_publication_ready.png'")

def main():
    """
    Main function to create all publication-ready visualizations
    """
    print("Creating publication-ready visualizations for individual stage feature importance...")
    
    # Create temporal evolution plot
    create_publication_ready_temporal_evolution_plot()
    
    # Create stage-specific rankings
    create_stage_specific_feature_rankings()
    
    # Create feature importance matrix
    create_feature_importance_comparison_matrix()
    
    print("\nAll publication-ready visualizations have been generated successfully!")
    print("\nGenerated files:")
    print("- temporal_feature_evolution_publication_ready.png")
    print("- stage_specific_feature_rankings_publication_ready.png")
    print("- feature_importance_matrix_publication_ready.png")

if __name__ == "__main__":
    main()