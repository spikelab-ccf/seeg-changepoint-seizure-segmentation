import pandas as pd
import numpy as np
from pathlib import Path

def extract_common_parameters():
    \"\"\"
    Extract common parameter sets for onset, transition, and termination detection
    \"\"\"
    # Load the optimized hyperparameters
    df = pd.read_csv("../optimized_hyperparameters_summary.csv")
    
    print("ANALYZING COMMON PARAMETERS ACROSS ALL FOLDS")
    print("="*60)
    
    # Extract window size and step parameters for each stage
    print("\\n1. WINDOW SIZE AND STEP PARAMETERS (Median values)")
    print("-" * 50)
    
    # Onset parameters
    onset_window_median = df['onset_window_size'].median()
    onset_step_median = df['onset_step'].median()
    onset_window_iqr = (df['onset_window_size'].quantile(0.25), df['onset_window_size'].quantile(0.75))
    onset_step_iqr = (df['onset_step'].quantile(0.25), df['onset_step'].quantile(0.75))
    
    print(f"Onset Detection:")
    print(f"  Window Size: {onset_window_median:.0f} ms (IQR: {onset_window_iqr[0]:.0f}-{onset_window_iqr[1]:.0f} ms)")
    print(f"  Step Size:   {onset_step_median:.0f} ms (IQR: {onset_step_iqr[0]:.0f}-{onset_step_iqr[1]:.0f} ms)")
    
    # Termination parameters
    term_window_median = df['term_window_size'].median()
    term_step_median = df['term_step'].median()
    term_window_iqr = (df['term_window_size'].quantile(0.25), df['term_window_size'].quantile(0.75))
    term_step_iqr = (df['term_step'].quantile(0.25), df['term_step'].quantile(0.75))
    
    print(f"\\nTermination Detection:")
    print(f"  Window Size: {term_window_median:.0f} ms (IQR: {term_window_iqr[0]:.0f}-{term_window_iqr[1]:.0f} ms)")
    print(f"  Step Size:   {term_step_median:.0f} ms (IQR: {term_step_iqr[0]:.0f}-{term_step_iqr[1]:.0f} ms)")
    
    # Transition parameters
    trans_window_median = df['trans_window_size'].median()
    trans_step_median = df['trans_step'].median()
    trans_window_iqr = (df['trans_window_size'].quantile(0.25), df['trans_window_size'].quantile(0.75))
    trans_step_iqr = (df['trans_step'].quantile(0.25), df['trans_step'].quantile(0.75))
    
    print(f"\\nTransition Detection:")
    print(f"  Window Size: {trans_window_median:.0f} ms (IQR: {trans_window_iqr[0]:.0f}-{trans_window_iqr[1]:.0f} ms)")
    print(f"  Step Size:   {trans_step_median:.0f} ms (IQR: {trans_step_iqr[0]:.0f}-{trans_step_iqr[1]:.0f} ms)")
    
    # Extract feature weights for each stage
    print("\\n\\n2. FEATURE WEIGHTS (Median values)")
    print("-" * 40)
    
    features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    stages = ['onset', 'term', 'trans']
    
    feature_weights_by_stage = {}
    
    for stage in stages:
        print(f"\\n{stage.capitalize()} Stage Feature Weights:")
        stage_weights = {}
        for feature in features:
            column_name = f'{stage}_weight_{feature}'
            if column_name in df.columns:
                median_weight = df[column_name].median()
                iqr_weight = (df[column_name].quantile(0.25), df[column_name].quantile(0.75))
                stage_weights[feature] = (median_weight, iqr_weight)
                print(f"  {feature.upper():>6}: {median_weight:.2f} (IQR: {iqr_weight[0]:.2f}-{iqr_weight[1]:.2f})")
        
        # Identify consistently low-weight features for this stage
        low_weight_features = [feat for feat, (weight, _) in stage_weights.items() if weight < 0.2]
        if low_weight_features:
            print(f"         → Low-weight features: {', '.join([f.upper() for f in low_weight_features])}")
        
        feature_weights_by_stage[stage] = stage_weights
    
    # Penalty parameters
    print("\\n\\n3. PENALTY PARAMETERS (Median values)")
    print("-" * 40)
    
    penalty_onset_median = df['penalty_onset'].median()
    penalty_term_median = df['penalty_term'].median()
    penalty_trans_median = df['penalty_trans'].median()
    
    print(f"Onset Penalty:    {penalty_onset_median:.0f}")
    print(f"Termination Penalty: {penalty_term_median:.0f}")
    print(f"Transition Penalty:  {penalty_trans_median:.0f}")
    
    # Save recommended common parameters
    print("\\n\\n4. RECOMMENDED COMMON PARAMETER SET")
    print("=" * 50)
    
    recommended_params = {
        'onset': {
            'window_size': int(round(onset_window_median, -2)),  # Round to nearest 100
            'step': int(round(onset_step_median, -1)),           # Round to nearest 10
            'penalty': int(round(penalty_onset_median)),
            'weights': {feature: round(feature_weights_by_stage['onset'][feature][0], 2) for feature in features}
        },
        'term': {
            'window_size': int(round(term_window_median, -2)),
            'step': int(round(term_step_median, -1)),
            'penalty': int(round(penalty_term_median)),
            'weights': {feature: round(feature_weights_by_stage['term'][feature][0], 2) for feature in features}
        },
        'trans': {
            'window_size': int(round(trans_window_median, -2)),
            'step': int(round(trans_step_median, -1)),
            'penalty': int(round(penalty_trans_median)),
            'weights': {feature: round(feature_weights_by_stage['trans'][feature][0], 2) for feature in features}
        }
    }
    
    print("RECOMMENDED FINAL MODEL PARAMETERS:")
    print("-" * 40)
    for stage in ['onset', 'term', 'trans']:
        params = recommended_params[stage]
        print(f"\\n{stage.upper()} DETECTION:")
        print(f"  Window Size: {params['window_size']} ms")
        print(f"  Step Size:   {params['step']} ms")
        print(f"  Penalty:     {params['penalty']}")
        print("  Feature Weights:")
        for feature, weight in params['weights'].items():
            print(f"    {feature.upper():>6}: {weight:.2f}")
    
    # Save to file
    with open('recommended_common_parameters.txt', 'w') as f:
        f.write("RECOMMENDED COMMON PARAMETERS FOR SEIZURE CHANGEPOINT DETECTION\\n")
        f.write("=" * 60 + "\\n\\n")
        
        f.write("WINDOW AND STEP PARAMETERS:\\n")
        f.write("-" * 30 + "\\n")
        f.write(f"Onset Detection:\\n")
        f.write(f"  Window Size: {recommended_params['onset']['window_size']} ms\\n")
        f.write(f"  Step Size:   {recommended_params['onset']['step']} ms\\n")
        f.write(f"  Penalty:     {recommended_params['onset']['penalty']}\\n\\n")
        
        f.write(f"Termination Detection:\\n")
        f.write(f"  Window Size: {recommended_params['term']['window_size']} ms\\n")
        f.write(f"  Step Size:   {recommended_params['term']['step']} ms\\n")
        f.write(f"  Penalty:     {recommended_params['term']['penalty']}\\n\\n")
        
        f.write(f"Transition Detection:\\n")
        f.write(f"  Window Size: {recommended_params['trans']['window_size']} ms\\n")
        f.write(f"  Step Size:   {recommended_params['trans']['step']} ms\\n")
        f.write(f"  Penalty:     {recommended_params['trans']['penalty']}\\n\\n")
        
        f.write("FEATURE WEIGHTS:\\n")
        f.write("-" * 20 + "\\n")
        for stage in ['onset', 'term', 'trans']:
            f.write(f"\\n{stage.upper()} STAGE:\\n")
            for feature, weight in recommended_params[stage]['weights'].items():
                f.write(f"  {feature.upper():>6}: {weight:.2f}\\n")
    
    print(f"\\nRecommended parameters saved to 'recommended_common_parameters.txt'")
    
    return recommended_params

def identify_redundant_features():
    \"\"\"
    Identify features that are consistently given low weights
    \"\"\"
    df = pd.read_csv("../optimized_hyperparameters_summary.csv")
    
    print("\\n\\n5. REDUNDANT FEATURE ANALYSIS")
    print("=" * 40)
    
    features = ['rms', 'theta', 'alpha', 'beta', 'gamma', 'll', 'se']
    stages = ['onset', 'term', 'trans']
    
    redundant_features = set()
    
    print("Features with consistently low weights (< 0.2):")
    print("-" * 45)
    
    for feature in features:
        low_weight_count = 0
        total_weights = 0
        
        for stage in stages:
            column_name = f'{stage}_weight_{feature}'
            if column_name in df.columns:
                # Count how many weights are below threshold
                low_weights = (df[column_name] < 0.2).sum()
                total = len(df[column_name])
                low_weight_count += low_weights
                total_weights += total
                
                # Print stage-specific info
                percentage = (low_weights / total) * 100 if total > 0 else 0
                print(f"{feature.upper():>6} {stage}: {low_weights}/{total} ({percentage:.1f}%) below 0.2")
        
        # Overall percentage
        overall_percentage = (low_weight_count / total_weights) * 100 if total_weights > 0 else 0
        print(f"       TOTAL: {low_weight_count}/{total_weights} ({overall_percentage:.1f}%) below 0.2")
        
        # Mark as redundant if consistently low
        if overall_percentage > 70:  # More than 70% of weights are low
            redundant_features.add(feature)
            print(f"       → REDUNDANT FEATURE")
        print()
    
    print(f"REDUNDANT FEATURES TO CONSIDER REMOVING: {', '.join([f.upper() for f in redundant_features])}")
    
    return list(redundant_features)

def main():
    \"\"\"
    Main function to extract and recommend common parameters
    \"\"\"
    print("Extracting common parameters from optimized hyperparameters...")
    
    # Extract common parameters
    recommended_params = extract_common_parameters()
    
    # Identify redundant features
    redundant_features = identify_redundant_features()
    
    print(f"\\nSUMMARY:")
    print("=" * 20)
    print(f"Recommended parameters extracted for all three detection stages")
    print(f"Redundant features identified: {[f.upper() for f in redundant_features]}")
    print(f"Files generated:")
    print(f"  - recommended_common_parameters.txt")

if __name__ == "__main__":
    main()