        results[name] = feature_results

    return results, epoch_time_axis

# --- Main Execution Block ---
if __name__ == '__main__':
    base_folder = '/mnt/isilon/w_nonlri/krishnblab/Users/Himanshu_Kumar/PYTHON_CHNGPT_CUSUM_new/'
    
    all_epoched_data, time_axis = aggregate_all_epoched_features(base_folder)
    
    if all_epoched_data:
        plot_all_features_subplots(all_epoched_data, time_axis)
    else:
        print("\nCould not generate plots because no valid data was found.")