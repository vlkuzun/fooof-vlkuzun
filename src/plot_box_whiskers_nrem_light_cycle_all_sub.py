import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from pymer4.models import Lmer

def analyze_nrem_exponents(pickle_files, csv_files, start_times, exclusion_ranges):
    """
    Analyze NREM 1/f exponents across time blocks
    
    Parameters:
    -----------
    pickle_files : list
        Paths to FOOOF result pickle files
    csv_files : list 
        Paths to sleep scoring CSV files
    start_times : list
        Recording start times for each subject
    exclusion_ranges : dict
        Time ranges to exclude for each subject
        
    Returns:
    --------
    tuple
        (model, comparisons, fig) - LMER model, statistical comparisons, and figure
    """
    all_bout_data = []

    for fg_pickle_path, csv_path, start_time in zip(pickle_files, csv_files, start_times):
        # Extract subject ID for labeling
        subject_id = fg_pickle_path.split('sub-')[1].split('_')[0]
        
        if not Path(csv_path).exists():
            print(f"Warning: Missing CSV file for {subject_id}")
            continue

        # Load FOOOF data
        with open(fg_pickle_path, 'rb') as f:
            fg_models = pickle.load(f)
        
        # Calculate average exponents
        exps_eeg1 = fg_models['EEG1'].get_params('aperiodic_params', 'exponent')
        exps_eeg2 = fg_models['EEG2'].get_params('aperiodic_params', 'exponent')
        avg_exps = (exps_eeg1 + exps_eeg2) / 2

        # Create resampling function
        def robust_mode(x):
            if len(x) == 0 or x.isna().all():
                return np.nan
            mode_result = x.mode()
            return mode_result[0] if len(mode_result) > 0 else np.nan

        # Load and filter sleep data
        sleep_data = pd.read_csv(csv_path)
        sleep_data['Timestamp'] = pd.to_datetime(sleep_data['Timestamp'])

        # Resample sleep stages to 10s intervals (mode)
        sleep_data_resampled = sleep_data.set_index('Timestamp').resample('10S').agg({
            'sleepStage': robust_mode
        })

        # Apply exclusions after resampling
        if subject_id in exclusion_ranges:
            for start_exc, end_exc in exclusion_ranges[subject_id]:
                mask = ~((sleep_data_resampled.index >= start_exc) & 
                        (sleep_data_resampled.index <= end_exc))
                sleep_data_resampled = sleep_data_resampled[mask]

        # Check if start time is aligned with 10s intervals
        start_seconds = pd.to_datetime(start_time).second
        is_aligned = start_seconds % 10 == 0

        # Only drop last row if not aligned
        if not is_aligned:
            sleep_data_resampled = sleep_data_resampled.iloc[:-1]

        # Create exponent series
        exp_series = pd.Series(avg_exps, 
                          index=pd.date_range(start=start_time, 
                                            periods=len(avg_exps), 
                                            freq='10S'))

        # Verify lengths match
        print(f"Subject {subject_id}:")
        print(f"exp_series length: {len(exp_series)}")
        print(f"sleep_data length: {len(sleep_data_resampled)}")

        # Process NREM bouts
        combined_data = pd.DataFrame({
            'timestamp': exp_series.index,
            'hour': exp_series.index.hour,
            'exponent': exp_series.values,
            'sleepStage': sleep_data_resampled['sleepStage']
        })

        nrem_data = combined_data[combined_data['sleepStage'] == 2].copy()
        nrem_data['zt'] = (nrem_data['hour'] - 9) % 24
        nrem_data['time_block'] = pd.cut(nrem_data['zt'],
                                        bins=[-0.1, 3, 6, 9, 12],
                                        labels=['ZT0-3', 'ZT3-6', 'ZT6-9', 'ZT9-12'])

        # Find NREM bouts
        nrem_bouts = []
        current_bout = []
        current_block = None

        sorted_data = nrem_data.sort_values('timestamp')
        
        for idx, row in sorted_data.iterrows():
            if not current_bout:
                current_bout = [row]
                current_block = row['time_block']
            else:
                time_diff = (row['timestamp'] - current_bout[-1]['timestamp']).total_seconds()
                
                if time_diff == 10.0:
                    current_bout.append(row)
                else:
                    if len(current_bout) > 0:
                        bout_mean = np.mean([r['exponent'] for r in current_bout])
                        nrem_bouts.append((current_block, bout_mean))
                    current_bout = [row]
                    current_block = row['time_block']

        if current_bout:
            bout_mean = np.mean([r['exponent'] for r in current_bout])
            nrem_bouts.append((current_block, bout_mean))

        # Create DataFrame for bout means and add subject ID
        bout_df = pd.DataFrame(nrem_bouts, columns=['time_block', 'bout_mean'])
        bout_df['subject'] = subject_id
        all_bout_data.append(bout_df)

    # Combine all bout data
    combined_bouts = pd.concat(all_bout_data, ignore_index=True)

    # Define valid time blocks and filter data
    time_blocks = ['ZT0-3', 'ZT3-6', 'ZT6-9', 'ZT9-12']
    filtered_bouts = combined_bouts[combined_bouts['time_block'].isin(time_blocks)].copy()

    # Verify filtering
    print("Original data shape:", combined_bouts.shape)
    print("Filtered data shape:", filtered_bouts.shape)
    print("\nUnique time blocks after filtering:", filtered_bouts['time_block'].unique())

    # Run LMER on filtered data
    filtered_bouts['subject'] = filtered_bouts['subject'].astype('category')
    filtered_bouts['time_block'] = filtered_bouts['time_block'].astype('category')

    model = Lmer("bout_mean ~ time_block + (1|subject)", data=filtered_bouts)
    model.fit(
        factors={"time_block": time_blocks},
        ordered=False
    )

    # Print ANOVA results
    print("\nANOVA Results:")
    print(model.anova())

    # Run post-hoc tests
    print("\nPost-hoc Tests:")
    marginal_estimates, comparisons = model.post_hoc(
        marginal_vars=["time_block"],
        p_adjust='fdr'
    )

    print("\nMarginal Estimates:")
    print(marginal_estimates)
    print("\nPairwise Comparisons:")
    print(comparisons)

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Define order and create visualization
    sns.boxplot(data=filtered_bouts, x='time_block', y='bout_mean',
                order=time_blocks,
                palette='tab10',
                showfliers=False,
                ax=ax)

    sns.stripplot(data=filtered_bouts, x='time_block', y='bout_mean',
                 order=time_blocks,
                 color='grey',
                 size=4,
                 alpha=0.2,
                 jitter=0.2,
                 ax=ax)

    # Add significance bars for contrasts 1, 4, and 6
    contrast_indices = [0, 3, 5]  # Indices for contrasts 1, 4, 6
    y_start = filtered_bouts['bout_mean'].max() + 0.1
    spacing = 0.1

    time_blocks_pvalue = ['(ZT0-3)', '(ZT3-6)', '(ZT6-9)', '(ZT9-12)']

    for idx, contrast_idx in enumerate(contrast_indices):
        p_value = comparisons.iloc[contrast_idx]['P-val']
        if p_value < 0.05:
            contrast = comparisons.iloc[contrast_idx]['Contrast']
            groups = contrast.split(' - ')
            
            # Get x-coordinates
            x1 = time_blocks_pvalue.index(groups[0])
            x2 = time_blocks_pvalue.index(groups[1])
            
            # Determine significance stars
            stars = '*'
            if p_value < 0.01:
                stars = '**'
            elif p_value < 0.001:
                stars = '***'
                
            # Plot significance bar
            y = y_start + (idx * spacing)
            plt.plot([x1, x2], [y, y], 'k-', linewidth=1.5)
            plt.text((x1 + x2)/2, y, stars, ha='center', va='bottom', fontsize=20)

    # Style plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('NREM 1/f Exponent Across Light Cycle (All Subjects)', fontsize=24, pad=30)
    plt.ylabel('1/f Exponent', fontsize=24)
    plt.xlabel('')  # Remove x-axis label
    plt.tick_params(axis='both', labelsize=24)
    plt.tight_layout()
    plt.savefig('/Volumes/harris/volkan/fooof/plots/nrem_exponent/combined_nrem_exponent_light_cycle.png', dpi=600, bbox_inches='tight')
    plt.show()

    return model, comparisons, fig

# Example usage:
model, comparisons, fig = analyze_nrem_exponents(
    pickle_files = [
    '/Volumes/harris/volkan/fooof/fooof_results/sub-007_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-010_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-011_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-015_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-016_ses-02_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl'
],
    csv_files = [
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv'
], 
    start_times = ['2024-07-02 15:13:12', '2024-09-16 16:01:50','2024-09-25 15:07:30','2024-11-18 12:35:00','2024-11-29 13:29:14','2024-12-16 17:50:47'],
    exclusion_ranges = {
    '015': [('2024-11-19 08:49:50', '2024-11-19 08:57:59')]
}
)

# Display results
print("\nANOVA Results:")
print(model.anova())
print("\nPairwise Comparisons:")
print(comparisons)
plt.show()