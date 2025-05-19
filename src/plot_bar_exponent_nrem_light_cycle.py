import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_nrem_exponents(fg_pickle_path, csv_path, start_time, end_time):
    # Load the pre-trained FOOOFGroup models from pickle file
    with open(fg_pickle_path, 'rb') as f:
        fg_models = pickle.load(f)

    # Extract aperiodic parameters for each channel
    exps_eeg1 = fg_models['EEG1'].get_params('aperiodic_params', 'exponent')
    exps_eeg2 = fg_models['EEG2'].get_params('aperiodic_params', 'exponent')
    avg_exps = (exps_eeg1 + exps_eeg2) / 2

    # Import the CSV file containing Timestamp and sleepStage data
    sleep_data = pd.read_csv(csv_path)

    # Ensure Timestamp column is in datetime format
    sleep_data['Timestamp'] = pd.to_datetime(sleep_data['Timestamp'])

    # Select a specific time range for analysis
    time_filtered_data = sleep_data[(sleep_data['Timestamp'] >= start_time) & 
                                    (sleep_data['Timestamp'] <= end_time)]

    # Resample sleepStage to the 10-second level and take the mode (most frequent value)
    sleep_data_resampled = time_filtered_data.set_index('Timestamp').resample('10S').agg({'sleepStage': lambda x: x.mode()[0]})

    # Filter for NREM data only (sleepStage == 2)
    nrem_data = time_filtered_data[time_filtered_data['sleepStage'] == 2]

    # Map 10-second avg_exps to the closest timestamps in sleepStage
    exp_series = pd.Series(avg_exps, index=pd.date_range(start=start_time, 
                                                         periods=len(avg_exps), 
                                                         freq='10S'))

    # Now, combine the sleepStage data with the exponents
    combined_data = pd.DataFrame({'sleepStage': sleep_data_resampled['sleepStage'],
                                  'exponent': exp_series})

    # Filter for NREM data only (sleepStage == 2)
    combined_nrem_data = combined_data[combined_data['sleepStage'] == 2]

    # Resample exponents to 1-minute intervals
    nrem_exps = combined_nrem_data['exponent'].resample('1T').mean()

    # Plot NREM exponents as bars
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Define distinct shades of gray - starting darker
    grays = ['#202020', '#404040', '#606060', '#808080']  # Very dark to lighter

    # Create full time range and series
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='1T')
    full_series = pd.Series(index=full_time_range, dtype=float)
    full_series[nrem_exps.index] = nrem_exps.values

    # Plot bars in 3-hour blocks with different colors
    for i in range(4):
        start_idx = i * 180
        end_idx = (i + 1) * 180
        block_data = full_series.iloc[start_idx:end_idx]
        x_positions = range(start_idx, end_idx)
        ax.bar(x_positions, block_data, color=grays[i], edgecolor=grays[i])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set x-axis limits and ticks
    ax.set_xlim(-5, 12*60+5)
    major_ticks = np.arange(0, 12*60+1, 60)
    ax.set_xticks(major_ticks)

    # Create labels: ZT values every 3 hours
    labels = [str(i) if i % 3 == 0 else '' for i in range(13)]
    ax.set_xticklabels(labels)

    plt.title('NREM 1/f Exponent Across Light Cycle', fontsize=24, pad=30)
    plt.xlabel('Zeitgeber Time', fontsize=24)
    plt.ylabel('1/f Exponent', fontsize=24)
    plt.tick_params(axis='both', labelsize=24)
    plt.tight_layout()
    plt.savefig('/Volumes/harris/volkan/fooof/plots/nrem_exponent/nrem_exponent_light_cycle.png', dpi=600)
    plt.show()

# Example usage
plot_nrem_exponents(
    fg_pickle_path='/Volumes/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl',
    csv_path='/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv',
    start_time='2024-12-18 09:00:00',
    end_time='2024-12-18 21:00:00',
)