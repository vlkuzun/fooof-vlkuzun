import pickle
import pandas as pd
import numpy as np
from neurodsp.spectral import compute_spectrum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.optimize as optimize

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_csv(file_path):
    return pd.read_csv(file_path)

def find_continuous_nrem_segments(sleep_stages):
    nrem_mask = sleep_stages['sleepStage'] == 2
    continuous_segments = []
    start_idx = None
    for i in range(len(nrem_mask)):
        if nrem_mask[i] and start_idx is None:
            start_idx = i
        elif not nrem_mask[i] and start_idx is not None:
            end_idx = i - 1
            continuous_segments.append((start_idx, end_idx))
            start_idx = None
    if start_idx is not None:
        continuous_segments.append((start_idx, len(nrem_mask) - 1))
    return continuous_segments

def compute_epoch_power(eeg_data, start_idx, end_idx, fs=512, freq_range=(0.5, 4)):
    eeg1_epoch = eeg_data['EEG1'][start_idx:end_idx]
    eeg2_epoch = eeg_data['EEG2'][start_idx:end_idx]
    
    f, Pxx_eeg1 = compute_spectrum(eeg1_epoch, fs, method='welch', nperseg=fs*2)
    f, Pxx_eeg2 = compute_spectrum(eeg2_epoch, fs, method='welch', nperseg=fs*2)
    
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    
    delta_power_eeg1 = np.mean(Pxx_eeg1[freq_mask])
    delta_power_eeg2 = np.mean(Pxx_eeg2[freq_mask])
    
    return delta_power_eeg1, delta_power_eeg2

def compute_baseline_power(eeg_data, sleep_stages, fs=512, freq_range=(0.5, 50)):
    segments = find_continuous_nrem_segments(sleep_stages)
    power_eeg1_list = []
    power_eeg2_list = []
    
    for start_idx, end_idx in segments:
        power_eeg1, power_eeg2 = compute_epoch_power(eeg_data, start_idx, end_idx, fs, freq_range)
        power_eeg1_list.append(power_eeg1)
        power_eeg2_list.append(power_eeg2)
    
    baseline_power_eeg1 = np.mean(power_eeg1_list)
    baseline_power_eeg2 = np.mean(power_eeg2_list)
    
    return baseline_power_eeg1, baseline_power_eeg2

def segment_epochs(sleep_stages, epoch_length=10, fs=512):
    epoch_samples = epoch_length * fs
    epochs = []
    for start in range(0, len(sleep_stages), epoch_samples):
        end = start + epoch_samples
        if end <= len(sleep_stages):
            epoch = sleep_stages[start:end]
            if all(epoch['sleepStage'] == 2):
                epochs.append((start, end))
    return epochs

def normalize_power(delta_power, baseline_power):
    return delta_power / baseline_power

def extract_epoch_exponents(fg_models, epochs):
    """Extract exponents for specific NREM epochs, 1 exponent per 10s."""
    exps_eeg1 = fg_models['EEG1'].get_params('aperiodic_params', 'exponent')
    exps_eeg2 = fg_models['EEG2'].get_params('aperiodic_params', 'exponent')
    
    epoch_exps = []
    samples_per_exp = 5120  # 10 seconds * 512 Hz
    
    for start_idx, end_idx in epochs:
        # Convert sample indices to exponent index (1 exp per 10s)
        exp_idx = start_idx // samples_per_exp
        
        # Get exponents for this epoch
        epoch_exp_eeg1 = exps_eeg1[exp_idx]
        epoch_exp_eeg2 = exps_eeg2[exp_idx]
        epoch_exps.append(np.mean([epoch_exp_eeg1, epoch_exp_eeg2]))
    
    return np.array(epoch_exps)

def find_continuous_bouts(epochs, samples_per_epoch=5120):
    """Group consecutive 10s epochs into bouts"""
    bouts = []
    current_bout = []
    
    for i in range(len(epochs)-1):
        current_start, current_end = epochs[i]
        next_start, next_end = epochs[i+1]
        
        if len(current_bout) == 0:
            current_bout.append(epochs[i])
            
        if next_start - current_end == 0:  # Consecutive epochs
            current_bout.append(epochs[i+1])
        else:  # Gap found
            if current_bout:
                bouts.append(current_bout)
                current_bout = []
    
    if current_bout:  # Add final bout
        bouts.append(current_bout)
        
    return bouts

def compute_bout_means(normalized_powers, exponents, epochs):
    """Calculate mean values for each bout"""
    bouts = find_continuous_bouts(epochs)
    bout_powers = []
    bout_exponents = []
    
    for bout in bouts:
        # Get indices within the epoch lists
        bout_indices = [epochs.index(epoch) for epoch in bout]
        
        # Calculate means for this bout
        bout_mean_power = np.mean([normalized_powers[i] for i in bout_indices])
        bout_mean_exponent = np.mean([exponents[i] for i in bout_indices])
        
        bout_powers.append(bout_mean_power)
        bout_exponents.append(bout_mean_exponent)
    
    return np.array(bout_powers), np.array(bout_exponents)

def exp_func(x, a, b, c):
    """Exponential function: a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def analyze_multiple_subjects(eeg_pickle_paths, sleep_stage_csv_paths, fg_models_pickle_paths, exclusion_ranges=None):
    """Analyze multiple subjects and create combined scatter plot"""
    all_bout_data = []
    
    for eeg_path, csv_path, fg_path in zip(eeg_pickle_paths, sleep_stage_csv_paths, fg_models_pickle_paths):
        # Extract subject ID
        subject_id = fg_path.split('sub-')[1].split('_')[0]
        
        eeg_data = load_pickle(eeg_path)
        sleep_stages = load_csv(csv_path)
        fg_models = load_pickle(fg_path)
        
        # Apply exclusions if specified
        if exclusion_ranges and subject_id in exclusion_ranges:
            for start_exc, end_exc in exclusion_ranges[subject_id]:
                mask = ~((sleep_stages['Timestamp'] >= start_exc) & 
                        (sleep_stages['Timestamp'] <= end_exc))
                sleep_stages = sleep_stages[mask]
        
        baseline_power_eeg1, baseline_power_eeg2 = compute_baseline_power(eeg_data, sleep_stages)
        average_baseline_power = np.mean([baseline_power_eeg1, baseline_power_eeg2])
        
        epochs = segment_epochs(sleep_stages)
        
        normalized_delta_powers = []
        for start_idx, end_idx in epochs:
            delta_power_eeg1, delta_power_eeg2 = compute_epoch_power(eeg_data, start_idx, end_idx)
            avg_delta_power = np.mean([delta_power_eeg1, delta_power_eeg2])
            normalized_power = normalize_power(avg_delta_power, average_baseline_power)
            normalized_delta_powers.append(normalized_power)
        
        epoch_exponents = extract_epoch_exponents(fg_models, epochs)
        
        if len(normalized_delta_powers) != len(epoch_exponents):
            print(f"Error: Length mismatch for subject {subject_id}")
            continue
            
        bout_powers, bout_exponents = compute_bout_means(normalized_delta_powers, epoch_exponents, epochs)
        
        # Store results with subject ID
        for power, exponent in zip(bout_powers, bout_exponents):
            all_bout_data.append({
                'subject': subject_id,
                'power': power,
                'exponent': exponent
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_bout_data)
    
    # Fit exponential curve
    popt, pcov = optimize.curve_fit(exp_func, df['power'], df['exponent'], 
                                  p0=[1, 0.1, 1], maxfev=10000)
    
    # Generate points for smooth curve
    x_fit = np.linspace(df['power'].min(), df['power'].max(), 100)
    y_fit = exp_func(x_fit, *popt)
    
    # Calculate R-squared
    residuals = df['exponent'] - exp_func(df['power'], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df['exponent'] - np.mean(df['exponent']))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate p-value using F-test
    n = len(df)
    p = 3  # number of parameters in exp_func
    f_stat = (r_squared / (p-1)) / ((1 - r_squared) / (n-p))
    p_value = 1 - stats.f.cdf(f_stat, p-1, n-p)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='power', y='exponent', alpha=0.6, s=80)
    plt.plot(x_fit, y_fit, 'r-')
    
    # Calculate p-value in scientific notation
    p_val_exp = int(np.floor(np.log10(p_value)))
    
    # Add stats text
    stats_text = (f'R² = {r_squared:.2f}\n'
                 f'p = 1x10^{p_val_exp}\n'
                 f'y = {popt[0]:.2f}*e^({popt[1]:.2f}x) + {popt[2]:.2f}')
    plt.text(0.05, 0.99, stats_text,
             transform=plt.gca().transAxes, 
             fontsize=18,
             verticalalignment='top')
    
    plt.title(f'1/f Exponent vs NREM Delta Power vs Exponent (All Subjects)', pad=20, fontsize=24)
    plt.xlabel('Normalized Delta Power', fontsize=24)
    plt.ylabel('1/f Exponent', fontsize=24, labelpad=10)
    plt.tick_params(axis='both', labelsize=24)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/ceph/harris/volkan/fooof/plots/delta_power/nrem_delta_vs_exponent_all_subjects_exponential.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    return df

# Example usage
eeg_files = [
    '/ceph/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-007_ses-01_recording-01.pkl',
    '/ceph/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-010_ses-01_recording-01.pkl',
    '/ceph/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-011_ses-01_recording-01.pkl',
    '/ceph/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-015_ses-01_recording-01.pkl',
    '/ceph/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
    '/ceph/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-017_ses-01_recording-01.pkl'
]

pickle_files = [
    '/ceph/harris/volkan/fooof/fooof_results/sub-007_ses-01_recording-01_fooof.pkl',
    '/ceph/harris/volkan/fooof/fooof_results/sub-010_ses-01_recording-01_fooof.pkl',
    '/ceph/harris/volkan/fooof/fooof_results/sub-011_ses-01_recording-01_fooof.pkl',
    '/ceph/harris/volkan/fooof/fooof_results/sub-015_ses-01_recording-01_fooof.pkl',
    '/ceph/harris/volkan/fooof/fooof_results/sub-016_ses-02_recording-01_fooof.pkl',
    '/ceph/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl'
]

csv_files = [
    '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_512Hz.csv',
    '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_512Hz.csv',
    '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_512Hz.csv',
    '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_512Hz_stitched.csv',
    '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_512Hz.csv',
    '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_512Hz.csv'
]

exclusion_ranges = {
    '015': [('2024-11-19 08:49:50', '2024-11-19 08:57:59')]
}

results_df = analyze_multiple_subjects(eeg_files, csv_files, pickle_files, exclusion_ranges)

