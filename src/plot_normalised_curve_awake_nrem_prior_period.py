import pandas as pd
import numpy as np
import pickle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

def find_dense_periods(df, stage_value, window_size=3600):
    stage_binary = (df['sleepStage'] == stage_value).astype(int)
    stage_density = []
    
    for i in range(len(df) - window_size + 1):
        density = np.mean(stage_binary[i:i + window_size])
        stage_density.append(density)
    
    peaks, _ = find_peaks(stage_density, height=0.66, distance=1200)
    return [df['Timestamp'].iloc[peak] for peak in peaks]

def get_prior_mean_exponents(period_starts, exp_df):
    prior_means = []
    for start in period_starts:
        window_start = start - timedelta(hours=2)
        mask = (exp_df['Timestamp'] >= window_start) & (exp_df['Timestamp'] < start)
        prior_means.append(exp_df.loc[mask, 'exponent'].mean())
    return prior_means

def process_single_subject(csv_path, pickle_path):
    # Load data
    sleep_df = pd.read_csv(csv_path)
    sleep_df['Timestamp'] = pd.to_datetime(sleep_df['Timestamp'])
    
    with open(pickle_path, 'rb') as f:
        fg_models = pickle.load(f)
    
    # Calculate average exponents
    exps_eeg1 = fg_models['EEG1'].get_params('aperiodic_params', 'exponent')
    exps_eeg2 = fg_models['EEG2'].get_params('aperiodic_params', 'exponent')
    avg_exps = (exps_eeg1 + exps_eeg2) / 2
    
    exp_times = pd.date_range(start=sleep_df['Timestamp'].iloc[0], 
                             periods=len(avg_exps), 
                             freq='10S')
    exp_df = pd.DataFrame({'Timestamp': exp_times, 'exponent': avg_exps})
    
    # Get dense periods
    nrem_periods = find_dense_periods(sleep_df, stage_value=2)
    awake_periods = find_dense_periods(sleep_df, stage_value=1)
    
    # Calculate prior exponents
    nrem_exps = get_prior_mean_exponents(nrem_periods, exp_df)
    awake_exps = get_prior_mean_exponents(awake_periods, exp_df)
    
    return nrem_exps, awake_exps

def process_all_subjects(csv_files, pickle_files):
    data_list = []
    
    for subject_id, (csv_file, pickle_file) in enumerate(zip(csv_files, pickle_files), 1):
        nrem_exps, awake_exps = process_single_subject(csv_file, pickle_file)
        
        data_list.extend([{
            'Subject': f'sub-{subject_id:03d}',
            'Stage': 'NREM Dense',
            'Exponent': exp
        } for exp in nrem_exps])
        
        data_list.extend([{
            'Subject': f'sub-{subject_id:03d}',
            'Stage': 'Awake Dense',
            'Exponent': exp
        } for exp in awake_exps])
    
    return pd.DataFrame(data_list)

def plot_exponent_distribution(df, save_path=None):
    plt.figure(figsize=(10, 8))

    # Define colors for each stage
    colors = {'Awake Dense': '#1f77b4', 'NREM Dense': '#ff7f0e'}

    # Create KDE plot for each stage
    for stage in ['Awake Dense', 'NREM Dense']:
        stage_data = df[df['Stage'] == stage]['Exponent']
        sns.kdeplot(data=stage_data, 
                    label=stage,
                    color=colors[stage],
                    fill=True,
                    alpha=0.4)

    plt.title('Distribution of Prior Period Exponent\nby Subsequent Vigilance Stage', 
              pad=20, 
              fontsize=26)
    plt.xlabel('1/f Exponent', 
              fontsize=26, 
              labelpad=15)
    plt.ylabel('Normalized Density', 
              fontsize=26, labelpad=15)

    # Customize the plot
    plt.tick_params(labelsize=26)
    plt.yticks([])

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600)
    
    plt.show()

# File paths
csv_files = [
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv',
    '/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv'
]

pickle_files = [
    '/Volumes/harris/volkan/fooof/fooof_results/sub-007_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-010_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-011_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-015_ses-01_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-016_ses-02_recording-01_fooof.pkl',
    '/Volumes/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl'
]

# Process data
df = process_all_subjects(csv_files, pickle_files)

# Plot and save figure
plot_exponent_distribution(df, save_path='/Volumes/harris/volkan/fooof/plots/prior_period/prior_period_dense_awake_nrem_normalised_distribution.png')