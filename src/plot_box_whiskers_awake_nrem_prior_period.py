import pandas as pd
import numpy as np
import pickle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pymer4.models import Lmer

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

def plot_exponents(csv_files, pickle_files, output_file):
    # Process data
    df = process_all_subjects(csv_files, pickle_files)
    
    # Run LME
    model = Lmer("Exponent ~ Stage + (1|Subject)", data=df)
    results = model.fit()
    
    print(results)

    # Create plot with significance marker
    plt.figure(figsize=(8, 10))
    
    # Set category order
    category_order = ['Awake Dense', 'NREM Dense']
    
    # Calculate y-axis limits
    y_min = df['Exponent'].min() - 0.1
    y_max = df['Exponent'].max() + 0.15  # Increased range for significance marker
    
    # Create plot with specified order
    sns.boxplot(data=df, x='Stage', y='Exponent', 
                order=category_order,
                showfliers=False,
                color='white',
                palette='tab10',
                width=0.9)
    
    sns.stripplot(data=df, x='Stage', y='Exponent',
                  order=category_order,
                  color='grey', 
                  alpha=0.3,
                  size=5,
                  jitter=0.2)
    
    # Add significance marker
    p_value = results.loc['StageNREM Dense', 'P-val']
    x1, x2 = 0, 1
    bar_height = y_max - 0.05
    
    plt.plot([x1, x1, x2, x2], 
             [bar_height-0.05, bar_height, bar_height, bar_height-0.05], 
             color='black', linewidth=1)
    
    # Add asterisks based on p-value
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'ns'
    
    plt.text((x1+x2)/2, bar_height+0.01, significance, 
             ha='center', va='bottom', fontsize=18)
    
    plt.ylim(y_min, y_max)
    plt.title('Mean Exponents in 2-hour Period \nPrior to Dense Vigilance Stages', pad=20, fontsize=24)
    plt.xlabel('Subsequent Vigilance Stage', color='red', fontsize=24, labelpad=15)
    plt.ylabel('1/f Exponent', fontsize=24)
    plt.tick_params(labelsize=24)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
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

# Call the function to process data and create plot
plot_exponents(csv_files, pickle_files, '/Volumes/harris/volkan/fooof/plots/prior_period/prior_period_dense_awake_nrem.png')