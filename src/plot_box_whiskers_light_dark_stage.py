import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison

def calculate_and_plot_exponents_light_dark_with_lmm(file_pairs):
    all_bout_data = []

    for pickle_path, csv_path in file_pairs:
        # Load the pre-trained FOOOFGroup models from pickle file
        with open(pickle_path, 'rb') as f:
            fg_models = pickle.load(f)

        # Extract aperiodic parameters for each channel
        exps_eeg1 = fg_models['EEG1'].get_params('aperiodic_params', 'exponent')
        exps_eeg2 = fg_models['EEG2'].get_params('aperiodic_params', 'exponent')
        avg_exps = (exps_eeg1 + exps_eeg2) / 2  # Average exponent across channels

        # Load the CSV file containing sleep stages
        sleep_data = pd.read_csv(csv_path)

        if 'sleepStage' not in sleep_data.columns or 'Timestamp' not in sleep_data.columns:
            raise ValueError(f"The CSV file {csv_path} must contain 'sleepStage' and 'Timestamp' columns.")

        if len(sleep_data) != len(avg_exps) * 10:
            raise ValueError(f"Mismatch between sleep stage data and exponent data length for {csv_path}.")

        # Determine the mode of the sleep stage for each 10-second interval
        sleep_stages_10s = []
        light_dark_periods = []
        for i in range(len(avg_exps)):
            interval = sleep_data['sleepStage'][i * 10:(i + 1) * 10]
            Timestamps = sleep_data['Timestamp'][i * 10:(i + 1) * 10]
            if len(interval) == 0 or len(Timestamps) == 0:
                raise ValueError(f"Empty interval at index {i} in {csv_path}.")
            sleep_stages_10s.append(mode(interval, keepdims=True).mode[0])

            first_time = pd.to_datetime(Timestamps.iloc[0])
            if 9 <= first_time.hour < 21:
                light_dark_periods.append('Light')
            else:
                light_dark_periods.append('Dark')

        data = pd.DataFrame({
            'Exponent': avg_exps,
            'SleepStage': sleep_stages_10s,
            'Period': light_dark_periods
        })

        data['State'] = data['SleepStage'].map({1: 'Awake', 2: 'NREM', 3: 'REM'})
        data['StatePeriod'] = data['Period'] + ' ' + data['State']
        data['Bout'] = (data['StatePeriod'] != data['StatePeriod'].shift()).cumsum()

        bout_data = data.groupby(['Bout', 'StatePeriod', 'Period', 'State'])['Exponent'].mean().reset_index()
        bout_data['Subject'] = pickle_path  # Tag with subject identifier
        all_bout_data.append(bout_data)

    combined_bout_data = pd.concat(all_bout_data, ignore_index=True)

    # LMM Analysis
    combined_bout_data['Subject'] = combined_bout_data['Subject'].astype('category')
    model = smf.mixedlm(
        "Exponent ~ State * Period", 
        combined_bout_data, 
        groups=combined_bout_data["Subject"]
    )
    lmm_result = model.fit()
    print(lmm_result.summary())
    
    # Perform pairwise comparisons using estimated marginal means

    combined_bout_data['StatePeriod'] = combined_bout_data['State'] + " " + combined_bout_data['Period']
    mc = MultiComparison(combined_bout_data['Exponent'], combined_bout_data['StatePeriod'])
    tukey_result = mc.tukeyhsd()
    print(tukey_result.summary())


    # Plotting
    plt.figure(figsize=(12, 10))
    categories = [
        'Light Awake', 'Dark Awake',
        'Light NREM', 'Dark NREM',
        'Light REM', 'Dark REM'
    ]

    sns.boxplot(
        x='StatePeriod', y='Exponent', data=combined_bout_data,
        palette='tab10', showfliers=False, width=0.9, order=categories
    )
    sns.stripplot(
        x='StatePeriod', y='Exponent', data=combined_bout_data,
        color='gray', alpha=0.2, jitter=0.3, size=4, order=categories
    )
    plt.xticks(ticks=range(len(categories)), labels=categories, fontsize=14, rotation=45)
    plt.ylabel('1/f Exponent', fontsize=18)
    plt.title('1/f Exponent by Sleep Stage and Period', fontsize=18, pad=20)
    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    sns.despine(top=True, right=True)

    # Save the plot
    #plt.savefig('/Volumes/harris/volkan/fooof/plots/stage_comparison/awake_nrem_rem_light_dark_boxplot_LMER.png', dpi=600)


    plt.tight_layout()
    plt.show()


# Example usage with the same file pairs
file_pairs = [('/Volumes/harris/volkan/fooof/fooof_results/sub-007_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv'),
    ("/Volumes/harris/volkan/fooof/fooof_results/sub-010_ses-01_recording-01_fooof.pkl", "/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv"),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-011_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv'),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-015_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv'),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-016_ses-02_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv'),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv')]

calculate_and_plot_exponents_light_dark_with_lmm(file_pairs)


