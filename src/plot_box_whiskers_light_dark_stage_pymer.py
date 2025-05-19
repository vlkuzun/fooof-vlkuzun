import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns
from pymer4.models import Lmer

def calculate_and_plot_exponents_light_dark_with_emm(file_pairs):
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

    # Prepare the data for pymer4 analysis
    combined_bout_data['Subject'] = combined_bout_data['Subject'].astype('category')

    # Fit the linear mixed model using pymer4
    model = Lmer("Exponent ~ State * Period + (1|Subject)", data=combined_bout_data)
    model.fit(
        factors={"State": ["Awake", "NREM", "REM"], "Period": ["Light", "Dark"]},
        ordered=False
    )
    print(model.anova())

    # Compute post-hoc tests
    marginal_estimates, comparisons = model.post_hoc(
        marginal_vars=["State"],
        p_adjust='fdr'
    )
    print(marginal_estimates)
    print(comparisons)

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

    # Extract p-values from comparisons for State groups
    p_values = comparisons["P-val"]
    pairs = comparisons["Contrast"]

    # Adjust the subplot layout to move the plot up
    plt.subplots_adjust(top=0.9)  # Increase the top margin (default is 0.8)

    # Create a mapping for the combined categories
    state_to_category = {
        'Awake': ['Light Awake', 'Dark Awake'],
        'NREM': ['Light NREM', 'Dark NREM'],
        'REM': ['Light REM', 'Dark REM']
    }

    # Set an initial y position for the first comparison
    y_position = 2.3
    spacing = 0.1  # Set the vertical spacing between comparisons

    # Annotating the plot with asterisks for significant comparisons
    for i, (pair, p_value) in enumerate(zip(pairs, p_values)):
        if p_value < 0.05:
            if p_value < 0.01:
                star = "***"
            elif p_value < 0.001:
                star = "****"
            else:
                star = "**"

            # Split the contrast string to get the two states (e.g., 'Awake' and 'NREM')
            first_state, second_state = pair.split(' - ')  # 'Awake - NREM' -> ['Awake', 'NREM']

            # Get the corresponding groups based on states
            first_group = state_to_category[first_state]
            second_group = state_to_category[second_state]

            # Get indices of the groups to draw lines and add text
            first_idx = [categories.index(group) for group in first_group]
            second_idx = [categories.index(group) for group in second_group]
        
            # Draw a line between the groups and add asterisk
            plt.plot([min(first_idx), max(second_idx)], [y_position, y_position], color='black', lw=1.5)
            plt.text((min(first_idx) + max(second_idx)) / 2, y_position - 0.02, star, ha='center', va='bottom', fontsize=18)

            # Increase y_position for the next comparison
            y_position += spacing

    # Determine the maximum y-value of the data
    y_min = combined_bout_data['Exponent'].min()
    y_max = combined_bout_data['Exponent'].max()

    # Set ylim to extend based on the number of comparisons
    plt.ylim(y_min - 0.1, y_max + 0.5)

    plt.xticks(ticks=range(len(categories)), labels=categories, fontsize=25, rotation=45)
    plt.ylabel('1/f Exponent', fontsize=25, labelpad=15)
    plt.title('1/f Exponent by Sleep Stage and Period', fontsize=25, pad=20)
    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    sns.despine(top=True, right=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig('/ceph/harris/volkan/fooof/plots/stage_comparison/awake_nrem_rem_light_dark_boxplot_LMM.png', dpi=600)

    
    plt.show()


# Example usage with the same file pairs
file_pairs = [('/ceph/harris/volkan/fooof/fooof_results/sub-007_ses-01_recording-01_fooof.pkl','/ceph/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv'),
    ("/ceph/harris/volkan/fooof/fooof_results/sub-010_ses-01_recording-01_fooof.pkl", "/ceph/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv"),
    ('/ceph/harris/volkan/fooof/fooof_results/sub-011_ses-01_recording-01_fooof.pkl','/ceph/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv'),
    ('/ceph/harris/volkan/fooof/fooof_results/sub-015_ses-01_recording-01_fooof.pkl','/ceph/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv'),
    ('/ceph/harris/volkan/fooof/fooof_results/sub-016_ses-02_recording-01_fooof.pkl','/ceph/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv'),
    ('/ceph/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl','/ceph/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv')]

calculate_and_plot_exponents_light_dark_with_emm(file_pairs)
