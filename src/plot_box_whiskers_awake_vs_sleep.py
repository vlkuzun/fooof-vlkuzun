import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def calculate_and_plot_exponents_multiple(file_pairs):
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

        if 'sleepStage' not in sleep_data.columns:
            raise ValueError(f"The CSV file {csv_path} must contain a 'sleepStage' column.")

        # Ensure the sleepStage column has the correct number of rows
        if len(sleep_data) != len(avg_exps) * 10:
            raise ValueError(f"Mismatch between sleep stage data and exponent data length for {csv_path}.")

        # Determine the mode of the sleep stage for each 10-second interval
        sleep_stages_10s = []
        for i in range(len(avg_exps)):
            interval = sleep_data['sleepStage'][i * 10:(i + 1) * 10]
            if len(interval) == 0:
                raise ValueError(f"Empty interval at index {i} in {csv_path}.")
            sleep_stages_10s.append(mode(interval, keepdims=True).mode[0])

        # Pair the modes with their respective average exponents
        data = pd.DataFrame({
            'Exponent': avg_exps,
            'SleepStage': sleep_stages_10s
        })

        # Group into continuous bouts of Awake or Sleep
        data['State'] = data['SleepStage'].map({1: 'Awake', 2: 'Sleep', 3: 'Sleep'})
        data['Bout'] = (data['State'] != data['State'].shift()).cumsum()

        # Calculate average exponents for each bout
        bout_data = data.groupby(['Bout', 'State'])['Exponent'].mean().reset_index()
        bout_data['Subject'] = pickle_path  # Tag with subject identifier
        all_bout_data.append(bout_data)

    # Combine all bout data into a single DataFrame
    combined_bout_data = pd.concat(all_bout_data, ignore_index=True)

    # Fit the mixed effects model with 'State' as fixed effect and 'Subject' as random effect
    model = smf.mixedlm('Exponent ~ State', combined_bout_data, groups=combined_bout_data['Subject'], re_formula='~State')
    result = model.fit()

    # Extract p-value for the 'State' fixed effect
    p_value = result.pvalues['State[T.Sleep]']

    # Display the results
    print(result.summary())

    # Set Matplotlib font settings
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Example: use serif fonts

    # Plotting
    plt.figure(figsize=(6, 10))

    # Define categories and desired x-tick positions
    categories = ['Awake', 'Sleep']

    # Create the boxplot
    sns.boxplot(
        x='State', y='Exponent', data=combined_bout_data,
        palette='tab10', showfliers=False, width=0.9, order=categories
    )

    # Get the current axis
    ax = plt.gca()

    # Create the stripplot
    sns.stripplot(
        x='State', y='Exponent', data=combined_bout_data,
        color='gray', alpha=0.2, jitter=0.3, size=4, order=categories
    )

    # Adjust the x-tick positions
    plt.xticks(ticks=[0,1], labels=categories, fontsize=18)

    # Adjust x-axis limits to center boxplots if necessary
    plt.xlim(-0.5, 1.5)  # Ensure spacing around boxplots

    # Add a horizontal bar between the x-ticks
    plt.plot([0, 1], [max(combined_bout_data['Exponent']) + 0.1] * 2, color='black', lw=1)

    # Add the p-value to the plot
    plt.text(0.5, max(combined_bout_data['Exponent']) + 0.15, f"p = {p_value:.2f}1", ha='center', fontsize=14)
    
    # Adjust the left margin to shift the plot
    plt.subplots_adjust(left=0.2)  # Increase the left margin (default is 0.125)

    # Formatting the plot
    plt.ylabel('1/f Exponent', fontsize=18)
    plt.title('1/f Exponent by Sleep Stage', fontsize=18, pad=20)
    plt.xlabel('')  # Remove the x-axis label
    plt.tick_params(axis='x', labelsize=18)  # Set the x-axis tick label size
    plt.tick_params(axis='y', labelsize=18)  # Set the y-axis tick label size

    # Remove the top and right spines
    sns.despine(top=True, right=True)  

    plt.savefig('/Volumes/harris/volkan/fooof/plots/stage_comparison/awake_vs_sleep_boxplot_LMER.png', dpi=600)

    plt.tight_layout()
    plt.show()
    


# Example usage with multiple files
file_pairs = [('/Volumes/harris/volkan/fooof/fooof_results/sub-007_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_1Hz.csv'),
    ("/Volumes/harris/volkan/fooof/fooof_results/sub-010_ses-01_recording-01_fooof.pkl", "/Volumes/harris/volkan/sleep_profile/downsample_auto_score/scoring_analysis/automated_state_annotationoutput_sub-010_ses-01_recording-01_time-0-69h_1Hz.csv"),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-011_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-011_ses-01_recording-01_time-0-72h_1Hz.csv'),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-015_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-015_ses-01_recording-01_time-0-49h_1Hz_stitched.csv'),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-016_ses-02_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-016_ses-02_recording-01_time-0-91h_1Hz.csv'),
    ('/Volumes/harris/volkan/fooof/fooof_results/sub-017_ses-01_recording-01_fooof.pkl','/Volumes/harris/volkan/sleep_profile/downsample_auto_score/automated_state_annotationoutput_sub-017_ses-01_recording-01_time-0-98h_1Hz.csv')]
calculate_and_plot_exponents_multiple(file_pairs)# Example usage
