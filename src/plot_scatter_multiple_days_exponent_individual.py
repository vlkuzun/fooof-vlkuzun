from neurodsp.spectral import compute_spectrum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def plot_exponent_values(start_time, epoch_duration, fg_pickle_path):

    # Convert start_time to datetime if it's not already
    start_time = pd.to_datetime(start_time)

    # Define the reference time (09:00)
    reference_time = pd.Timestamp('09:00:00').time()

    # Load the pre-trained FOOOFGroup models from pickle file
    with open(fg_pickle_path, 'rb') as f:
        fg_models = pickle.load(f)

    # Extract aperiodic parameters for each channel
    exps_eeg1 = fg_models['EEG1'].get_params('aperiodic_params', 'exponent')
    exps_eeg2 = fg_models['EEG2'].get_params('aperiodic_params', 'exponent')
    avg_exps = (exps_eeg1 + exps_eeg2) / 2  # Use the average exponent across channels for future analysis

    # Function to calculate duration in hours
    def calculate_duration_hours(epoch_duration, index):
        # Calculate the time difference in seconds from the start
        time_diff_seconds = index * epoch_duration
        # Convert time difference to hours
        duration_hours = time_diff_seconds / 3600  # 3600 seconds in an hour
        return duration_hours

    # Function to calculate ZT value
    def calculate_zt(start_time, epoch_duration, index):
        # Calculate the time difference from start_time
        time_diff_seconds = index * epoch_duration
        current_time = start_time + pd.Timedelta(seconds=time_diff_seconds)
        # Calculate the time difference from 09:00
        reference_minutes = reference_time.hour * 60 + reference_time.minute
        current_minutes = current_time.hour * 60 + current_time.minute + current_time.second / 60
        time_diff = current_minutes - reference_minutes
        # Convert time difference to ZT value
        zt = (time_diff % 1440) / 60  # 1440 minutes in a day
        return zt

    # Calculate duration_hours and ZT values for each row
    duration_hours_values = [calculate_duration_hours(epoch_duration, i) for i in range(len(avg_exps))]
    zt_values = [calculate_zt(start_time, epoch_duration, i) for i in range(len(avg_exps))]

    # Convert avg_exps to a DataFrame to add the duration_hours and ZT columns
    avg_exps_df = pd.DataFrame(avg_exps)
    avg_exps_df['duration_hours'] = duration_hours_values
    avg_exps_df['ZT'] = zt_values

    # Rename the first column of avg_exps_df to 'avg_exp'
    avg_exps_df.rename(columns={avg_exps_df.columns[0]: 'avg_exp'}, inplace=True)

    ## Plot the average exponent values across ZT - scatter plot every 10 minutes and showing entire recording length

    # Group by every 60 rows and calculate the mean for avg_exp
    grouped_df = avg_exps_df.groupby(avg_exps_df.index // 60).agg({
        'avg_exp': 'mean',
        'duration_hours': 'mean',
        'ZT': 'first'
    })

    # Calculate the rolling average
    rolling_window_size = 5  # Adjust this value as needed
    grouped_df['rolling_avg'] = grouped_df['avg_exp'].rolling(window=rolling_window_size).mean()

    # Create the scatter plot
    plt.figure(figsize=(20, 6))

    # Plot background shading based on ZT column
    current_color = 'orange' if grouped_df['ZT'].iloc[0] < 12 else 'grey'
    start_duration = grouped_df['duration_hours'].iloc[0]

    for i in range(1, len(grouped_df)):
        end_duration = grouped_df['duration_hours'].iloc[i]
        new_color = 'orange' if grouped_df['ZT'].iloc[i] < 12 else 'grey'
    
        if new_color != current_color:
            plt.axvspan(start_duration, end_duration, color=current_color, alpha=0.3)
            start_duration = end_duration
            current_color = new_color

    # Add the last span
    plt.axvspan(start_duration, grouped_df['duration_hours'].iloc[-1] + (grouped_df['duration_hours'].iloc[1] - grouped_df['duration_hours'].iloc[0]), color=current_color, alpha=0.3)

    # Variable for scatter plot marker size
    marker_size = 7  # Adjust this value to reduce or increase the size of the scatter plot markers

    # Scatter plot of mean exponent values vs. first duration_hours
    plt.scatter(grouped_df['duration_hours'], grouped_df['avg_exp'], c='blue', s=marker_size, alpha=0.4)

    # Plot the rolling average line
    plt.plot(grouped_df['duration_hours'], grouped_df['rolling_avg'], c='black', label='Rolling Average')

    # Add labels and title
    plt.xlabel('Time (hours)', fontsize=22)
    plt.ylabel('1/f Exponent', fontsize=22)
    plt.title('Exponent Value Across Light/Dark Cycle', fontsize=26, pad=30)

    # Set x-ticks every 12 hours
    max_duration = grouped_df['duration_hours'].max()
    plt.xticks(np.arange(0, max_duration, 12), fontsize=22)

    # Set x-axis limits to cover the entirety of the edges of the plot box
    plt.xlim(0, max_duration)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(axis='y', labelsize=22)

    plt.tight_layout()
    # Show the plot
    plt.show()

# Example usage
plot_exponent_values(
    start_time=input("Input in HH:MM:SS start time of recording: "),
    epoch_duration=int(input("Enter length of epoch for exponent analysis in seconds: ")),
    fg_pickle_path=input("Enter the path to the pre-trained FOOOFGroup pickle file: ")
)
