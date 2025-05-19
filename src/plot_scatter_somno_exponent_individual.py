from neurodsp.spectral import compute_spectrum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import mode
import matplotlib.patches as mpatches

def plot_exponent_values(start_time, epoch_duration, fg_pickle_path, fs, somno_scoring_csv):

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

    ## Add somno scoring to avg_exps_dfs

    # Load somno_scoring data from the CSV file
    somno_scoring = pd.read_csv(somno_scoring_csv)

    # Calculate the number of rows per epoch
    rows_per_epoch = int(epoch_duration * fs)

    # Add a column to indicate the epoch
    somno_scoring['epoch'] = somno_scoring.index // rows_per_epoch

    # Downsample the sleepStage column by taking the mode for each epoch
    downsampled_sleep_stage = somno_scoring.groupby('epoch').agg({
        "sleepStage": lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index(drop=True)

    # Check if the number of rows are identical
    if len(avg_exps_df) != len(downsampled_sleep_stage):
        print("Error: The number of rows in avg_exps_df and downsampled_sleep_stage are not identical.")
    else:
        # Add the downsampled sleepStage values as a new column to avg_exps_df
        avg_exps_df['sleepStage'] = downsampled_sleep_stage['sleepStage'].values

    ## Plot the somno scoring
    
    # Filter data within the 57–69 hours range - provide some context for the window size and interval
    filtered_df = avg_exps_df[(avg_exps_df['duration_hours'] >= 56.9) & 
                            (avg_exps_df['duration_hours'] <= 69.1)].copy()

    # Define constants for window size and interval
    half_window = 5 / 60  # 5 minutes in fractional hours
    interval = 10 / 60    # 10 minutes in fractional hours

    # Create a time range with 10-minute intervals starting at 57 - provide extra for end point
    time_points = np.arange(57, 69.1, interval)

    # Compute the average for each time window
    means = []
    for tp in time_points:
        # Define the start and end of the averaging window
        window_start = tp - half_window
        window_end = tp + half_window
    
        # Select data within this window
        window_data = filtered_df[(filtered_df['duration_hours'] >= window_start) & 
                              (filtered_df['duration_hours'] <= window_end)]
    
        # Compute the mean avg_exp for this window
        means.append(window_data['avg_exp'].mean())

    # Create a new DataFrame for grouped data
    grouped_df = pd.DataFrame({
        'duration_hours': time_points,
        'avg_exp': means
    })

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(26, 6), constrained_layout=True)

    # Plot data points and line
    ax.scatter(grouped_df['duration_hours'], grouped_df['avg_exp'], color='black', label='Data Points', marker='o', s=50)
    ax.plot(grouped_df['duration_hours'], grouped_df['avg_exp'], color='black', linestyle='-', linewidth=2)

    # Adjust the position of the main plot to take less vertical space
    pos = ax.get_position()
    new_ax_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height * 0.9]  # Shrink height and move up
    ax.set_position(new_ax_pos)

    # Add inset axis for rectangle bars and position it dynamically below the main plot
    inset_bar_height = 0.05  # Adjust height of the bar
    bar_gap = 0.15           # Gap between the main plot and bar

    # Sort data by duration_hours
    filtered_df = filtered_df.sort_values('duration_hours')

    # Get unique stages and colors
    unique_stages = filtered_df['sleepStage'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))

    # Manually map colors to stages
    stage_color_map = {
        1: colors[1],  # 2nd color
        2: colors[2],  # 3rd color
        3: colors[0]   # 1st color
    }

    # Modified segment creation with gaps
    segments = []
    current_stage = filtered_df.iloc[0]['sleepStage']
    start_hour = filtered_df.iloc[0]['duration_hours']

    for i in range(1, len(filtered_df)):
        if filtered_df.iloc[i]['sleepStage'] != current_stage:
            segments.append({
                'start': start_hour,
                'end': filtered_df.iloc[i]['duration_hours'],  # Subtract gap
                'stage': current_stage
            })
            start_hour = filtered_df.iloc[i]['duration_hours']
            current_stage = filtered_df.iloc[i]['sleepStage']

    # Add final segment
    segments.append({
        'start': start_hour,
        'end': filtered_df.iloc[-1]['duration_hours'],  # No gap needed for last segment
        'stage': current_stage
    })

    # Plot segments with modified boundaries
    for segment in segments:
        color = stage_color_map[segment['stage']]
        ax.axvspan(segment['start'], segment['end'], facecolor=color, alpha=0.25, antialiased=False, edgecolor='none')

    # Create legend patches with new names and sort them in the desired order
    legend_patches = [mpatches.Patch(color=stage_color_map[stage], alpha=0.3, label={1: 'Awake', 2: 'NREM', 3: 'REM'}[stage]) 
                    for stage in unique_stages]

    # Sort legend patches in the order of Awake, NREM, REM
    order = ['Awake', 'NREM', 'REM']
    legend_patches = sorted(legend_patches, key=lambda x: order.index(x.get_label()))

    # Add legend in top right corner, closer to plot, and increase font size
    ax.legend(handles=legend_patches,
            loc='upper right', 
            bbox_to_anchor=(1.09, 1),
            borderaxespad=0.1,
            fontsize=18)

    inset_ax = fig.add_axes([
    new_ax_pos[0],                      # Same x-start as main plot
    new_ax_pos[1] - inset_bar_height - bar_gap,  # Position below the main plot
    new_ax_pos[2],                      # Same width as main plot
    inset_bar_height                   # Bar height
    ])

    # Manually add boundaries (spines) for the inset axis
    inset_ax.spines['top'].set_visible(True)
    inset_ax.spines['right'].set_visible(True)
    inset_ax.spines['left'].set_visible(True)
    inset_ax.spines['bottom'].set_visible(True)

    # Plot the bar
    inset_ax.axhspan(0, 1, xmin=0, xmax=(63 - 57) / (69 - 57), color='grey', alpha=0.5)
    inset_ax.axhspan(0, 1, xmin=(63 - 57) / (69 - 57), xmax=1, color='orange', alpha=0.5)   

    # Format axes
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    
    ax.set_xlim(56.95, 69.05)
    ax.set_ylim(grouped_df['avg_exp'].min() - 0.1, grouped_df['avg_exp'].max() + 0.1)

    # Labels and title
    ax.set_xlabel('Time (hours)', fontsize=24)
    ax.set_ylabel('1/f Exponent', fontsize=24)
    ax.set_title('Exponent Value Across Vigilance Stage', fontsize=26, pad=10)

    # Set x-ticks to start at 57 and at every second value until 69
    ax.set_xticks(range(57, 70, 2))
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Remove top and right spines of main plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
    plt.savefig('/ceph/harris/volkan/fooof/plots/multiple_days_exponent/exponent_zoom_57-69h_sub-017.png')

# Example usage
plot_exponent_values(
    start_time=input("Input in HH:MM:SS start time of recording: "),
    epoch_duration=int(input("Enter length of epoch for exponent analysis in seconds: ")),
    fg_pickle_path=input("Enter the path to the pre-trained FOOOFGroup pickle file: "),
    fs=1,
    somno_scoring_csv=input("Enter the path to the somno scoring CSV file: ")
)
