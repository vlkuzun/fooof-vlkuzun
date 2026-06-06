import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_publication_style():
    """Apply publication-quality defaults with Arial font for vector export."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,  # Keep text editable in Illustrator
            "ps.fonttype": 42,
        }
    )


def plot_exponent_values(start_time, epoch_duration, fg_pickle_path, output_pdf_path=None):
    apply_publication_style()

    # Convert start_time to datetime if it's not already
    start_time = pd.to_datetime(start_time)

    # Define the reference time (09:00)
    reference_time = pd.Timestamp("09:00:00").time()

    # Load the pre-trained FOOOFGroup models from pickle file
    with open(fg_pickle_path, "rb") as f:
        fg_models = pickle.load(f)

    # Extract aperiodic parameters for each channel
    exps_eeg1 = fg_models["EEG1"].get_params("aperiodic_params", "exponent")
    exps_eeg2 = fg_models["EEG2"].get_params("aperiodic_params", "exponent")
    avg_exps = (exps_eeg1 + exps_eeg2) / 2

    def calculate_duration_hours(epoch_duration_seconds, index):
        # Calculate the time difference in seconds from the start
        time_diff_seconds = index * epoch_duration_seconds
        return time_diff_seconds / 3600  # 3600 seconds in an hour

    def calculate_zt(start_time_value, epoch_duration_seconds, index):
        # Calculate the time difference from start_time
        time_diff_seconds = index * epoch_duration_seconds
        current_time = start_time_value + pd.Timedelta(seconds=time_diff_seconds)
        # Calculate the time difference from 09:00
        reference_minutes = reference_time.hour * 60 + reference_time.minute
        current_minutes = (
            current_time.hour * 60 + current_time.minute + current_time.second / 60
        )
        time_diff = current_minutes - reference_minutes
        return (time_diff % 1440) / 60  # 1440 minutes in a day

    # Calculate duration_hours and ZT values for each row
    duration_hours_values = [
        calculate_duration_hours(epoch_duration, i) for i in range(len(avg_exps))
    ]
    zt_values = [calculate_zt(start_time, epoch_duration, i) for i in range(len(avg_exps))]

    # Convert avg_exps to a DataFrame to add the duration_hours and ZT columns
    avg_exps_df = pd.DataFrame(avg_exps)
    avg_exps_df["duration_hours"] = duration_hours_values
    avg_exps_df["ZT"] = zt_values

    # Rename the first column of avg_exps_df to 'avg_exp'
    avg_exps_df.rename(columns={avg_exps_df.columns[0]: "avg_exp"}, inplace=True)

    ## Plot the average exponent values across ZT - scatter plot every 10 minutes and showing entire recording length

    # Group by every 60 rows and calculate the mean for avg_exp
    grouped_df = avg_exps_df.groupby(avg_exps_df.index // 60).agg(
        {"avg_exp": "mean", "duration_hours": "mean", "ZT": "first"}
    )

    # Calculate the rolling average
    rolling_window_size = 5  # Adjust this value as needed
    grouped_df["rolling_avg"] = grouped_df["avg_exp"].rolling(window=rolling_window_size).mean()

    # Create the scatter plot
    plt.figure(figsize=(20, 6))

    # Plot background shading to match the line plot:
    # ZT 0-1 and 12-13 are highlighted first-hour blocks.
    def phase_style(zt_value):
        if zt_value < 1:
            return ("#FFD1A1", 0.8)
        if zt_value < 12:
            return ("orange", 0.5)
        if zt_value < 13:
            return ("#C0C0C0", 0.5)
        return ("gray", 0.5)

    span_kwargs = {
        "linewidth": 0,
        "edgecolor": "none",
        "antialiased": False,
        "zorder": 0,
    }

    current_style = phase_style(grouped_df["ZT"].iloc[0])
    start_duration = grouped_df["duration_hours"].iloc[0]

    for i in range(1, len(grouped_df)):
        end_duration = grouped_df["duration_hours"].iloc[i]
        new_style = phase_style(grouped_df["ZT"].iloc[i])

        if new_style != current_style:
            color, alpha = current_style
            plt.axvspan(
                start_duration,
                end_duration,
                color=color,
                alpha=alpha,
                **span_kwargs,
            )
            start_duration = end_duration
            current_style = new_style

    # Add the last span up to the right edge of the final time bin.
    if len(grouped_df) > 1:
        step = grouped_df["duration_hours"].iloc[1] - grouped_df["duration_hours"].iloc[0]
    else:
        step = 0

    color, alpha = current_style
    plt.axvspan(
        start_duration,
        grouped_df["duration_hours"].iloc[-1] + step,
        color=color,
        alpha=alpha,
        **span_kwargs,
    )

    # Variable for scatter plot marker size
    marker_size = 7  # Adjust this value to reduce or increase the size of the scatter plot markers

    # Scatter plot of mean exponent values vs. first duration_hours
    plt.scatter(grouped_df["duration_hours"], grouped_df["avg_exp"], c="blue", s=marker_size, alpha=0.4)

    # Plot the rolling average line
    plt.plot(grouped_df["duration_hours"], grouped_df["rolling_avg"], c="black", label="Rolling Average")

    # Add labels and title
    plt.xlabel("Time (hours)", fontsize=22)
    plt.ylabel("1/f Exponent", fontsize=22)

    # Set x-ticks every 12 hours
    max_duration = grouped_df["duration_hours"].max()
    plt.xticks(np.arange(0, max_duration, 12), fontsize=22)

    # Set x-axis limits to cover the entirety of the edges of the plot box
    plt.xlim(0, max_duration)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tick_params(axis="y", labelsize=22)
    plt.tight_layout()

    if output_pdf_path is None or str(output_pdf_path).strip() == "":
        output_pdf_path = (
            Path(fg_pickle_path).with_suffix("").name + "_exponent_scatter_publication.pdf"
        )

    output_pdf_path = Path(output_pdf_path)
    if output_pdf_path.suffix.lower() != ".pdf":
        output_pdf_path = output_pdf_path.with_suffix(".pdf")

    plt.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved publication PDF to: {output_pdf_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    plot_exponent_values(
        start_time=input("Input HH:MM:SS start time of recording: "),
        epoch_duration=int(
            input("Enter length of epoch for exponent analysis in seconds: ")
        ),
        fg_pickle_path=input("Enter path to the pre-trained FOOOFGroup pickle file: "),
        output_pdf_path=input(
            "Enter output PDF path (or press Enter for auto name): "
        ),
    )
