import os
import pickle
import numpy as np
from neurodsp.spectral import compute_spectrum
from fooof import FOOOFGroup

def process_eeg_signals(subject_files, save_location, fs, epoch_duration, target_channels, fooof_params):
    """
    Processes EEG signals from pickle files, computes PSD, runs FOOOF modeling, and saves results.

    Parameters:
        subject_files (dict): Dictionary where keys are subject IDs and values are file paths to pickle files.
        save_location (str): Directory to save the FOOOF model pickle files.
        fs (int): Sampling rate of the EEG signals in Hz.
        epoch_duration (int): Duration of each epoch in seconds.
        target_channels (list): List of channels to process.
        fooof_params (dict): Parameters for the FOOOF fitting.

    Returns:
        None
    """
    # Ensure the save location exists
    os.makedirs(save_location, exist_ok=True)

    # Define a small number (epsilon) to add to 0 values
    epsilon = 1e-10

    for subject_id, file_path in subject_files.items():
        print(f"Processing subject {subject_id} from {file_path}...")

        # Load EEG data from the pickle file
        with open(file_path, 'rb') as f:
            eeg_data = pickle.load(f)

        # Prepare storage for PSD and FOOOF results
        psd_values_dict = {}
        frequencies = None

        # Compute PSD for each target channel
        for channel, sig in eeg_data.items():
            if channel not in target_channels:
                continue

            # Initialize storage for the current channel
            psd_values_dict[channel] = []

            # Determine number of epochs and samples per epoch
            recording_seconds = len(sig) / fs
            num_bins = int(recording_seconds // epoch_duration)
            samples_per_bin = fs * epoch_duration

            # Loop over each epoch
            for i in range(num_bins):
                start = i * samples_per_bin
                end = start + samples_per_bin
                bin_data = sig[start:end]

                # Compute the PSD using Welch's method
                freqs, psd = compute_spectrum(bin_data, fs, method='welch', avg_type='mean', nperseg=fs*2)
                psd_values_dict[channel].append(psd)

                # Store frequencies once
                if frequencies is None:
                    frequencies = freqs

            # Convert PSD values to NumPy array and replace zeros with epsilon
            psd_values_dict[channel] = np.array(psd_values_dict[channel])
            psd_values_dict[channel][psd_values_dict[channel] == 0] = epsilon

        # Fit the FOOOF model across all PSDs for each channel
        fooof_results = {}
        for channel, psd_matrix in psd_values_dict.items():
            # Initialize a FOOOFGroup for modeling
            fg = FOOOFGroup(**fooof_params)
            fg.fit(frequencies, psd_matrix, [2, 40])
            fooof_results[channel] = fg

        # Save the FOOOF results for the subject
        save_path = os.path.join(save_location, f"{subject_id}_fooof.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(fooof_results, f)

        print(f"Saved FOOOF results for {subject_id} to {save_path}")

# Example usage
subject_files = {
    'sub-007_ses-01_recording-01': '/ceph/harris/somnotate/to_score_set/pickle_eeg_somno/eeg_data_sub-007_ses-01_recording-01.pkl',
    'sub-010_ses-01_recording-01': '/ceph/harris/somnotate/to_score_set/pickle_eeg_somno/eeg_data_sub-010_ses-01_recording-01.pkl',
    'sub-011_ses-01_recording-01': '/ceph/harris/somnotate/to_score_set/pickle_eeg_somno/eeg_data_sub-011_ses-01_recording-01.pkl',
    'sub-015_ses-01_recording-01': '/ceph/harris/somnotate/to_score_set/pickle_eeg_somno/eeg_data_sub-015_ses-01_recording-01.pkl',
    'sub-016_ses-02_recording-01': '/ceph/harris/somnotate/to_score_set/pickle_eeg_somno/eeg_data_sub-016_ses-02_recording-01.pkl',
    'sub-017_ses-01_recording-01': '/ceph/harris/somnotate/to_score_set/pickle_eeg_somno/eeg_data_sub-017_ses-01_recording-01.pkl'
}
save_location = '/ceph/harris/volkan/fooof/fooof_results'
fs = 512  # Sampling rate in Hz
epoch_duration = 10  # Epoch duration in seconds
target_channels = ['EEG1', 'EEG2']
fooof_params = {
    'peak_width_limits': [1.0, 8.0],
    'max_n_peaks': 6,
    'min_peak_height': 0.1,
    'peak_threshold': 2.0,
    'aperiodic_mode': 'fixed'
}

process_eeg_signals(subject_files, save_location, fs, epoch_duration, target_channels, fooof_params)
