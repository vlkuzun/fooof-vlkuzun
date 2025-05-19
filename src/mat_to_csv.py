# import required libraries
import numpy as np
import pandas as pd
import h5py
import os
import glob

## This code is modified from Arya & Francesca's example

# Ask the user to input the directory path
input_directory = input("Please enter the directory path: ")

# Validate if the entered path is a valid directory
if os.path.isdir(input_directory):
    print(f"The directory {input_directory} exists and is valid.")
else:
    print(f"The directory {input_directory} does not exist. Please check the path.")


def get_output_directory(input_directory):

    ''' Ask whether they would like to create their own output directory path or whether it should automatically create an output directory path '''
    
    # Ask the user if they want to specify their own output directory
    user_input = input("Would you like to specify your own output directory? (yes/no): ").strip().lower()

    if user_input == 'yes':
        # If yes, prompt the user for the output directory path
        output_directory = input("Please provide the output directory path: ").strip()
        
        # Check if the directory exists, and create it if necessary
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Output directory '{output_directory}' created.")
        else:
            print(f"Output directory '{output_directory}' already exists.")
    else:
        # If no, create an 'output_mat_to_csv' subfolder in the input directory
        output_directory = os.path.join(input_directory, "output_mat_to_csv")
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Subfolder output directory '{output_directory}' created.")
        else:
            print(f"Subfolder output directory '{output_directory}' already exists.")
    
    return output_directory

output_directory = get_output_directory(input_directory)

# Ask the user to input sampling rate and manual annotations sleep epochs
while True:
    try:
        sampling_rate = int(input("Please enter the sampling rate in Hz: "))
        sleep_stage_resolution = int(input("Please enter the sleep stage resolution in seconds: "))
        print(f"You entered the sampling rates: {sampling_rate} and sleep stage resolution: {sleep_stage_resolution}")
        break  # Exit loop if both inputs are valid
    except ValueError:
        print("One or both of your inputs are not valid integers. Please try again.")


def mat_to_csv(input_directory, output_directory, sampling_rate, sleep_stage_resolution): 
    '''
    Converts .mat files extraced from Spike2 into .csv files to be used in the fooof analysis.
    Inputs:
    directory_path: str, path to the directory containing the .mat files
    output_directory_path: str, path to the directory where the .csv files should be saved
    sampling_rate: int, Hz (samples per second)
    sleep_stage_resolution: int, seconds 

    '''
    mat_files = glob.glob(os.path.join(input_directory, '*.mat'))

    for file_path in mat_files:
        # Extract the base filename without the extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        with h5py.File(file_path, 'r') as raw_data:
            print(f'Processing file: {file_path}')
            # print('Type and size of structure:', type(raw_data), np.size(raw_data), raw_data.keys()) # for debugging

            # Determine the sampling rate based on the filename
            sampling_rate = sampling_rate # Hz (samples per second)

            # Define the sleep stage resolution
            sleep_stage_resolution = sleep_stage_resolution # seconds

            # Initialize variables to store data
            eeg1_data, eeg2_data, sleep_stages = None, None, None

            # Iterate over all keys in the HDF5 file to extract data
            for key in raw_data.keys():
                if key.endswith('_EEG_EEG1A_B'):
                    eeg1_data = np.array(raw_data[key]['values'])
                elif key.endswith('_EEG_EEG2A_B'):
                    eeg2_data = np.array(raw_data[key]['values'])
                elif key.endswith('_Stage_1_'):
                    sleep_stages = np.array(raw_data[key]['codes'])
                    sleep_stages = sleep_stages[0, :]
            
        # Check if the data was found
            if eeg1_data is not None:
                print("EEG1 data extracted successfully.")
            if eeg2_data is not None:
                print("EEG2 data extracted successfully.")
            if sleep_stages is not None:
                print("Sleep stage data extracted successfully.")

            # format data for saving to a CSV file
            eeg1_flattened = eeg1_data.flatten()
            eeg2_flattened = eeg2_data.flatten()
            assert eeg1_flattened.shape == eeg2_flattened.shape, "The flattened shapes of the two EEG channels do not match"

            # upsample the sleep stages to match the resolution of the EEG and EMG data
            upsampled_sleep_stages = np.repeat(sleep_stages, sampling_rate * sleep_stage_resolution)
            if len(upsampled_sleep_stages) != len(eeg1_flattened):
                print(f"Length of upsampled sleep stages ({len(upsampled_sleep_stages)}) does not match length of EEG data ({len(eeg1_flattened)}) by {len(eeg1_flattened) - len(upsampled_sleep_stages)} samples") 
                if len(upsampled_sleep_stages) > len(eeg1_flattened):
                    upsampled_sleep_stages = upsampled_sleep_stages[:len(eeg1_flattened)]
                    print("Upsampled sleep stages truncated to match length of EEG data")
                else:
                    padding_length = len(eeg1_flattened) - len(upsampled_sleep_stages)
                    upsampled_sleep_stages = np.pad(upsampled_sleep_stages, (0, padding_length), mode='constant')
                    print("Upsampled sleep stages padded with zeros to match length of EEG data")
                assert len(upsampled_sleep_stages) == len(eeg1_flattened), "Length of upsampled sleep stages does not match length of EEG data after truncation" 
                print("Length of upsampled sleep stages matches length of EEG data after truncation")


            extracted_data = {
                'sleepStage': upsampled_sleep_stages,
                'EEG1': eeg1_flattened,
                'EEG2': eeg2_flattened,

            }

            df = pd.DataFrame(extracted_data)

            # Save DataFrame to a CSV file
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_file_path = os.path.join(output_directory, base_filename + '.csv')
            df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    mat_to_csv(input_directory, output_directory, sampling_rate=sampling_rate, sleep_stage_resolution=sleep_stage_resolution)