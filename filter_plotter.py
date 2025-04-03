import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def load_ecg_data(file_path):
    # Load file
    df = pd.read_csv(file_path)
    
    # Replace non-numeric characters with NaNs
    df.replace('!', np.nan, regex=True, inplace=True)    
    df['ECG Value'] = pd.to_numeric(df['ECG Value'], errors='coerce')
    
    # Remove rows with NaN values or where ECG value equals 4095 or 0
    df = df[~df['ECG Value'].isin([4095, 0])]

    # Replace extreme outlier values (e.g., less than -1000 or greater than 3000) with NaN
    df['ECG Value'] = df['ECG Value'].apply(lambda x: np.nan if abs(x) > 2500 else x)
    
    # Interpolate missing values or drop them
    df['ECG Value'].interpolate(method='linear', inplace=True)  # Interpolation
    df.dropna(subset=['ECG Value'], inplace=True)  # Drop rows if interpolation fails
    
    return df['ECG Value'].values  

# Plotting the filtered ECG signal
def plot_ecg(original_data, filtered_data, fs):
    time = np.arange(len(original_data)) / fs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, original_data, label='Original ECG')
    plt.title('Original ECG Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_data, label='Filtered ECG', color='orange')
    plt.title('Filtered ECG Signal (Bandpass 0.5-50Hz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

# Example usage
fs = 250  # Sampling rate of 250Hz
file_path = '/Users/jasaro/Documents/Python_files/CPEG398/ecg_data_no_adhesive.csv'  

# Load and filter data
ecg_data = load_ecg_data(file_path)
filtered_ecg = bandpass_filter(ecg_data, lowcut=0.5, highcut=50.0, fs=fs)

# Plot the results
plot_ecg(ecg_data, filtered_ecg, fs)
