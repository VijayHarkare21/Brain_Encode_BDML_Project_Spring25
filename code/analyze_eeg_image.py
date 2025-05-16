import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne
from mne.channels import make_standard_montage
from mne.time_frequency import psd_array_multitaper
import h5py
import os
from collections import OrderedDict
from data_loading_helpers_modified import load_matlab_string
import random
import argparse

# SENT_NUM = 0
# SUB_ID = 'YAC'

parser = argparse.ArgumentParser(description='Analyze EEG data and plot topographic maps.')
parser.add_argument('--subject_id', type=int, default=1, help='Subject ID')
parser.add_argument('--image_num', type=int, default=0, help='Sentence number')

args = parser.parse_args()

SUB_ID = args.subject_id
IMAGE_NUM = args.image_num

montage_path = r"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ImageNetEEG\actiCAP_snap_CACS_CAS_GACS-v2\actiCap_snap_CACS_CAS_GACS\actiCap_slim_for actiChamp_Plus\CACS-128\CACS-128_NO_REF.bvef"

def preprocess_eeg(eeg_data, orig_sfreq=1000):
    # Apply bandpass filter for 0.5 to 75 Hz, and notch filter at 50 Hz
    nyq = orig_sfreq / 2.0
    low_cut = 0.5 / nyq
    high_cut = 75.0 / nyq
    sos_band = signal.butter(4, [low_cut, high_cut], btype='band', output='sos')
    eeg_bandpassed = signal.sosfiltfilt(sos_band, eeg_data, axis=0)

    notch_freq = 50.0  # Hz
    Q = 30.0         # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq/nyq, Q)
    sos_notch = signal.tf2sos(b_notch, a_notch)
    eeg_filtered = signal.sosfiltfilt(sos_notch, eeg_bandpassed, axis=0)

    return eeg_filtered

# Following function is referred from https://raphaelvallat.com/bandpower.html
def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simpson
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp

def analyze_eeg_modified(eeg_data, sfreq=1000, electrode_names=None, save_dir=None):
    """
    Performs frequency analysis, electrode analysis, and other relevant analyses on EEG data,
    using specified frequency bands.

    Args:
        eeg_data (numpy.ndarray): EEG data of shape (time_steps, num_channels).
        sfreq (int, optional): Sampling frequency in Hz. Defaults to 500.
        electrode_names (list of str, optional): List of electrode names. Defaults to None.
        save_dir (str, optional): Directory to save plots. Defaults to None.
    """

    num_channels = eeg_data.shape[1]
    time_steps = eeg_data.shape[0]
    time = np.arange(time_steps) / sfreq

    if electrode_names is None:
        electrode_names = [f"Channel {i+1}" for i in range(num_channels)]

    # 1. Frequency Analysis (Power Spectral Density)
    print("\n--- Frequency Analysis ---")
    psd, frequencies = psd_array_multitaper(eeg_data.T, sfreq, fmin=0.5, fmax=75)
    print(frequencies.shape)
    print(psd.shape)

    plt.figure(figsize=(12, 6))
    for i in range(num_channels):
        plt.plot(frequencies, psd[i], label=electrode_names[i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (uV^2/Hz)")
    plt.title("Power Spectral Density")
    plt.xlim(0, 80)  # Extended frequency range
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "frequency_analysis.png"))
    plt.show()

    # 2. Electrode Analysis (Mean Power)
    # print("\n--- Electrode Analysis (Mean Power) ---")

    # plt.figure(figsize=(12, 6))
    # plt.title("PSD for All Channels")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("PSD (V^2/Hz)")
    # plt.grid(True)

    # for i in range(num_channels):
    #     psd_chan, freqs_chan = psd_array_multitaper(eeg_data[:, i].T, sfreq, fmin=0.5, fmax=75)
    #     plt.plot(freqs_chan, psd_chan.transpose(), label=electrode_names[i])

    # plt.tight_layout()
    # if save_dir:
    #     plt.savefig(os.path.join(save_dir, "electrode_mean_power.png"))
    # plt.show()

    # 3. Time-Domain Visualization (do for random 5 channels)
    # channel_indices = random.sample(list(range(num_channels)), 5)
    # channel_indices = [104] + channel_indices  # Add Cz channel (index 104) to the list
    # print("\n--- Time-Domain Visualization (Random Channels) ---")
    # for i in channel_indices:
    #     print("\n--- Time-Domain Visualization (Channel 1) ---")
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(time, eeg_data[:, i])  # Plot the first channel
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Amplitude (micro V)")
    #     plt.title(f"Time-Domain Signal ({electrode_names[i]})")
    #     if save_dir:
    #         plt.savefig(os.path.join(save_dir, f"time_domain_channel{i+1}.png"))
    #     plt.show()

    # 4. Correlation Analysis (Channel-wise)
    print("\n--- Correlation Analysis (Channel-wise) ---")
    correlation_matrix = np.corrcoef(eeg_data.T)  # Transpose to get channels as rows
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(np.arange(num_channels), electrode_names, rotation=45, ha="right")
    plt.yticks(np.arange(num_channels), electrode_names)
    plt.title("Channel Correlation Matrix")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "channel_correlation.png"))
    plt.show()

    # 6. Frequency Band Power Analysis (Modified Bands)
    print("\n--- Frequency Band Power Analysis ---")
    bands = {"Delta": (0.5, 4), "Alpha": (8, 12), "Beta": (13, 30), "Gamma": (30, 75)}
    band_powers = {}

    for band_name, (low, high) in bands.items():
        print(f"\n--- {band_name} Band ---")
        relative_power = []
        for i in range(num_channels):
            power = bandpower(eeg_data[:, i].T, sfreq, [low, high], method='multitaper', relative=True)
            print(f"{electrode_names[i]}: {power:.4f}")
            relative_power.append(power)
        band_powers[band_name] = relative_power
    
    # save the band powers
    np.save(os.path.join(save_dir, "band_powers.npy"), band_powers, allow_pickle=True)

    for band_name, (low, high) in bands.items():
        plt.figure(figsize=(12, 6))
        plt.title(f"PSD for All Channels ({band_name} Band)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (uV^2/Hz)")
        plt.grid(True)

        print(f"\n--- {band_name} Band ---")
        for i in range(num_channels):
            psd_chan, freqs_chan = psd_array_multitaper(eeg_data[:, i].T, sfreq, fmin=low, fmax=high)
            plt.plot(freqs_chan, psd_chan.transpose(), label=electrode_names[i])
    
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"electrode_psd_{band_name}.png"))
        plt.show()

    # 6. Electrode Analysis (Mean power)
    print("\n--- Electrode Analysis (Mean Power) ---")
    
    # calculate power for each band
    # here mean_power is basically psd
    mean_power = {}
    for band_name, (low, high) in bands.items():
        mean_power_band = []
        for i in range(num_channels):
            power = bandpower(eeg_data[:, i].T, sfreq, [low, high], method='multitaper', relative=False)
            mean_power_band.append(power)
        mean_power[band_name] = mean_power_band

    # plot mean power for each channel, for each band
    for band_name, (low, high) in bands.items():
        plt.figure(figsize=(12, 6))
        plt.bar(electrode_names, mean_power[band_name])
        plt.xlabel("Electrodes")
        plt.ylabel("PSD (uV^2/Hz)")
        plt.title(f"PSD in {band_name} Band by Electrodes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"electrode_mean_power_{band_name}.png"))
        plt.show()

def get_min_max_v(raw):
    """
    Calculates the minimum and maximum microvolt values across all channels of an MNE Raw object.

    Args:
        raw (mne.io.Raw): MNE Raw object containing EEG data.

    Returns:
        tuple: A tuple containing the minimum and maximum microvolt values.
    """
    # Convert raw data to numpy array (in volts)
    data = raw.get_data()

    # Calculate min and max
    min_v = np.min(data)
    max_v = np.max(data)

    return max(abs(min_v), abs(max_v))

def visualize_eeg_time_domain(eeg_data, sampling_rate, channel_names=None, title="EEG Time Series"):
    """
    Visualizes EEG data in the time domain.

    Args:
        eeg_data (numpy.ndarray): EEG data, where rows are channels and columns are time points.
        sampling_rate (float): Sampling rate of the EEG data in Hz.
        channel_names (list, optional): List of channel names. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "EEG Time Series".
    """

    num_samples, num_channels = eeg_data.shape
    time = np.arange(num_samples) / sampling_rate

    plt.figure(figsize=(12, 6))

    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(num_channels)]

    for i in range(num_channels - 1, num_channels - 16, -1):
        plt.plot(time, eeg_data[:, i] + i * np.max(eeg_data), label=channel_names[i]) #Offset the channels to appear on the same graph

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (µV) + Offset")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eeg_time_series.png"))
    plt.show()

def plot_topomap_mne_modified_array_h5(eeg_data, sfreq=1000, save_dir=None):
    """
    Plots topographic maps using MNE for channel locations from an HDF5 file.

    Args:
        eeg_data (numpy.ndarray): EEG data of shape (time_steps, num_channels).
        sfreq (int, optional): Sampling frequency in Hz. Defaults to 500.
        h5_file_path (str): Path to the HDF5 file containing channel locations.
        save_dir (str, optional): Directory to save plots. Defaults to None.
    """

    num_channels = eeg_data.shape[1]
    time_steps = eeg_data.shape[0]
    time = np.arange(time_steps) / sfreq

    # find min and max in eeg_data
    min_v = np.min(eeg_data)
    max_v = np.max(eeg_data)
    print(f"Min: {min_v}, Max: {max_v}")

    # EEG data is in microvolts, convert it to volts
    eeg_data = eeg_data / 1e6  # Convert microvolts to volts

    min_v = np.min(eeg_data)
    max_v = np.max(eeg_data)
    print(f"Min: {min_v}, Max: {max_v}")

    # Read channel locations from HDF5 file
    # with h5py.File(h5_file_path, 'r') as f:
    # ch_pos = OrderedDict(chanlocs)

    # Create DigMontage from chanlocs
    # montage = mne.channels.make_dig_montage(ch_pos=chanlocs, coord_frame='head')
    montage = mne.channels.read_custom_montage(montage_path, coord_frame='head')

    # Extract channel names from the montage
    electrode_names = montage.ch_names[1:]

    # Create MNE Info object
    info = mne.create_info(electrode_names, sfreq, ch_types='eeg')

    # Create Raw object from numpy array
    raw = mne.io.RawArray(eeg_data.T, info)

    # Set the montage and reference
    raw.set_montage(montage)
    raw.set_eeg_reference(ref_channels=['FCz'])

    # Calculate power spectral density (PSD) using psd_array_multitaper
    # fmin, fmax = 0.5, 75

    # apply bandpass filter to raw data
    raw.filter(l_freq=0.5, h_freq=75)
    raw.notch_filter(freqs=50)

    # psds, freqs = psd_array_multitaper(eeg_data.T, sfreq, fmin=fmin, fmax=fmax, output='power')

    # Average PSD across time
    # psds = np.mean(psds, axis=1)

    # Plot topographic map for the average PSD
    bands = {"Delta": (0.5, 4), "Alpha": (8, 12), "Beta": (13, 30), "Gamma": (30, 75)}
    # fig = raw.plot_topomap(times=0, average=True, ch_type='eeg', sensors=True, show=False)
    for band, (fmin, fmax) in bands.items():
        fig = raw.compute_psd(fmin=fmin, fmax=fmax).plot_topomap(bands={band:(fmin, fmax)}, ch_type='eeg', show=True, outlines='head', show_names=True)
        fig.suptitle('PSD Topomap for ' + band)
        fig.savefig(os.path.join(save_dir, f"average_psd_topomap_{band}.png"))

    duration = time_steps // sfreq
    max_v = get_min_max_v(raw)
    fig = raw.plot(n_channels=16, show=False, duration=duration, scalings={'eeg': max_v})
    fig.suptitle('EEG Data')
    plt.show()
    fig.savefig(os.path.join(save_dir, "eeg_data.png"))

    visualize_eeg_time_domain(eeg_data, sfreq, channel_names=electrode_names, title="EEG Time Series")
    
    # fig = raw.plot_sensors(show_names=True, show=False, ch_type='eeg')
    # fig.suptitle('Sensor Locations')
    # fig.savefig(os.path.join(save_dir, "sensor_locations.png"))    

# --- Example Usage with Sample EEG Data ---

complete_data = np.load(r"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ImageNetEEG\processed_eeg_signals.npy", allow_pickle=True)
complete_data = complete_data.item()
eeg_data = [x for x in complete_data['rawData'] if (x['subject'] == SUB_ID and x['image'] == IMAGE_NUM)][0]['eeg']

# transpose the data to have shape (time_steps, num_channels)
eeg_data = eeg_data.T
eeg_data = np.array(eeg_data)

print(eeg_data.shape)  # Shape should be (4872, 105)

sfreq = 1000

montage = mne.channels.read_custom_montage(montage_path, coord_frame='head')

chanlocs = montage.ch_names[1:]

save_dir = rf"D:\Vijay\NYU\Spring_25\BDMLS\Project\code\figs\imageEEG\{SUB_ID}\image{IMAGE_NUM}"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Run the analysis functions
# analyze_eeg_modified(eeg_data, sfreq=sfreq, save_dir=save_dir, electrode_names=chanlocs)
plot_topomap_mne_modified_array_h5(eeg_data, sfreq=sfreq, save_dir=save_dir)