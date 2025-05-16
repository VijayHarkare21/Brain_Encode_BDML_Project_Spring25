import numpy as np
from scipy.signal import butter, sosfiltfilt, tf2sos, iirnotch, resample
import pickle as pkl

def process_eeg(eeg, orig_sfreq):
    """
    Process an EEG signal by resampling, filtering, and segmenting.
    
    Parameters:
    -----------
    eeg : np.ndarray
        Input EEG array with shape (n_samples, n_channels). For example (4872, 105).
    orig_sfreq : float
        Original sampling frequency of the EEG signal.
    
    Returns:
    --------
    processed_eeg : np.ndarray
        Processed EEG with shape (1, n_channels, n_segments, 200), where each segment is 1 second (200 samples).
    """
    target_sfreq = 200  # target sampling frequency in Hz

    # 1. Resample to target sampling frequency along the time axis (axis=0)
    num_samples_target = int(eeg.shape[0] * target_sfreq / orig_sfreq)
    
    # 2. Apply bandpass filter (0.3 Hz to 75 Hz)
    nyq = orig_sfreq / 2.0
    low_cut = 0.3 / nyq
    high_cut = 75.0 / nyq
    sos_band = butter(4, [low_cut, high_cut], btype='band', output='sos')
    eeg_bandpassed = sosfiltfilt(sos_band, eeg, axis=0)

    # 3. Apply notch filter at 60 Hz
    notch_freq = 60.0  # Hz
    Q = 30.0         # Quality factor
    b_notch, a_notch = iirnotch(notch_freq/nyq, Q)
    sos_notch = tf2sos(b_notch, a_notch)
    eeg_filtered = sosfiltfilt(sos_notch, eeg_bandpassed, axis=0)

    eeg_resampled = resample(eeg_filtered, num_samples_target, axis=0)
    eeg_resampled /= 100.0
    
    # 4. Create 1-second segments (200 samples per segment)
    segment_length = target_sfreq  # 200 samples
    n_timepoints = eeg_resampled.shape[0]
    remainder = n_timepoints % segment_length
    if remainder != 0:
        pad_width = segment_length - remainder
        pad_array = np.zeros((pad_width, eeg_resampled.shape[1]))
        eeg_resampled = np.concatenate([eeg_resampled, pad_array], axis=0)
    
    # Calculate number of segments and reshape:
    n_segments = eeg_resampled.shape[0] // segment_length
    # Reshape to (n_segments, segment_length, n_channels)
    eeg_segments = eeg_resampled.reshape(n_segments, segment_length, eeg_resampled.shape[1])
    # Rearrange to shape (1, n_channels, n_segments, segment_length)
    eeg_segments = np.transpose(eeg_segments, (2, 0, 1))  # now (n_channels, n_segments, segment_length)
    processed_eeg = np.expand_dims(eeg_segments, axis=0)
    
    return processed_eeg

# Test
# path_to_eeg = r"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_YAC.pickle"
# with open(path_to_eeg, 'rb') as f:
#     eeg_data = pkl.load(f)

# sfreq_original = 500

# print(eeg_data['YAC'][0]['rawData'].shape)

# preprocessed_eeg = process_eeg(eeg_data['YAC'][0]['rawData'], sfreq_original)
# print(preprocessed_eeg.shape)