# Description: This file contains utility functions for EGG signal processing.
# noise deduction
# band filter
# normalisation


import numpy as np
from scipy.signal import butter, lfilter
import librosa
import os

def EGG_noise_deduction(EGG_signal, noise_threshold):
    """
    This function removes noise from the EGG signal.
    
    Parameters:
    EGG_signal: EGG signal to be processed.
    noise_threshold: Threshold value for noise detection.
    
    Returns:
    EGG_signal: EGG signal after noise removal.
    """
    
    # Calculate the mean of the EGG signal.
    EGG_mean = np.mean(EGG_signal)
    
    # Calculate the standard deviation of the EGG signal.
    EGG_std = np.std(EGG_signal)
    
    # Calculate the noise threshold.
    noise_threshold = EGG_mean + noise_threshold * EGG_std
    
    # Remove noise from the EGG signal.
    EGG_signal[EGG_signal < noise_threshold] = 0
    
    return EGG_signal

def band_filter(signal, lowcut, highcut, fs, order=5):
    """
    This function applies band-pass filtering to the EGG signal.
    
    Parameters:
    EGG_signal: EGG signal to be processed.
    lowcut: Lower cut-off frequency.
    highcut: Higher cut-off frequency.
    fs: Sampling frequency.
    order: Order of the filter.

    Returns:
    EGG_signal: EGG signal after band-pass filtering.
    """
        
    # Calculate the Nyquist frequency.
    nyquist = 0.5 * fs
    
    # Calculate the low and high cut-off frequencies.
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Apply band-pass filtering to the EGG signal.
    b, a = butter(order, [low, high], btype='band')
    signal = lfilter(b, a, signal)
    
    return signal

def normalise(signal):
    """
    This function min-max normalises the signal.
    
    Parameters:
    signal: Signal to be normalised.
    
    Returns:
    signal: Normalised signal.
    """
    
    # Calculate the minimum and maximum values of the signal.
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Normalise the signal.
    signal = (signal - min_val) / (max_val - min_val)
    
    return signal

# Function to log configuration details to TensorBoard
def tensorboard_log_config(writer, model, input_type, sample_rate, sample_length, batch_size):
    config_text = f"Model type: {type(model).__name__}\nInput type: {input_type}\nSample rate: {sample_rate} Hz \nSample length: {sample_length} samples\nBatch size: {batch_size}"
    notes = input("Enter your notes for this experiment setup: ")
    writer.add_text('Training Configuration', config_text, 0)
    writer.add_text('Notes', notes, 0)

def reshape_to_fit_LSTM(input, num_frames):
    """
    Reshape an input of shape (n_feature, total_samples) to (num_samples, num_frames, n_feature).

    Args:
    input (numpy.ndarray): The input to reshape.
    num_frames (int): The number of frames to include in each sample.

    Returns:
    numpy.ndarray: The reshaped input.
    """
    n_feature, total_samples = input.shape
    # Calculate the number of samples
    num_samples = total_samples // num_frames
    # Calculate the new shape
    new_shape = (num_samples, num_frames, n_feature)
    # Reshape the input
    reshaped_input = input[:, :num_samples * num_frames].reshape(new_shape)
    return reshaped_input

def split_data(audio, egg, train_frac=0.8, val_frac=0.1):
    total_length = len(audio)
    train_end = int(total_length * train_frac)
    val_end = int(total_length * (train_frac + val_frac))
    
    audio_train = audio[:train_end]
    egg_train = egg[:train_end]
    
    audio_val = audio[train_end:val_end]
    egg_val = egg[train_end:val_end]
    
    audio_test = audio[val_end:]
    egg_test = egg[val_end:]
    
    return (audio_train, egg_train), (audio_val, egg_val), (audio_test, egg_test)

def preprocess_data(
        audio_wav, 
        egg_wav, 
        n_fft, 
        hop_length, 
        n_harmonics, 
        window, 
        mag_min=None, 
        mag_max=None):
    
    # Compute the STFT
    stft_result_audio = librosa.stft(audio_wav, n_fft=n_fft, hop_length=hop_length, window=window)
    stft_result_egg = librosa.stft(egg_wav, n_fft=n_fft, hop_length=hop_length, window=window)

    # only keep the first n_harmonics
    stft_result_audio = stft_result_audio[:n_harmonics, :]
    stft_result_egg = stft_result_egg[:n_harmonics, :]

    # Extract magnitude and phase
    magnitude_audio = np.real(stft_result_audio)
    phase_audio = np.imag(stft_result_audio)
    magnitude_egg = np.real(stft_result_egg)
    phase_egg = np.imag(stft_result_egg)
 

    # Calculate min and max for normalization if not provided
    if mag_min is None:
        mag_min = min(np.min(magnitude_audio), np.min(magnitude_egg))
    if mag_max is None:
        mag_max = max(np.max(magnitude_audio), np.max(magnitude_egg))

    # Normalize the magnitude into range [-1,1]
    

    # calculate sin and cos of phase
    phase_audio_sin = np.sin(phase_audio)
    phase_audio_cos = np.cos(phase_audio)
    phase_egg_sin = np.sin(phase_egg)
    phase_egg_cos = np.cos(phase_egg)

    # Unwrap the phase
    # phase_audio = np.unwrap(phase_audio)
    # phase_egg = np.unwrap(phase_egg)

    # Add magnitude and phase into one feature vector
    complex_audio = np.concatenate((magnitude_audio, phase_audio_sin, phase_audio_cos), axis=0)
    complex_egg = np.concatenate((magnitude_egg, phase_egg_sin, phase_egg_cos), axis=0)

    return complex_audio, complex_egg, mag_min, mag_max

def denormalize(value, min_val, max_val):
    """Reverses the normalization"""
    return (value * (max_val - min_val)) + min_val

def reconstruct_signal(magnitude, phase, hop_length, window):
    """Reconstructs an audio signal from magnitude and phase"""
    # Convert magnitude and phase into a complex STFT matrix
    stft_matrix = magnitude * np.exp(1j * phase)
    # Reconstruct the audio signal
    return librosa.istft(stft_matrix, hop_length=hop_length, window=window)

def get_audio_length(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return librosa.get_duration(y=y, sr=sr)

def total_audio_length(directory):
    total_length = 0.0 # in hours
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                file_path = os.path.join(root, file)
                total_length += get_audio_length(file_path)
    return total_length

def main():
    directory = r'F:\TTS_audio\EGG'
    total_length = total_audio_length(directory)
    print(f"Total audio length: {total_length} seconds")

if __name__ == '__main__':
    main()
