# Description: This file contains utility functions for EGG signal processing.
# noise deduction
# band filter
# normalisation


import numpy as np
from scipy.signal import butter, lfilter

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