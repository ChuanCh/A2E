import numpy as np
from scipy.signal import hilbert, find_peaks


def phase_tracker(signal, fs):
    def detect_zero_crossings(signal):
        # Detects zero-crossings in the signal
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return zero_crossings

    def extract_instantaneous_features(signal):
        """Extracts amplitude envelope and normalized phase using the Hilbert transform."""
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        normalized_phase = instantaneous_phase / np.max(np.abs(instantaneous_phase))
        return amplitude_envelope, normalized_phase


    def validate_cycles(phase, zero_crossings, fs):
        valid_boundaries = []
        for i in range(1, len(zero_crossings)):  # Start from 1 to safely use zero_crossings[i - 1]
            start_idx = zero_crossings[i - 1]
            end_idx = zero_crossings[i]
            
            # Extract the phase segment for the current cycle
            cycle_phase = phase[start_idx:end_idx]
            
            # Find positive and negative peaks within the cycle
            positive_peaks, _ = find_peaks(cycle_phase, height=0.5, prominence=0.2)
            negative_peaks, _ = find_peaks(-cycle_phase, height=0.5, prominence=0.2)

            
            # Validate the cycle by checking for exactly one positive and one negative peak
            if positive_peaks.size == 1 and negative_peaks.size == 1 and any(cycle_phase > 0.9) and any(cycle_phase < -0.9):
                valid_boundaries.append(end_idx)  # Append the start index of the cycle
                i += 1  # Skip the next cycle since it is already validated
        
        # Cycles longer than 20 ms (882 samples) or shorter than 0.23 ms (10 samples) are rejected at this stage.
        periods = []
        for i in range(len(valid_boundaries)-1):
            start = valid_boundaries[i]
            end = valid_boundaries[i+1]
            if end - start < 0.02 * fs and end - start > 0.00023 * fs:
                periods.append((start, end))

        return  periods
    
    # Extract instantaneous features
    amplitude_envelope, phase = extract_instantaneous_features(signal)
    zero_crossings = detect_zero_crossings(signal)
    cycle_boundaries = validate_cycles(phase, zero_crossings, fs)

    return cycle_boundaries