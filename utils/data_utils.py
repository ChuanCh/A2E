import os
import librosa
from scipy.signal import butter, lfilter, firwin, filtfilt
from tqdm import tqdm
import numpy as np
import soundfile as sf


def load_data(path, samplerate=16000):
    audios = []
    eggs = []

    # Check if the path is a directory
    if os.path.isdir(path):
        # Walk through the folder
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    try:
                        # Load file with librosa
                        wave, sr = librosa.load(audio_path, sr=samplerate, mono=False)
                        if wave.shape[0] == 2:  # Ensure it is stereo
                            audio, egg = wave[0], wave[1]
                            audios.append(audio)
                            eggs.append(egg)
                        else:
                            print(f"Skipped {audio_path}: Audio is not stereo.")
                    except Exception as e:
                        print(f"Error loading {audio_path}: {str(e)}")
    elif os.path.isfile(path) and path.endswith('.wav'):
        # It's a single file, load directly
        try:
            wave, sr = librosa.load(path, sr=samplerate, mono=False)
            if wave.shape[0] == 2:  # Ensure it is stereo
                audio, egg = wave[0], wave[1]
                audios.append(audio)
                eggs.append(egg)
            else:
                print(f"Skipped {path}: Audio is not stereo.")
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    else:
        print(f"Error: The path provided is neither a directory nor a .wav file.")

    # Concatenate all collected data into single arrays
    audio_concat = np.concatenate(audios) if audios else np.array([])
    egg_concat = np.concatenate(eggs) if eggs else np.array([])
    return audio_concat, egg_concat

# +
import os
import numpy as np
import soundfile as sf
from scipy.signal import resample

def load_audio_file_sf(file_path):
    try:
        wave, original_sr = sf.read(file_path, always_2d=True)
        if wave.shape[1] == 2:  # Ensure it is stereo
            if original_sr != 16000:
                wave = resample(wave, int(wave.shape[0] * 16000 / original_sr))
            audio, egg = wave[:, 0], wave[:, 1]
            return audio, egg
        else:
            print(f"Skipped {file_path}: Audio is not stereo.")
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
    return None, None

def load_data_sf(path):
    audios = []
    eggs = []

    files_to_load = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    files_to_load.append(os.path.join(root, file))
    elif os.path.isfile(path) and path.endswith('.wav'):
        files_to_load.append(path)
    else:
        print(f"Error: The path provided is neither a directory nor a .wav file.")
        return np.array([]), np.array([])

    for file_path in files_to_load:
        audio, egg = load_audio_file_sf(file_path)
        if audio is not None and egg is not None:
            audios.append(audio)
            eggs.append(egg)

    audio_concat = np.concatenate(audios) if audios else np.array([])
    egg_concat = np.concatenate(eggs) if eggs else np.array([])

    return audio_concat, egg_concat

def load_audio_file_sf_mono(file_path):
    try:
        wave, original_sr = sf.read(file_path, always_2d=True)
        if wave.shape[1] == 1:  # Ensure it is mono
            if original_sr != 16000:
                wave = resample(wave.flatten(), int(wave.shape[0] * 16000 / original_sr))
            audio = wave.flatten()
            return audio
        else:
            print(f"Skipped {file_path}: Audio is not mono.")
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
    return None

def segment_audio(audio, sr, frame_length_ms=12, hop_length_samples=160, batch_size=1000):
    frame_length_samples = int(sr * frame_length_ms / 1000)
    total_samples = len(audio)

    # Calculate the total number of frames that will be created
    num_frames = (total_samples - frame_length_samples) // hop_length_samples + 1

    # Use a list to collect frames
    frames = []
    
    # Process each batch with a tqdm progress bar
    for i in tqdm(range(0, num_frames, batch_size), desc="Processing frames"):
        # Calculate the batch range
        batch_end = i + batch_size
        if batch_end > num_frames:
            batch_end = num_frames
        
        # Process each frame in the batch
        for j in range(i, batch_end):
            start_idx = j * hop_length_samples
            end_idx = start_idx + frame_length_samples
            frames.append(audio[start_idx:end_idx])

    # Convert list of frames to a numpy array
    return np.array(frames)

def load_and_preprocess_data(audio_path, egg_path, samplerate=16000, frame_length_ms=12, hop_length_samples=1):
    def butter_highpass(cutoff, sample_rate, order=2):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, sample_rate, order=2):
        b, a = butter_highpass(cutoff, sample_rate, order=order)
        y = lfilter(b, a, data)
        return y

    def voice_preprocess(voice, sample_rate):
        voice = highpass_filter(voice, 30, sample_rate)
        return voice

    def process_EGG_signal(egg_signal, sample_rate, threshold_dB=-40, expansion_ratio=1/4):
        # High pass filter
        filtered_signal = apply_high_pass_filter(egg_signal, sample_rate)
        
        # Low pass filter
        filtered_signal = apply_low_pass_filter(filtered_signal, sample_rate)
        
        return filtered_signal

    def apply_high_pass_filter(signal, sample_rate=44100, numtaps=1025, cutoff=80):
        # Design the FIR filter
        fir_coeff = firwin(numtaps, cutoff, pass_zero=False, fs=sample_rate, window='hamming')

        # Apply the filter to the signal using filtfilt to avoid phase shift
        filtered_signal = filtfilt(fir_coeff, 1.0, signal)
        
        return filtered_signal

    def apply_low_pass_filter(signal, sample_rate=44100, cutoff_hz=4000, numtaps=1025):
        # Design the low-pass FIR filter with a cutoff of 10 kHz
        fir_coeff = firwin(numtaps, cutoff_hz, fs=sample_rate, window='hamming', pass_zero=True)

        # Apply the filter to the signal using filtfilt to avoid phase shift
        filtered_signal = filtfilt(fir_coeff, 1.0, signal)

        return filtered_signal
    
    def process_file(audio, egg, sr):
        # Preprocess audio and EGG data
        audio = voice_preprocess(audio, sr)
        egg = process_EGG_signal(egg, sr)

        # Segment audio and EGG data
        # audio_frames = segment_audio_temp(audio, sr, frame_length_ms=frame_length_ms, hop_length_samples=hop_length_samples)
        # egg_frames = segment_audio_temp(egg, sr, frame_length_ms=frame_length_ms, hop_length_samples=hop_length_samples)
        audio_frames = segment_audio(audio, sr, frame_length_ms=frame_length_ms, hop_length_samples=hop_length_samples, batch_size=1000)
        egg_frames = segment_audio(egg, sr, frame_length_ms=frame_length_ms, hop_length_samples=hop_length_samples, batch_size=1000)
        
        return audio_frames, egg_frames

    def normalize(frames):
        # Normalize frames
        return frames / np.max(np.abs(frames), axis=0, keepdims=True)

    all_audio_frames, all_egg_frames = [], []
    
    if os.path.isfile(audio_path) and audio_path.endswith('.wav'):
        # It's a single file, load directly
        wave, sr = librosa.load(audio_path, sr=samplerate, mono=False)
        if wave.shape[0] == 2:  # Ensure it is stereo
            audio, egg = wave[0], wave[1]
            audio_frames, egg_frames = process_file(audio, egg, sr)
            all_audio_frames.append(audio_frames)
            all_egg_frames.append(egg_frames)
        else:
            print(f"Skipped {audio_path}: Audio is not stereo.")
    
    elif os.path.isdir(audio_path):
        # Walk through the folder
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    wave, sr = librosa.load(audio_path, sr=samplerate, mono=False)
                    if wave.shape[0] == 2:  # Ensure it is stereo
                        audio, egg = wave[0], wave[1]
                        audio_frames, egg_frames = process_file(audio, egg, sr)
                        all_audio_frames.append(audio_frames)
                        all_egg_frames.append(egg_frames)
                    else:
                        print(f"Skipped {audio_path}: Audio is not stereo.")

    # Concatenate all frames and normalize
    all_audio_frames = normalize(np.concatenate(all_audio_frames, axis=0))
    all_egg_frames = normalize(np.concatenate(all_egg_frames, axis=0))

    return all_audio_frames, all_egg_frames



def voice_preprocess(voice, sample_rate):
    def butter_highpass(cutoff, sample_rate, order=2):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, sample_rate, order=2):
        b, a = butter_highpass(cutoff, sample_rate, order=order)
        y = lfilter(b, a, data)
        return y
    voice = highpass_filter(voice, 30, sample_rate)
    return voice

def process_EGG_signal(egg_signal, sample_rate, threshold_dB=-40, expansion_ratio=1/4):
    def apply_high_pass_filter(signal, sample_rate=44100, numtaps=1025, cutoff=80):
        # Design the FIR filter
        fir_coeff = firwin(numtaps, cutoff, pass_zero=False, fs=sample_rate, window='hamming')

        # Apply the filter to the signal using filtfilt to avoid phase shift
        filtered_signal = filtfilt(fir_coeff, 1.0, signal)
        
        return filtered_signal

    def apply_low_pass_filter(signal, sample_rate=44100, cutoff_hz=4000, numtaps=1025):
        # Design the low-pass FIR filter with a cutoff of 10 kHz
        fir_coeff = firwin(numtaps, cutoff_hz, fs=sample_rate, window='hamming', pass_zero=True)

        # Apply the filter to the signal using filtfilt to avoid phase shift
        filtered_signal = filtfilt(fir_coeff, 1.0, signal)

        return filtered_signal
    
    # High pass filter
    filtered_signal = apply_high_pass_filter(egg_signal, sample_rate)
    
    # Low pass filter
    filtered_signal = apply_low_pass_filter(filtered_signal, sample_rate)
    
    return filtered_signal

def create_experiment_folders(base_dir, experiment_name):
    """Create experiment directories for checkpoints, models, and logs."""
    chkpt_dir = os.path.join(base_dir, experiment_name, 'chkpt')
    models_dir = os.path.join(base_dir, experiment_name, 'models')
    logs_dir = os.path.join(base_dir, experiment_name, 'logs')
    
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return chkpt_dir, models_dir, logs_dir
