'''
Pre process audio and EGG data:
Audio files move, rename, segment, check integrity
Normalization
[Feature extraction]
'''

import os
from pydub import AudioSegment, silence
from scipy.signal import butter, lfilter, firwin, medfilt

def move_audio_files(audio_dir, output_dir):
    '''
    Move the needed audio files to a new directory.
    
    Parameters:
    audio_dir (str): Directory containing the audio files.
    output_dir (str): Directory to save the audio files.

    Structure:
    a dir looks like this:
    F:\A2E\children-trimmed
    where it contains the audio files for each person from B01 to B20, in the 
    subfolders only the wav files are needed. Move those files to output_dir.
    '''
    for subdir, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                os.rename(file_path, os.path.join(output_dir, file))

def rename(dir):
    '''
    Remove the first two number sequences from the file names
    For the same person, M01 for example, merge the SRP1,2,... files into one file SRP.wav, and VRP1,2,... files into VRP.wav.
    '''
    # get the wav files
    wav_files = [f for f in os.listdir(dir) if f.endswith('.wav')]

    # remove the string '_C_' from the file names
    for file in wav_files:
        os.rename(os.path.join(dir, file), os.path.join(dir, file.replace('_C_', '_')))

def segment_audio(file_path, min_silence_len=30, silence_thresh=-60, target_length=15):
    # Load audio file
    audio = AudioSegment.from_wav(file_path)
    audio_channel,egg_channel = audio.split_to_mono()
    # Detect non-silent chunks
    chunks = silence.split_on_silence(audio_channel, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # print total chunks time in seconds
    total_time = sum([len(chunk) for chunk in chunks]) / 1000
    # print(f"Total chunks time: {total_time} seconds")


    # Create segments around 10 seconds
    target_duration = target_length * 1000  # convert to milliseconds
    segment_start = 0
    segment_end = 0

    for chunk in chunks:
        if chunk.duration_seconds > target_duration:
            segment_end = segment_start + target_duration
            yield audio_channel[segment_start:segment_end], egg_channel[segment_start:segment_end]
            segment_start = segment_end
        else:
            current_length = segment_end - segment_start + len(chunk)
            if current_length < target_duration:
                segment_end += len(chunk)
            else:
                yield audio_channel[segment_start:segment_end], egg_channel[segment_start:segment_end]
                segment_start = segment_end


def process_folder(folder_path, output_dir_audio, output_dir_egg):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            base_name = file_name.split('.')[0][:8]

            for i, (audio_seg, egg_seg) in enumerate(segment_audio(file_path)):
                audio_file_name = f"{base_name}_Voice_{i}.wav"  # Format the new file name
                egg_file_name = f"{base_name}_EGG_{i}.wav"

                if audio_seg.duration_seconds < 1:
                    print(f"Segment {audio_file_name} is too short")

                audio_output_path = os.path.join(output_dir_audio, audio_file_name)
                egg_output_path = os.path.join(output_dir_egg, egg_file_name)

                audio_seg.export(audio_output_path, format='wav')
                egg_seg.export(egg_output_path, format='wav')

def compute_total_duration(folder_path):
    total_duration = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            audio = AudioSegment.from_wav(file_path)
            total_duration += len(audio)
    return total_duration / 1000

def check_integrity(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            audio = AudioSegment.from_wav(file_path)
            if audio.channels != 1:
                print(f"File {file_name} has {audio.channels} channels")
            if audio.frame_rate != 16000:
                print(f"File {file_name} has {audio.frame_rate} frame rate")
            if audio.sample_width != 2:
                print(f"File {file_name} has {audio.sample_width} sample width")


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
    # high pass filter
    filtered_signal = apply_high_pass_filter(egg_signal)
    
    # low pass filter
    filtered_signal = apply_low_pass_filter(filtered_signal)
    
    # a nine-point running median filter

    return filtered_signal

def apply_high_pass_filter(signal, sample_rate=44100, numtaps=1025, cutoff=80):
    # Design the FIR filter
    fir_coeff = firwin(numtaps, cutoff, pass_zero=False, fs=sample_rate, window='hamming')

    # Apply the filter to the signal using lfilter, which applies the filter in a linear-phase manner
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    
    return filtered_signal

def apply_low_pass_filter(signal, sample_rate=44100, cutoff_hz=10000, numtaps=1025):
    # Design the low-pass FIR filter with a cutoff of 10 kHz
    fir_coeff = firwin(numtaps, cutoff_hz, fs=sample_rate, window='hamming', pass_zero=True)

    # Apply the filter to the signal
    filtered_signal = lfilter(fir_coeff, 1.0, signal)

    return filtered_signal

def delete_0s_audio(audio_dir, sr):
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(audio_dir, file_name)
            audio = AudioSegment.from_wav(file_path)
            if len(audio) < sr:
                os.remove(file_path)
                print(f"File {file_name} deleted")

def main():
    folder_path = r'F:\audio\SRP'
    output_dir_audio = r'F:\audio\SRP_segmented\Voice'
    output_dir_egg = r'F:\audio\SRP_segmented\EGG'
    if not os.path.exists(output_dir_audio):
        os.makedirs(output_dir_audio)
    if not os.path.exists(output_dir_egg):
        os.makedirs(output_dir_egg)
    process_folder(folder_path, output_dir_audio, output_dir_egg)

if __name__ == '__main__':
    main()