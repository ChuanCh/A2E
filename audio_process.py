'''
Pre process audio and EGG data
Normalization
Segmentation
[Feature extraction]
'''

import os
import librosa
import numpy as np
from pydub import AudioSegment, silence
import matplotlib.pyplot as plt

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

import os
from pydub import AudioSegment, silence

def segment_audio(file_path, min_silence_len=200, silence_thresh=-60, target_length=10):
    # Load audio file
    audio = AudioSegment.from_wav(file_path)
    audio_channel,egg_channel = audio.split_to_mono()
    # Detect non-silent chunks
    chunks = silence.split_on_silence(audio_channel, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # print total chunks time in seconds
    total_time = sum([len(chunk) for chunk in chunks]) / 1000
    print(f"Total chunks time: {total_time} seconds")


    # Create segments around 10 seconds
    target_duration = target_length * 1000  # convert to milliseconds
    segment_start = 0
    segment_end = 0

    for chunk in chunks:
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

                audio_output_path = os.path.join(output_dir_audio, audio_file_name)
                egg_output_path = os.path.join(output_dir_egg, egg_file_name)

                audio_seg.export(audio_output_path, format='wav')
                egg_seg.export(egg_output_path, format='wav')

audio_dir = r'F:\A2E git\A2E\audio\VRP'
voice_dir = r'F:\A2E git\A2E\audio\VRP_segmented\Voice'
egg_dir = r'F:\A2E git\A2E\audio\VRP_segmented\EGG'
process_folder(audio_dir, voice_dir, egg_dir)