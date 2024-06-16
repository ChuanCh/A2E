import librosa
import os


def get_audio_length(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return librosa.get_duration(y=y, sr=sr)

def check_valid_wave_files_and_delete(directory):
    # Check if all the wave files in the directory are longer than 0.1 second, if not delete them
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio_length = get_audio_length(file_path)
                if audio_length < 0.1:
                    os.remove(file_path)
                    print(f"File {file} is too short and has been removed")