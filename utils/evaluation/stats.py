import os
from pydub import AudioSegment, silence

def compute_total_duration(folder_path):
    total_duration = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            audio = AudioSegment.from_wav(file_path)
            total_duration += len(audio)
    return total_duration / 1000

