import torch
from torch import nn
import os
import librosa
import numpy as np
from torch.utils.data import Dataset

# test file path
audio_path = r'F:\Audio\Audio'
egg_path = r'F:\Audio\EGG'
audio_list = os.listdir(audio_path)
egg_list = os.listdir(egg_path)

samplerate = 16000

class AudioEGGDataset(Dataset):
    def __init__(self, audio_path, egg_path, transform=None):
        self.audio_path = audio_path
        self.egg_path = egg_path
        self.audio_list = os.listdir(audio_path)
        self.egg_list = os.listdir(egg_path)
        self.transform = transform

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        try:
            audio_file = os.path.join(self.audio_path, self.audio_list[idx])
            egg_file = os.path.join(self.egg_path, self.egg_list[idx])

            # Load audio and EGG data
            audio, sr = librosa.load(audio_file, sr=samplerate)  # None for native sampling rate, or replace with specific rate
            egg, _ = librosa.load(egg_file, sr=samplerate)      # Assume same sample rate as audio

            # Find the maximum length in the dataset or a predetermined max length
            max_length = 160000  # This could also be dynamically calculated or set based on your data
            # Pad or truncate to the maximum length
            audio = librosa.util.fix_length(audio, size=max_length)
            egg = librosa.util.fix_length(egg, size=max_length)

            if self.transform:
                audio = self.transform(audio)
                egg = self.transform(egg)

            # Convert to PyTorch tensors and add channel dimension
            audio = torch.from_numpy(audio).float().unsqueeze(0)  # Add channel dimension
            egg = torch.from_numpy(egg).float().unsqueeze(0)

        except Exception as e:
            print(f"Error loading {audio_file} and {egg_file}: {e}")
            return None

        return audio, egg
    
class WaveNet(nn.Module):
    def __init__(self, input_channels, dilation_channels):
        super(WaveNet, self).__init__()
        self.dilation_channels = dilation_channels
        self.receptive_field_size = 1
        self.dilated_convs = nn.ModuleList()

        dilations = [2**i for i in range(6)]
        self.dilated_convs.append(nn.Conv1d(input_channels, 2 * dilation_channels, kernel_size=3, padding=dilations[0]))
        for dilation in dilations[1:]:
            padding = dilation * (3 - 1) // 2
            self.dilated_convs.append(nn.Conv1d(dilation_channels, 2 * dilation_channels, kernel_size=3, padding=padding, dilation=dilation))
            self.receptive_field_size += dilation * 2

        self.output_conv = nn.Conv1d(dilation_channels, 1, kernel_size=1)

    def forward(self, x):
        for conv in self.dilated_convs:
            out = conv(x)
            # Splitting the output of the convolution into filter and gate parts
            filter, gate = torch.split(out, self.dilation_channels, dim=1)  # Correct dimension for splitting is 1 (channels)
            x = torch.tanh(filter) * torch.sigmoid(gate)

        return self.output_conv(x)

# load data
test_dataset = AudioEGGDataset(audio_path, egg_path)

# Create a DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaveNet(1, 32)  # Use the same parameters as used during training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate and optimizer type as needed

# chkpt/checkpoint_40.pt
checkpoint_path = 'chkpt\checkpoint_40.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.to(device)
model.eval()

# Predict
with torch.no_grad():
    for i, (audio, egg) in enumerate(test_loader):
        audio = audio.to(device)
        egg = egg.to(device)
        output = model(audio)   
        print(f'Predicted: {output}')
        print(f'Actual: {egg}')
        if i == 5:
            break

# draw the audio, egg, and predicted egg waveforms
import matplotlib.pyplot as plt

# Convert tensors to numpy arrays
audio = audio.squeeze().cpu().numpy()
egg = egg.squeeze().cpu().numpy()
output = output.squeeze().cpu().numpy()

# Plot the audio, EGG, and predicted EGG waveforms
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(audio)
plt.subplot(3, 1, 2)
plt.plot(egg)
plt.subplot(3, 1, 3)
plt.plot(output)
plt.legend(['Predicted EGG', 'Actual EGG', 'Audio'])
plt.show()

# spectrogram
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.specgram(audio, Fs=samplerate)
plt.subplot(3, 1, 2)
plt.specgram(egg, Fs=samplerate)
plt.subplot(3, 1, 3)
plt.specgram(output, Fs=samplerate)
plt.legend(['Predicted EGG', 'Actual EGG', 'Audio'])
plt.show()

