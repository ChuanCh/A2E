import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

samplerate = 16000

# load data
audio_path = r'F:\Audio\Audio'
egg_path = r'F:\Audio\EGG'
audio_list = os.listdir(audio_path)
egg_list = os.listdir(egg_path)

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

            # divide by max value to normalize
            audio = audio / np.max(np.abs(audio))
            egg = egg / np.max(np.abs(egg))

            # Convert to PyTorch tensors and add channel dimension
            audio = torch.from_numpy(audio).float().unsqueeze(0)  # Add channel dimension
            egg = torch.from_numpy(egg).float().unsqueeze(0)

        except Exception as e:
            print(f"Error loading {audio_file} and {egg_file}: {e}")
            return None

        return audio, egg

dataset = AudioEGGDataset(audio_path, egg_path)
# Create train and validation and test sets
batch_size = 2  # Adjust as necessary
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

# Instantiate the model
channels = 32  # You may need to tune this based on your dataset
model = WaveNet(input_channels=1, dilation_channels=channels)


# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

from torch import optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(100):  # Adjust the number of epochs based on your needs
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        audio, egg = data
        audio = audio.to(device)
        egg = egg.to(device)

        optimizer.zero_grad()
        output = model(audio)
        loss = criterion(output, egg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:  # Log every 10 batches
            print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')

    scheduler.step()

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            audio, egg = data
            audio = audio.to(device)
            egg = egg.to(device)
            output = model(audio)
            loss = criterion(output, egg)
            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_dataloader)
    print(f'Epoch {epoch}, Validation Loss: {val_loss}')

    # Early stopping and saving best model based on validation loss
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # Save checkpoint
    if epoch % 10 == 0:  # Save every 10 epochs in chkpt folder
        checkpoint_path = os.path.join('chkpt', f'checkpoint_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(dataloader),
            'val_loss': val_loss,
        }, checkpoint_path)

    # Save the best model
    if val_loss < early_stopping.val_loss_min:
        best_model_path = 'WaveNetbest_model.pt'
        torch.save(model.state_dict(), best_model_path)