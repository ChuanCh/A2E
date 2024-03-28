import numpy as np
import librosa  
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

top_db = 30
n_mels = 32
n_fft = 256
hop_length = 256
samplerate = 16000


# 函数的目的，是将音频数据转换为特征向量。
# 原始输入应该是1.音频数据，2.EGG数据
# 输出应该是1.音频特征向量，2.EGG特征向量

class VoiceDataset(Dataset):
    def __init__(self, segment_length_in_samples):
        """
        Initialize the VoiceDataset class.

        Args:
            segment_length_in_samples (int): Length of audio segments.
        """
        self.segments = None
        self.length = None
        self.top_db = top_db
        self.segment_length_samples = segment_length_in_samples
        
    def load_dataset(self, dataset_name, samplerate):
        '''Load the dataset based on its name.'''
        file_paths = {
            'test': [r'F:\A2E\test file\test_Voice_EGG.wav'],
            'F01': ['F01_VRP_C_Voice_EGG.wav'],
            'one_singing': [
                r'F:\A2E\adults-trimmed\F01\190306_095129_F01_VRP1_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_095940_F01_VRP2_C_Voice_EGG.wav'
            ],
            'one_reading': [
                r'F:\A2E\adults-trimmed\F01\190306_094138_F01_SRP1_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_094516_F01_SRP2_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_094802_F01_SRP3_C_Voice_EGG.wav'
            ],
            'one_everything': [
                r'F:\A2E\adults-trimmed\F01\190306_095129_F01_VRP1_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_095940_F01_VRP2_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_094138_F01_SRP1_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_094516_F01_SRP2_C_Voice_EGG.wav',
                r'F:\A2E\adults-trimmed\F01\190306_094802_F01_SRP3_C_Voice_EGG.wav'
            ],
            'all_singing': 'all_VRP',
            'all_reading': 'all_SRP',
            'all_everything': 'all'
        }

        # Special handling for datasets that require walking through directories
        if dataset_name in ['all_singing', 'all_reading', 'all_everything']:
            wav_data = self._load_from_directory(dataset_name, samplerate)
        else:
            wav_files = file_paths.get(dataset_name)
            if wav_files is None:
                raise ValueError('Invalid dataset name.')
            wav_data = np.concatenate([librosa.load(file, sr=samplerate, mono=False)[0] for file in wav_files], axis=1)

        return wav_data
    
    def _load_from_directory(self, dataset_name, samplerate):
        pattern = {
            'all_VRP': 'VRP',
            'all_SRP': 'SRP',
            'all': ''
        }.get(dataset_name)

        temp = []
        for root, _, files in os.walk(adult_directory):
            for file in files:
                if file.endswith('Voice_EGG.wav') and (pattern in file):
                    file_path = os.path.join(root, file)
                    audio_data, _ = librosa.load(file_path, sr=samplerate, mono=False)
                    temp.append(audio_data)
        return np.concatenate(temp, axis=1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        segment = self.segments[idx]

        # Convert to tensors
        input_tensor = torch.tensor(segment[0], dtype=torch.float32)
        target_tensor = torch.tensor(segment[1], dtype=torch.float32)

        # Ensure tensors have correct shape
        assert input_tensor.shape == (self.segment_length_samples, n_mels)
        assert target_tensor.shape == (self.segment_length_samples, n_mels)

        return input_tensor, target_tensor

    def _remove_silence(self, wav):
        # remove silence in the first channel
        non_silent_intervals = librosa.effects.split(wav[0], top_db=top_db)
        # Apply these intervals to both channels
        non_silent_wavs = []
        for channel in range(wav.shape[0]):  # Assuming wav has shape (channels, samples)
            channel_data = wav[channel]
            # Applying the same non-silent intervals to each channel
            non_silent_channel_wav = np.concatenate([channel_data[interval[0]:interval[1]] for interval in non_silent_intervals])
            non_silent_wavs.append(non_silent_channel_wav)        
        
        non_silent_wav = np.array(non_silent_wavs)
        return non_silent_wav
    
    def _min_max_normalization(self, wav):
        # min-max normalization for each channel separately
        for i in range(wav.shape[0]):
            wav[i] = (wav[i] - wav[i].min()) / (wav[i].max() - wav[i].min())
        return wav

    def preprocess(self, wav, denoise = False, preprocess_type = 'raw'):
        ''' general preprocessing
         Returns:
            numpy.ndarray: Preprocessed audio data. '''
        # remove silence if needed
        if denoise:
            wav = self._remove_silence(wav)
            
        # min-max normalization
        wav = self._min_max_normalization(wav)

        if preprocess_type == 'raw':
            # create segments
            self.segments = self.create_segments(wav)
            self.length = len(self.segments)

        elif preprocess_type == 'Mel':
            temp = self.Mel(wav)
            input = np.transpose(temp[0])
            target = np.transpose(temp[1])
            paired_segments = list(zip(input, target))
            self.segments = paired_segments
            self.length = len(self.segments)


        elif preprocess_type == 'MFCC_process':
            wav = self.MFCC(wav)

            pass
        elif preprocess_type == 'fourier_descriptor':
            wav = self.fourier_descriptor(wav)
            pass
               
        return wav
      
    def plot(self, wav):
        # plot the wav for visualization, first channel then second channel
        plt.figure(figsize=(16, 4))
        plt.plot(wav[0])
        plt.title('Acoustic')
        plt.show()
        plt.figure(figsize=(16, 4))
        plt.plot(wav[1])
        plt.title('EGG')
        plt.show()
        
    def check_shape(self, wav):
        # check the shape of the wav
        print(wav.shape)
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        total_segments = len(self.segments)
        train_end = int(total_segments * train_ratio)
        val_end = train_end + int(total_segments * val_ratio)

        train_segments = self.segments[:train_end]
        val_segments = self.segments[train_end:val_end]
        test_segments = self.segments[val_end:]

        return train_segments, val_segments, test_segments
    
    def raw(self, wav):
        return wav
    
    def Mel(self, wav, n_mels=32, convert_to_db=True):
        # Calculate hop_length for 20ms frames
        hop_length = int(samplerate * 0.02)  # for 20 ms, ensure this is an integer

        # Calculate n_fft (the window size) as the next power of 2 greater than or equal to window_length_in_samples
        window_length_in_seconds = 0.025  # Typical window length for speech processing is 25ms
        n_fft = 2 ** int(np.ceil(np.log2(samplerate * window_length_in_seconds)))

        # Compute Mel spectrogram for each channel
        mel_spectrograms = []
        for channel in range(wav.shape[0]):
            S = librosa.feature.melspectrogram(y=wav[channel], sr=samplerate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

            # Convert to decibels if required
            if convert_to_db:
                S = librosa.power_to_db(S)

            mel_spectrograms.append(S)

        return mel_spectrograms
    
    def MFCC(self, wav):
        return wav
    
    def fourier_descriptor(self, wav):
        return wav

# Function to log configuration details to TensorBoard
def tensorboard_log_config(writer, model, input_type, sample_rate, sample_length, batch_size):
    config_text = f"Model type: {type(model).__name__}\nInput type: {input_type}\nSample rate: {sample_rate} Hz \nSample length: {sample_length} samples\nBatch size: {batch_size}"
    writer.add_text('Training Configuration', config_text, 0)


# FNN
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, second_hidden_size, output_size, dropout_prob):
        super(Feedforward, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fc2 = torch.nn.Linear(hidden_size, second_hidden_size)
        self.fc3 = torch.nn.Linear(second_hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# CNN
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, num_layers, dropout_prob=0.5):
        super(CNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
    
        # Convolutional layers for encoding
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Convolutional layers for decoding
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        # Reshape input for CNN: [batch_size, features, sequence_length]
        x = x.transpose(1, 2)

        # Encoding
        encoded = self.encoder(x)

        # Decoding
        decoded = self.decoder(encoded)

        # Reshape output to original shape: [batch_size, sequence_length, features]
        output = decoded.transpose(1, 2)

        return output

# LSTM
class LSTMmodel(nn.Module):
    def __init__(self, n_mels, hidden_size, num_layers=1):
        super(LSTMmodel, self).__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(input_size=n_mels, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Fully connected layer that maps from hidden state space to output space
        self.fc = nn.Linear(hidden_size, n_mels)
    
    def forward(self, x):
        # Initialize hidden and cell states
        # Dimensions: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # Output shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Reshape output to (batch_size*seq_len, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        
        # Pass through fully connected layer and reshape to (batch_size, seq_len, n_mels)
        out = self.fc(out).view(x.size(0), -1, self.n_mels)
        return out



# Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_size, nhead=num_heads, 
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward)
        self.pos_encoder = PositionalEncoding(input_size)
        self.decoder = nn.Linear(input_size, output_size)

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough 'pe' matrix that can be sliced according to input length
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# WaveNet
    


def train_model(writer, model, train_loader, val_loader, num_epochs, learning_rate, patience=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    lr_rates = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        lr_rates.append(current_lr)

        print(f'Epoch: {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {current_lr}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model checkpoint if desired
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Plotting the training and validation losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lr_rates, label='Learning Rate')
    plt.title("Learning Rate Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.show()

    return model