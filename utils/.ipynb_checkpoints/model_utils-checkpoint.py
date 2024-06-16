import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



class WaveNet(nn.Module):
    def __init__(self, input_channels, dilation_channels):
        super(WaveNet, self).__init__()
        self.dilation_channels = dilation_channels
        self.receptive_field_size = 1
        self.dilated_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()

        dilations = [2**i for i in range(10)] 
        self.dilated_convs.append(nn.Conv1d(input_channels, 2 * dilation_channels, kernel_size=3, padding=dilations[0], dilation=1))
        self.batch_norms.append(nn.BatchNorm1d(2 * dilation_channels)) 
        self.dropouts.append(nn.Dropout(p=0.5))
        self.attentions.append(MultiHeadSelfAttention(dilation_channels, heads=4)) 

        for dilation in dilations[1:]:
            padding = dilation * (3 - 1) // 2
            self.dilated_convs.append(nn.Conv1d(dilation_channels, 2 * dilation_channels, kernel_size=3, padding=padding, dilation=dilation))
            self.receptive_field_size += dilation * 2
            self.batch_norms.append(nn.BatchNorm1d(2 * dilation_channels))
            self.dropouts.append(nn.Dropout(p=0.5))
            self.attentions.append(MultiHeadSelfAttention(dilation_channels, heads=4))

        self.output_conv = nn.Conv1d(dilation_channels, 1, kernel_size=1)

    def forward(self, x):
        for conv, bn, do, attn in zip(self.dilated_convs, self.batch_norms, self.dropouts, self.attentions):
            out = conv(x)
            out = bn(out)
            filter, gate = torch.split(out, self.dilation_channels, dim=1)
            x = torch.tanh(filter) * torch.sigmoid(gate)
            x = do(x)
            x = attn(x)

        return self.output_conv(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, heads=4, reduction=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.query_convs = nn.ModuleList([nn.Conv1d(channels, channels // reduction, kernel_size=1) for _ in range(heads)])
        self.key_convs = nn.ModuleList([nn.Conv1d(channels, channels // reduction, kernel_size=1) for _ in range(heads)])
        self.value_convs = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=1) for _ in range(heads)])
        self.scale = (channels // reduction) ** -0.5

    def forward(self, x):
        batch, channels, width = x.size()
        context = x
        for query_conv, key_conv, value_conv in zip(self.query_convs, self.key_convs, self.value_convs):
            query = query_conv(x).view(batch, -1, width)
            key = key_conv(x).view(batch, -1, width)
            value = value_conv(x).view(batch, -1, width)

            scores = torch.bmm(query.permute(0, 2, 1), key) * self.scale
            attn = F.softmax(scores, dim=-1)
            context = context + torch.bmm(value, attn).view(batch, channels, width)

        return context + x

def initialize_model(input_channels, dilation_channels):
    """
    Initialize the WaveNet model with specified configurations.
    
    Args:
    input_channels (int): Number of input channels (e.g., 1 for mono audio).
    dilation_channels (int): Number of dilation channels in WaveNet.

    Returns:
    nn.Module: Initialized WaveNet model.
    """
    model = WaveNet(input_channels=input_channels, dilation_channels=dilation_channels)
    return model

class AudioEGGDataset(Dataset):
    def __init__(self, audio_frames, egg_frames):
        """
        Initializes the dataset with pre-loaded data.
        :param audio_frames: A list or array of preprocessed and segmented audio frames.
        :param egg_frames: A list or array of preprocessed and segmented EGG frames.
        :param transform: Optional transform to be applied on a sample.
        """
        assert len(audio_frames) == len(egg_frames), "Audio and EGG frames must be the same length"
        self.audio_frames = audio_frames
        self.egg_frames = egg_frames

    def __len__(self):
        return len(self.audio_frames)

    def __getitem__(self, idx):
        audio_frame = self.audio_frames[idx]
        egg_frame = self.egg_frames[idx]

        # Convert arrays to PyTorch tensors
        audio_tensor = torch.from_numpy(audio_frame).float().unsqueeze(0)  # Add channel dimension if needed
        egg_tensor = torch.from_numpy(egg_frame).float().unsqueeze(0)      # Add channel dimension if needed

        return audio_tensor, egg_tensor
