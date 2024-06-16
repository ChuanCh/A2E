import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class WaveNet_a2e(nn.Module):
    def __init__(self, input_channels, dilation_channels, dilation_layers, dropout):
        super(WaveNet_a2e, self).__init__()
        self.dilation_channels = dilation_channels
        self.receptive_field_size = 1
        self.dilated_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()     
        self.attentions = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        dilations = [2**i for i in range(dilation_layers)]
        self.dilated_convs.append(nn.Conv1d(input_channels, 2 * dilation_channels, kernel_size=3, padding=dilations[0]))
        self.batch_norms.append(nn.BatchNorm1d(2 * dilation_channels))  # Initialize batch norm for the first conv layer
        self.dropouts.append(nn.Dropout(p=dropout))  # Dropout layer with 10% probability
        self.attentions.append(SelfAttention(dilation_channels))
        self.residual_convs.append(nn.Conv1d(dilation_channels, dilation_channels, kernel_size=1))


        for dilation in dilations[1:]:
            padding = dilation * (3 - 1) // 2
            self.dilated_convs.append(nn.Conv1d(dilation_channels, 2 * dilation_channels, kernel_size=3, padding=padding, dilation=dilation))
            self.receptive_field_size += dilation * 2
            self.batch_norms.append(nn.BatchNorm1d(2 * dilation_channels))
            # self.layer_norms.append(nn.LayerNorm(2 * dilation_channels))
            self.dropouts.append(nn.Dropout(p=dropout)) 
            self.attentions.append(SelfAttention(dilation_channels))
            self.residual_convs.append(nn.Conv1d(dilation_channels, dilation_channels, kernel_size=1))


        self.output_conv = nn.Conv1d(dilation_channels, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []
        for conv, bn, do, attn, res_conv in zip(self.dilated_convs, self.batch_norms, self.dropouts, self.attentions, self.residual_convs):
            out = conv(x)
            out = bn(out)
            filter, gate = torch.split(out, self.dilation_channels, dim=1)
            activated = torch.tanh(filter) * torch.sigmoid(gate)
            out = do(activated)
            out = attn(out)
            x = res_conv(out) + x  # Adding skip connection from residual block
            skip_connections.append(out)

        total = sum(skip_connections)
        return self.output_conv(total)


class SelfAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(channel, channel // reduction, kernel_size=1)
        self.key_conv = nn.Conv1d(channel, channel // reduction, kernel_size=1)
        self.value_conv = nn.Conv1d(channel, channel, kernel_size=1)
        self.scale = (channel // reduction) ** -0.5

    def forward(self, x):
        batch, channels, width = x.size()
        query = self.query_conv(x).view(batch, -1, width)
        key = self.key_conv(x).view(batch, -1, width)
        value = self.value_conv(x).view(batch, -1, width)

        scores = torch.bmm(query.permute(0, 2, 1), key) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(value, attn)

        return context.view(batch, channels, width) + x




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

