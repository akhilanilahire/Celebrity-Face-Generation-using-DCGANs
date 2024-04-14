import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.de_conv1 = nn.Sequential(
            # Transposed Convolution Layer
            nn.ConvTranspose2d(
                in_channels = z_size,                 # Input Channels = 100
                out_channels = conv_dim * 8,          # Output Channels = 1024 (128 * 8)
                kernel_size = 4,                      # Size of the convolutional kernel = 4 x 4
                stride = 1,
                padding = 0,
                bias = False
            ),
            # Batch Normalization Layer
            nn.BatchNorm2d(conv_dim * 8),           # Number of Features = 1024 (128 * 8)
        )
        self.de_conv2 = nn.Sequential(
            # Transposed Convolution Layer
            nn.ConvTranspose2d(
                in_channels = conv_dim * 8,           # Input Channels = 1024 (128 * 8)
                out_channels = conv_dim * 4,          # Output Channels = 512 (128 * 4)
                kernel_size = 4,                      # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False
            ),
            # Batch Normalization Layer
            nn.BatchNorm2d(conv_dim * 4),           # Number of Features = 512 (128 * 4)
        )
        self.de_conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = conv_dim * 4,           # Input Channels = 512 (128 * 4)
                out_channels = conv_dim * 2,          # Output Channels = 256 (128 * 2)
                kernel_size = 4,                      # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(conv_dim * 2),           # Number of Features = 256 (128 * 2)
        )
        self.de_conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = conv_dim * 2,           # Input Channels = 512 (128 * 4)
                out_channels = conv_dim,              # Output Channels = 256 (128 * 2)
                kernel_size = 4,                      # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(conv_dim),               # Number of Features = 128
        )
        self.de_conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = conv_dim,               # Input Channels = 128
                out_channels = 3,                     # Output Channels = 3
                kernel_size = 4,                      # Size of the convolutional kernel = 4 x 4
                stride=2,
                padding=1,
                bias=False
            ),
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, input):
        x = self.dropout(input)
        x1 = F.relu(self.de_conv1(x))
        x2 = F.relu(self.de_conv2(x1))
        x3 = F.relu(self.de_conv3(x2))
        x4 = F.relu(self.de_conv4(x3))
        x5 = F.tanh(self.de_conv5(x4))
        
        return x5
