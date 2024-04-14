import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            # Convolutional Layer:
            nn.Conv2d(
                in_channels = 3,                     # Input Channels = 3 [RGB image]
                out_channels = conv_dim,             # Output Channels = 64
                kernel_size = 4,                     # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False
            )
        )

        self.conv2 = nn.Sequential(
            # Convolutional Layer:
            nn.Conv2d(
                in_channels = conv_dim,                 # Input Channels = 64
                out_channels = conv_dim * 2,            # Output Channels = 128 (64 * 2)
                kernel_size = 4,                        # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False
            ),
            # Batch Normalization Layer:
            nn.BatchNorm2d(num_features = conv_dim * 2)  # Number of Features = 128
        )

        self.conv3 = nn.Sequential(
            # Convolutional Layer:
            nn.Conv2d(
                in_channels = conv_dim * 2,             # Input Channels = 128  (64 * 2)
                out_channels = conv_dim * 4,            # Output Channels = 256 (64 * 4)
                kernel_size = 4,                        # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False),
            # Batch Normalization Layer:
            nn.BatchNorm2d(num_features = conv_dim * 4)  # Number of Features = 256
        )

        self.conv4 = nn.Sequential(
            #Convolutional Layer:
            nn.Conv2d(
                in_channels = conv_dim * 4,             # Input Channels = 256  (64 * 4)
                out_channels = conv_dim * 8,            # Output Channels = 512 (64 * 8)
                kernel_size = 4,                        # Size of the convolutional kernel = 4 x 4
                stride = 2,
                padding = 1,
                bias = False),
            # Batch Normalization Layer:
            nn.BatchNorm2d(num_features = conv_dim * 8)  # Number of Features = 512
        )

        self.conv5 = nn.Sequential(
            #Convolutional Layer:
            nn.Conv2d(
                in_channels = conv_dim * 8,          # Input Channels = 512 (64 * 8)
                out_channels = 1,                    # Output Channels = 1  [For Binary Classification]
                kernel_size = 4,                     # Size of the convolutional kernel = 4 x 4
                stride = 1,
                padding = 0,
                bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):

        x1 = F.leaky_relu(self.conv1(input), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        x5 = self.conv5(x4)

        return x5
