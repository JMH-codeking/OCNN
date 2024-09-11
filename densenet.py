import torch.nn as nn
from layers import *
import torch.nn.functional as F

# Sobel filtering layer as a PyTorch module
class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        # Define Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        # Register kernels as parameters but without learning (frozen parameters)
        self.weight_x = nn.Parameter(sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x):
        # Apply Sobel filters on the input image
        Gx = F.conv2d(x, self.weight_x, padding=1)
        Gy = F.conv2d(x, self.weight_y, padding=1)
        # G = torch.sqrt(Gx**2 + Gy**2)
        # return G
        return torch.cat((Gx, Gy), dim=1)  # Concatenate along the channel dimension

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super(FCDenseNet, self).__init__()

        self.sobel_filt = SobelLayer()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        skip_connection_channels = []
        
        #initial conv
        self.conv0 = nn.Conv2d(2, 4, kernel_size = 3, padding = 1)
        self.firstconv = nn.Conv2d(4, out_chans_first_conv, kernel_size=3, stride=1, padding=1)
        cur_channels = out_chans_first_conv

        #downsampling path
        self.denseBlocksDown = nn.ModuleList()
        self.transDownBlocks = nn.ModuleList()
        for n_layers in down_blocks:
            self.denseBlocksDown.append(DenseBlock(cur_channels, growth_rate, n_layers))
            cur_channels += growth_rate * n_layers
            skip_connection_channels.insert(0, cur_channels)
            self.transDownBlocks.append(TransitionDown(cur_channels))

        #bottleneck
        self.bottleneck = Bottleneck(cur_channels, growth_rate, bottleneck_layers)
        cur_channels += growth_rate * bottleneck_layers

        #upsampling path
        self.transUpBlocks = nn.ModuleList()
        self.denseBlocksUp = nn.ModuleList()
        prev_block_channels = growth_rate * bottleneck_layers

        for i, n_layers in enumerate(up_blocks[:-1]):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels = prev_block_channels + skip_connection_channels[i]
            self.denseBlocksUp.append(DenseBlock(cur_channels, growth_rate, n_layers, upsample=True))
            prev_block_channels = growth_rate * n_layers
            cur_channels += prev_block_channels

        #final up and denseblock
        self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels = prev_block_channels + skip_connection_channels[-1]
        self.denseBlocksUp.append(DenseBlock(cur_channels, growth_rate, up_blocks[-1], upsample=False))
        cur_channels += growth_rate * up_blocks[-1]

        self.finalConv = nn.Conv2d(cur_channels, n_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.sobel_filt(x)
        # print (x.shape)

        x = self.conv0(x)
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.softmax(out)

        return out.squeeze(1)

