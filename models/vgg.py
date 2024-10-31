import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        # Modified reg_layer with parallel paths for multi-scale feature extraction
        self.reg_layer_3x3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.reg_layer_5x5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        self.reg_layer_7x7 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        
        # Concatenated output from the parallel paths will be reduced to a single channel
        self.output_layer = nn.Conv2d(128 * 3, 1, kernel_size=1)

    def forward(self, x):
        # Feature extraction from VGG backbone
        x = self.features(x)
        
        # Upsample the features before applying regression layers
        x = F.upsample_bilinear(x, scale_factor=2)
        
        # Apply parallel convolutional paths
        x_3x3 = self.reg_layer_3x3(x)
        x_5x5 = self.reg_layer_5x5(x)
        x_7x7 = self.reg_layer_7x7(x)
        
        # Concatenate along the channel dimension
        x = torch.cat((x_3x3, x_5x5, x_7x7), dim=1)  # Resulting in 128 * 3 channels
        
        # Reduce channels to 1 for density map output
        x = self.output_layer(x)
        
        return torch.abs(x)  # Ensuring density values are non-negative

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
