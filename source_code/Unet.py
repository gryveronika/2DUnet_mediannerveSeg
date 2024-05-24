from collections import OrderedDict
from torch import nn, cat, manual_seed

from source_code.config import Unet_levels, num_classes
#from config import Unet_levels, num_classes

# Define U-Net architecture based on Unet_levels configuration
class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=32):
        super(Unet, self).__init__()
        features = init_features

        # Define encoder and decoder blocks
        self.encoder1 = Unet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Unet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Unet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Unet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if Unet_levels == 6:
            self.encoder5 = Unet._block(features * 8, features * 16, name="enc5")
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bottleneck = Unet._block(features * 16, features * 32, name="bottleneck")
            self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
            self.decoder5 = Unet._block(features * 32, features * 16, name="dec5")

        else:
            self.bottleneck = Unet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Unet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Unet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Unet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Unet._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        pool1 = self.pool1(encoder1)
        encoder2 = self.encoder2(pool1)
        pool2 = self.pool2(encoder2)
        encoder3 = self.encoder3(pool2)
        pool3 = self.pool3(encoder3)
        encoder4 = self.encoder4(pool3)
        pool4 = self.pool4(encoder4)

        if Unet_levels == 6:
            encoder5 = self.encoder5(pool4)
            pool5 = self.pool5(encoder5)
            bottleneck = self.bottleneck(pool5)
            upconv5 = self.upconv5(bottleneck)
            decoder5 = self.decoder5(cat([upconv5, encoder5], 1))
        else:
            bottleneck = self.bottleneck(pool4)

        upconv4 = self.upconv4(bottleneck)
        decoder4 = self.decoder4(cat([upconv4, encoder4], 1))
        upconv3 = self.upconv3(decoder4)
        decoder3 = self.decoder3(cat([upconv3, encoder3], 1))
        upconv2 = self.upconv2(decoder3)
        decoder2 = self.decoder2(cat([upconv2, encoder2], 1))
        upconv1 = self.upconv1(decoder2)
        decoder1 = self.decoder1(cat([upconv1, encoder1], 1))

        output = self.conv(decoder1)
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (name + "_conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1)),
            (name + "_norm1", nn.BatchNorm2d(features)),
            (name + "_relu1", nn.ReLU(inplace=True)),
            (name + "_conv2", nn.Conv2d(features, features, kernel_size=3, padding=1)),
            (name + "_norm2", nn.BatchNorm2d(features)),
            (name + "_relu2", nn.ReLU(inplace=True))
        ]))

def count_parameters(model):
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate and print the number of parameters in the U-Net model
unet = Unet(in_channels=1, out_channels=num_classes, init_features=32)
print(count_parameters(unet))
