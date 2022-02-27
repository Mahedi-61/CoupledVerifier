import torch
import torch.nn as nn
from torchsummary import summary 
from . import resnet
from . import densenet



class Mapper(nn.Module):
    def __init__(self, pre_network, img_dim, out_dim):
        super().__init__()

        # let's start with encoder
        if pre_network == "resnet18":
            self.backbone = resnet.get_model(img_dim, out_dim)

        elif pre_network == "densenet":
            self.backbone = densenet.get_model(img_dim, out_dim)

        # now decoder
        if pre_network == "resnet18":
            in_channels = 512 
            out_channels = in_channels // 2
            num_blocks = 8

        if pre_network == "densenet":
            in_channels = 1024 
            out_channels = in_channels // 4
            num_blocks = 8 

        decoder = [] 
        for _ in range(num_blocks):
            decoder += [
                nn.ConvTranspose2d(in_channels, out_channels, 
                        kernel_size=3, stride=2, 
                        padding=1, output_padding=1),

                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]

            in_channels = out_channels
            out_channels = in_channels // 2

 
        decoder += [nn.ReflectionPad2d(3), 
                    nn.Conv2d(out_channels*2, img_dim, kernel_size=7),
                    nn.Tanh()]
    
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x):
        img, embd = self.backbone(x)
        img =  self.decoder(img)
        return img, embd


    def EncodeImage(self, x):
        return self.backbone(x)

    def DecodeImage(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_size=256):
        super().__init__()

        self.shared_conv = nn.Sequential(
            *self.discriminator_block(in_channels, 16, bn=False),
            *self.discriminator_block(16, 32),
            *self.discriminator_block(32, 64),
            *self.discriminator_block(64, 128)
        )

        #image pixel in down-sampled features map
        input_node = 128 * (img_size // (2**4))**2 

        self.D1 = nn.Linear(input_node, 1)

    def discriminator_block(self, in_filters, out_filters, bn=True):

        block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]

        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        block.extend([nn.LeakyReLU(0.2, inplace=True), 
                        nn.Dropout2d(0.25)])

        return block

    def forward(self, x):
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)
        x = self.D1(x)
        return x 



if __name__ == "__main__":
    model = Mapper("resnet18", 1, 256)
    summary(model, (1, 256, 256), device="cpu") 
