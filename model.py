import torch 
from torch import nn 
from torchvision import models 
import torch.nn.functional as F 
#from torchsummary import summary

class Mapper(nn.Module):
    def __init__(self, pre_network, join_type, img_dim, out_dim=256):
        super().__init__()

        self.join_type = join_type

        # let's start with encoder
        model = getattr(models, pre_network)(pretrained=False)
        model.conv1 = nn.Conv2d(img_dim, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        
        if self.join_type == "concat":
            model.avgpool =  nn.AdaptiveAvgPool2d(output_size=(1, 2))

        model = list(model.children())[:-1]
        self.backbone = nn.Sequential(*model)

        if pre_network == "resnet50": nfc = 2048
        elif pre_network == "resnet18": nfc = 512 

        if self.join_type == "concat":
            self.fc1 = nn.Linear(nfc*2, out_dim)
        else:
            self.fc1 = nn.Linear(nfc, out_dim)

        # now decoder
        decoder = [] 
        in_channels = nfc 
        out_channels = in_channels // 2

        num_blocks = 8 
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
        x = self.backbone(x)
        img =  self.decoder(x)

        x = x.view(x.size(0), -1)
        embedding = self.fc1(x)

        return img, embedding

    def EncodeImage(self, x):
        return self.backbone(x)

    def DecodeImage(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, join_type, in_channels, img_size=256):
        super().__init__()

        self.join_type = join_type 
        self.shared_conv = nn.Sequential(
            *self.discriminator_block(in_channels, 16, bn=False),
            *self.discriminator_block(16, 32),
            *self.discriminator_block(32, 64),
            *self.discriminator_block(64, 128)
        )

        #image pixel in down-sampled features map
        if self.join_type == "concat":
            input_node = 128 * (img_size // (2**4)) * (img_size // (2**3))
        else:
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


class FingerIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(256, 1))

    def forward(self, embd1, embd2):
        dis = torch.abs(embd1 - embd2)
        return self.model(dis)



if __name__ == "__main__":
    m = Mapper(pre_network="resnet18", join_type = "concat", img_dim=2).cuda()
    print(m)

    #m = Discriminator(in_channels=1).cuda()
    m = FingerIdentity()
    x1 = torch.randn((256))
    x2 = torch.randn((256))
    #summary(m, input_size=(256, 256))