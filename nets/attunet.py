import torch 
from torch import nn 
from .network_utils import conv_block, up_conv, Attention_block

class AttU_Net(nn.Module):
    def __init__(self,img_dim, out_dim=256, features=32): #org 64
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_dim,ch_out=features)
        self.Conv2 = conv_block(ch_in=features,ch_out=features*2)
        self.Conv3 = conv_block(ch_in=features*2, ch_out=features*4)
        self.Conv4 = conv_block(ch_in=features*4,ch_out=features*8)
        self.Conv5 = conv_block(ch_in=features*8,ch_out=features*16)
        
        self.avrgpool = nn.AdaptiveAvgPool2d((2, 2))  
        self.fc = nn.Linear(features*16*4, out_dim) 

        self.Up5 = up_conv(ch_in=features*16,ch_out=features*8)
        self.Att5 = Attention_block(F_g=features*8,F_l=features*8,F_int=features*4)
        self.Up_conv5 = conv_block(ch_in=features*16, ch_out=features*8)

        self.Up4 = up_conv(ch_in=features*8,ch_out=features*4)
        self.Att4 = Attention_block(F_g=features*4,F_l=features*4,F_int=features*2)
        self.Up_conv4 = conv_block(ch_in=features*8, ch_out=features*4)
        
        self.Up3 = up_conv(ch_in=features*4,ch_out=features*2)
        self.Att3 = Attention_block(F_g=features*2,F_l=features*2,F_int=features)
        self.Up_conv3 = conv_block(ch_in=features*4, ch_out=features*2)
        
        self.Up2 = up_conv(ch_in=features*2,ch_out=features)
        self.Att2 = Attention_block(F_g=features,F_l=features,F_int=features//2)
        self.Up_conv2 = conv_block(ch_in=features*2, ch_out=features)

        self.Conv_1x1 = nn.Conv2d(features,img_dim,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        embd = self.avrgpool(x5)
        embd = embd.view(embd.size(0), -1)
        embd = self.fc(embd)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1, embd 