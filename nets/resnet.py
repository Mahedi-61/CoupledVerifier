import torch 
from torch import nn 
from torchvision import models 
from torchsummary import summary
import os 

is_multi_gpus = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_load = True 
save_w_dir = "."

def def_resnet(img_dim):
    cnn_block = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1])
    cnn_block[0] = nn.Conv2d(img_dim, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)

    return cnn_block


class get_model(nn.Module):
    def __init__(self, img_dim, out_dim):
        super().__init__() 

        self.backbone = def_resnet(img_dim)
        self.fc1 = nn.Linear(512, out_dim)

    def forward(self, x):
        img = self.backbone(x)
        x = img.view(img.size(0), -1)
        embd= self.fc1(x)

        return img, embd


def load_resnet():
    model = get_model(img_dim=1, out_dim=256)
    model.to(device)

    if is_multi_gpus:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    if is_load:
        print("loading pretrained NIST-302 weights")
        model_file = os.path.join(save_w_dir, "best_model_000.pth")
        checkpoint = torch.load(model_file)

        model.load_state_dict(checkpoint["net_print"])

    return model


if __name__ == "__main__":
    model = load_resnet()
