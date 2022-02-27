import torch 
from torch import nn 
from torchsummary import summary
import config 
import os 

from nets import resnet, resnet_unet, densenet, attunet
from nets.efficientnet_v2 import EfficientNet
from nets import mapper


def get_model(w_name, img_dim):
    network_arch = w_name.split("_")[-1]

    if network_arch == "A1":
        net_photo = mapper.Mapper(pre_network="resnet18", 
                            img_dim = img_dim, out_dim=config.feature_dim) 

        net_print = mapper.Mapper(pre_network="resnet18", 
                            img_dim = img_dim, out_dim=config.feature_dim) 

    elif network_arch == "A2":
        net_photo = attunet.AttU_Net(img_dim = img_dim, out_dim=config.feature_dim)
        net_print = attunet.AttU_Net(img_dim = img_dim, out_dim=config.feature_dim) 

    elif network_arch == "A3":
        net_photo = EfficientNet(img_dim = img_dim, out_dim=config.feature_dim)
        net_print = EfficientNet(img_dim = img_dim, out_dim=config.feature_dim) 

    elif network_arch == "A4":
        net_photo = resnet_unet.ResNetUNet(img_dim = img_dim, out_dim=config.feature_dim)
        net_print = resnet_unet.ResNetUNet(img_dim = img_dim, out_dim=config.feature_dim) 

    elif network_arch == "A5":
        net_photo = mapper.Mapper(pre_network="densenet", 
                            img_dim = img_dim,  out_dim=config.feature_dim) 

        net_print = mapper.Mapper(pre_network="densenet", 
                            img_dim = img_dim,  out_dim=config.feature_dim) 

    # load in GPU and do parallel
    net_photo.to(config.device)
    net_print.to(config.device)

    if config.multi_gpus:
        net_photo = torch.nn.DataParallel(net_photo, device_ids=[0, 1])
        net_print = torch.nn.DataParallel(net_print, device_ids=[0, 1])

    return net_photo, net_print


def get_one_model(w_name, img_dim):
    network_arch = w_name.split("_")[-1]

    if network_arch == "A1":
        net_print = resnet.get_model(img_dim = img_dim, out_dim=config.feature_dim)
                            

    elif network_arch == "A5":
        net_print = densenet.get_model(img_dim = img_dim, out_dim=config.feature_dim) 

    # load in GPU and do parallel
    net_print.to(config.device)

    if config.multi_gpus:
        net_print = torch.nn.DataParallel(net_print, device_ids=[0, 1])

    return net_print


def get_discriminator(img_dim):
    disc_photo = mapper.Discriminator(in_channels=img_dim)
    disc_photo.to(config.device)
        
    disc_print = mapper.Discriminator(in_channels=img_dim)
    disc_print.to(config.device)

    if config.multi_gpus:
        disc_photo = torch.nn.DataParallel(disc_photo, device_ids=[0, 1])
        disc_print = torch.nn.DataParallel(disc_print, device_ids=[0, 1])

    return disc_photo, disc_print


# loading save_w_name to finetune on single or multi-finger, and same or another dataset
def load_saved_model_for_finetuing(net_photo, net_print, disc_photo, disc_print, 
                                    is_disc_load=True, partial_finetune=False):
    
    if config.is_finetune == False:
        if config.dataset_name == "wvu_old":
            save_w_dir = os.path.join(config.dataset_cp_dir, config.save_w_name)

        if config.dataset_name == "wvu_new":
             save_w_dir = os.path.join(config.new_weights_dir, config.save_w_name)

    else:
        save_w_dir = config.save_w_dir


    if config.is_convert_one_to_many == False: 
        loaded_model_file = os.path.join(save_w_dir, "best_model_000.pth")
        checkpoint = torch.load(loaded_model_file)
        net_photo.load_state_dict(checkpoint["net_photo"])
        net_print.load_state_dict(checkpoint["net_print"])

        if is_disc_load:
            disc_photo.load_state_dict(checkpoint["disc_photo"])
            disc_print.load_state_dict(checkpoint["disc_print"])

        if partial_finetune == True:
            if config.num_join_fingers >=1:
                print("Allah help me")
                network_arch = config.save_w_name.split("_")[-1]

                if network_arch == "A1":
                    # setting generator gradient false
                    for name, param in net_photo.named_parameters():
                        #print(name)
                        if (name=="module.backbone.7.1.conv2.weight" or
                            name=="module.backbone.7.1.bn2.weight" or 
                            name=="module.backbone.7.1.bn2.bias"):
                            param.requires_grad = True 

                        if name=="module.fc1.weight" or name=="module.fc1.bias":
                            param.requires_grad = True

                        else:  param.requires_grad = False
                        """
                        if name=="module.decoder.25.weight" or name=="module.decoder.25.bias":
                            param.requires_grad = True
                        """
                        
                    for name, param in net_print.named_parameters():
                        if (name=="module.backbone.7.1.conv2.weight" or
                            name=="module.backbone.7.1.bn2.weight" or 
                            name=="module.backbone.7.1.bn2.bias"):
                            param.requires_grad = True 

                        if name=="module.fc1.weight" or name=="module.fc1.bias":
                            param.requires_grad = True

                        else:  param.requires_grad = False

                        """
                        if name=="module.decoder.25.weight" or name=="module.decoder.25.bias":
                            param.requires_grad = True
                        """
                        
                    # setting discriminator gradient false
                    for param in disc_photo.parameters():
                        param.requires_grad = False

                    for param in disc_print.parameters():
                        param.requires_grad = False


    elif config.is_convert_one_to_many == True:
        print("intitializing single finger weights ...")
        if config.num_join_fingers >= 2:
            loaded_model_file = os.path.join(save_w_dir, "best_model_000.pth")

        checkpoint = torch.load(loaded_model_file)
        compatible_load(checkpoint, net_photo, net_print, 
                        disc_photo, disc_print, config.save_w_name, is_disc_load)
   



def compatible_load(checkpoint, net_photo, net_print, disc_photo, 
                                    disc_print, w_name, is_disc_load=True):
    network_arch = w_name.split("_")[-1]

    if network_arch == "A1":
        # photo
        net_photo_state_dict = checkpoint["net_photo"]
        first_conv_key = "module.backbone.backbone.0.weight"
        first_conv_weights = [net_photo_state_dict[first_conv_key] 
                                        for _ in range(config.num_join_fingers)]

        net_photo_state_dict[first_conv_key] = torch.cat(first_conv_weights, dim=1)


        last_conv_key = "module.decoder.25.weight"
        last_conv_weights = [net_photo_state_dict[last_conv_key] 
                                        for _ in range(config.num_join_fingers)]
        
        net_photo_state_dict[last_conv_key] = torch.cat(last_conv_weights, dim=0)


        last_bias_key = "module.decoder.25.bias"
        last_bias = [net_photo_state_dict[last_bias_key] 
                                        for _ in range(config.num_join_fingers)]

        net_photo_state_dict[last_bias_key] = torch.cat(last_bias, dim=0)
        net_photo.load_state_dict(net_photo_state_dict)


        # print
        net_print_state_dict = checkpoint["net_print"]
        first_conv_key = "module.backbone.backbone.0.weight"
        first_conv_weights = [net_print_state_dict[first_conv_key] 
                                        for _ in range(config.num_join_fingers)]
        
        net_print_state_dict[first_conv_key] = torch.cat(first_conv_weights, dim=1)


        last_conv_key = "module.decoder.25.weight"
        last_conv_weights = [net_print_state_dict[last_conv_key] 
                                        for _ in range(config.num_join_fingers)]

        net_print_state_dict[last_conv_key] = torch.cat(last_conv_weights, dim=0)
        

        last_bias_key = "module.decoder.25.bias"
        last_bias = [net_print_state_dict[last_bias_key]  
                                        for _ in range(config.num_join_fingers)]

        net_print_state_dict[last_bias_key] = torch.cat(last_bias, dim=0)
        net_print.load_state_dict(net_print_state_dict)


    elif network_arch == "A2":
        # photo
        net_photo_state_dict = checkpoint["net_photo"]
        first_conv_weights = net_photo_state_dict["module.Conv1.conv.0.weight"]
        net_photo_state_dict["module.Conv1.conv.0.weight"] = torch.cat((
                                    first_conv_weights, first_conv_weights), dim=1)

        last_conv_weights = net_photo_state_dict["module.Conv_1x1.weight"]

        if config.num_join_fingers == 2:
            net_photo_state_dict["module.Conv_1x1.weight"] = torch.cat((
                                    last_conv_weights, last_conv_weights), dim=0)

        elif config.num_join_fingers == 3:
            net_photo_state_dict["module.Conv_1x1.weight"] = torch.cat((
                last_conv_weights, last_conv_weights, last_conv_weights), dim=0)

        elif config.num_join_fingers == 4:
            net_photo_state_dict["module.Conv_1x1.weight"] = torch.cat((
            last_conv_weights, last_conv_weights, last_conv_weights, last_conv_weights), dim=0)

        last_bias = net_photo_state_dict["module.Conv_1x1.bias"]

        if config.num_join_fingers == 2:
            net_photo_state_dict["module.Conv_1x1.bias"] = torch.cat((
                                                    last_bias, last_bias), dim=0)
        elif config.num_join_fingers == 3:
            net_photo_state_dict["module.Conv_1x1.bias"] = torch.cat((
                                        last_bias, last_bias, last_bias), dim=0)

        elif config.num_join_fingers == 4:
            net_photo_state_dict["module.Conv_1x1.bias"] = torch.cat((
                            last_bias, last_bias, last_bias, last_bias), dim=0)

        net_photo.load_state_dict(net_photo_state_dict)

        # print
        net_print_state_dict = checkpoint["net_print"]
        first_conv_weights = net_print_state_dict["module.Conv1.conv.0.weight"]
        net_print_state_dict["module.Conv1.conv.0.weight"] = torch.cat((
                                    first_conv_weights, first_conv_weights), dim=1)

        last_conv_weights = net_print_state_dict["module.Conv_1x1.weight"]

        if config.num_join_fingers == 2:
            net_print_state_dict["module.Conv_1x1.weight"] = torch.cat((
                                last_conv_weights, last_conv_weights), dim=0)

        elif config.num_join_fingers == 3:
            net_print_state_dict["module.Conv_1x1.weight"] = torch.cat((
                last_conv_weights, last_conv_weights, last_conv_weights), dim=0)

        elif config.num_join_fingers == 4:
            net_print_state_dict["module.Conv_1x1.weight"] = torch.cat((
            last_conv_weights, last_conv_weights, last_conv_weights, last_conv_weights), dim=0)

        last_bias = net_print_state_dict["module.Conv_1x1.bias"]
        if config.num_join_fingers == 2:
            net_print_state_dict["module.Conv_1x1.bias"] = torch.cat((
                                                last_bias, last_bias), dim=0)
        elif config.num_join_fingers == 3:
            net_print_state_dict["module.Conv_1x1.bias"] = torch.cat((
                                    last_bias, last_bias, last_bias), dim=0)

        elif config.num_join_fingers == 4:
            net_print_state_dict["module.Conv_1x1.bias"] = torch.cat((
                        last_bias, last_bias, last_bias, last_bias), dim=0)


        net_print.load_state_dict(net_print_state_dict) 
        print("A2 here .............................")


    elif network_arch == "A3":
        pass

    if is_disc_load:
        # disc_photo
        disc_photo_state_dict = checkpoint["disc_photo"]
        first_conv_weights = disc_photo_state_dict["module.shared_conv.0.weight"]

        if config.num_join_fingers == 2:
            disc_photo_state_dict["module.shared_conv.0.weight"] = torch.cat((
                                    first_conv_weights, first_conv_weights), dim=1)

        elif config.num_join_fingers == 3:
            disc_photo_state_dict["module.shared_conv.0.weight"] = torch.cat((
                first_conv_weights, first_conv_weights, first_conv_weights), dim=1)

        elif config.num_join_fingers == 4:
            disc_photo_state_dict["module.shared_conv.0.weight"] = torch.cat((
            first_conv_weights, first_conv_weights, first_conv_weights, first_conv_weights), dim=1)

        disc_photo.load_state_dict(disc_photo_state_dict)

        # disc_print
        disc_print_state_dict = checkpoint["disc_print"]
        first_conv_weights = disc_print_state_dict["module.shared_conv.0.weight"]

        if config.num_join_fingers == 2:
            disc_print_state_dict["module.shared_conv.0.weight"] = torch.cat((
                                    first_conv_weights, first_conv_weights), dim=1)

        elif config.num_join_fingers == 3:
            disc_print_state_dict["module.shared_conv.0.weight"] = torch.cat((
                first_conv_weights, first_conv_weights, first_conv_weights), dim=1)

        elif config.num_join_fingers == 4:
            disc_print_state_dict["module.shared_conv.0.weight"] = torch.cat((
            first_conv_weights, first_conv_weights, first_conv_weights, first_conv_weights), dim=1)

        disc_print.load_state_dict(disc_print_state_dict)



def load_single_model_for_finetuing(net_print):
    save_w_dir = config.save_w_dir


    if config.is_convert_one_to_many == False: 
        loaded_model_file = os.path.join(save_w_dir, "best_model_000.pth")
        checkpoint = torch.load(loaded_model_file)
        net_print.load_state_dict(checkpoint["net_print"])
        


class FingerIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(256, 1))

    def forward(self, embd1, embd2):
        dis = torch.abs(embd1 - embd2)
        return self.model(dis)


class WeightedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(10, 1))

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    model = mapper.Mapper(pre_network="resnet18", 
                            img_dim = 1, out_dim=config.feature_dim) 

    print(model.backbone)
    summary(model, input_size=(1, 256, 256), device="cpu")