import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import config 
from model import *
import random 
import os
import utils_wvu_old
from PIL import Image 


## Special Class for Fixed Test set and results
class WVUOldVerifierForTest(Dataset):
    def __init__(self):
        super().__init__()

        print("test data")
        if config.num_join_fingers == 1:
            self.dict_photo, self.dict_print = utils_wvu_old.get_img_dict(
            config.test_photo_dir, config.test_print_dir)

        elif config.num_join_fingers == 2:
            self.dict_photo, self.dict_print = utils_wvu_old.get_two_img_dict(
            config.test_photo_dir, config.test_print_dir, config.fnums)

        self.num_photo_samples = len(self.dict_photo)
        print("Join Type: ", config.join_type)

        mean = [0.5]
        std = [0.5]

        self.test_trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


    def __len__(self):
        return self.num_photo_samples * config.num_pair_test 

    def __getitem__(self, index):
        num = index % config.num_pair_test 
        # genuine pair 
        if (num == 0):
            finger_id, photo_image = self.dict_photo[index // config.num_pair_test]
            class_id = finger_id
            same_class = True

        # imposter pair
        elif (num > 0):
            finger_id, photo_image = self.dict_photo[index // config.num_pair_test]
            same_class = False 

            class_id = list(self.dict_print.keys())[random.randint(0, 
                                        len(self.dict_print) - 1)]

            while finger_id == class_id:
                class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)] 

        if config.num_join_fingers == 1:
            num_print_images = len(self.dict_print[class_id])
            pos_print =  random.randint(0, num_print_images-1) 

            ph_f = Image.open(photo_image).convert("L")
            pr_f = Image.open((self.dict_print[class_id])[pos_print]).convert("L")

            img1 = self.test_trans(ph_f)
            img2 = self.test_trans(pr_f)

        elif config.num_join_fingers == 2:
            print_image = self.dict_print[class_id]

            ph_f1 = self.test_trans(Image.open(photo_image[0]).convert("L")) 
            ph_f2 = self.test_trans(Image.open(photo_image[1]).convert("L"))
            pr_f1 = self.test_trans(Image.open(print_image[0]).convert("L")) 
            pr_f2 = self.test_trans(Image.open(print_image[1]).convert("L"))

            if config.join_type == "concat":
                img1 = torch.cat([ph_f1, ph_f2], dim=2)
                img2 = torch.cat([pr_f1, pr_f2], dim=2)

            elif config.join_type == "channel": 
                img1 = torch.cat([ph_f1, ph_f2], dim=0)
                img2 = torch.cat([pr_f1, pr_f2], dim=0)

        return img1, img2, same_class


class VerifTest:
    def __init__(self):
        print("loading dataset ...")
        self.test_loader = DataLoader(
            WVUOldVerifierForTest(),
            batch_size=config.batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers= 4  
        )

        self.net_photo = Mapper(pre_network="resnet18", 
                            join_type = config.join_type,
                            img_dim = config.img_dim, 
                            out_dim=config.feature_dim) 
        self.net_photo.to(config.device)

        self.net_print = Mapper(pre_network="resnet18", 
                            join_type = config.join_type,
                            img_dim = config.img_dim, 
                            out_dim=config.feature_dim) 
        self.net_print.to(config.device)

        if config.multi_gpus:
            self.net_photo = torch.nn.DataParallel(self.net_photo, device_ids=[0, 1])
            self.net_print = torch.nn.DataParallel(self.net_print, device_ids=[0, 1])

        if config.is_load_model:
            print("loading all models")
            all_models = sorted(os.listdir(config.weights_one_dir), 
                    key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            for model in all_models:
                model_file = os.path.join(config.weights_one_dir, model)
                checkpoint = torch.load(model_file)

                # photo
                net_photo_state_dict = checkpoint["net_photo"]
                first_conv_weights = net_photo_state_dict["module.backbone.0.weight"]
                net_photo_state_dict["module.backbone.0.weight"] = torch.cat((first_conv_weights, 
                                                                    first_conv_weights), dim=1)

                last_conv_weights = net_photo_state_dict["module.decoder.25.weight"]
                net_photo_state_dict["module.decoder.25.weight"] = torch.cat((last_conv_weights, 
                                                                    last_conv_weights), dim=0)

                last_bias = net_photo_state_dict["module.decoder.25.bias"]
                net_photo_state_dict["module.decoder.25.bias"] = torch.cat((last_bias, last_bias), dim=0)
                self.net_photo.load_state_dict(net_photo_state_dict)

                # print
                net_print_state_dict = checkpoint["net_print"]
                first_conv_weights = net_print_state_dict["module.backbone.0.weight"]
                net_print_state_dict["module.backbone.0.weight"] = torch.cat((first_conv_weights, 
                                                                    first_conv_weights), dim=1)

                last_conv_weights = net_print_state_dict["module.decoder.25.weight"]
                net_print_state_dict["module.decoder.25.weight"] = torch.cat((last_conv_weights, 
                                                                    last_conv_weights), dim=0)

                last_bias = net_print_state_dict["module.decoder.25.bias"]
                net_print_state_dict["module.decoder.25.bias"] = torch.cat((last_bias, last_bias), dim=0)
                self.net_print.load_state_dict(net_print_state_dict)
                
                #self.net_photo.load_state_dict(checkpoint["net_photo"])
                #self.net_print.load_state_dict(checkpoint["net_print"])

                print(model_file)
                self.test()


    def test(self):
        self.net_photo.eval()
        self.net_print.eval()

        ls_sq_dist = []
        ls_labels = []
            
        for img_photo, img_print, label in self.test_loader:
            label = label.type(torch.float)

            img_photo = img_photo.to(config.device)
            img_print = img_print.to(config.device)
            label = label.to(config.device)

            _, embd_photo = self.net_photo(img_photo)
            _, embd_print = self.net_print(img_print)

            dist_sq = torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1) #torch.sqrt()
            ls_sq_dist.append(dist_sq.data)
            ls_labels.append((1 - label).data)
            
        utils_wvu_old.calculate_scores(ls_labels, ls_sq_dist)
        #self.plot_roc()

    # plotting roc curve 
    def plot_roc(self):
        pass 

if __name__ == "__main__":
    v = VerifTest()