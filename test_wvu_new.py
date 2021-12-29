import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import config 
from model import *
import random 
import os
import utils_wvu_new
from PIL import Image 


## Special Class for Fixed Test set and results
class WVUNewVerifierForTest(Dataset):
    def __init__(self):
        super().__init__()

        print("test data")
        if config.num_join_fingers == 1:
            self.dict_photo, self.dict_print = utils_wvu_new.get_img_dict(
            config.test_photo_dir, config.test_print_dir)

        elif config.num_join_fingers == 2:
            self.dict_photo, self.dict_print = utils_wvu_new.get_two_img_dict(
            config.test_photo_dir, config.test_print_dir, config.fnums)

        self.num_photo_samples = len(self.dict_photo)
        print("Number of Fingers:", config.num_join_fingers)
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
            ph_f = Image.open(photo_image).convert("L")
            pr_f = Image.open((self.dict_print[class_id])).convert("L")

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
            WVUNewVerifierForTest(),
            batch_size=config.batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers= 4  
        )

        self.net_photo, self.net_print = get_model(config.w_name)

        if config.is_load_model:
            print("loading all models")
            if config.is_finetune == True and config.is_convert_one_to_many == True:
                w_dir = config.old_weights_one_dir

            elif config.is_finetune == True and config.is_convert_one_to_many == False:
                if config.num_join_fingers == 1: w_dir = config.old_weights_one_dir
                elif config.num_join_fingers == 2: w_dir = config.old_weights_two_dir

            elif config.is_finetune == False:
                if config.num_join_fingers == 1: w_dir = config.new_weights_one_dir
                elif config.num_join_fingers == 2: w_dir = config.new_weights_two_dir

            all_models = sorted(os.listdir(w_dir), 
                    key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            for model in all_models:
                model_file = os.path.join(w_dir, model)
                checkpoint = torch.load(model_file)

                #loads weights trained of one finger to the weights for multi-finger
                if (config.is_finetune == True and 
                    config.is_convert_one_to_many == True and
                    config.num_join_fingers == 2 and 
                    config.join_type == "channel"):

                    compatible_load(checkpoint, self.net_photo, self.net_print, disc_photo = None, 
                                 disc_print = None, w_name = config.w_name, is_disc_load=False)    

                else:            
                    self.net_photo.load_state_dict(checkpoint["net_photo"])
                    self.net_print.load_state_dict(checkpoint["net_print"])

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
            
        utils_wvu_new.calculate_scores(ls_labels, ls_sq_dist)
        #self.plot_roc()

    # plotting roc curve 
    def plot_roc(self):
        pass 

if __name__ == "__main__":
    v = VerifTest()

    """
    vt = WVUNewVerifierForTest()
    img1, img2, label = vt.__getitem__(13)
    utils_wvu_new.plot_tensors([img1, img2], title=label)
    """