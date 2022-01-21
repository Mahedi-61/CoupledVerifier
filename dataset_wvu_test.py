import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import config 
from PIL import Image 
from collections import OrderedDict
import random 

if config.dataset_name == "wvu_old":
    from utils_wvu_old import *

elif config.dataset_name == "wvu_new":
    from utils_wvu_new import *


## Special Class for Fixed Test set and results
class WVUFingerDatasetForTest(Dataset):
    def __init__(self, is_fixed = True):
        super().__init__()

        print("test data loading ...")
        if config.num_join_fingers == 1 and config.is_one_fid == False:
            self.dict_photo, self.dict_print = get_img_dict(
            config.test_photo_dir, config.test_print_dir)

        elif config.num_join_fingers >= 2 or config.is_one_fid == True:
            self.dict_photo, self.dict_print = get_multiple_img_dict(
            config.test_photo_dir, config.test_print_dir, config.fnums)

        self.is_fixed = is_fixed 
        print("Dataset: ", config.dataset_name)
        print("experiment type: test")
        print("Number of Fingers IDs: ", len(self.dict_photo))
        print("Number of Fingers:", config.num_join_fingers)
        print("Network Arch:", config.w_name.split("_")[-1])
        print("loading models from: ", config.w_name)
        print("Number of imposter pair: ", config.num_pair_test)

        self.test_trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.dict_print = OrderedDict(sorted(self.dict_print.items()))
        self.new_imposter_pos = 0


    def __len__(self):
        return  len(self.dict_photo) * config.num_pair_test 

    def __getitem__(self, index):
        num = index % config.num_pair_test 
        id_position = (index // config.num_pair_test) 

        # genuine pair 
        if (num == 0):
            finger_id, photo_image = self.dict_photo[index // config.num_pair_test]
            class_id = finger_id
            same_class = True
            #print("Photo: " + str(id_position) + "| Print: " + str(id_position) + " genuine")
            self.new_imposter_pos = 0

        # imposter pair
        elif (num > 0):
            finger_id, photo_image = self.dict_photo[index // config.num_pair_test]
            same_class = False 

            # fixed test set
            if self.is_fixed:
                if (id_position + num <  len(self.dict_photo)):
                    class_id = list(self.dict_print.keys())[id_position + num]
                    #print("Photo: " + str(id_position) + "| Print: " + str(id_position + num))

                else:
                    class_id = list(self.dict_print.keys())[self.new_imposter_pos]
                    #print("Photo: " + str(id_position) + "| Print: " + str(self.new_imposter_pos))
                    self.new_imposter_pos += 1

            # random test set
            else:
                class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)]

                while finger_id == class_id:
                    class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)] 

        if config.num_join_fingers == 1:
            ph_f = Image.open(photo_image).convert("L")
            pr_f = Image.open((self.dict_print[class_id])[0]).convert("L")

            img1 = self.test_trans(ph_f)
            img2 = self.test_trans(pr_f)

        elif config.num_join_fingers >= 2:
            print_image = self.dict_print[class_id]

            ph_f1 = self.test_trans(Image.open(photo_image[0]).convert("L")) 
            ph_f2 = self.test_trans(Image.open(photo_image[1]).convert("L"))
            pr_f1 = self.test_trans(Image.open(print_image[0]).convert("L")) 
            pr_f2 = self.test_trans(Image.open(print_image[1]).convert("L"))

            if config.num_join_fingers == 2:
                img1 = torch.cat([ph_f1, ph_f2], dim=0)
                img2 = torch.cat([pr_f1, pr_f2], dim=0)

            else:
                ph_f3 = self.test_trans(Image.open(photo_image[2]).convert("L"))
                pr_f3 = self.test_trans(Image.open(print_image[2]).convert("L"))

                if config.num_join_fingers == 3:
                    img1 = torch.cat([ph_f1, ph_f2, ph_f3], dim=0)
                    img2 = torch.cat([pr_f1, pr_f2, pr_f3], dim=0)

                else:
                    ph_f4 = self.test_trans(Image.open(photo_image[3]).convert("L"))
                    pr_f4 = self.test_trans(Image.open(print_image[3]).convert("L"))

                    if config.num_join_fingers == 4:
                        img1 = torch.cat([ph_f1, ph_f2, ph_f3, ph_f4], dim=0)
                        img2 = torch.cat([pr_f1, pr_f2, pr_f3, pr_f4], dim=0)

        return img1, img2, same_class
