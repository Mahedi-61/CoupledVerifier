from operator import truediv
import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import config 
from PIL import Image 
from collections import OrderedDict
import random 
import torchvision.transforms.functional as TF


if config.dataset_name == "wvu_old":
    from utils_wvu_old import *

elif config.dataset_name == "wvu_new":
    from utils_wvu_new import *


## Special Class for Fixed Test set and results
class WVUFingerDatasetForTest(Dataset):
    def __init__(self, test_aug_id, is_fixed = True):
        super().__init__()

        if config.num_join_fingers == 1 and config.is_one_fid == False:
            self.dict_photo, self.dict_print = get_img_dict(
            config.test_photo_dir, config.test_print_dir)

        elif config.num_join_fingers >= 2 or config.is_one_fid == True:
            self.dict_photo, self.dict_print = get_multiple_img_dict(
            config.test_photo_dir, config.test_print_dir, config.fnums)

        self.is_fixed = is_fixed 

        if config.is_display == True:
            print("Dataset: ", config.dataset_name)
            print("experiment type: test")
            print("Number of Fingers IDs: ", len(self.dict_photo))
            print("Number of Fingers:", config.num_join_fingers)
            print("Network Arch:", config.w_name.split("_")[-1])
            print("loading models from: ", config.w_name)
            print("Number of imposter pair: ", config.num_pair_test)
            config.is_display = False 

        self.dict_print = OrderedDict(sorted(self.dict_print.items()))        
        self.new_imposter_pos = 0
        self.test_aug_id = test_aug_id


    def test_trans(self, photo, print):
        """
         0: only (256, 256) resized images
         1: (256, 256) h-flip resized images
         2: (286, 286) resized (256, 256) center crop images
        """

        fill_photo = (255,)
        fill_print = (255,)

        if config.dataset_name == "wvu_new" and config.photo_type == "vfp":
            fill_photo = (0, )

        # Resize
        if self.test_aug_id <= 1:
            resize = transforms.Resize(size=(config.img_size, config.img_size))
            photo = resize(photo)
            print = resize(print)
        
        else:
            resize = transforms.Resize(size=(286, 286))
            photo = resize(photo)
            print = resize(print)

        # Random horizontal flipping
        if self.test_aug_id == 1 or self.test_aug_id == 3:
            photo = TF.hflip(photo)
            print = TF.hflip(print)

        """
        if self.test_aug_id == 2 or self.test_aug_id == 3: 
            # center crop
            i, j = (15, 15)

            h=config.img_size
            w=config.img_size

            photo = TF.crop(photo, i, j, h, w)
            print = TF.crop(print, i, j, h, w)
        """

        if self.test_aug_id == 2 and self.test_aug_id == 3:
            if self.test_aug_id == 2: angle = 10
            elif self.test_aug_id == 3: angle = -10
            photo = TF.rotate(photo, angle, fill=fill_photo)
            print = TF.rotate(print, angle, fill=fill_print)
        

        # Transform to tensor
        photo = TF.to_tensor(photo)
        print = TF.to_tensor(print)

        # normalize
        normalize = transforms.Normalize(mean = [0.5], std = [0.5])
        photo = normalize(photo)
        print = normalize(print)

        return photo, print


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
            img1, img2 = self.test_trans(ph_f, pr_f)

        elif config.num_join_fingers >= 2:
            print_image = self.dict_print[class_id]

            ph_f1 = Image.open(photo_image[0]).convert("L") 
            ph_f2 = Image.open(photo_image[1]).convert("L")
            pr_f1 = Image.open(print_image[0]).convert("L") 
            pr_f2 = Image.open(print_image[1]).convert("L")

            ph_f1, pr_f1 = self.test_trans(ph_f1, pr_f1)
            ph_f2, pr_f2 = self.test_trans(ph_f2, pr_f2)

            if config.num_join_fingers == 2:
                img1 = torch.cat([ph_f1, ph_f2], dim=0)
                img2 = torch.cat([pr_f1, pr_f2], dim=0)

            else:
                ph_f3 = Image.open(photo_image[2]).convert("L")
                pr_f3 = Image.open(print_image[2]).convert("L")
                ph_f3, pr_f3 = self.test_trans(ph_f3, pr_f3)

                if config.num_join_fingers == 3:
                    img1 = torch.cat([ph_f1, ph_f2, ph_f3], dim=0)
                    img2 = torch.cat([pr_f1, pr_f2, pr_f3], dim=0)

                else:
                    ph_f4 = Image.open(photo_image[3]).convert("L")
                    pr_f4 = Image.open(print_image[3]).convert("L")
                    ph_f4, pr_f4 = self.test_trans(ph_f4, pr_f4)

                    if config.num_join_fingers == 4:
                        img1 = torch.cat([ph_f1, ph_f2, ph_f3, ph_f4], dim=0)
                        img2 = torch.cat([pr_f1, pr_f2, pr_f3, pr_f4], dim=0)

        return img1, img2, same_class



if __name__ == "__main__":
    db = WVUFingerDatasetForTest(is_fixed=True)