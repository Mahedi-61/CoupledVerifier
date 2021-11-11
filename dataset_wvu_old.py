import  random  
import numpy as np 
import os 
from PIL import Image
import torch  
from torch.utils.data import Dataset 
from torchvision import transforms 
import config
import utils_wvu_old 

class WVUOldVerifier(Dataset):
    def __init__(self, train = True):
        super().__init__()

        if (train == True): 
            print("trainning data loading")
            if config.num_join_fingers == 1:
                self.dict_photo, self.dict_print = utils_wvu_old.get_img_dict(
                    config.train_photo_dir, config.train_print_dir)
            
            elif config.num_join_fingers == 2:
                self.dict_photo, self.dict_print = utils_wvu_old.get_two_img_dict(
                    config.train_photo_dir, config.train_print_dir, config.fnums)

        elif(train == False):
            print("validation data loading")
            if config.num_join_fingers == 1:
                self.dict_photo, self.dict_print = utils_wvu_old.get_img_dict(
                    config.test_photo_dir, config.test_print_dir)
            
            elif config.num_join_fingers == 2:
                self.dict_photo, self.dict_print = utils_wvu_old.get_two_img_dict(
                    config.test_photo_dir, config.test_print_dir, config.fnums)

        self.num_photo_samples = len(self.dict_photo)

        mean = [0.5] 
        std = [0.5] 
        fill_white = (255,)

        self.train_trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            #transforms.RandomAffine(3),
            #transforms.Pad(16),
            #transforms.RandomCrop(256),
            #transforms.ColorJitter(brightness=0.2),
            transforms.RandomRotation(degrees=(0, 20), fill=fill_white),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.test_trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.trans = (self.train_trans if train else self.test_trans)


    def __len__(self):
        return self.num_photo_samples * config.num_imposter


    def __getitem__(self, index):
        num = index % config.num_imposter 
        if num == 0: #or num == 5:
            same_class = True
        else: 
            same_class = False 

        finger_id, photo_image = self.dict_photo[index // config.num_imposter]

        # genuine pair 
        if same_class:
            class_id = finger_id

        # imposter pair
        else:
            class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)]

            while finger_id == class_id:
                class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)]  


        # single finger
        if config.num_join_fingers == 1:
            num_print_images = len(self.dict_print[class_id])
            pos_print =  random.randint(0, num_print_images-1) 

            ph_f = Image.open(photo_image).convert("L")
            pr_f = Image.open((self.dict_print[class_id])[pos_print]).convert("L")

            img1 = self.trans(ph_f)
            img2 = self.trans(pr_f)

        # two fingers
        elif config.num_join_fingers == 2:
            num = index % config.num_imposter

            # take another finger 
            class_id2 = list(self.dict_print.keys())[random.randint(0, 
                                len(self.dict_print) - 1)]

            while class_id == class_id2:
                class_id2 = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)]  

            # Making two genuine pairs
            if num == 0:
                print_image = self.dict_print[class_id]
            
            """
            elif num == 5:
                ph = [photo_image[1], photo_image[0]]
                photo_image = ph

                img = self.dict_print[class_id]
                print_image = [img[1], img[0]]
            """

            # Making imposter pairs
            # one same print
            if num == 1:
                img1, img2 = self.dict_print[finger_id], self.dict_print[class_id]
                print_image = [img1[0], img2[1]]

            elif num == 2:
                img1, img2 = self.dict_print[finger_id], self.dict_print[class_id]
                print_image = [img2[0], img1[1]]

            # no same print
            elif num == 3:
                img = self.dict_print[class_id]
                print_image = [img[0], img[1]]

            """
            elif num == 4:
                img1, img2 = self.dict_print[class_id], self.dict_print[class_id2]
                print_image = [img1[0], img2[1]]

            # opposite
            elif num == 6:
                img = self.dict_print[finger_id]
                print_image = [img[1], img[0]]
            """


            ph_f1 = self.trans(Image.open(photo_image[0]).convert("L")) 
            ph_f2 = self.trans(Image.open(photo_image[1]).convert("L"))
            pr_f1 = self.trans(Image.open(print_image[0]).convert("L")) 
            pr_f2 = self.trans(Image.open(print_image[1]).convert("L"))

            if config.join_type == "concat":
                img1 = torch.cat([ph_f1, ph_f2], dim=2)
                img2 = torch.cat([pr_f1, pr_f2], dim=2)

            elif config.join_type == "channel": 
                img1 = torch.cat([ph_f1, ph_f2], dim=0)
                img2 = torch.cat([pr_f1, pr_f2], dim=0)

        return img1, img2, same_class


if __name__ == "__main__":
    data = WVUOldVerifier()
    img1, img2, same_class = data.__getitem__(31)

    title = ("genuine pair" if same_class else "imposter pair")
    utils_wvu_old.plot_tensors(img2, title) 