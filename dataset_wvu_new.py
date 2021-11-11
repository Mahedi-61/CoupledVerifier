import  random
from matplotlib.pyplot import fill  
import numpy as np 
import os 
from torch.utils.data import Dataset 
from torchvision import transforms 
import config
import utils_wvu_new 
import torch 

class WVUNewVerifierOne(Dataset):
    def __init__(self, train = True):
        super().__init__()
        self.train = train
        if self.train == True:
            print("trainning phase ...")
            if config.num_join_fingers == 1:
                self.dict_photo, self.dict_print = utils_wvu_new.get_one_img_dict(
                           config.train_photo_dir, config.train_print_dir)

            elif config.num_join_fingers == 2:
                self.dict_photo, self.dict_print = utils_wvu_new.get_two_img_dict(
                        config.train_photo_dir, config.train_print_dir, config.fnums)


        elif self.train == False:
            print("testing phase ...")
            if config.num_join_fingers == 1:
                self.dict_photo, self.dict_print = utils_wvu_new.get_one_img_dict(
                           config.test_photo_dir, config.test_print_dir)

            elif config.num_join_fingers == 2:
                self.dict_photo, self.dict_print = utils_wvu_new.get_two_img_dict(
                        config.test_photo_dir, config.test_print_dir, config.fnums)                

        self.num_photo_samples = len(self.dict_photo)
        mean = [0.5] 
        std = [0.5]
        fill_white = (255, )

        self.train_trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            #transforms.RandomAffine(3),
            #transforms.Pad(16),
            #transforms.RandomCrop(256),
            #transforms.ColorJitter(brightness=0.2),
            transforms.RandomRotation(20, fill=fill_white),
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
        if index % config.num_imposter == 0: same_class = True
        else: same_class = False 
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

        if config.num_join_fingers == 1:
            img1 = self.trans(photo_image)
            img2 = self.trans(self.dict_print[class_id])

        elif config.num_join_fingers == 2:
            print_image = self.dict_print[class_id]
            ph_f1, ph_f2 = self.trans(photo_image[0]), self.trans(photo_image[1])
            pr_f1, pr_f2  = self.trans(print_image[0]), self.trans(print_image[1])

            if config.join_type == "concat":
                img1 = torch.cat([ph_f1, ph_f2], dim=2)
                img2 = torch.cat([pr_f1, pr_f2], dim=2)

            elif config.join_type == "channel": 
                img1 = torch.cat([ph_f1, ph_f2], dim=0)
                img2 = torch.cat([pr_f1, pr_f2], dim=0)

        return img1, img2, same_class


if __name__ == "__main__":
    data = WVUNewVerifierOne(train = config.is_train)
    img1, img2, same_class = data.__getitem__(90)
    print(img1.shape)
    #title = ("genuine pair" if same_class else "imposter pair")
    #utils_wvu_new.plot_tensors([img1, img2], title) 