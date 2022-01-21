import  random  
import numpy as np 
from PIL import Image
import torch  
from torch.utils.data import Dataset 
from torchvision import transforms 
import config
import torchvision.transforms.functional as TF

# load dataset: wvu_new, wvu_old
if config.dataset_name == "wvu_old":
    from utils_wvu_old import get_img_dict, get_multiple_img_dict

elif config.dataset_name == "wvu_new":
    from utils_wvu_new import get_img_dict, get_multiple_img_dict


class WVUFingerDataset(Dataset):
    def __init__(self, train = True):
        super().__init__()
        self.train = train 

        if (self.train == True): 
            print("trainning data loading")
            if config.num_join_fingers == 1 and config.is_one_fid == False:
                self.dict_photo, self.dict_print = get_img_dict(
                    config.train_photo_dir, config.train_print_dir)
            
            elif config.num_join_fingers >= 2 or config.is_one_fid == True:
                if (config.is_all_pairs == True): ls_fnums = config.all_fnums
                else: ls_fnums = config.fnums

                self.dict_photo, self.dict_print = get_multiple_img_dict(
                    config.train_photo_dir, config.train_print_dir, ls_fnums)

        elif(self.train == False):
            print("\nvalidation data loading ...")
            if config.num_join_fingers == 1 and config.is_one_fid == False:
                self.dict_photo, self.dict_print = get_img_dict(
                    config.test_photo_dir, config.test_print_dir)
            
            elif config.num_join_fingers >= 2 or config.is_one_fid == True:
                self.dict_photo, self.dict_print = get_multiple_img_dict(
                    config.test_photo_dir, config.test_print_dir, config.fnums)

        self.num_photo_samples = len(self.dict_photo)


    def trans(self, photo, print, train=True):
        fill_photo = (255,)
        fill_print = (255,)

        if config.dataset_name == "wvu_new" and config.photo_type == "vfp":
            fill_photo = (0, )

        if train == True:
            # Resize
            resize = transforms.Resize(size=(286, 286))
            photo = resize(photo)
            print = resize(print)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                            photo, output_size=(config.img_size, config.img_size))
            photo = TF.crop(photo, i, j, h, w)
            print = TF.crop(print, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                photo = TF.hflip(photo)
                print = TF.hflip(print)

            # Random rotation
            angle = transforms.RandomRotation.get_params(degrees=(0, 10))
            photo = TF.rotate(photo, angle, fill=fill_photo)
            print = TF.rotate(print, angle, fill=fill_print)

        elif train == False:
            # Resize
            resize = transforms.Resize(size=(config.img_size, config.img_size))
            photo = resize(photo)
            print = resize(print)

        # Transform to tensor
        photo = TF.to_tensor(photo)
        print = TF.to_tensor(print)

        # normalize
        normalize = transforms.Normalize(mean = [0.5], std = [0.5])
        photo = normalize(photo)
        print = normalize(print)

        return photo, print


    def __len__(self):
        if self.train: 
            return self.num_photo_samples * config.num_imposter
        else:
            return self.num_photo_samples * 2


    def __getitem__(self, index):
        if self.train: 
            num = index % config.num_imposter 
            finger_id, photo_image = self.dict_photo[index // config.num_imposter]

        else:
            num = index % 2
            finger_id, photo_image = self.dict_photo[index // 2]
       
        if num == 0:
            same_class = True
            class_id = finger_id

        else: 
            same_class = False 
            class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)]

            while finger_id == class_id:
                class_id = list(self.dict_print.keys())[random.randint(0, 
                                            len(self.dict_print) - 1)]  

        # single finger
        if config.num_join_fingers == 1:
            ph_f = Image.open(photo_image).convert("L")
            pr_f = Image.open((self.dict_print[class_id])[0]).convert("L")

            img1, img2 = self.trans(ph_f, pr_f, self.train)

        # two fingers
        elif config.num_join_fingers >= 2:
            print_image = self.dict_print[class_id]

            ph_f1 = Image.open(photo_image[0]).convert("L") 
            ph_f2 = Image.open(photo_image[1]).convert("L")
            pr_f1 = Image.open(print_image[0]).convert("L") 
            pr_f2 = Image.open(print_image[1]).convert("L")

            ph_f1, pr_f1 = self.trans(ph_f1, pr_f1, self.train)
            ph_f2, pr_f2 = self.trans(ph_f2, pr_f2, self.train)

            if config.num_join_fingers == 2:
                img1 = torch.cat([ph_f1, ph_f2], dim=0)
                img2 = torch.cat([pr_f1, pr_f2], dim=0)

            else:
                ph_f3 = Image.open(photo_image[2]).convert("L")
                pr_f3 = Image.open(print_image[2]).convert("L")
                ph_f3, pr_f3 = self.trans(ph_f3, pr_f3, self.train)

                if config.num_join_fingers == 3:
                    img1 = torch.cat([ph_f1, ph_f2, ph_f3], dim=0)
                    img2 = torch.cat([pr_f1, pr_f2, pr_f3], dim=0)
                
                else:
                    ph_f4 = Image.open(photo_image[3]).convert("L")
                    pr_f4 = Image.open(print_image[3]).convert("L")
                    ph_f4, pr_f4 = self.trans(ph_f4, pr_f4, self.train)

                    if config.num_join_fingers == 4:
                        img1 = torch.cat([ph_f1, ph_f2, ph_f3, ph_f4], dim=0)
                        img2 = torch.cat([pr_f1, pr_f2, pr_f3, pr_f4], dim=0)
                            
        return img1, img2, same_class


    def find_mean_std(self):
        #ph_mean = [0.70751] ph_std = [0.22236] 
        #pr_mean = [0.63939] pr_std = [0.2373]
        ph_img_ls = [self.dict_photo[i][0] for i in range(self.num_photo_samples)]
        pil_img = [Image.open(self.dict_print[ph_img][0]).convert("L") for ph_img in ph_img_ls]

        images = np.stack([np.asarray(pil_img[0])/255.0 for img in pil_img])
        images = images.reshape(images.shape[0], -1)

        mean_val = images.mean(axis=1)
        std_val = images.std(axis=1)

        print(mean_val.mean())
        print(std_val.mean())
        return mean_val, std_val


if __name__ == "__main__":
    vt = WVUFingerDataset()
    img1, img2, same_class = vt.__getitem__(90)
    print(img1.shape)
    #title = ("genuine pair" if same_class else "imposter pair")
    #utils_wvu_new.plot_tensors([img1, img2], title) 
