import  random  
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset 
from torchvision import transforms 
import config
import torchvision.transforms.functional as TF
import os 
import utils_wvu_old


class PretrainFingerDataset(Dataset):
    def __init__(self, train = True):
        super().__init__()
        self.train = train 

        if (self.train == True): 
            if config.num_join_fingers == 1:
                self.dict_print = get_img_dict(config.train_print_dir)
            

        elif(self.train == False):
            print("\nvalidation data loading ...")
            if config.num_join_fingers == 1:
                self.dict_print = get_img_dict(config.test_print_dir)
            
        self.num_print_samples = len(self.dict_print)


    def trans(self, print_img1, print_img2, train=True):
        fill_print = (255,)

        if train == True:
            # Resize
            if random.random() > 0.5:
                resize = transforms.Resize(size=(286, 286))
                print_img1 = resize(print_img1)
                print_img2 = resize(print_img2)

                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(print_img1, 
                                        output_size=(config.img_size, config.img_size))
                print_img1 = TF.crop(print_img1, i, j, h, w)
                print_img2 = TF.crop(print_img2, i, j, h, w)

            else:
                resize = transforms.Resize(size=(config.img_size, config.img_size))
                print_img1 = resize(print_img1)
                print_img2 = resize(print_img2)

            # Random horizontal flipping
            if random.random() > 0.5:
                print_img1 = TF.hflip(print_img1)
                print_img2 = TF.hflip(print_img2)


            # Random rotation
            angle = transforms.RandomRotation.get_params(degrees=(-10, 10))
            print_img1 = TF.rotate(print_img1, angle, fill=fill_print)
            print_img2 = TF.rotate(print_img2, angle, fill=fill_print)

        elif train == False:
            # Resize
            resize = transforms.Resize(size=(config.img_size, config.img_size))
            print_img1 = resize(print_img1)
            print_img2 = resize(print_img2)

        # Transform to tensor
        print_img1 = TF.to_tensor(print_img1)
        print_img2 = TF.to_tensor(print_img2)

        # normalize
        normalize = transforms.Normalize(mean = [0.5], std = [0.5])
        print_img1 = normalize(print_img1)
        print_img2 = normalize(print_img2)

        return print_img1, print_img2


    def __len__(self):
        if self.train: 
            return self.num_print_samples * 2


    def __getitem__(self, index):
        if self.train: 
            num = index % 2
            finger_id, print_img1_dir = self.dict_print[index // 2]
       
        if num == 0:
            same_class = True
            print_img2_dir = print_img1_dir 

        else: 
            same_class = False 
            class_id, print_img2_dir = self.dict_print[random.randint(0, len(self.dict_print) - 1)]

            while finger_id == class_id:
                class_id, print_img2_dir = self.dict_print[random.randint(0, len(self.dict_print) - 1)]  

        # single finger
        if config.num_join_fingers == 1:

            ls_img1 = os.listdir(print_img1_dir)
            if len(ls_img1) == 1: pos_img1 = 0
            else: pos_img1 = random.randint(0, len(ls_img1) - 1)

            img1_path = os.path.join(print_img1_dir, ls_img1[pos_img1])
            print_img1 = Image.open(img1_path).convert("L")

            ls_img2 = os.listdir(print_img2_dir)
            if len(ls_img2) == 1: pos_img2 = 0
            else: 
                pos_img2 = random.randint(0, len(ls_img2) - 1)
                """
                if same_class == True:
                    while pos_img1 == pos_img2:
                        pos_img2 = random.randint(0, len(ls_img2) - 1)
                """

            img2_path = os.path.join(print_img2_dir, ls_img2[pos_img2])
            print_img2 = Image.open(img2_path).convert("L")

            img1, img2 = self.trans(print_img1, print_img2, self.train)     

        return img1, img2, same_class


    def find_mean_std(self):
        #ph_mean = [0.70751] ph_std = [0.22236] 
        #pr_mean = [0.63939] pr_std = [0.2373]
        ph_img_ls = [self.dict_print[i][0] for i in range(self.num_print_samples)]
        pil_img = [Image.open(self.dict_print[ph_img][0]).convert("L") for ph_img in ph_img_ls]

        images = np.stack([np.asarray(pil_img[0])/255.0 for img in pil_img])
        images = images.reshape(images.shape[0], -1)

        mean_val = images.mean(axis=1)
        std_val = images.std(axis=1)

        print(mean_val.mean())
        print(std_val.mean())
        return mean_val, std_val



class PretrainFingerDatasetForTest(Dataset):
    def __init__(self, test_aug_id, is_fixed = True):
        super().__init__()

        if config.num_join_fingers == 1:
            self.dict_print = get_img_dict(config.test_print_dir)
            
        self.is_fixed = is_fixed 

        if config.is_display == True:
            print("Dataset: ", config.dataset_name)
            print("experiment type: test")
            print("Number of Fingers IDs: ", len(self.dict_print))
            print("Number of Fingers:", config.num_join_fingers)
            print("Network Arch:", config.w_name.split("_")[-1])
            print("loading models from: ", config.w_name)
            print("Number of imposter pair: ", config.num_pair_test)
            config.is_display = False 
    
        self.new_imposter_pos = 0
        self.test_aug_id = test_aug_id
        self.num_print_samples = len(self.dict_print)


    def trans(self, print_img1, print_img2):
        fill_print = (255,)
    
        # Resize
        if self.test_aug_id <=1 :
            resize = transforms.Resize(size=(config.img_size, config.img_size))
            print_img1 = resize(print_img1)
            print_img2 = resize(print_img2)

        # Random horizontal flipping
        if self.test_aug_id == 1 or self.test_aug_id == 3:
            print_img1 = TF.hflip(print_img1)
            print_img2 = TF.hflip(print_img2)


        # Transform to tensor
        print_img1 = TF.to_tensor(print_img1)
        print_img2 = TF.to_tensor(print_img2)

        # normalize
        normalize = transforms.Normalize(mean = [0.5], std = [0.5])
        print_img1 = normalize(print_img1)
        print_img2 = normalize(print_img2)

        return print_img1, print_img2


    def __len__(self):
        return self.num_print_samples * config.num_pair_test


    def __getitem__(self, index):
        num = index % config.num_pair_test 
        id_position = (index // config.num_pair_test) 
       
        if num == 0:
            finger_id, print_img1_dir = self.dict_print[index // config.num_pair_test]
            same_class = True
            print_img2_dir = print_img1_dir 
            self.new_imposter_pos = 0

        elif (num > 0): 
            finger_id, print_img1_dir = self.dict_print[index // config.num_pair_test]
            same_class = False 

            if self.is_fixed:
                if (id_position + num <  len(self.dict_print)):
                    class_id, print_img2_dir = self.dict_print[id_position + num]

                else: 
                    class_id, print_img2_dir = self.dict_print[self.new_imposter_pos]
                    self.new_imposter_pos += 1

            # random test 
            else: 
                class_id, print_img2_dir = self.dict_print[random.randint(0, len(self.dict_print) - 1)]

                while finger_id == class_id:
                    class_id, print_img2_dir = self.dict_print[random.randint(0, len(self.dict_print) - 1)]  

        # single finger
        if config.num_join_fingers == 1:

            ls_img1 = os.listdir(print_img1_dir)
            if len(ls_img1) == 1: pos_img1 = 0
            else: pos_img1 = random.randint(0, len(ls_img1) - 1)

            img1_path = os.path.join(print_img1_dir, ls_img1[pos_img1])
            print_img1 = Image.open(img1_path).convert("L")

            ls_img2 = os.listdir(print_img2_dir)
            if len(ls_img2) == 1: pos_img2 = 0
            else: pos_img2 = random.randint(0, len(ls_img2) - 1)
                
            img2_path = os.path.join(print_img2_dir, ls_img2[pos_img2])
            print_img2 = Image.open(img2_path).convert("L")

            img1, img2 = self.trans(print_img1, print_img2)     

        return img1, img2, same_class


# loading images in a dictionary
def get_img_dict(print_path):
    print_finger_dict = {}
    index = 0
    print_fingers =  sorted(os.listdir(print_path))

    for finger_id in print_fingers:
        id_dir = os.path.join(print_path, finger_id) 
        print_finger_dict[index] = [finger_id, id_dir]
        index += 1

    return print_finger_dict


if __name__ == "__main__":
    vt = PretrainFingerDataset()

    for i in range(200):
        img1, img2, same_class = vt.__getitem__(i)
        utils_wvu_old.plot_tensors([img1, img2], title=same_class)
