import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import config 
from model import *
import random 
import os
import utils_wvu_old
#import utils_wvu_new
from PIL import Image 


## Special Class for Fixed Test set and results
class WVUVerifierForTest(Dataset):
    def __init__(self):
        super().__init__()

        print("test data")
        if config.num_join_fingers >= 2:
            self.dict_photo, self.dict_print = utils_wvu_old.get_multiple_img_dict(
                    config.test_photo_dir, config.test_print_dir, config.fnums)

        self.num_photo_samples = len(self.dict_photo)
        print("\nNumber of Fingers: ", config.num_join_fingers)
        print("Network Arch:", config.w_name.split("_")[-1])

        self.test_trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
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

        if config.num_join_fingers >= 2:
            print_image = self.dict_print[class_id]

            # for 2 fingers
            ph_f1 = self.test_trans(Image.open(photo_image[0]).convert("L")) 
            ph_f2 = self.test_trans(Image.open(photo_image[1]).convert("L"))
            pr_f1 = self.test_trans(Image.open(print_image[0]).convert("L")) 
            pr_f2 = self.test_trans(Image.open(print_image[1]).convert("L"))

            img1 = torch.cat([ph_f1, ph_f2], dim=0)
            img2 = torch.cat([pr_f1, pr_f2], dim=0)

            ls_photo_fs = [ph_f1, ph_f2]
            ls_print_fs = [pr_f1, pr_f2]

            # for 3 fingers (only adding additional finger's data)
            if config.num_join_fingers == 3:
                ph_f3 = self.test_trans(Image.open(photo_image[2]).convert("L"))
                pr_f3 = self.test_trans(Image.open(print_image[2]).convert("L"))

                ls_photo_fs.append(ph_f3)
                ls_print_fs.append(pr_f3)

                img1 = torch.cat([ph_f1, ph_f2, ph_f3], dim=0)
                img2 = torch.cat([pr_f1, pr_f2, pr_f3], dim=0)

        return  ls_photo_fs, ls_print_fs, img1, img2, same_class


class VerifTest:
    def __init__(self):
        print("loading dataset ...")
        self.test_loader = DataLoader(
            WVUVerifierForTest(),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
        )

        # loading single finger model
        self.net_photo, self.net_print = get_model(config.w_name, img_dim=1)
        self.net_photo_combine, self.net_print_combine = get_model(
                            config.combined_w_name, img_dim=config.img_dim)

        if config.is_load_model:
            """
            print("loading combined models")
            # for combined weight
            combine_model = os.path.join(config.combined_w_dir, "best_model_000.pth")
            checkpoint = torch.load(combine_model)
            self.net_photo_combine.load_state_dict(checkpoint["net_photo"])
            self.net_print_combine.load_state_dict(checkpoint["net_print"])
            """

            print("loading singer finger models")
            w_dir = config.w_dir   # single fingers saved dir
            all_models = sorted(os.listdir(w_dir), 
                        key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            for model in all_models:
                model_file = os.path.join(w_dir, model)
                checkpoint = torch.load(model_file)

                self.net_photo.load_state_dict(checkpoint["net_photo"])
                self.net_print.load_state_dict(checkpoint["net_print"])

                print(model_file)
                self.test()
                


    def test(self):
        self.net_photo.eval()
        self.net_print.eval()
        self.net_photo_combine.eval()
        self.net_print_combine.eval()

        ls_sq_dist = []
        ls_labels = []

        with torch.no_grad():
            for ls_photo_fs, ls_print_fs, img_photo, img_print, label in self.test_loader:
                label = label.type(torch.float)

                # for multiple fingers
                ls_each_finger_dist = []
                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

                """
                _, embd_photo = self.net_photo_combine(img_photo)
                _, embd_print = self.net_print_combine(img_print)

                ls_each_finger_dist.append(
                        torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1))
                """
                        
                #for single fingers
                for img_photo, img_print in zip(ls_photo_fs, ls_print_fs):
                    img_photo = img_photo.to(config.device)
                    img_print = img_print.to(config.device)

                    _, embd_photo = self.net_photo(img_photo)
                    _, embd_print = self.net_print(img_print)

                    ls_each_finger_dist.append(
                        torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1)) #torch.sqrt()

                # normalization
                ls_each_finger_dist = min_max_normalization(ls_each_finger_dist)
    
                # fusion
                dist_sq = simple_average(ls_each_finger_dist)

                ls_sq_dist.append(dist_sq.data)
                ls_labels.append((1 - label).data)
            
        utils_wvu_old.calculate_scores(ls_labels, ls_sq_dist)
        #self.plot_roc()

    # plotting roc curve 
    def plot_roc(self):
        pass 


def min_max_normalization(ls_each_finger_dist):
    normalized_dist = []
    assert (len(ls_each_finger_dist) > 1), "Must have multiple fingers"

    for fdist in ls_each_finger_dist:
        fdist -= fdist.min()
        fdist /= fdist.max()
        normalized_dist.append(fdist)
    return normalized_dist

def z_score_normalization(ls_each_finger_dist):
    normalized_dist = []
    assert (len(ls_each_finger_dist) > 1), "Must have multiple fingers"

    for fdist in ls_each_finger_dist:
        fdist -= fdist.mean()
        fdist /= fdist.std()
        normalized_dist.append(fdist)
    return normalized_dist

# Efficient approach to Normalization of Multimodal Biometric Scores, 2011
def htangent_normalization(ls_each_finger_dist):
    normalized_dist = []
    assert (len(ls_each_finger_dist) > 1), "Must have multiple fingers"

    for fdist in ls_each_finger_dist:
        fdist -= fdist.mean()
        fdist /= fdist.std()
        fdist = 0.5 * (torch.tanh(0.01 * fdist) + 1)
        normalized_dist.append(fdist)
    return normalized_dist


def simple_average(ls_each_finger_dist):
    return torch.stack(ls_each_finger_dist, dim=0).mean(dim=0)



if __name__ == "__main__":
    v = VerifTest()
