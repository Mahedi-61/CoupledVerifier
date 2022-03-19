import numpy as np 
import torch
from torch.utils.data import DataLoader 
import config 
from model import *
import random 
import os
from collections import OrderedDict
import random 
import torchvision.transforms.functional as TF
import dataset_wvu_test

if config.dataset_name == "wvu_old":
    from utils_wvu_old import *
    w_dir = config.old_weights_dir

elif config.dataset_name == "wvu_new":
    from utils_wvu_new import *
    w_dir = config.new_weights_dir

f1_dir = "F1_D1_ID%s_A1" % config.fnums[0][0]
f2_dir = "F1_D1_ID%s_A1" % config.fnums[0][1]

if config.num_join_fingers >= 3:
    f3_dir = "F1_D1_ID%s_A1" % config.fnums[0][2]

    if config.num_join_fingers == 4:
         f4_dir = "F1_D1_ID%s_A1" % config.fnums[0][3]


is_combine = False         
is_final_ensemble = True                                


class VerifTest:
    def __init__(self):
        print("loading test dataset ...")

        # loading single finger model
        self.net_photo_f1, self.net_print_f1 = get_model(config.w_name, img_dim=1)
        self.net_photo_f2, self.net_print_f2 = get_model(config.w_name, img_dim=1)

        if config.num_join_fingers >= 3:
            self.net_photo_f3, self.net_print_f3 = get_model(config.w_name, img_dim=1)

            if config.num_join_fingers == 4:
                self.net_photo_f4, self.net_print_f4 = get_model(config.w_name, img_dim=1)

        if is_combine:
            self.net_photo_combine, self.net_print_combine = get_model(
                        config.combined_w_name, img_dim=config.img_dim)

        if config.is_load_model:
            if is_combine: 
                print("loading combined models")
                # for combined weight
                combine_model = os.path.join(config.combined_w_dir, "best_model_000.pth")
                checkpoint = torch.load(combine_model)
                self.net_photo_combine.load_state_dict(checkpoint["net_photo"])
                self.net_print_combine.load_state_dict(checkpoint["net_print"])
            

            print("loading 10 singer finger models")
            w_dir_f1 = os.path.join(w_dir, f1_dir)
            w_dir_f2 = os.path.join(w_dir, f2_dir)

            all_models_f1 = sorted(os.listdir(w_dir_f1), 
                    key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            all_models_f2 = sorted(os.listdir(w_dir_f2), 
                    key= lambda x: int((x.split("_")[-1]).split(".")[0]))  


            assert (len(all_models_f1) == len(all_models_f2)), "# finger models must be equal size"

            if config.num_join_fingers >=3:
                w_dir_f3 = os.path.join(w_dir, f3_dir)
                all_models_f3 = sorted(os.listdir(w_dir_f3), 
                    key= lambda x: int((x.split("_")[-1]).split(".")[0])) 

                assert (len(all_models_f1) == len(all_models_f3)), "# finger models must be equal"

                if config.num_join_fingers == 4:
                    w_dir_f4 = os.path.join(w_dir, f4_dir)
                    all_models_f4 = sorted(os.listdir(w_dir_f4), 
                        key= lambda x: int((x.split("_")[-1]).split(".")[0])) 

                    assert (len(all_models_f1) == len(all_models_f4)), "# finger models must be equal"


            # for all models
            ls_final_ensemble = []
            for i, model in enumerate(all_models_f1):
                print("Finger model: ", i + 1)
                # for all variations of test data
                ls_each_test_aug_dist = []
                for j in range(config.num_test_aug):
                    self.test_loader = DataLoader(
                        dataset_wvu_test.WVUFingerDatasetForTest(test_aug_id=j, is_fixed=True),
                        batch_size=config.batch_size, 
                        shuffle=False,
                        pin_memory=True,
                        num_workers= 6  
                    )

                    model_file_f1 = os.path.join(w_dir_f1, model)
                    checkpoint_f1 = torch.load(model_file_f1)

                    model_file_f2 = os.path.join(w_dir_f2, all_models_f2[i]) #model
                    checkpoint_f2 = torch.load(model_file_f2)

                    if config.num_join_fingers >= 3:
                        model_file_f3 = os.path.join(w_dir_f3, all_models_f3[i]) #model
                        checkpoint_f3 = torch.load(model_file_f3)

                        if config.num_join_fingers == 4:
                            model_file_f4 = os.path.join(w_dir_f4, all_models_f4[i])  #model
                            checkpoint_f4 = torch.load(model_file_f4)


                    self.net_photo_f1.load_state_dict(checkpoint_f1["net_photo"])
                    self.net_print_f1.load_state_dict(checkpoint_f1["net_print"])

                    self.net_photo_f2.load_state_dict(checkpoint_f2["net_photo"])
                    self.net_print_f2.load_state_dict(checkpoint_f2["net_print"])

                    if config.num_join_fingers >=3:
                        self.net_photo_f3.load_state_dict(checkpoint_f3["net_photo"])
                        self.net_print_f3.load_state_dict(checkpoint_f3["net_print"])

                        if config.num_join_fingers == 4:
                            self.net_photo_f4.load_state_dict(checkpoint_f4["net_photo"])
                            self.net_print_f4.load_state_dict(checkpoint_f4["net_print"])

                    if j == 0:
                        print(model_file_f1)
                        print(model_file_f2)
                        if config.num_join_fingers >=3:
                            print(model_file_f3)
                            if config.num_join_fingers ==4: print(model_file_f4)

                    ls_sq_dist, ls_labels = self.test()
                    ls_sq_dist = torch.cat(ls_sq_dist, dim=0)
                    ls_each_test_aug_dist.append(ls_sq_dist)

                if config.is_test_augment == True:
                    ls_sq_dist = simple_average(min_max_normalization(ls_each_test_aug_dist))
                    ls_labels = torch.cat(ls_labels, 0)

                    print("augmented result: ", end="")
                    calculate_scores(ls_labels, ls_sq_dist, is_ensemble=True)
                    if is_final_ensemble: ls_final_ensemble.append(ls_sq_dist)
                

            if is_final_ensemble:
                ls_sq_dist = simple_average(min_max_normalization(ls_final_ensemble))
                print(">>>>>>>>>>>>>>>>>>> Final ensemble <<<<<<<<<<<<<<<<")
                calculate_scores(ls_labels, ls_sq_dist, is_ensemble=True)


    def test(self):
        self.net_photo_f1.eval()
        self.net_print_f1.eval()
        self.net_photo_f2.eval()
        self.net_print_f2.eval()

        if config.num_join_fingers >= 3:
            self.net_photo_f3.eval()
            self.net_print_f3.eval()

            if config.num_join_fingers == 4:
                self.net_photo_f4.eval()
                self.net_print_f4.eval()

        if is_combine:
            self.net_photo_combine.eval()
            self.net_print_combine.eval()

        ls_sq_dist = []
        ls_labels = []

        with torch.no_grad():
            for img_photo, img_print, label in self.test_loader:
                label = label.type(torch.float)

                # for multiple fingers
                ls_each_finger_dist = []
                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

                ls_photo_fs = torch.split(img_photo, 1, dim=1)
                ls_print_fs = torch.split(img_print, 1, dim=1)

                if is_combine:
                    _, embd_photo = self.net_photo_combine(img_photo)
                    _, embd_print = self.net_print_combine(img_print)

                    ls_each_finger_dist.append(
                            torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1))
                
                #for single fingers
                _, embd_photo_f1 = self.net_photo_f1(ls_photo_fs[0].to(config.device))
                _, embd_print_f1 = self.net_print_f1(ls_print_fs[0].to(config.device))

                ls_each_finger_dist.append(
                    torch.sum(torch.pow(embd_photo_f1 - embd_print_f1, 2), dim=1)) 

                _, embd_photo_f2 = self.net_photo_f2(ls_photo_fs[1].to(config.device))
                _, embd_print_f2 = self.net_print_f2(ls_print_fs[1].to(config.device))

                ls_each_finger_dist.append(
                    torch.sum(torch.pow(embd_photo_f2 - embd_print_f2, 2), dim=1))

                if config.num_join_fingers >= 3:
                    _, embd_photo_f3 = self.net_photo_f3(ls_photo_fs[2].to(config.device))
                    _, embd_print_f3 = self.net_print_f3(ls_print_fs[2].to(config.device))

                    ls_each_finger_dist.append(
                        torch.sum(torch.pow(embd_photo_f3 - embd_print_f3, 2), dim=1)) 

                    if config.num_join_fingers == 4:
                        _, embd_photo_f4 = self.net_photo_f4(ls_photo_fs[3].to(config.device))
                        _, embd_print_f4 = self.net_print_f4(ls_print_fs[3].to(config.device))

                        ls_each_finger_dist.append(
                            torch.sum(torch.pow(embd_photo_f4 - embd_print_f4, 2), dim=1)) 

                # normalization
                ls_each_finger_dist = z_score_normalization(ls_each_finger_dist)
    
                # fusion
                dist_sq = simple_average(ls_each_finger_dist)

                ls_sq_dist.append(dist_sq.data)
                ls_labels.append((1 - label).data)
            
        calculate_scores(ls_labels, ls_sq_dist, is_ensemble=False)
        return ls_sq_dist, ls_labels


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
