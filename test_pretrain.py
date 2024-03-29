import numpy as np 
import torch
from torch.utils.data import DataLoader 
import config 
from model import *
import os
import dataset_pretrain
import dataset_wvu_test
from utils_wvu_new import * 

class VerifTest:
    def __init__(self):

        self.net_print = get_one_model(config.w_name, img_dim=config.img_dim)

        if config.is_load_model:
            print("loading all models")
            
            if config.is_finetune: w_dir = config.save_w_dir
            else: w_dir = config.w_dir

            all_models = sorted(os.listdir(w_dir), 
                        key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            ls_each_finger_dist = []
            for model in all_models:
                model_file = os.path.join(w_dir, model)
                checkpoint = torch.load(model_file)

                #loads weights trained of one finger to the weights for multi-finger
                if (config.is_convert_one_to_many == False):
                    self.net_print.load_state_dict(checkpoint["net_print"])

                print(model_file)
                del checkpoint
                ls_each_finger_per_model_dist = []
            
                for i in range(config.num_test_aug):
                    self.test_loader = DataLoader(
                        #dataset_pretrain.PretrainFingerDatasetForTest(test_aug_id=i, is_fixed = True),
                        dataset_wvu_test.WVUFingerDatasetForTest(test_aug_id=i, is_fixed = True),
                        batch_size=config.batch_size, 
                        shuffle=False,
                        pin_memory=True,
                        num_workers= 1)

                    ls_sq_dist, ls_labels = self.test()
                    ls_sq_dist = torch.cat(ls_sq_dist, dim=0)
                    ls_each_finger_per_model_dist.append(ls_sq_dist)

                if config.is_test_augment == True:
                    ls_sq_dist = simple_average(min_max_normalization(ls_each_finger_per_model_dist))
                    ls_labels = torch.cat(ls_labels, 0)

                    print("augmented result: ", end="")
                    calculate_scores(ls_labels, ls_sq_dist, is_ensemble=True)
                
                if config.is_ensemble == True:
                    ls_each_finger_dist.append(ls_sq_dist)


            if config.is_ensemble == True: 
                print(">>>>>>>>>>>>>>> Fusion <<<<<<<<<<<<<<<")
                ls_each_finger_dist = min_max_normalization(ls_each_finger_dist)
                ls_sq_dist = simple_average(ls_each_finger_dist)

                if config.is_test_augment == False: ls_labels = torch.cat(ls_labels, 0)
                calculate_scores(ls_labels, ls_sq_dist, is_ensemble=True)

    def test(self):
        self.net_print.eval()
        ls_sq_dist = []
        ls_labels = []

        with torch.no_grad():
            for img_print1, img_print2, label in self.test_loader:
                label = label.type(torch.float)
                img_print1 = img_print1.to(config.device)
                img_print2 = img_print2.to(config.device)
                label = label.to(config.device)

                img1, embd_print1 = self.net_print(img_print1)
                img2, embd_print2 = self.net_print(img_print2)

                dist_sq = torch.sum(torch.pow(embd_print1 - embd_print2, 2), dim=1) #torch.sqrt()
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


def simple_average(ls_each_finger_dist):
    return torch.stack(ls_each_finger_dist, dim=0).mean(dim=0)


if __name__ == "__main__":
    v = VerifTest()
