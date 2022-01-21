import numpy as np 
import torch 
from torch.utils.data import DataLoader 
import config 
from model import *
import os
import utils_wvu_new
import dataset_wvu_test

is_ensemble = True         

class VerifTest:
    def __init__(self):
        self.test_loader = DataLoader(
            dataset_wvu_test.WVUFingerDatasetForTest(is_fixed = True),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
        )

        self.net_photo, self.net_print = get_model(config.w_name, img_dim=config.img_dim)

        if config.is_load_model:
            print("loading all models")
            if config.is_finetune == False:
                w_dir = config.w_dir 

            all_models = sorted(os.listdir(w_dir), 
                    key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            ls_each_finger_dist = []
            for model in all_models:
                model_file = os.path.join(w_dir, model)
                checkpoint = torch.load(model_file)

                #loads weights trained of one finger to the weights for multi-finger
                if (config.is_finetune == False and 
                    config.is_convert_one_to_many == True and
                    config.num_join_fingers >= 2):

                    compatible_load(checkpoint, self.net_photo, self.net_print, disc_photo = None, 
                                 disc_print = None, w_name = config.w_name, is_disc_load=False)    

                else:            
                    self.net_photo.load_state_dict(checkpoint["net_photo"])
                    self.net_print.load_state_dict(checkpoint["net_print"])

                print(model_file)
                del checkpoint

                if is_ensemble == False:
                    ls_sq_dist, ls_labels = self.test()
                
                elif is_ensemble == True:
                    ls_sq_dist, ls_labels = self.test()
                    ls_sq_dist = torch.cat(ls_sq_dist, dim=0)
                    ls_each_finger_dist.append(ls_sq_dist)

            if is_ensemble == True: 
                print(">>>>>>>>>>>>>>> Fusion <<<<<<<<<<<<<<<")
                ls_each_finger_dist = min_max_normalization(ls_each_finger_dist)
                ls_sq_dist = simple_average(ls_each_finger_dist)
                utils_wvu_new.calculate_scores(ls_labels, ls_sq_dist, is_ensemble=True)


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
            
        utils_wvu_new.calculate_scores(ls_labels, ls_sq_dist, is_ensemble=False)
        return ls_sq_dist, ls_labels
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


def simple_average(ls_each_finger_dist):
    return torch.stack(ls_each_finger_dist, dim=0).mean(dim=0)


if __name__ == "__main__":
    v = VerifTest()
    """
    vt = WVUNewVerifierForTest()
    for i in range(400, 579):
        phi, pi, sc = vt.__getitem__(i)
    """

    #vt = WVUNewVerifierForTest()
    #img1, img2, label = vt.__getitem__(13)
    #utils_wvu_new.plot_tensors([img1, img2], title=label)