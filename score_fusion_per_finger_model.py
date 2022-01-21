import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import config 
from model import *
import random 
import os
from PIL import Image 
from collections import OrderedDict
import random 

if config.dataset_name == "wvu_old":
    from utils_wvu_old import *
    w_dir = config.old_weights_dir

elif config.dataset_name == "wvu_new":
    from utils_wvu_new import *
    w_dir = config.new_weights_dir

f1_dir = "F1_D2vfp_ID%s_A1" % config.fnums[0][0]
f2_dir = "F1_D2vfp_ID%s_A1" % config.fnums[0][1]

if config.num_join_fingers >= 3:
    f3_dir = "F1_D2vfp_ID%s_A1" % config.fnums[0][2]

    if config.num_join_fingers == 4:
         f4_dir = "F1_D2vfp_ID%s_A1" % config.fnums[0][3]

is_combine = True                    

## Special Class for Fixed Test set and results
class WVUVerifierForTest(Dataset):
    def __init__(self, is_fixed = True):
        super().__init__()

        print("test data")
        if config.num_join_fingers >= 2:
            self.dict_photo, self.dict_print = get_multiple_img_dict(
                    config.test_photo_dir, config.test_print_dir, config.fnums)

        self.is_fixed = is_fixed 
        print("Dataset: ", config.dataset_name)
        print("experiment type: test")
        print("Number of Fingers IDs: ", len(self.dict_photo))
        print("Number of Fingers:", config.num_join_fingers)
        print("Network Arch:", config.w_name.split("_")[-1])
        if is_combine: print("loading models from: ", config.combined_w_name)
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
        if config.num_join_fingers >= 3:
            ph_f3 = self.test_trans(Image.open(photo_image[2]).convert("L"))
            pr_f3 = self.test_trans(Image.open(print_image[2]).convert("L"))

            ls_photo_fs.append(ph_f3)
            ls_print_fs.append(pr_f3)

            img1 = torch.cat([ph_f1, ph_f2, ph_f3], dim=0)
            img2 = torch.cat([pr_f1, pr_f2, pr_f3], dim=0)

            if config.num_join_fingers == 4:
                ph_f4 = self.test_trans(Image.open(photo_image[3]).convert("L"))
                pr_f4 = self.test_trans(Image.open(print_image[3]).convert("L"))

                ls_photo_fs.append(ph_f4)
                ls_print_fs.append(pr_f4)

                img1 = torch.cat([ph_f1, ph_f2, ph_f3, ph_f4], dim=0)
                img2 = torch.cat([pr_f1, pr_f2, pr_f3, pr_f4], dim=0)

        return  ls_photo_fs, ls_print_fs, img1, img2, same_class


class VerifTest:
    def __init__(self):
        print("loading test dataset ...")
        self.test_loader = DataLoader(
            WVUVerifierForTest(),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
        )

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


            for i, model in enumerate(all_models_f1):
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

                print(model_file_f1)
                print(model_file_f2)
                if config.num_join_fingers >=3:
                    print(model_file_f3)
                    if config.num_join_fingers ==4: print(model_file_f4)

                self.test()
                

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
            for ls_photo_fs, ls_print_fs, img_photo, img_print, label in self.test_loader:
                label = label.type(torch.float)

                # for multiple fingers
                ls_each_finger_dist = []
                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

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
