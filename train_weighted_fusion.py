import numpy as np 
import torch
from torch.utils.data import DataLoader 
import config 
from model import *
import os
from  torch.optim.lr_scheduler import ExponentialLR

if config.dataset_name == "wvu_old":
    from utils_wvu_old import *
elif config.dataset_name == "wvu_new":
    from utils_wvu_new import * 


num_fusion_models = 10
do_test = True # for testing
is_load = True # for loading
is_save_model = False    

if do_test == False: import dataset_wvu
elif do_test == True: import dataset_wvu_test


class ModelFusion:
    def __init__(self):

        if do_test == False:
            self.data_loader = DataLoader(
            dataset_wvu.WVUFingerDataset(train = True),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6)


        self.wfusion = WeightedFusion()
        self.wfusion.to(config.device)
        self.wfusion.train()

        if config.multi_gpus:
            self.wfusion = torch.nn.DataParallel(self.wfusion)

        self.L2_Norm_loss = torch.nn.MSELoss()
        self.criterion =  torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            list(self.wfusion.parameters()), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)

        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)


        if is_load == True:
            e = 450
            file_dir = "w_fusion_" + config.w_name

            print("loading pretrained model: ", file_dir)
            model_file = os.path.join(config.dataset_cp_dir, file_dir, "wf_model_" + str(e) + ".pth")
            checkpoint = torch.load(model_file)
            self.wfusion.load_state_dict(checkpoint["wfusion"])
            #self.optimizer.load_state_dict(checkpoint["optimizer"])



    def load_all_models(self):
        if config.is_load_model:
            print("loading all models")
            w_dir = config.w_dir
            all_models = sorted(os.listdir(w_dir), 
                            key= lambda x: int((x.split("_")[-1]).split(".")[0]))  

            ls_net_photo = []
            ls_net_print = []

            for model in all_models:
                net_photo, net_print = get_model(config.w_name, img_dim=config.img_dim)

                model_file = os.path.join(w_dir, model)
                checkpoint = torch.load(model_file)

                #loads weights trained of one finger to the weights for multi-finger
                if (config.is_finetune == False and 
                    config.is_convert_one_to_many == True and
                    config.num_join_fingers >= 2):

                    compatible_load(checkpoint, net_photo, net_print, disc_photo = None, 
                                disc_print = None, w_name = config.w_name, is_disc_load=False) 

                else:
                    net_photo.load_state_dict(checkpoint["net_photo"])
                    net_print.load_state_dict(checkpoint["net_print"])

                if do_test: print(model_file)
                del checkpoint
                ls_net_photo.append(net_photo)
                ls_net_print.append(net_print)

            return ls_net_photo, ls_net_print



    def train(self):

        ls_per_model_sq_dist = []
        ls_net_photo, ls_net_print = self.load_all_models()

        for epoch in range(config.num_epochs):
            loss_m = AverageMeter()

            for i in range(num_fusion_models):
                ls_sq_dist, ls_labels = self.calculate_scores(ls_net_photo[i], ls_net_print[i])
                ls_sq_dist = torch.cat(ls_sq_dist, dim=0)
                ls_labels = torch.cat(ls_labels, 0)
                ls_per_model_sq_dist.append(ls_sq_dist)


            input_data = torch.stack(([torch.stack([ls_per_model_sq_dist[model][i] 
                        for model in range(num_fusion_models)]) for i in range(len(ls_labels))]))


            self.optimizer.zero_grad() 
            scores = self.wfusion(input_data)
            scores = torch.squeeze(scores, dim=1)
            loss = self.criterion(scores, ls_labels)
            loss.backward()
            self.optimizer.step() 

            loss_m.update(loss.item())

            # at the end of each epoch 
            epoch = epoch + 1
            print("Epoch {} | Loss {:.4f}". format(epoch, loss_m.avg)) 

            if (is_save_model and epoch >= config.start_saving_epoch):
                save_wfusion_model(self.wfusion, self.optimizer, epoch)

            if (epoch !=0 and epoch % 50 == 0):
                self.scheduler.step()
                print("learning rate ", self.optimizer.param_groups[0]["lr"])
              

    def calculate_scores(self, net_photo, net_print):
        net_photo.eval()
        net_print.eval()

        ls_sq_dist = []
        ls_labels = []

        with torch.no_grad():
            for img_photo, img_print, label in self.data_loader:
                label = label.type(torch.float)

                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

                _, embd_photo = net_photo(img_photo)
                _, embd_print = net_print(img_print)

                dist_sq = torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1) #torch.sqrt()
                ls_sq_dist.append(dist_sq.data)
                ls_labels.append((1 - label).data)
        
        return ls_sq_dist, ls_labels


    def test(self):

        ls_per_model_sq_dist = []
        ls_net_photo, ls_net_print = self.load_all_models()

        for i in range(num_fusion_models):
            print("testing model: ", i)
            ls_per_tes_aug_dist = []

            for i in range(config.num_test_aug):
                self.data_loader = DataLoader(
                    dataset_wvu_test.WVUFingerDatasetForTest(test_aug_id=i, is_fixed = True),
                    batch_size=config.batch_size, 
                    shuffle=False,
                    pin_memory=True,
                    num_workers= 6)

                ls_sq_dist, ls_labels = self.calculate_scores(ls_net_photo[i], ls_net_print[i])
                ls_sq_dist = torch.cat(ls_sq_dist, dim=0)
                ls_per_tes_aug_dist.append(ls_sq_dist)

            ls_sq_dist = simple_average(min_max_normalization(ls_per_tes_aug_dist))
            ls_labels = torch.cat(ls_labels, 0)
            ls_per_model_sq_dist.append(ls_sq_dist)



        input_data = torch.stack(([torch.stack([ls_per_model_sq_dist[model][i] 
                    for model in range(num_fusion_models)]) for i in range(len(ls_labels))]))


        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Bismillah <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        scores = self.wfusion(input_data)
        scores = torch.squeeze(scores, dim=1)
        calculate_scores(ls_labels, scores, is_ensemble=True)



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
    v = ModelFusion()
    if do_test == False: v.train()
    elif do_test == True: v.test()


    """
    for i in range(101):
        phi, pi = vt.__getitem__(i)
        print(phi[0].split("/")[-3:-1], phi[1].split("/")[-3:-1])
        print("\n", pi[0].split("/")[-3:-1], pi[1].split("/")[-3:-1])
    """