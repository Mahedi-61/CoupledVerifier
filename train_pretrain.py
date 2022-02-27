import numpy as np 
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import config 
from model import *
import torch.nn.functional as F
import dataset_pretrain 
from  torch.optim.lr_scheduler import ExponentialLR
from utils_wvu_old import * 

class VerifTrain:
    def __init__(self):
        self.train_loader = DataLoader(
            dataset_pretrain.PretrainFingerDataset(train = config.is_train),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
            )

        print("experiment type: train")
        print("Dataset: %s and Type: %s" % (config.dataset_name, config.train_dataset))
        print("Number of Fingers: ", config.num_join_fingers)
        print("Network Arch:", config.w_name.split("_")[-1])
        print("Savging model in: ", config.w_name)
        
        self.net_print = get_one_model(config.w_name, img_dim=config.img_dim)
        self.net_print.train()

        self.optimizer_G = torch.optim.Adam(
            self.net_print.parameters(), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)


        self.scheduler = ExponentialLR(self.optimizer_G, gamma=0.9)


        if config.is_load_model:
            print("loading pretrained model from: ", config.save_w_name)
            load_single_model_for_finetuing(self.net_print)

    
    """
    def validate(self):
        self.net_print.eval()

        ls_sq_dist = []
        ls_labels = []
        
        with torch.no_grad(): 
            for img_photo, img_print, label in self.val_loader:
                label = label.type(torch.float)

                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

                _, embd_photo = self.net_photo(img_photo)
                _, embd_print = self.net_print(img_print)

                dist_sq = torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1) #torch.sqrt()
                ls_sq_dist.append(dist_sq.data)
                ls_labels.append((1 - label).data)

        auc, eer =  calculate_scores(ls_labels, ls_sq_dist, is_ensemble=False)
        self.net_print.train() 
        return auc, eer
    """

    def contrastive_loss(self, embd_photo, embd_print, label):
        # euclidean distance
        dist_sq = torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1)

        margin = (torch.ones_like(dist_sq, device= config.device) * 
                        config.margin)

        #mid_dist = margin - torch.sqrt(dist_sq + config.eps)
        mid_dist = margin - dist_sq

        true_loss =  label * (dist_sq)
        #wrong_loss = (1 - label) * torch.pow(F.relu(mid_dist), 2)
        wrong_loss = (1 - label) * F.relu(mid_dist)
        con_loss =  (true_loss + wrong_loss).mean()

        return dist_sq, con_loss



    def backward_G(self, img_print1, img_print2, label):
        img1, embd_print1 = self.net_print(img_print1)
        img2, embd_print2 = self.net_print(img_print2)


        # contrastive loss 
        dist_sq, con_loss = self.contrastive_loss(embd_print1, embd_print2, label)

        con_loss.backward()
        return dist_sq, con_loss

        

    def train(self):
        train_loop = tqdm(self.train_loader)

        for epoch in range(config.num_epochs):
            acc_m = AverageMeter()
            ls_con = AverageMeter()
            
            for img_print1, img_print2, label in train_loop:
                label = label.type(torch.float)

                img_print1 = img_print1.to(config.device)
                img_print2 = img_print2.to(config.device)
                label = label.to(config.device)


                ## generator
                self.optimizer_G.zero_grad() 
                dist_sq, con_loss = self.backward_G(img_print1, img_print2, label)
                self.optimizer_G.step() 

                # calculating acc, generator loss and 
                acc = (dist_sq < config.margin).type(torch.float)
                acc = (acc == label).type(torch.float)
                acc_m.update(acc.mean())

                # saving loss 
                ls_con.update(con_loss.item())


            # at the end of each epoch 
            epoch = epoch + 1
            print("Epoch {} | Acc. {:.4f}".format(epoch, acc_m.avg)) 
            print("Con loss {:.4f}".format(ls_con.avg))

            
            if (config.is_save_model and epoch >= config.start_saving_epoch):
                save_one_model(self.net_print, self.optimizer_G, epoch)

            if (epoch !=0 and epoch % 50 == 0):
                self.scheduler.step()
                print("learning rate ", self.optimizer_G.param_groups[0]["lr"])


if __name__ == "__main__":
    t = VerifTrain()
    t.train()

