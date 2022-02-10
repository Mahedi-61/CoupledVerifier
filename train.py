import numpy as np 
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import config 
from torch.autograd import Variable
from model import *
import torch.nn.functional as F
import dataset_wvu
from  torch.optim.lr_scheduler import ExponentialLR

if config.dataset_name == "wvu_old":
    from utils_wvu_old import *

elif config.dataset_name == "wvu_new":
    from utils_wvu_new import *

class VerifTrain:
    def __init__(self):
        self.train_loader = DataLoader(
            dataset_wvu.WVUFingerDataset(train = config.is_train),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
        )

        self.val_loader = DataLoader(
            dataset_wvu.WVUFingerDataset(train = not config.is_train),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
        )

        print("experiment type: train")
        print("Dataset: ", config.dataset_name)
        print("Number of Fingers: ", config.num_join_fingers)
        print("Network Arch:", config.w_name.split("_")[-1])
        print("Savging model in: ", config.w_name)
        
        self.net_photo, self.net_print = get_model(config.w_name, img_dim=config.img_dim)
        self.net_photo.train()
        self.net_print.train()

        self.disc_photo, self.disc_print = get_discriminator(img_dim=config.img_dim)
        self.disc_photo.train()
        self.disc_print.train()

        self.fidentity = FingerIdentity()
        self.fidentity.to(config.device)
        self.fidentity.train()

        if config.multi_gpus:
            self.fidentity = torch.nn.DataParallel(self.fidentity)

        self.adversarial_loss =  torch.nn.MSELoss()
        self.L2_Norm_loss = torch.nn.MSELoss()
        self.criterion =  torch.nn.BCEWithLogitsLoss()
        self.iter = 0

        if config.partial_finetune == False:
            self.optimizer_G = torch.optim.Adam(
                list(self.net_photo.parameters()) + 
                    list(self.net_print.parameters()) + 
                    list(self.fidentity.parameters()), 
                lr = config.learning_rate,
                weight_decay = config.weight_decay)

        else:
            print("Identity loss is not calculating ...")
            self.optimizer_G = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, self.net_photo.parameters())) + 
                list(filter(lambda p: p.requires_grad, self.net_print.parameters())), 
                lr = config.learning_rate,
                weight_decay = config.weight_decay)

    
        self.optimizer_D = torch.optim.Adam(
            list(self.disc_photo.parameters()) + list(self.disc_print.parameters()), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)

        self.scheduler = ExponentialLR(self.optimizer_G, gamma=0.9)

        if config.is_load_model:
            print("loading pretrained model from: ", config.save_w_name)
            load_saved_model_for_finetuing(self.net_photo, self.net_print, 
                                    self.disc_photo, self.disc_print, 
                                    is_disc_load=True, partial_finetune=config.partial_finetune) 


    def validate(self):
        self.net_photo.eval()
        self.net_print.eval()

        ls_sq_dist = []
        ls_labels = []
        
        with torch.no_grad(): 
            for img_photo, img_print, label in self.val_loader:
                label = label.type(torch.float)

                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)
                plot_tensors([img_photo[2], img_print[2]], title="check")

                _, embd_photo = self.net_photo(img_photo)
                _, embd_print = self.net_print(img_print)

                dist_sq = torch.sum(torch.pow(embd_photo - embd_print, 2), dim=1) #torch.sqrt()
                ls_sq_dist.append(dist_sq.data)
                ls_labels.append((1 - label).data)

        auc, eer =  calculate_scores(ls_labels, ls_sq_dist, is_ensemble=False)
        self.net_photo.train()
        self.net_print.train() 
        return auc, eer


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


    def backward_D(self, img_photo, img_print, valid, fake):
        fake_photo, embd_photo = self.net_photo(img_photo)
        fake_print, embd_print = self.net_print(img_print)

        pred_real_photo = self.disc_photo(img_photo)
        pred_real_print = self.disc_print(img_print)

        # fake 
        pred_fake_photo = self.disc_photo(fake_photo.detach())
        pred_fake_print = self.disc_print(fake_print.detach())

        D_loss = (
            self.adversarial_loss(pred_real_photo, valid) + 
            self.adversarial_loss(pred_real_print, valid) + 
            self.adversarial_loss(pred_fake_photo, fake) +
            self.adversarial_loss(pred_fake_print, fake)
        ) / 4 

        D_loss.backward()
        return D_loss 


    def backward_G(self, img_photo, img_print, valid, fake, label):
        fake_photo, embd_photo = self.net_photo(img_photo)
        fake_print, embd_print = self.net_print(img_print)

        pred_fake_photo = self.disc_photo(fake_photo)
        pred_fake_print = self.disc_print(fake_print)

        #l2 loss
        l2_loss = (self.L2_Norm_loss(fake_print, img_print) + 
                    self.L2_Norm_loss(fake_photo, img_photo)) /2
                    
        # gan loss
        gan_loss = (self.adversarial_loss(pred_fake_photo, valid) + 
                    self.adversarial_loss(pred_fake_print, valid)) / 2


        # contrastive loss 
        dist_sq, con_loss = self.contrastive_loss(embd_photo, embd_print, label)

        #identity loss
        out = self.fidentity(embd_photo, embd_print)
        out = torch.squeeze(out, dim=1)

        if config.partial_finetune == False:
            id_loss = self.criterion(out, label)
        else:
            id_loss = 0.0

        G_loss =  (con_loss + gan_loss*config.delta_gan + 
                            l2_loss*config.delta_l2 + id_loss)

        G_loss.backward()
        return dist_sq, con_loss,l2_loss, id_loss, G_loss

        
    def train(self):
        train_loop = tqdm(self.train_loader)
        Tensor = torch.cuda.FloatTensor

        for epoch in range(config.num_epochs):
            loss_m_d = AverageMeter()
            loss_m_g = AverageMeter()
            acc_m = AverageMeter()
            ls_con = AverageMeter()
            ls_l2 = AverageMeter()
            ls_id = AverageMeter() 
            
            for img_photo, img_print, label in train_loop:
                label = label.type(torch.float)

                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

                # valid = 1; fake = 0
                valid = Variable(Tensor(img_photo.size(0), 1).fill_(1.0), 
                                    requires_grad=False)
                fake = Variable(Tensor(img_photo.size(0), 1).fill_(0.0), 
                                    requires_grad=False)

                ## generator
                self.optimizer_G.zero_grad() 
                dist_sq, con_loss,l2_loss, id_loss, G_loss = self.backward_G(
                                        img_photo, img_print, valid, fake, label)
                self.optimizer_G.step() 

                # calculating acc, generator loss and 
                acc = (dist_sq < config.margin).type(torch.float)
                acc = (acc == label).type(torch.float)
                acc_m.update(acc.mean())

                # discriminator 
                if config.partial_finetune == False: 
                    self.optimizer_D.zero_grad()
                    D_loss = self.backward_D(img_photo, img_print, valid, fake)
                    self.optimizer_D.step() 

                loss_m_g.update(G_loss.item())
                ls_con.update(con_loss.item())
                ls_l2.update(l2_loss.item())

                if config.partial_finetune == False: 
                    ls_id.update(id_loss.item())
                else:
                    ls_id.update(id_loss)

                if config.partial_finetune == False: loss_m_d.update(D_loss.item())
                else: loss_m_d.update(0.00)
   

            # at the end of each epoch 
            epoch = epoch + 1
            print("Epoch {} | Acc. {:.4f} | D_loss {:.4f} | G_loss {:.4f}".
                    format(epoch, acc_m.avg, loss_m_d.avg, loss_m_g.avg)) 

            print("Con loss {:.4f} | l2 loss {:.4f} | id loss {:.4f}".format(
                                            ls_con.avg, ls_l2.avg, ls_id.avg))

            
            if (config.is_save_model and epoch >= config.start_saving_epoch):

                save_model(self.net_photo, self.net_print, self.fidentity, self.optimizer_G, 
                        self.disc_photo, self.disc_print, epoch, is_best=False)

            if (epoch !=0 and epoch % 50 == 0):
                self.scheduler.step()
                print("learning rate ", self.optimizer_G.param_groups[0]["lr"])
         

if __name__ == "__main__":
    t = VerifTrain()
    t.train()