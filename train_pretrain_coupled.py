import numpy as np 
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import config 
from torch.autograd import Variable
from model import *
import torch.nn.functional as F
import dataset_pretrain
from  torch.optim.lr_scheduler import ExponentialLR
from collections import OrderedDict
from utils_wvu_new import *

class VerifTrain:
    def __init__(self):
        self.train_loader = DataLoader(
            dataset_pretrain.PretrainFingerDataset(train = config.is_train),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 6  
        )

        print("Dataset: ", config.dataset_name)
        print("Experiment Type: Train | Set:", config.train_dataset)
        print("Pretrian type: ", config.pretrain_type)
        print("Data: ", config.train_print_dir)
        print("Number of Fingers: ", config.num_join_fingers)
        print("Network Arch:", config.w_name.split("_")[-1])
        print("Savging model in: ", config.w_name)
        
        self.net_print = get_one_model_full(config.w_name, img_dim=config.img_dim)
        self.net_print.train()

        self.disc_print = get_one_discriminator(img_dim=config.img_dim)
        self.disc_print.train()

        self.adversarial_loss =  torch.nn.MSELoss()
        self.L2_Norm_loss = torch.nn.MSELoss()

        self.optimizer_G = torch.optim.Adam(
            self.net_print.parameters(), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)
    
        self.optimizer_D = torch.optim.Adam(
            list(self.disc_print.parameters()), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)

        self.scheduler = ExponentialLR(self.optimizer_G, gamma=0.9)
        #print(self.net_photo.state_dict().keys())

        """
        if config.load_pretrain_weights and config.is_finetune:
            print("loading pretrained weights form", config.save_w_dir.split("/")[-3:])
            model_file = os.path.join(config.save_w_dir, "best_model_000.pth")
            checkpoint = torch.load(model_file)

            ### remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint["net_print"].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            self.net_print.module.backbone.load_state_dict(new_state_dict)
        """

        if config.is_load_model:
            print("loading pretrained model from: ", config.save_w_name)
            load_single_model_for_finetuing_coupled(self.net_print, self.disc_print)
        


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


    def backward_D(self, img_print1, img_print2, valid, fake):
        fake_photo, embd_photo = self.net_print(img_print1)
        fake_print, embd_print = self.net_print(img_print2)

        pred_real_photo = self.disc_print(img_print1)
        pred_real_print = self.disc_print(img_print2)

        # fake 
        pred_fake_photo = self.disc_print(fake_photo.detach())
        pred_fake_print = self.disc_print(fake_print.detach())

        D_loss = (
            self.adversarial_loss(pred_real_photo, valid) + 
            self.adversarial_loss(pred_real_print, valid) + 
            self.adversarial_loss(pred_fake_photo, fake) +
            self.adversarial_loss(pred_fake_print, fake)
        ) / 4 

        D_loss.backward()
        return D_loss 


    def backward_G(self, img_print1, img_print2, valid, fake, label):
        fake_photo, embd_photo = self.net_print(img_print1)
        fake_print, embd_print = self.net_print(img_print2)

        pred_fake_photo = self.disc_print(fake_photo)
        pred_fake_print = self.disc_print(fake_print)

        #l2 loss
        l2_loss = (self.L2_Norm_loss(fake_photo, img_print1) + 
                    self.L2_Norm_loss(fake_print, img_print2)) /2
                    
        # gan loss
        gan_loss = (self.adversarial_loss(pred_fake_photo, valid) + 
                    self.adversarial_loss(pred_fake_print, valid)) / 2


        # contrastive loss 
        dist_sq, con_loss = self.contrastive_loss(embd_photo, embd_print, label)


        G_loss =  (con_loss + gan_loss*config.delta_gan + l2_loss*config.delta_l2)
        G_loss.backward()
        return dist_sq, con_loss,l2_loss, G_loss

        
    def train(self):
        train_loop = tqdm(self.train_loader)
        Tensor = torch.cuda.FloatTensor

        for epoch in range(config.num_epochs):
            loss_m_d = AverageMeter()
            loss_m_g = AverageMeter()
            acc_m = AverageMeter()
            ls_con = AverageMeter()
            ls_l2 = AverageMeter()
            
            for img_print1, img_print2, label in train_loop:
                label = label.type(torch.float)

                img_print1 = img_print1.to(config.device)
                img_print2 = img_print2.to(config.device)
                label = label.to(config.device)

                # valid = 1; fake = 0
                valid = Variable(Tensor(img_print1.size(0), 1).fill_(1.0), 
                                    requires_grad=False)
                fake = Variable(Tensor(img_print1.size(0), 1).fill_(0.0), 
                                    requires_grad=False)

                ## generator
                self.optimizer_G.zero_grad() 
                dist_sq, con_loss,l2_loss, G_loss = self.backward_G(
                                        img_print1, img_print2, valid, fake, label)
                self.optimizer_G.step() 

                # calculating acc, generator loss and 
                acc = (dist_sq < config.margin).type(torch.float)
                acc = (acc == label).type(torch.float)
                acc_m.update(acc.mean())

                # discriminator 
                if config.partial_finetune == False: 
                    self.optimizer_D.zero_grad()
                    D_loss = self.backward_D(img_print1, img_print2, valid, fake)
                    self.optimizer_D.step() 

                loss_m_g.update(G_loss.item())
                ls_con.update(con_loss.item())
                ls_l2.update(l2_loss.item())


                if config.partial_finetune == False: loss_m_d.update(D_loss.item())
                else: loss_m_d.update(0.00)
   

            # at the end of each epoch 
            epoch = epoch + 1
            print("Epoch {} | Acc. {:.4f} | D_loss {:.4f} | G_loss {:.4f}".
                    format(epoch, acc_m.avg, loss_m_d.avg, loss_m_g.avg)) 

            print("Con loss {:.4f} | l2 loss {:.4f}".format(ls_con.avg, ls_l2.avg))

            
            if (config.is_save_model and epoch >= config.start_saving_epoch):
                save_one_model(self.net_print, self.optimizer_G, epoch)

            if (epoch !=0 and epoch % 50 == 0):
                self.scheduler.step()
                print("learning rate ", self.optimizer_G.param_groups[0]["lr"])
         

if __name__ == "__main__":
    t = VerifTrain()
    t.train()

#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