import numpy as np 
import torch 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import config 
from torch.autograd import Variable
from model import *
from utils_wvu_old import *
import dataset_wvu_old 
from  torch.optim.lr_scheduler import ExponentialLR

class VerifTrain:
    def __init__(self):
        print("loading dataset ...")
        self.train_loader = DataLoader(
            dataset_wvu_old.WVUOldVerifier(train = config.is_train),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 4  
        )

        self.val_loader = DataLoader(
            dataset_wvu_old.WVUOldVerifier(train = not config.is_train),
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers= 4  
        )

        print("Join Type: ", config.join_type)

        self.net_photo = Mapper(pre_network="resnet18", 
                            join_type = config.join_type,
                            img_dim = config.img_dim, 
                            out_dim=config.feature_dim) 
        self.net_photo.to(config.device)
        self.net_photo.train()

        self.net_print = Mapper(pre_network="resnet18", 
                            join_type = config.join_type,
                            img_dim = config.img_dim, 
                            out_dim=config.feature_dim) 
        self.net_print.to(config.device)
        self.net_print.train()

        self.disc_photo = Discriminator(join_type = config.join_type, in_channels=config.img_dim)
        self.disc_photo.to(config.device)
        self.disc_photo.train()

        self.disc_print = Discriminator(join_type = config.join_type, in_channels=config.img_dim)
        self.disc_print.to(config.device)
        self.disc_print.train()

        self.fidentity = FingerIdentity()
        self.fidentity.to(config.device)
        self.fidentity.train()

        if config.multi_gpus:
            self.net_photo = torch.nn.DataParallel(self.net_photo)
            self.net_print = torch.nn.DataParallel(self.net_print)
            self.disc_photo = torch.nn.DataParallel(self.disc_photo)
            self.disc_print = torch.nn.DataParallel(self.disc_print)
            self.fidentity = torch.nn.DataParallel(self.fidentity)
        

        self.adversarial_loss =  torch.nn.MSELoss()
        self.L2_Norm_loss = torch.nn.MSELoss()
        self.criterion =  torch.nn.BCEWithLogitsLoss()
        self.iter = 0

        self.optimizer_G = torch.optim.Adam(
            list(self.net_photo.parameters()) + list(self.net_print.parameters()), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)

        self.optimizer_D = torch.optim.Adam(
            list(self.disc_photo.parameters()) + list(self.disc_print.parameters()), 
            lr = config.learning_rate,
            weight_decay = config.weight_decay)


        self.scheduler = ExponentialLR(self.optimizer_G, gamma=0.9)

        if config.is_load_model:
            print("loading pretrained model for one finger")
            if config.num_join_fingers == 1:
                checkpoint = load_one_checkpoint() 

            elif config.num_join_fingers == 2:
                checkpoint = load_two_checkpoint()

            self.net_photo.load_state_dict(checkpoint["net_photo"])
            self.net_print.load_state_dict(checkpoint["net_print"])
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            self.disc_photo.load_state_dict(checkpoint["disc_photo"])
            self.disc_print.load_state_dict(checkpoint["disc_print"])


    def validate(self):
        self.net_photo.eval()
        self.net_print.eval()

        ls_sq_dist = []
        ls_labels = []
            
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

        auc, eer = calculate_scores(ls_labels, ls_sq_dist)
        self.net_photo.train()
        self.net_print.train() 
        return auc, eer


    def train(self):
        train_loop = tqdm(self.train_loader)
        Tensor = torch.cuda.FloatTensor
        best_auc = 0

        for epoch in range(config.num_epochs):
            loss_m_d = AverageMeter()
            loss_m_g = AverageMeter()
            acc_m = AverageMeter()
            ls_labels = []
            ls_sq_dist = []
            
            for img_photo, img_print, label in train_loop:
                label = label.type(torch.float)

                img_photo = img_photo.to(config.device)
                img_print = img_print.to(config.device)
                label = label.to(config.device)

                # implementing coupled gan loss
                # Mapper return gen_image & embedding 
                # valid = 1; fake = 0
                valid = Variable(Tensor(img_photo.size(0), 1).fill_(1.0), 
                                    requires_grad=False)
                fake = Variable(Tensor(img_photo.size(0), 1).fill_(0.0), 
                                    requires_grad=False)

                ## generator
                self.optimizer_G.zero_grad() 
                fake_photo, embd_photo = self.net_photo(img_photo)
                fake_print, embd_print = self.net_print(img_print)

                pred_fake_photo = self.disc_photo(fake_photo)
                pred_fake_print = self.disc_print(fake_print)

                #identity loss
                out = self.fidentity(embd_photo, embd_print)
                out = torch.squeeze(out, dim=1)
                identity_loss = self.criterion(out, label)

                #l2 loss
                l2_loss = (self.L2_Norm_loss(fake_photo, img_photo) + 
                          self.L2_Norm_loss(fake_print, img_print)) / 2
                         

                # gan loss
                gan_loss = (self.adversarial_loss(pred_fake_photo, valid) + 
                           self.adversarial_loss(pred_fake_print, valid)) / 2
                
                # contrastive loss 
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

               
                identity_loss.backward(retain_graph = True)
                G_loss =  con_loss + gan_loss*config.delta_1 + l2_loss*config.delta_2 + identity_loss
                G_loss.backward()
                self.optimizer_G.step() 

                # calculating acc, generator loss and 
                acc = (dist_sq < config.margin).type(torch.float)
                acc = (acc == label).type(torch.float)
                acc_m.update(acc.mean())

                loss_m_g.update(G_loss.item())
                ls_sq_dist.append(dist_sq.data)
                ls_labels.append((1- label).data)

                # discriminator 
                # real
                self.optimizer_D.zero_grad()
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
                self.optimizer_D.step() 
                loss_m_d.update(D_loss.item())
   

            # at the end of each epoch 
            epoch = epoch + 1
            print("Epoch {} | Acc. {} | D_loss {} | G_loss {}".format(
                epoch, acc_m.avg, loss_m_d.avg, loss_m_g.avg ))
            print("Con loss {} | gan loss {} | l2 loss {}| identity loss {}".format(
                            con_loss, gan_loss, l2_loss, identity_loss))

            
            if (config.is_save_model and epoch >= 50):
                auc, eer = self.validate() 
                if (auc > best_auc):
                    best_auc = auc  
                    save_model(self.net_photo, self.net_print, self.optimizer_G, 
                        self.disc_photo, self.disc_print, epoch, is_best=True)

            if (config.is_save_model and epoch >=30 and epoch%10 == 0):
                save_model(self.net_photo, self.net_print, self.optimizer_G, 
                    self.disc_photo, self.disc_print, epoch, is_best=False)

            if (epoch !=0 and epoch % 80 == 0):
                self.scheduler.step()
                print("learning rate ", self.optimizer_G.param_groups[0]["lr"])
                #self.save_images() 
            #"""
            

    def save_images(self):
        print("saving images ")
        self.net_photo.eval()
        self.net_print.eval() 

        # making folders
        self.iter += 1
        saved_dir = os.path.join(config.saved_img_dir, str(self.iter))
        os.makedirs(saved_dir, exist_ok=True)
        
        for b_id, (img_photo, img_print, label) in enumerate(self.train_loader):

            img_photo = img_photo.to(config.device)
            syntheic_fprint = self.net_print.module.DecodeImage(
                self.net_photo.module.EncodeImage(img_photo)
            )

            # saving images in checkpoint directory 
            sample_img = torch.cat((img_photo, syntheic_fprint), dim=-2)
            save_image(sample_img, "%s/%s.png" %(saved_dir, b_id+1), 
                        nrow=img_photo.size(0), normalize=True)

        self.net_photo.train()
        self.net_print.train()
         

if __name__ == "__main__":
    t = VerifTrain()