from matplotlib import pyplot as plt 
from torchvision.utils import save_image 
import torch 
import config 
import os 
from sklearn import metrics
import numpy as np 

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count 

# min-max normalization of numpy array
def norm_minmax(x):
    return (x - x.min()) / (x.max() - x.min())


# input: list of tensor
def plot_tensors(tensors, title):
    for i in range(len(tensors)):
        ten_numpy = tensors[i].cpu().detach().numpy().squeeze()
        ten_numpy = norm_minmax(ten_numpy)

        if len(ten_numpy.shape) > 2:
            ten_numpy = ten_numpy.transpose(1, 2, 0)

        plt.subplot(1, len(tensors), i+1)
        plt.title(title)
        plt.imshow(ten_numpy, cmap="gray")
    
    plt.show() 


def save_model(net_photo, net_print, fidentity, optimizer_G, disc_photo, disc_print, epoch, is_best):
    checkpoint = {}

    checkpoint["net_photo"] = net_photo.state_dict()
    checkpoint["net_print"] = net_print.state_dict()
    checkpoint["fidentity"] = fidentity.state_dict()
    checkpoint["optimizer_G"] = optimizer_G.state_dict()
    checkpoint["disc_photo"] = disc_photo.state_dict()
    checkpoint["disc_print"] = disc_print.state_dict() 

    if is_best == False:
        print("saving model for epoch: ", str(epoch))
        torch.save(checkpoint, config.model_file + str(epoch) + ".pth")
    
    elif is_best == True:
        print("saving best model so far")
        torch.save(checkpoint, config.best_model + str(epoch) + ".pth")



# loading images in a dictionary
def get_img_dict(photo_path, print_path):
    photo_finger_dict = {}
    print_finger_dict = {}

    index = 0
    for sub_id in os.listdir(photo_path):
        sub_dir = os.path.join(photo_path, sub_id) 
        
        for img_file in os.listdir(sub_dir):
            finger_id = img_file.split(".")[0]
            photo_finger_dict[index] = [finger_id, os.path.join(sub_dir, img_file)] 

            index += 1

        # for finger print
        sub_dir = os.path.join(print_path, sub_id)

        if(os.path.isdir(sub_dir)):
            for img_file in os.listdir(sub_dir):
                finger_id = img_file.split(".")[0]
                print_finger_dict[finger_id] = [os.path.join(sub_dir, img_file)]

    return photo_finger_dict, print_finger_dict


def get_multiple_img_dict(photo_path, print_path, ls_fnums):
    photo_finger_dict = {}
    print_finger_dict = {}

    index = 0
    for sub_id in os.listdir(photo_path):
        for fnums in ls_fnums:
            sub_dir = os.path.join(photo_path, sub_id) 

            first_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[0] + ".png")
            if config.num_join_fingers == 1:
                photo_finger_dict[index] = [sub_id + "_" + str(index), first_f_dir]

            else:
                second_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[1] + ".png")
                if config.num_join_fingers == 2:
                    photo_finger_dict[index] = [sub_id + "_" + str(index), [first_f_dir, second_f_dir]] 

                else:
                    third_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[2] + ".png")
                    if config.num_join_fingers == 3:
                        photo_finger_dict[index] = [sub_id + "_" + str(index), [first_f_dir, second_f_dir, third_f_dir]]

                    else:
                        fourth_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[3] + ".png")
                        if config.num_join_fingers == 4:
                            photo_finger_dict[index] = [sub_id + "_" + str(index), [first_f_dir, 
                                                            second_f_dir, third_f_dir, fourth_f_dir]]

            # for finger print
            sub_dir = os.path.join(print_path, sub_id)
            if(os.path.isdir(sub_dir)):

                first_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[0] + ".png")
                if config.num_join_fingers == 1:
                    print_finger_dict[sub_id + "_" + str(index)] = [first_f_dir] 

                elif config.num_join_fingers >= 2:
                    second_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[1] + ".png")

                    if config.num_join_fingers == 2:
                        print_finger_dict[sub_id + "_" + str(index)] = [first_f_dir, second_f_dir] 

                    elif config.num_join_fingers >= 3:
                        third_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[2] + ".png")

                        if config.num_join_fingers == 3:
                            print_finger_dict[sub_id + "_" + str(index)] = [first_f_dir, second_f_dir, third_f_dir] 

                        elif config.num_join_fingers == 4:
                            fourth_f_dir = os.path.join(sub_dir, sub_id + "_" + fnums[3] + ".png")
                            print_finger_dict[sub_id + "_" + str(index)] = [first_f_dir, second_f_dir, 
                                                                        third_f_dir, fourth_f_dir] 

            index += 1

    print("Joint Fingers ID: ", ls_fnums)
    print("Number of Data: ", len(photo_finger_dict))
    print(len(print_finger_dict))
    return photo_finger_dict, print_finger_dict



# calculating scores 
def calculate_scores(ls_labels, ls_sq_dist, is_ensemble):
    
    if is_ensemble == False: 
        pred_ls = torch.cat(ls_sq_dist, 0)
    elif is_ensemble == True:
        pred_ls = ls_sq_dist
        
    true_label = torch.cat(ls_labels, 0)
    pred_ls = pred_ls.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy() 
    
    # sklearn always takes (y_true, y_pred)
    fprs, tprs, threshold = metrics.roc_curve(true_label, pred_ls)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    auc = metrics.auc(fprs, tprs)
    print("AUC {:.4f} | EER {:.4f}".format(auc, eer))
    #np.save("%s/lbl_test.npy" %(config.saved_data_dir), true_label)
    #np.save("%s/dist_test.npy" %(config.saved_data_dir), pred_ls)
    
    return auc, eer 


if __name__ == "__main__":
    #t = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
    #a = AverageMeter()
    ph_dict, pr_dict =  get_multiple_img_dict(config.train_photo_dir, 
                            config.train_print_dir, [["7", "8", "9"]]) 
