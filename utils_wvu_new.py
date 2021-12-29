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


def save_model(net_photo, net_print, optimizer_G, disc_photo, disc_print, epoch, is_best):
    checkpoint = {}

    checkpoint["net_photo"] = net_photo.state_dict()
    checkpoint["net_print"] = net_print.state_dict()
    checkpoint["optimizer_G"] = optimizer_G.state_dict()
    checkpoint["disc_photo"] = disc_photo.state_dict()
    checkpoint["disc_print"] = disc_print.state_dict() 

    if is_best == False:
        print("saving model for epoch: ", str(epoch))
        torch.save(checkpoint, config.model_file + str(epoch) + ".pth")
    
    elif is_best == True:
        print("saving best model so far")
        torch.save(checkpoint, config.best_model + ".pth")



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
                print_finger_dict[finger_id] = os.path.join(sub_dir, img_file)

    return photo_finger_dict, print_finger_dict


def get_two_img_dict(photo_path, print_path, fnums):
    photo_finger_dict = {}
    print_finger_dict = {}

    index = 0
    for sub_id in os.listdir(photo_path):
        sub_dir = os.path.join(photo_path, sub_id) 

        first_finger_dir = os.path.join(sub_dir, sub_id + "_" + fnums[0] + ".png")
        second_finger_dir = os.path.join(sub_dir, sub_id + "_" + fnums[1] + ".png")

        photo_finger_dict[index] = [sub_id, [first_finger_dir, second_finger_dir]] 

        index += 1

        # for finger print
        sub_dir = os.path.join(print_path, sub_id)
        if(os.path.isdir(sub_dir)):
            first_finger_dir = os.path.join(sub_dir, sub_id + "_" + fnums[0] + ".png")
            second_finger_dir = os.path.join(sub_dir, sub_id + "_" + fnums[1] + ".png")

            print_finger_dict[sub_id] = [first_finger_dir, second_finger_dir] 

    print("Joint Fingers ID: ", fnums)
    print("Number of Data: ", len(photo_finger_dict))
    return photo_finger_dict, print_finger_dict



def load_one_checkpoint():
    loaded_model_file = os.path.join(config.weights_one_dir, 
                                    "best_model_000.pth")
    checkpoint = torch.load(loaded_model_file)
    return checkpoint

def load_two_checkpoint():
    loaded_model_file = os.path.join(config.weights_two_dir, 
                                    "model_res18_m75_270_f.pth")
    checkpoint = torch.load(loaded_model_file)
    return checkpoint


# calculating scores 
def calculate_scores(ls_labels, ls_sq_dist):
    pred_ls = torch.cat(ls_sq_dist, 0)
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


if __name__ == "__main__":
    #t = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
    #a = AverageMeter()
    phdict, prdict =  get_two_img_dict(config.train_photo_dir, 
                            config.train_print_dir, config.fnums) 
    
    print(prdict["6664111"])