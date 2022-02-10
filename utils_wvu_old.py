from matplotlib import pyplot as plt 
from torchvision.utils import save_image 
import torch 
import config 
import os 
from sklearn import metrics
import numpy as np 
import math
from collections import OrderedDict
irange = range

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


def save_model_only_encoder(net_print, net_syn_print, optimizer_AE, epoch, is_best):
    checkpoint = {}

    checkpoint["net_print"] = net_print.state_dict()
    checkpoint["net_syn_print"] = net_syn_print.state_dict()
    checkpoint["optimizer_G"] = optimizer_AE.state_dict()

    dir = "F1_D1_A3_final"
    model_file = os.path.join(config.root_dir, "checkpoints", "wvu_old", dir)
    best_model = os.path.join(config.root_dir, "checkpoints", "wvu_old", dir)
    if is_best == False:
        print("saving model for epoch: ", str(epoch))
        torch.save(checkpoint, model_file + str(epoch) + ".pth")
    
    elif is_best == True:
        print("saving best model so far")
        torch.save(checkpoint, best_model + ".pth")


# loading images in a dictionary
def get_img_dict(photo_path, print_path):
    photo_finger_dict = {}
    print_finger_dict = {}

    index = 0
    for finger_id in os.listdir(photo_path):
        id_dir = os.path.join(photo_path, finger_id) 
        
        #for img_file in os.listdir(id_dir):
        #photo_finger_dict[index] = [finger_id, os.path.join(id_dir, img_file)]
        first_f_img = os.path.join(id_dir, os.listdir(id_dir)[0])  
        photo_finger_dict[index] = [finger_id, first_f_img]
        index += 1

        # for finger print
        id_dir = os.path.join(print_path, finger_id)
        if(os.path.isdir(id_dir)):
            #images_path = [os.path.join(id_dir, img_file) for img_file in os.listdir(id_dir)]
            #print_finger_dict[finger_id] = [img_path for img_path in images_path]
            first_f_img = os.path.join(id_dir, os.listdir(id_dir)[0]) 
            print_finger_dict[finger_id] = [first_f_img]
    
    return photo_finger_dict, print_finger_dict


def get_multiple_img_dict(photo_path, print_path, ls_fnums):
    photo_finger_dict = {}
    print_finger_dict = {}

    index = 0
    all_sub_finger = {}

    for finger_id in os.listdir(photo_path):
        finger_id_dir = os.path.join(photo_path, finger_id) 
        sub_id, fnum = finger_id.split("_")[0], finger_id.split("_")[1]

        if sub_id not in list(all_sub_finger.keys()):
            all_sub_finger[sub_id] = [fnum]

        else:
            all_sub_finger[sub_id].append(fnum)

    index = 0
    all_sub_finger = OrderedDict(sorted(all_sub_finger.items()))
    for key in list(all_sub_finger.keys()):
        for fnums in ls_fnums:
            if set(fnums).issubset(set(all_sub_finger[key])):
                first_f_dir = os.path.join(photo_path, key + "_" + fnums[0])
                first_f_img = os.path.join(first_f_dir, os.listdir(first_f_dir)[0])

                dict_key = key + "_" + str(index)
                
                if config.num_join_fingers == 1:
                    photo_finger_dict[index] = [dict_key, first_f_img]

                else:
                    second_f_dir = os.path.join(photo_path, key + "_" + fnums[1])
                    second_f_img = os.path.join(second_f_dir, os.listdir(second_f_dir)[0]) 

                    if config.num_join_fingers == 2:
                        photo_finger_dict[index] = [dict_key, [first_f_img, second_f_img]]
                    
                    else:
                        third_f_dir = os.path.join(photo_path, key + "_" + fnums[2])
                        third_f_img = os.path.join(third_f_dir, os.listdir(third_f_dir)[0])

                        if config.num_join_fingers == 3:
                            photo_finger_dict[index] = [dict_key, [first_f_img, 
                                                                second_f_img, third_f_img]] 

                        else:
                            fourth_f_dir = os.path.join(photo_path, key + "_" + fnums[3])
                            fourth_f_img = os.path.join(fourth_f_dir, os.listdir(fourth_f_dir)[0])

                            if config.num_join_fingers == 4:
                                photo_finger_dict[index] = [dict_key, [first_f_img, 
                                                    second_f_img, third_f_img, fourth_f_img]] 

                # for print
                first_f_dir = os.path.join(print_path, key + "_" + fnums[0])
                first_f_img = os.path.join(first_f_dir, os.listdir(first_f_dir)[0])

                if config.num_join_fingers == 1:
                    print_finger_dict[dict_key] = [first_f_img]
                        
                elif config.num_join_fingers >= 2:
                    second_f_dir = os.path.join(print_path, key + "_" + fnums[1])
                    second_f_img = os.path.join(second_f_dir, os.listdir(second_f_dir)[0]) 

                    if config.num_join_fingers == 2:
                        print_finger_dict[dict_key] = [first_f_img, second_f_img]
                        
                    elif config.num_join_fingers >= 3:
                        third_f_dir = os.path.join(print_path, key + "_" + fnums[2])
                        third_f_img = os.path.join(third_f_dir, os.listdir(third_f_dir)[0]) 

                        if config.num_join_fingers == 3:
                            print_finger_dict[dict_key] = [first_f_img, second_f_img, third_f_img]
                            
                        elif config.num_join_fingers == 4:
                            fourth_f_dir = os.path.join(print_path, key + "_" + fnums[3])
                            fourth_f_img = os.path.join(fourth_f_dir, os.listdir(fourth_f_dir)[0]) 

                            print_finger_dict[dict_key] = [first_f_img, 
                                                    second_f_img, third_f_img, fourth_f_img]
                
                index += 1

    if config.is_test_augment == False: 
        print("Joint Fingers ID: ", ls_fnums)
        print("Number of Data: ", len(photo_finger_dict))
        
    return photo_finger_dict, print_finger_dict


# calculating scores 
def calculate_scores(ls_labels, ls_sq_dist, is_ensemble):
    
    if is_ensemble == False: 
        pred_ls = torch.cat(ls_sq_dist, 0)
        true_label = torch.cat(ls_labels, 0)

    elif is_ensemble == True:
        pred_ls = ls_sq_dist
        true_label = ls_labels

    pred_ls = pred_ls.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy() 

    # sklearn always takes (y_true, y_pred)
    fprs, tprs, threshold = metrics.roc_curve(true_label, pred_ls)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    auc = metrics.auc(fprs, tprs)

    print("AUC {:.4f} | EER {:.4f}".format(auc, eer))
    return auc, eer 
    #np.save("%s/lbl_test.npy" %(config.saved_data_dir), true_label)
    #np.save("%s/dist_test.npy" %(config.saved_data_dir), pred_ls)



def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, fp, nrow=8, padding=2, normalize=False, range=None, 
                scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


if __name__ == "__main__":
    t = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
    #a = AverageMeter()

    ph_d, pr_d = get_multiple_img_dict(config.train_photo_dir, 
                    config.train_print_dir, [["2", "3", "8", "7"]])
    
    print(len(ph_d))
    print(len(pr_d))
