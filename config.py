import os
import torch 

#conditions
dataset_name = "pretrain" #wvu_new, wvu_old
photo_type = ""
if dataset_name == "wvu_new": photo_type = "v" #vfp

is_train = True                                                                                                                                                                                                                                                       
is_load_model = False                                                                                             
is_one_fid = False              # train and test only on the finger ID 
partial_finetune = False  # only finetuning the last layer of the pretrained model
is_finetune = True                           
is_convert_one_to_many = False                                                                                    
num_join_fingers = 1

fnums =  [[""]]                                                                    
w_name = "bin_F1_D3_A1"  #weight save --> train | weights load --> test

 #weights for initialization
save_w_name = ""
conversion_type = ""
pretrain_type = "bin" #gr, g


# for testing 
is_all_pairs = False                   
combined_w_name = "" #require for score fusion   
num_test_aug = 2
is_test_augment = True
is_ensemble = True                  

load_pretrain_weights = False     
multi_gpus = True  
img_dim = num_join_fingers
is_save_model = is_train 
if is_train == True: is_test_augment = False       
if is_test_augment == False: num_test_aug = 1


# training parameters
is_display = True 
num_imposter = 2
num_pair_test = 70
batch_size = 544        
learning_rate = 0.002
weight_decay = 5e-4
num_epochs = 550
start_saving_epoch = 400  

if is_all_pairs:
    if dataset_name == "wvu_old":
        if num_join_fingers == 2:
            all_fnums = [["2", "3"], ["3", "4"], ["7", "8"], ["8", "9"], ["2", "7"]]

        elif num_join_fingers == 3:
            all_fnums =  [["2", "3", "4"], ["2", "3", "7"], 
                          ["2", "7", "8"], ["3", "7", "8"]]

        elif num_join_fingers == 4:
            all_fnums = [["2", "3", "7", "8"]]

    if dataset_name == "wvu_new":
        if num_join_fingers == 2:
            all_fnums = [["7", "8"], ["8", "9"], ["9", "10"], ["7", "9"]]
        
        if num_join_fingers == 3:
            all_fnums = [["7", "8", "9"], ["8", "9", "10"], ["7", "9", "10"]]
        
        if num_join_fingers == 4:
            all_fnums = [["7", "8", "9", "10"]]

# model hyperparameters
feature_dim = 256 
delta_l1 = 100
delta_l2 = 1
delta_gan = 1
# 65 == wvu_old; wvu_new vfp; 
margin = 65
img_size = 256
eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# directories
root_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(root_dir, "..", "datasets")


if dataset_name == "wvu_old":
    train_dataset = "amar_clean_all_as_train" 
    test_dataset =  "amar_clean_r60_as_test" #clean

elif dataset_name == "wvu_new":
    train_dataset = "wvu_new_train"
    test_dataset =  "wvu_new_combine"

elif dataset_name == "pretrain":
    if pretrain_type == "bin": train_dataset = "bin_pretrain" 
    elif pretrain_type == "gr": train_dataset = "gr_pretrain"  
    elif pretrain_type == "g": train_dataset = "g_pretrain" 
    test_dataset = "wvu_new_combine"


train_photo_dir = os.path.join(datasets_dir, train_dataset, "photo")
train_print_dir = os.path.join(datasets_dir, train_dataset, "print")

test_photo_dir = os.path.join(datasets_dir, test_dataset, "photo")
test_print_dir = os.path.join(datasets_dir, test_dataset, "print")

dataset_cp_dir = os.path.join(root_dir, "checkpoints", dataset_name)
saved_img_dir = os.path.join(dataset_cp_dir, "images")
saved_data_dir = os.path.join(dataset_cp_dir, "data")

old_weights_dir = os.path.join(root_dir, "checkpoints", "wvu_old")
new_weights_dir = os.path.join(root_dir, "checkpoints", "wvu_new", photo_type)
pretrain_weights_dir = os.path.join(root_dir, "checkpoints", "pretrain")


# weights, model, best model
if dataset_name == "wvu_old":
    w_dir = os.path.join(old_weights_dir, w_name)
    combined_w_dir = os.path.join(old_weights_dir, combined_w_name)

elif dataset_name == "wvu_new":
    w_dir = os.path.join(new_weights_dir, w_name)
    combined_w_dir = os.path.join(new_weights_dir, combined_w_name)

elif dataset_name == "pretrain":
    w_dir = os.path.join(pretrain_weights_dir, w_name)


if w_name.split("_")[-1] == "A1":
    model_file = os.path.join(w_dir, "model_res18_m%d_" %margin)

if w_name.split("_")[-1] == "A2":
    model_file = os.path.join(w_dir, "model_atten_m%d_" %margin)

if w_name.split("_")[-1] == "A5":
    model_file = os.path.join(w_dir, "model_dense_m%d_" %margin)

best_model = os.path.join(w_dir, "best_model_" )


if is_finetune == True:
    if conversion_type == "PvsO":
         save_w_dir = os.path.join(pretrain_weights_dir, save_w_name)

    if conversion_type == "OvsVFP" or conversion_type == "OvsV":
        save_w_dir = os.path.join(old_weights_dir, save_w_name)

    if conversion_type == "VFPvsV":
        save_w_dir =  os.path.join(root_dir, "checkpoints", "wvu_new", "vfp", save_w_name)