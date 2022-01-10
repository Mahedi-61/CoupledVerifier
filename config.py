import os
import torch 

#conditions
dataset_name = "wvu_new" #wvu_new, wvu_old
if dataset_name == "wvu_new": photo_type = "vfp" #vfp

is_train = False                                      
is_load_model = True                                                                
is_finetune = False #on other dataset (same number of finger)
is_one_fid = False #train and test only on finger ID 
partial_finetune = False  # only finetuning the last layer of the pretrained model
is_convert_one_to_many = False                                                               
num_join_fingers = 1

fnums = [["2", "3", "7", "8"]]                                                                    
w_name = "F1_D2vfp_A2"  # weight save --> train | weights load --> test

 #weights for initialization
if is_convert_one_to_many or partial_finetune: 
    save_w_name = "F3_D1_IDALL_A1"

is_all_pairs = False   
combined_w_name = "F3_D1_IDALL_A1" #require for score fusion   
multi_gpus = True  
img_dim = num_join_fingers
is_save_model = is_train 


# training parameters
num_imposter = 2
num_pair_test = 10
batch_size = 48
learning_rate = 0.00009
weight_decay = 5e-4
num_epochs = 250
start_saving_epoch = 20

if is_all_pairs:
    if dataset_name == "wvu_old":
        if num_join_fingers == 2:
            all_fnums = [["2", "3"], ["3", "4"], ["7", "8"], ["8", "9"]]

        elif num_join_fingers == 3:
            all_fnums =  [["2", "3", "4"], ["2", "3", "7"], 
                          ["2", "7", "8"], ["3", "7", "8"]]

        elif num_join_fingers == 4:
            all_fnums = [["2", "3", "7", "8"]]

    if dataset_name == "wvu_new":
        if num_join_fingers == 2:
            all_fnums = [["7", "8"], ["8", "9"], ["9", "10"]]


# model hyperparameters
feature_dim = 256 
delta_l1 = 100
delta_l2 = 1
delta_gan = 1
# 65 == wvu_old; 
margin = 65 #### 55 = g_vs_vfp (attention_unet) + g_vs_v (all) 
img_size = 256
eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# directories
root_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(root_dir, "..", "datasets")

#if dataset_name == "wvu_new": dataset_type = "g_vs_gr"  #"g_vs_v", "g_vs_vfp"

if dataset_name == "wvu_old":
    train_dataset = "clean_12_as_train"  #clean
    test_dataset =  "clean_13_as_test" #clean


elif dataset_name == "wvu_new":
    train_dataset = "wvu_new_train"
    test_dataset =  "wvu_new_test"

train_photo_dir = os.path.join(datasets_dir, train_dataset, "photo")
train_print_dir = os.path.join(datasets_dir, train_dataset, "print")

test_photo_dir = os.path.join(datasets_dir, test_dataset, "photo")
test_print_dir = os.path.join(datasets_dir, test_dataset, "print")
checkpoint_dir = os.path.join(root_dir, "checkpoints", dataset_name)

saved_img_dir = os.path.join(checkpoint_dir, "images")
saved_data_dir = os.path.join(checkpoint_dir, "data")

old_weights_dir = os.path.join(root_dir, "checkpoints", "wvu_old")
new_weights_dir = os.path.join(root_dir, "checkpoints", "wvu_new")


# weights, model, best model
if dataset_name == "wvu_old":
    w_dir = os.path.join(old_weights_dir, w_name)
    combined_w_dir = os.path.join(old_weights_dir, combined_w_name)

elif dataset_name == "wvu_new":
    if is_finetune == False:
        if is_convert_one_to_many == False: 
            w_dir = os.path.join(new_weights_dir, w_name)

        elif is_convert_one_to_many == True:
             w_dir = os.path.join(new_weights_dir, w_name)
             combined_w_dir = os.path.join(new_weights_dir, combined_w_name)


if w_name.split("_")[-1] == "A1":
    model_file = os.path.join(w_dir, "model_res18_m%d_" %margin)

if w_name.split("_")[-1] == "A2":
    model_file = os.path.join(w_dir, "model_atten_m%d_" %margin)

best_model = os.path.join(w_dir, "best_model_" )