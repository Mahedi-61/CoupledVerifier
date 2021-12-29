import os
import torch 

#conditions
dataset_name = "wvu_old" #wvu_new, wvu_old
is_train = False                                     
is_finetune = False #on other dataset (same number of finger)
is_convert_one_to_many = False                                                  
num_join_fingers = 3
join_type =  "channel" #channel, concat, none 
fnums = [["2", "3", "4"]]
multi_gpus = True  
is_save_model = is_train 
is_load_model = True                                                                        
img_dim = (1 if join_type == "concat" else num_join_fingers)   

w_name = "F3_D1_IDALL_A1"
if is_convert_one_to_many: save_w_name = "F1_D1_A1" #weights for initialization
is_all_pairs = False  

# training parameters
num_imposter = 2
num_pair_test = 10
batch_size = 64
learning_rate = 0.0003
weight_decay = 5e-4
num_epochs = 350
start_saving_epoch = 30

if is_all_pairs:
    if num_join_fingers == 2:
        all_fnums = [["2", "3"], ["3", "4"], ["7", "8"], ["8", "9"]]

    elif num_join_fingers == 3:
        all_fnums =  [["2", "3", "4"], ["2", "3", "7"], ["2", "7", "8"], ["3", "7", "8"]]

# model hyperparameters
feature_dim = 256 
delta_l1 = 100
delta_l2 = 1
delta_gan = 1
margin = 65 
img_size = 256
eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# directories
root_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(root_dir, "..", "datasets")

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

elif dataset_name == "wvu_new":
    if num_join_fingers == 1: 
        w_dir = os.path.join(new_weights_dir, "weights_one")

    elif num_join_fingers == 2: 
        w_dir = os.path.join(new_weights_dir, "weights_two")

model_file = os.path.join(w_dir, "model_res18_m75_")
best_model = os.path.join(w_dir, "best_model_" )