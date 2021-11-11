import os
import torch 

# directories
dataset_name = "wvu_old" #wvu_new
root_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(root_dir, "..", "datasets")

if dataset_name == "wvu_old":
    train_dataset = "clean_12_as_train" 
    test_dataset =  "clean_13_as_test" 

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
weights_one_dir = os.path.join(checkpoint_dir, "weights_one")
weights_two_dir = os.path.join(checkpoint_dir, "weights_two")


#conditions
is_train = False                     
num_join_fingers = 2
join_type =  "channel" #channel, concat, none 
fnums = ["3", "4"]
img_dim = (1 if join_type == "concat" else num_join_fingers)   
multi_gpus = True  
is_save_model = is_train 
is_load_model = True                                               

# model hyperparameters
num_imposter = 4
num_pair_test = 10

batch_size = 96
learning_rate = 0.0002
weight_decay = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 300

# training parameters
feature_dim = 256 
delta_1 = 0.01
delta_2 = 1
margin = 75 
img_size = 256
eps = 1e-8

if num_join_fingers == 1:
    model_file = os.path.join(weights_one_dir, "model_res18_m75_")
    best_model = os.path.join(weights_one_dir, "best_model_000" )

if num_join_fingers == 2:
    model_file = os.path.join(weights_two_dir, "model_res18_m75_")
    best_model = os.path.join(weights_two_dir, "best_model_000" )
    