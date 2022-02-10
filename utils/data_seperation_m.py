#################################################################
# Utils package
# prepare raw_data to raw_data_clean by separating photos and prnts
#################################################################

import os
import numpy as np
from shutil import copyfile


source_dir = "/media/lab320/SSDDrive/reproduce/datasets/my_train_add/"
output_dir = "/media/lab320/SSDDrive/reproduce/datasets/amar_clean_extra_as_train/"

def get_finger_type(number):
    if number == 2:
        return 'index', 'right'

    elif number == 3:
        return 'middle', 'right'

    elif number == 4:
        return 'ring', 'right'

    elif number == 7:
        return 'index', 'left'

    elif number == 8:
        return 'middle', 'left'

    elif number == 9:
        return 'ring', 'left'
    else:
        raise ValueError('finger index is not valid', str(number))


c = 0
for file in os.listdir(source_dir):
    # first get the fingerprint
    if "300LC" in file:
        # get id of the person
        fsplit = file.split('_')

        tag_num = fsplit[0]
        id = fsplit[2]
        finger = fsplit[4]

        photo_finger_dir = os.path.join(output_dir, "photo", id + "_" + finger)
        print_finger_dir = os.path.join(output_dir, "print", id + "_" + finger)

        os.makedirs(photo_finger_dir, exist_ok=True)
        os.makedirs(print_finger_dir, exist_ok=True)


        # copy fingerprint to the new directory
        copyfile(os.path.join(source_dir, file), os.path.join(print_finger_dir, file))

        # find corresponding photo
        finger_name, hand_side = get_finger_type(int(finger)) 

        for fphoto in os.listdir(source_dir):
            if (id in fphoto) and (finger_name in fphoto) and (tag_num in fphoto) and (hand_side in fphoto):
                copyfile(os.path.join(source_dir, fphoto), 
                os.path.join(photo_finger_dir, fphoto))

    c += 1
    if (c % 100 == 0): print(c)

print("data preparation successful !!")