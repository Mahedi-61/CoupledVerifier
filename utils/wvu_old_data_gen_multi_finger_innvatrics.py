import os
import shutil 


ls_fnums = ["8", "9"]
photo_dir = "./photo"
print_dir = "./print"

dst_ph_dir = "./wvu_old_prep/89/photo"
dst_pr_dir = "./wvu_old_prep/89/print"

os.makedirs(dst_ph_dir, exist_ok=True)
os.makedirs(dst_pr_dir, exist_ok=True)

all_sub_finger = {}

for finger_id in os.listdir(photo_dir):
    sub_id, fnum = finger_id.split("_")[0], finger_id.split("_")[1]

    if sub_id not in list(all_sub_finger.keys()):
        all_sub_finger[sub_id] = [fnum]

    else:
        all_sub_finger[sub_id].append(fnum)

for key in list(all_sub_finger.keys()):

    if set(ls_fnums).issubset(set(all_sub_finger[key])):
        first_fph_dir = os.path.join(photo_dir, key + "_" + ls_fnums[0])
        first_fph_img_dir = os.path.join(first_fph_dir, os.listdir(first_fph_dir)[0])

        first_fpr_dir = os.path.join(print_dir, key + "_" + ls_fnums[0])
        first_fpr_img_dir = os.path.join(first_fpr_dir, os.listdir(first_fpr_dir)[0])
        img_1 =  key + "_" + ls_fnums[0] + ".png"

        #copying photo and prints
        shutil.copy(first_fph_img_dir, os.path.join(dst_ph_dir, img_1))
        shutil.copy(first_fpr_img_dir, os.path.join(dst_pr_dir, img_1))

        if len(ls_fnums) >= 2:
            second_fph_dir = os.path.join(photo_dir, key + "_" + ls_fnums[1])
            second_fph_img_dir = os.path.join(second_fph_dir, os.listdir(second_fph_dir)[0])

            second_fpr_dir = os.path.join(print_dir, key + "_" + ls_fnums[1])
            second_fpr_img_dir = os.path.join(second_fpr_dir, os.listdir(second_fpr_dir)[0])

            img_2 =  key + "_" + ls_fnums[1] + ".png"
            shutil.copy(second_fph_img_dir, os.path.join(dst_ph_dir, img_2))
            shutil.copy(second_fpr_img_dir, os.path.join(dst_pr_dir, img_2))

            if len(ls_fnums) >= 3:
                third_fph_dir = os.path.join(photo_dir, key + "_" + ls_fnums[2])
                third_fph_img_dir = os.path.join(third_fph_dir, os.listdir(third_fph_dir)[0])
            
                third_fpr_dir = os.path.join(print_dir, key + "_" + ls_fnums[2])
                third_fpr_img_dir = os.path.join(third_fpr_dir, os.listdir(third_fpr_dir)[0])

                img_3 =  key + "_" + ls_fnums[2] + ".png"
                shutil.copy(third_fph_img_dir, os.path.join(dst_ph_dir, img_3))
                shutil.copy(third_fpr_img_dir, os.path.join(dst_pr_dir, img_3))

                if len(ls_fnums) == 4:
                    fourth_fph_dir = os.path.join(photo_dir, key + "_" + ls_fnums[3])
                    fourth_fph_img_dir = os.path.join(fourth_fph_dir, os.listdir(fourth_fph_dir)[0])

                    fourth_fpr_dir = os.path.join(print_dir, key + "_" + ls_fnums[3])
                    fourth_fpr_img_dir = os.path.join(fourth_fpr_dir, os.listdir(fourth_fpr_dir)[0])

                    img_4 =  key + "_" + ls_fnums[3] + ".png"
                    shutil.copy(fourth_fph_img_dir, os.path.join(dst_ph_dir, img_4))
                    shutil.copy(fourth_fpr_img_dir, os.path.join(dst_pr_dir, img_4))


assert (len(os.listdir(dst_ph_dir)) == len(os.listdir(dst_pr_dir))); "failed"
print("success")
