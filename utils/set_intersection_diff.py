import os 
import shutil

"""
test_dir = "/media/lab320/SSDDrive/reproduce/datasets/amar_clean_r60_as_test"
org_test_ph_dir = os.path.join(test_dir, "photo")
org_test_pr_dir = os.path.join(test_dir, "print")
org_test_sub = [sub for sub in (os.listdir(org_test_ph_dir))] #.split("_")[0] 
"""

train_dir = "/media/lab320/SSDDrive/reproduce/datasets/amar_clean_r60_as_train"
org_train_ph_dir = os.path.join(train_dir, "photo")
org_train_pr_dir = os.path.join(train_dir, "print")
#org_train_sub = [sub for sub in (os.listdir(org_train_ph_dir))]



fingerprint_dir = "/mnt/tea/Fingerprint"
photo_dir = "/mnt/tea/Databases/Old/Cropped_2012"
dst_dir = "/mnt/tea/Databases/Old/insh"
#org_train_sub = [sub.split("_")[0] for sub in (os.listdir(org_train_dir))]


print_sub = set(os.listdir(fingerprint_dir))
photo_sub = set(os.listdir(photo_dir))
common = list(photo_sub.intersection(print_sub))


a = [shutil.copytree(os.path.join(photo_dir, c), os.path.join(dst_dir, c)) for c in common]


"""
all_sub_finger = {}
for finger_id in os.listdir(org_test_ph_dir):
    finger_id_dir = os.path.join(org_test_ph_dir, finger_id) 
    sub_id, fnum = finger_id.split("_")[0], finger_id.split("_")[1]

    if sub_id not in list(all_sub_finger.keys()):
        all_sub_finger[sub_id] = [fnum]

    else:
        all_sub_finger[sub_id].append(fnum)

print(len(d_sub))
print(len(list((all_sub_finger.keys()))))
"""
"""
for key in all_sub_finger.keys():
    if key in d_sub:
        fingers = all_sub_finger[key]
        for f in fingers:
            src_ph = os.path.join(org_test_ph_dir, key+"_"+f)
            src_pr = os.path.join(org_test_pr_dir, key+"_"+f)

            shutil.move(src_ph, org_train_ph_dir)
            shutil.move(src_pr, org_train_pr_dir)
"""