import os 
import shutil
import cv2 
import numpy as np 


test_dir = "/media/lab320/SSDDrive/reproduce/datasets/amar_clean_r60_as_test"
test_ph_dir = os.path.join(test_dir, "photo")
test_pr_dir = os.path.join(test_dir, "print")
test_sub = set([sub.split("_")[0] for sub in (os.listdir(test_ph_dir))])

print("total test subjects: ", len(test_sub))

fingerprint_dir = "/mnt/tea/Fingerprint"
print_sub = set(os.listdir(fingerprint_dir))
common = test_sub.intersection(print_sub)
diff = print_sub.difference(test_sub)

output_dir = "/mnt/tea/UPKE"
print("total print sub", len(print_sub))
print("total_common: ", len(common))
print("total difference: ", len(diff))

diff = list(diff)

final_finger_dirs = [os.path.join(fingerprint_dir, folder) for folder in diff]
c = 0 

for finger_dir in final_finger_dirs:
    date_dirs = [os.path.join(finger_dir, date) for date in os.listdir(finger_dir)]
   
    #out_dir = os.path.join(output_dir, finger_dir.split("/")[-1])
    #os.makedirs(out_dir, exist_ok=True)
    #print(finger_dir)
    for date in date_dirs:
        cmv_dir = os.path.join(date, "UPEK EikonTouch 700")
        cmv_session_dirs = [os.path.join(cmv_dir, s) for s in os.listdir(cmv_dir)]
        for cmv_session in cmv_session_dirs:
            imgs_dir = [os.path.join(cmv_session, img) 
                        for img in os.listdir(cmv_session) if img != "Thumbs.db"]
            
            for img in imgs_dir:
                fprint = cv2.imread(img)
                if np.any(fprint) == False: print(img)
                else: 
                    c += 1
                    dst_img = (img.split("/")[-1]).split(".")[0] + ".png"
                    dst_img_dir = os.path.join(output_dir, dst_img) 
                    cv2.imwrite(dst_img_dir, fprint)

           #a = [shutil.copy(src_img, out_dir) for src_img in imgs_dir]
                
print(c)
print("Alhamdullilah")


"""
a = [shutil.copytree(os.path.join(photo_dir, c), os.path.join(dst_dir, c)) for c in common]

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