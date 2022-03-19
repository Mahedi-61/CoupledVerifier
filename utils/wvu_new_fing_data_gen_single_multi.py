import os
import shutil 

ls = [10] 
print_dir = "/media/lab320/SSDDrive/reproduce/datasets/wvu_new_combine/print"
photo_dir = "/media/lab320/SSDDrive/reproduce/datasets/wvu_new_combine/photo"

dst_pr_dir = "/media/lab320/SSDDrive/reproduce/datasets/wvu_new_combine/wvu_new_gr/%d/print" % (ls[0])
dst_ph_dir = "/media/lab320/SSDDrive/reproduce/datasets/wvu_new_combine/wvu_new_gr/%d/photo" % (ls[0])
os.makedirs(dst_pr_dir, exist_ok=True)
os.makedirs(dst_ph_dir, exist_ok=True)

for f in os.listdir(print_dir):
    for number in ls: 
        src_img = f + "_" + str(number) + ".png"
        dst_img = f + "_" + str(number) + ".bmp"
        src_pr_img_dir = os.path.join(print_dir, f,  src_img)
        src_ph_img_dir = os.path.join(photo_dir, f, dst_img)

        shutil.copy(src_pr_img_dir, dst_pr_dir)
        shutil.copy(src_ph_img_dir, dst_ph_dir)

"""
assert (os.listdir(dst_ph_dir) == os.listdir(dst_pr_dir)), "folders' don't match"
print("Alhamdullilah")
"""