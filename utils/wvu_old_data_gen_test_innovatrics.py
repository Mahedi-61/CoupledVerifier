import os
import shutil 

root = "/media/lab320/SSDDrive/reproduce/datasets/amar_clean_r60_as_test"
print_dir = os.path.join(root, "print")
photo_dir = os.path.join(root, "photo")

dst_pr_dir = os.path.join(root, "test/print") 
dst_ph_dir = os.path.join(root, "test/photo")

os.makedirs(dst_pr_dir, exist_ok=True)
os.makedirs(dst_ph_dir, exist_ok=True)


for f in os.listdir(print_dir):
    sub_id, fid = f.split("_")

    # print
    src_pr_dir = os.path.join(print_dir, f)
    src_pr_img = os.path.join(src_pr_dir, os.listdir(src_pr_dir)[0])
    dst_pr_img = os.path.join(dst_pr_dir, f + ".png")


    # photo
    src_ph_dir = os.path.join(photo_dir, f)
    src_ph_img = os.path.join(src_ph_dir, os.listdir(src_ph_dir)[0])
    dst_ph_img = os.path.join(dst_ph_dir,  f + ".png")

    shutil.copy(src_pr_img, dst_pr_img)
    shutil.copy(src_ph_img, dst_ph_img)

"""
dst_pr_dir = "./wvu_old/print"
dst_ph_dir = "./wvu_old/photo"
l_ph = [f.split(".")[0] for f in os.listdir(dst_ph_dir)]
l_pr = [f.split(".")[0] for f in os.listdir(dst_pr_dir)]

assert (l_ph == l_pr), "images' don't match"
print("success")
"""