import os
import shutil 

"""
p_files = os.listdir("photo")
pvfp_files = os.listdir("photo_v")
assert (p_files == pvfp_files), "folders' don't match"
"""


ls = [7, 9]
print_dir = "./print"
photo_dir = "./photo"

dst_pr_dir = "./wvu_new/v/79/print"
dst_ph_dir = "./wvu_new/v/79/photo"
os.makedirs(dst_pr_dir, exist_ok=True)
os.makedirs(dst_ph_dir, exist_ok=True)

for f in os.listdir(print_dir):
    for number in ls: 
        src_img = f + "_" + str(number) + ".png"
        src_pr_img_dir = os.path.join(print_dir, f,  src_img)
        src_ph_img_dir = os.path.join(photo_dir, f, src_img)

        shutil.copy(src_pr_img_dir, dst_pr_dir)
        shutil.copy(src_ph_img_dir, dst_ph_dir)


assert (os.listdir(dst_ph_dir) == os.listdir(dst_pr_dir)), "folders' don't match"
print("success")