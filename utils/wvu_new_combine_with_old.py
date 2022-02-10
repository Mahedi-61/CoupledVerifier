import os
import shutil

"""
p_files = os.listdir("photo")
pvfp_files = os.listdir("photo_vfp")

print(p_files)
print(pvfp_files)

assert (p_files == pvfp_files), "folders' don't match"
print("success !!")
"""

src_dir = "./photo"
dst_dir = "./wvu_combine/photo"

for sid in os.listdir(src_dir):
    sid_dir = os.path.join(src_dir, sid)

    for fid in os.listdir(sid_dir):
        fid_dir = os.path.join(sid_dir, fid)
        dst_fid_dir = os.path.join(dst_dir, "n_" + fid.split(".")[0])
        os.makedirs(dst_fid_dir, exist_ok=True)

        shutil.copy(fid_dir, dst_fid_dir)
