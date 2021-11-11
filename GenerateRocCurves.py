import os
import numpy
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt 


delta1_list = [1]
delta2_list = [200]
margin_list = [75]
feature_dims = [256]
# feature_dims = [128]
# 75 > 25 > 50 > 100

# To run tests with conditional GAN I need to run dataset_prepare_for_trainingresults.py
# Then I need to run test_verif_fingerVSfinger.py with the same verifier model used while training
# The conditional GAN which was model_resnet18_75_0.001_1.0_256.pt

# Delta_1:0.001, delta_2:1.000, margin:75, feature_dims:256
# The images don't really look good with this, they look better with higher delta values.

# Delta_1:6, delta_2:1.000, margin:75, feature_dims:256
# works pretty good

# This fixed my errors where my drivers were the incorrect version.
# conda install pytorch torchvision cudatoolkit=10 -c pytorch
# for delta1 in delta1_list:
#     for delta2 in delta2_list:
#         for margin in margin_list:
#             for feature_dim in feature_dims:
#                 print("delta_1 %f delta_2 %f margin %d" % (delta1, delta2, margin))
#                 os.system("python train_verif.py --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))

# for delta1 in delta1_list:
#     for delta2 in delta2_list:
#         for margin in margin_list:
#             for feature_dim in feature_dims:
#                 print("delta_1 %f delta_2 %f margin %d" % (delta1, delta2, margin))
#                 os.system("python test_verif_fingerVSfinger.py --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))
#                 # os.system("python test_verif_fingerVSphoto.py --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))

# for delta1 in delta1_list:
#     for delta2 in delta2_list:
#         for margin in margin_list:
#             for feature_dim in feature_dims:
#                 print("delta_1 %f delta_2 %f margin %d" % (delta1, delta2, margin))
#                 os.system("python SaveImages.py --batch_size 10 --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))

def loadRocCurve(file):
    tpr = list()
    fpr = list()
    with open(file, 'r') as lines:
        for line in lines:
            tpr.append(line.split(",")[0].strip())
            fpr.append(line.split(",")[1].strip())
    return numpy.asanyarray(tpr, dtype=numpy.float32), numpy.asanyarray(fpr, dtype=numpy.float32)

def getEER(tpr, fpr):
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


fpr = dict()
tpr = dict()
roc_auc = dict()
result_dirs = dict()
roc_eer = dict()
i = 0


################## All Experiment ##################
# This is the commercial matcher matching fingerphoto vs fingerprint images
tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/TPRFPR_FphotovsFprint.csv")
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
roc_eer[i] = getEER(tpr[i], fpr[i])
result_dirs[i] = "Fphoto vs Fprint - Commercial Matcher"
i = i + 1

delta1 = 0.001
delta2 = 1.0
margin = 75
feature_dim = 256

print("delta_1 %f delta_2 %f margin %d feat_dim %d" % (delta1, delta2, margin, feature_dim))
dist = numpy.load("./src_verifier/data/" + str(margin) + "_" + str(float(delta1)) + 
                  "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_dist_test.npy")

lbl = numpy.load("./src_verifier/data/" + str(margin) + "_" + str(float(delta1)) + "_" + 
                str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_lbl_test.npy")

fpr[i], tpr[i], threshold = metrics.roc_curve(lbl, dist)
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
roc_eer[i] = getEER(tpr[i], fpr[i])
result_dirs[i] = "FPhoto vs Fprint - CpGAN Matcher"
i = i + 1


tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/ROCPLOT_Coarse1000.csv")
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
roc_eer[i] = getEER(tpr[i], fpr[i])
result_dirs[i] = ("PDRM - Synthetic Fprint vs Fprint - Commercial Matcher")
i = i + 1


tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/CGAN_Synthetic2Contact_Fixed.csv")
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
roc_eer[i] = getEER(tpr[i], fpr[i])
result_dirs[i] = ("PDRM w/o CpGAN+MinutiaeNet - Synthetic Fprint vs Fprint - Commercial Matcher")
i = i + 1


tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/ROCVal_finetuning_cgan3.csv")
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
roc_eer[i] = getEER(tpr[i], fpr[i])
result_dirs[i] = ("PDRM w/o MinutiaeNet - Synthetic Fprint vs Fprint - Commercial Matcher")
i = i + 1


tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/ROCPLOT_CoarseWOVerifier.csv")
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
roc_eer[i] = getEER(tpr[i], fpr[i])
result_dirs[i] = ("PDRM w/o CpGAN - Synthetic Fprint vs Fprint - Commercial Matcher")
i = i + 1


tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/TPRFPR_SyntheticFprintvsFprint.csv")
result_dirs[i] = ("Synthetic Fprint vs Fprint (Innovatrics Matcher - Ali)")
roc_auc[i] = 0
i = i + 1


fpr[i], tpr[i] = loadRocCurve("./src_verifier/data/SynthFprintvsFprint_DeepMatcher.csv")
result_dirs[i] = ("Synthetic Fprint vs Fprint (DeepMatcher - Ali)")
roc_auc[i] = 0
i = i + 1

tpr[i], fpr[i] = loadRocCurve("./src_verifier/data/ROC_Plot_VerandCoarse.csv")
result_dirs[i] = ("Synthetic Fprint vs Fprint (Innovatrics Matcher - Alex, and MinutiaeNet)")
roc_auc[i] = 0
i = i + 1

"""
# # This is the synthetic Fprint vs Fprint (Deep matcher - Alex). 
This is my verifier after using the conditional GAN outputs
# delta1 = 1
# delta2 = 100
# margin = 75
# feature_dim = 256
# print("delta_1 %f delta_2 %f margin %d feat_dim %d" % (delta1, delta2, margin, feature_dim))
# dist = numpy.load("data/" + str(margin) + "_" + str(float(delta1)) + "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_dist_test.npy")
# lbl = numpy.load("data/" + str(margin) + "_" + str(float(delta1)) + "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_lbl_test.npy")
# fpr[i], tpr[i], threshold = metrics.roc_curve(lbl, dist)
# roc_auc[i] = metrics.auc(fpr[i], tpr[i])
# # result_dirs[i] = "delta_1 %0.5f delta_2 %0.3f margin %d: %0.3f" % (delta1, delta2, margin, roc_auc[i])
# result_dirs[i] = "Synthetic Fprint vs Fprint - Coupled GAN"
# i = i + 1

# tpr[i], fpr[i] = loadRocCurve("./data/ROC_prints_fake_alex.csv")
# result_dirs[i] = ("Synthetic Fprint vs Fprint (Innovatrics Matcher - Alex) - Me Recreating Ali's Work")
# roc_auc[i] = 0
# i = i + 1
# tpr[i], fpr[i] = loadRocCurve("./data/ROC_prints_fake_ali.csv")
# result_dirs[i] = ("Synthetic Fprint vs Fprint (Innovatrics Matcher - Ali) - His Work")
# roc_auc[i] = 0
# i = i + 1
"""

plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Fingerprint vs Fingerphoto Matching ROC')
plt.plot([0, 1], [0, 1], 'k--')

for i in range(len(result_dirs)):
    plt.plot(fpr[i], tpr[i], label=result_dirs[i], linewidth=3)
# for i in range(len(roc_eer)):
#     plt.plot([roc_eer[i], roc_eer[i]],[0, 1-roc_eer[i]], 'r-')
plt.legend(loc="lower right")
plt.show()

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

#a = [1, 2, 3] x b = [3, 4, 5]