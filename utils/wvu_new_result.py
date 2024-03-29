import numpy as np
import sklearn.metrics
import os 
import sys 

root_dir = "/media/lab320/SSDDrive/reproduce/src_verifier/result_innovatrics/wvu_new"
type_dir = "gr"
result = sys.argv[1]
result_dir = os.path.join(root_dir, type_dir, result)

with open(result_dir) as f:
    result = f.read()
    f.close()

scores = result.split(",")[:-1]
total_number = int(np.sqrt(len(scores)))
print("total row: ", total_number)

y_true_ls = []
y_pred_ls = []

for row in range(total_number):
    y_true = []
    y_pred = []

    for col in range(total_number):
        if (row == col): y_true.append(1.0)
        else: y_true.append(0.0)
        y_pred.append(int(scores[row*total_number + col]))

    y_true_ls += y_true
    y_pred_ls += y_pred  

fprs, tprs, thresholds = sklearn.metrics.roc_curve(y_true_ls, y_pred_ls)
eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
auc = sklearn.metrics.auc(fprs, tprs)

print("AUC: ", auc)
print("EER: ", eer)