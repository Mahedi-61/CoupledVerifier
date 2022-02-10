import numpy as np
import sklearn.metrics

result_dir = "../result_innovatrics/wvu_old/wvu_old_r60/res_wvu_old_r60_test.csv"
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