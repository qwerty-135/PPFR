from clf_xgb import xgb_pre_process, xgboost_train, xgb_pre_process_for_20, xgboost_train_for_20
from disk_smart_parameter import PARAMETER_NUM
import csv
import time
DISK_MODEL = "ST4000DM000"

result_path = "./clf_xgb/productive_for_80_"+DISK_MODEL+"/"
result_file_path = result_path + "result.csv"
data_path = "clf_xgb/data-80.csv"
output_csv = csv.writer(open(result_file_path, "w", encoding="utf-8", newline=""))
output_csv.writerow(["Boost Round", "Precision", "Recall", "F1", "Accuracy", "AUC"])
positive_path = "./failure_disk_data_" + DISK_MODEL + "/"
negative_path = "./data_" + DISK_MODEL + "/"
# Preprocess
start_time = time.time()
xgb_pre_process(PARAMETER_NUM[DISK_MODEL], 80, positive_path, negative_path)


# Productive Model
result, bst = xgboost_train(DISK_MODEL, data_path, result_path + "xgb-productive.png", 5000)
# result, bst = xgboost_train_for_20(DISK_MODEL, data_path, result_path + "xgb-productive.png", 5000)
output_csv.writerow(result)
# bst.save_model(result_path + "xgb-model")
end_time = time.time()
execution_time = end_time - start_time

score = bst.get_score()
score_smart = {}
for key in score:
    i = key.rfind("S")
    score_smart.setdefault(key[i + 1:], 0)
    score_smart[key[i + 1:]] += score[key]
with open(result_path + "svm-score.txt", "w", encoding="utf-8") as file:
    file.write(str(score_smart))
    file.write("\n")
    file.write(str(score))
