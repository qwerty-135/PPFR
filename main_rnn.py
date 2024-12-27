import os

from utils import generate_dataset_pt, rnn_train, test_result
from disk_smart_parameter import ST12000NM0008_REALIST, ST4000DM000_REALIST
from rnn import PredictorRNN, PredictorLSTM,PredictorTransformer
from disk_smart_parameter import PARAMETER_NUM, XGBOOST_SELECT

DISK = "ST4000DM000" #
# INPUT_SIZE = PARAMETER_NUM[DISK] + 1 #len(XGBOOST_SELECT[DISK])
INPUT_SIZE = PARAMETER_NUM[DISK] +1#len(XGBOOST_SELECT[DISK])
print(INPUT_SIZE)
HIDDEN_SIZE = INPUT_SIZE * 4
BATCH_SIZE = 40  #
OUTPUT_SIZE = 40
EPOCH = 10

LR = 1e-4
# TIME_STEP = 15
DROP_RATE = 0.1
LAYER = 5

MODEL_NAME = "Transformer-80"
MODEL_RESULT_PATH = "./rnn/"

output_absolute_path = MODEL_RESULT_PATH + "failure_"
data_absolute_path = "./failure_disk_data_" + DISK + "/*.csv"
# DISK_SMART_PARAMETER_NUM = len(eval(DISK + "_REALIST"))
# FLOW_SIZE = 40
format_data_path = MODEL_RESULT_PATH + "failure_"
output_path = MODEL_RESULT_PATH + "result/" + DISK + "/"

os.makedirs(output_path, exist_ok=True)

# #Dataset Making
generate_dataset_pt(data_absolute_path, output_absolute_path, DISK, MODEL_NAME, BATCH_SIZE, test_rate=1)
#
#Train Model
predictor = PredictorTransformer(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCH, LR, LAYER, DROP_RATE, DISK, MODEL_NAME, )
# predictor = PredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCH, LR, LAYER, DROP_RATE, DISK, MODEL_NAME, )
# predictor = PredictorRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCH, LR, LAYER, DROP_RATE, DISK, MODEL_NAME, )
rnn_train(predictor, format_data_path, output_path, timestep=BATCH_SIZE )

#Test Model
DATASET_PATH = MODEL_RESULT_PATH + "failure_" + DISK + "_" + MODEL_NAME + "_train.pt"
model_path = MODEL_RESULT_PATH + "result/" + DISK + "/" + MODEL_NAME
test_result(DATASET_PATH, output_path, model_path, BATCH_SIZE, DISK, MODEL_NAME)
