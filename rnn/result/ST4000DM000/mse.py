import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error


ls=pd.read_csv("output_ST4000DM000_Transformer-80.csv")
result=ls.iloc[:,0].tolist()
real=ls.iloc[:,1].tolist()
rmse = math.sqrt(np.mean((np.array(result) - np.array(real)) ** 2))
print(f"RMSE: {rmse}")
mae = mean_absolute_error(np.array(result), np.array(real))
print(f"MAE: {mae}")
