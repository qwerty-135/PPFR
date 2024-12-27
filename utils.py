import copy
import csv
import os
from glob import glob
from math import floor
import random
import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from disk_smart_parameter import PARAMETER_NUM, XGBOOST_SELECT
import pandas as pd
from torch.utils.data import DataLoader
EXTEND_TIMES = 2


class DiskLifeLoss(nn.Module):
    def __init__(self, flow_size):
        super(DiskLifeLoss, self).__init__()
        self.mean_flow = flow_size * EXTEND_TIMES / 2

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y) * (torch.abs(y - self.mean_flow) + 1))

def insert_at_backslash(s):
    try:
        index = s.index('\\')
        return s[:index] + '_pseudo_labels' + s[index:]
    except ValueError:
        return s + '_pseudo_labels'

def rnn_train(predictor, format_data_path, output_path, timestep):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device:", "cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(20240311)
    rnn = predictor.to(device,dtype=torch.float64)


    train_dataset = torch.load(format_data_path + rnn.DISK + "_" + rnn.MODEL_NAME + "_train.pt")
    train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = torch.load(format_data_path + rnn.DISK + "_" + rnn.MODEL_NAME + "_test.pt")
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=rnn.LR)
    # loss_fn = DiskLifeLoss(timestep)  # nn.L1Loss()  # nn.MSELoss()
    loss_fn = nn.L1Loss()

    torch.set_default_dtype(torch.float64)
    min_v_loss = 1000000
    min_t_loss = 1000000
    t_losses, v_losses = [], []
    # Loop over epochs
    best_model_no = None
    counter = 0

    for epoch in tqdm(range(rnn.EPOCH), "Epoch", leave=False, position=0):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        rnn.train()
        # Loop over train dataset
        for x, y in tqdm(train_dataset, "Train", leave=False, position=1):
            # print(counter)
            optimizer.zero_grad()
            rnn.hidden_cell = (torch.zeros(1, 1, rnn.HIDDEN_SIZE),
                               torch.zeros(1, 1, rnn.HIDDEN_SIZE))
            # move inputs to device
            # x = x.reshape(-1,1)
            x = x.to(device)  # .to(torch.float32)
            y = y.to(device)  # .to(torch.float32)
            counter += 1
            # Forward Pass
            try:
                preds = rnn.to(device)(x).squeeze()
                # print(preds)
                # print(y.squeeze())
            except Exception as e:
                print(x, y)
                print(counter)
                print(len(x))
                print(e)

            if preds.dim() == 1:
                preds = preds.unsqueeze(0)
            loss = loss_fn(preds, y.squeeze())  # compute batch loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss / len(train_dataset)
        t_losses.append(epoch_loss)

        # validation step
        rnn.eval()
        # Loop over validation dataset
        for x, y in tqdm(test_dataset, "Test", leave=False, position=1):
            with torch.no_grad():
                # x, y = x.to(device).to(torch.float32), y.squeeze().to(device).to(torch.float32)
                x, y = x.to(device), y.squeeze().to(device)
                preds = rnn.to(device)(x).squeeze()
                error = loss_fn(preds, y)
            valid_loss += error.item()
        valid_loss = valid_loss / len(test_dataset)
        v_losses.append(valid_loss)
        if valid_loss < min_v_loss:  # and train_loss < min_t_loss:
            print("\n")
            print(epoch_loss, valid_loss)
            min_v_loss = valid_loss
            min_t_loss = train_loss
            torch.save(rnn, output_path + rnn.MODEL_NAME)
            best_model_no = epoch

        # print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')

    plt.plot(t_losses, label="Training loss")
    plt.plot(v_losses, label="Validation loss")
    plt.ylim(min(min(t_losses), min(v_losses)), max(max(t_losses), max(v_losses)))
    plt.xlim(0, rnn.EPOCH)
    plt.vlines(best_model_no, 0, max(v_losses[best_model_no], 0.1), colors='gray', linestyles='dashed',
               label="Best:" + str(best_model_no))
    plt.legend()
    plt.title("Losses-" + rnn.MODEL_NAME)
    plt.savefig(output_path + rnn.MODEL_NAME + '.png')
    plt.show()
    print("Best:", best_model_no)

    with open(output_path + "/life_losses_result_" + rnn.DISK + "_" + rnn.MODEL_NAME + ".txt", "w",
              encoding='utf-8') as f:
        f.write(str(v_losses) + "\n")
        f.write(str(t_losses) + '\n')
        f.write("Best model no:" + str(best_model_no))


def csv_to_list(file: str):
    result = []
    csv_file = csv.reader(open(file, "r", encoding="utf-8"))
    file2 = insert_at_backslash(file)
    # print(file2)
    qs = pd.read_csv(file2,header=None)
    lenth = len(qs) + 1
    flag = 0
    check1 = lenth - flag
    check2 = qs[2][flag]
    check =check2 - check1
    for row in csv_file:
        templist = row[3:]
        templist.append(qs[2][flag])
        result.append(templist)
        flag = flag + 1
    return result,check


def single_raw_dataset_loading(file: str, smart_min_max: list, disk: str, flow_size: int = 20):  # 加载一块硬盘的数据为python list
    x_tensor_list = []
    y_tensor_list = []

    raw_data,check = csv_to_list(file)

    raw_data.reverse()

    n = len(raw_data)

    if n < flow_size * EXTEND_TIMES:
        return [], []
    if not flow_size:
        return [], []

    timestep = int(EXTEND_TIMES - 1) * flow_size
    # print(flow_size)

    for k in range(timestep):
        x = []
        for i in range(k, k + flow_size):
            # print(smart_min_max[1][2])
            x.append(
                [torch.tensor((int(raw_data[i][_]) - smart_min_max[_][0]) / smart_min_max[_][2],dtype=torch.float64) for _ in range(PARAMETER_NUM[disk]+1)]

            )
        y = [(_ + 1) / (flow_size * EXTEND_TIMES) for _ in range(k, k + flow_size)]

        dt = list(zip(x, y))
        # random.shuffle(dt)
        x, y = zip(*dt)
        x = list(x)
        y = list(y)
        for i in range(flow_size):
            x[i] = torch.tensor(x[i],dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)
        x_tensor_list.append(torch.stack(x))
        y_tensor_list.append(torch.stack([copy.deepcopy(y) for _ in range(flow_size)]))

    return x_tensor_list, y_tensor_list


def min_max(input_path="./failure_disk_data", smart_number=21):
    smart_min_max = [[1000000000000, -1] for _ in range(smart_number)]
    input_path2 = input_path+"_pseudo_labels"
    for file in tqdm(os.listdir(input_path)):

        csv_file = csv.reader(open(input_path + "/" + file, "r", encoding="utf-8"))
        # csv_file2 = csv.reader(open(input_path2 + "/" + file, "r", encoding="utf-8"))
        qs = pd.read_csv(input_path2 + "/" + file,header=None)
        # print(qs)
        lenth = len(qs) + 1
        flag = 0
        for row in csv_file:
            templist = row[3:]
            templist.append(lenth - flag)
            p = 0
            for item in templist:
                smart_min_max[p][0] = min(smart_min_max[p][0], int(item))
                smart_min_max[p][1] = max(smart_min_max[p][1], int(item))
                p += 1
            flag = flag + 1

    for i in range(len(smart_min_max)):
        smart_min_max[i].append(smart_min_max[i][1] - smart_min_max[i][0]+1e-2)

    return smart_min_max


def generate_dataset_pt(data_absolute_path: str, output_absolute_path: str, disk: str, model_name: str, flow_size: int,
                        test_rate: float = 0.1):
    random.seed(20240129)
    x_tensor = []
    y_tensor = []
    all_raw_data = glob(data_absolute_path)
    disk_smart_parameter = PARAMETER_NUM[disk] + 1

    smart_min_max = min_max(data_absolute_path.replace("/*.csv", ""), smart_number=disk_smart_parameter)
    # print(smart_min_max)
    for file in tqdm(all_raw_data):
        x, y = single_raw_dataset_loading(file, smart_min_max, disk, flow_size)

        if not x:
            continue
        x_tensor.extend(x)
        y_tensor.extend(y)



    test_size = floor(len(x_tensor) * test_rate)
    dt = list(zip(x_tensor, y_tensor))
    # random.shuffle(dt)
    x_tensor, y_tensor = zip(*dt)


    test_dataset = torch.utils.data.TensorDataset(
        torch.stack(x_tensor[:test_size]),
        torch.stack(y_tensor[:test_size]),
    )

    torch.save(test_dataset, output_absolute_path + disk + "_" + model_name + "_test.pt")


def test_result(dataset, output_path, model, flow_size, disk, model_name):
    output = csv.DictWriter(
        open(output_path + "output2_" + disk + "_" + model_name + ".csv", "w", encoding="gbk", newline=""),
        ['result', "real"])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.load(dataset)
    # x = DataLoader(x, batch_size=1, shuffle=True)
    model = torch.load(model,map_location='cuda:0').to(device)
    output.writeheader()

    for xi, yi in tqdm(x):

        xi = torch.unsqueeze(xi, 0)
        xh = model(xi.to(device))
        xh = list(torch.squeeze(xh))
        p = 0

        for tensor in list(model(xi.to(device)))[-1]:
        # for tensor in xh[-1]:
            row = {'result': round(tensor.item() * flow_size * EXTEND_TIMES, 1),
                   "real": round(yi[-1][p].item() * flow_size * EXTEND_TIMES, 1)}
            p += 1
            output.writerow(row)
