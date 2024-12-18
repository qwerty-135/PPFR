import random
from torch import nn
import torch
random.seed(20240129)
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class PredictorTransformer(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCH, LR, LAYER, DROP_RATE, DISK, MODEL_NAME):
        super(PredictorTransformer, self).__init__()

        d_model=HIDDEN_SIZE
        # embed_dim = head_dim * num_heads?
        self.input_fc = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)

        self.tsm_fc = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.output_fc = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.pos_emb = PositionalEncoding(HIDDEN_SIZE)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=4,
            dim_feedforward=4 * INPUT_SIZE,
            batch_first=True,
            dropout=0.2,
            # device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=4,
            dropout=0.2,
            dim_feedforward=4 * INPUT_SIZE,
            batch_first=True,
            # device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=3)
        # self.fc = nn.Linear(args.output_size * args.d_model, args.output_size)
        self.fc1 = nn.Linear(40 * HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        # self.fc3 = nn.Linear(800, OUTPUT_SIZE)
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DISK = DISK
        self.EPOCH = EPOCH
        self.LR = LR
        self.MODEL_NAME = MODEL_NAME
        # self.fc3 = nn.Linear(args.seq_len * 3 * args.d_model, 3*args.d_model)
        # self.fc4 = nn.Linear(3 * args.d_model, args.d_model)
        # self.fc5 = nn.Linear(args.d_model, args.output_size)
        # self.day_in_week_emb = nn.Parameter(
        #     torch.empty(12, 32))
        # nn.init.xavier_uniform_(self.day_in_week_emb)
        # self.time_in_day_emb = nn.Parameter(
        #     torch.empty(24, 32))
        # nn.init.xavier_uniform_(self.time_in_day_emb)
        # self.node_emb = nn.Parameter(
        #     torch.empty(args.input_size, 32))
        # nn.init.xavier_uniform_(self.node_emb)
    def forward(self, x):

        x = self.input_fc(x)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class PredictorRNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCH, LR, LAYER, DROP_RATE, DISK, MODEL_NAME):
        super(PredictorRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE // 2,
            num_layers=LAYER,
            dropout=DROP_RATE,
            bidirectional=True,
            batch_first=True,  # 为True则输入输出格式为（Batch，seq_len，feature），否则Batch和Seq_len颠倒
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)  # 最后一个时序的输出接一个全连接层
        self.dropout = nn.Dropout(p=DROP_RATE)
        self.activation = nn.Tanh()
        self.h_s = None
        self.h_c = None
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DISK = DISK
        self.EPOCH = EPOCH
        self.LR = LR
        self.MODEL_NAME = MODEL_NAME


    def forward(self, x):  # X是输入数据集
        # print(x.size())
        # print("aa")
        x, h = self.rnn(x)
        x = self.hidden_out(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class PredictorLSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCH, LR, LAYER, DROP_RATE, DISK, MODEL_NAME):
        super(PredictorLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE // 2,
            num_layers=LAYER,
            dropout=DROP_RATE,
            bidirectional=True,
            batch_first=True,  # 为True则输入输出格式为（Batch，seq_len，feature），否则Batch和Seq_len颠倒
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)  # 最后一个时序的输出接一个全连接层
        self.dropout = nn.Dropout(p=DROP_RATE)
        self.activation = nn.Tanh()
        self.h_s = None
        self.h_c = None
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DISK = DISK
        self.EPOCH = EPOCH
        self.LR = LR
        self.MODEL_NAME = MODEL_NAME

    def forward(self, x):  # X是输入数据集
        x, h = self.rnn(x)
        x = self.hidden_out(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
