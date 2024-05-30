from einops import rearrange
from os.path import join as pjoin
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import math
import torch
import pywt


def swish(x):
    return x * torch.sigmoid(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
import torch
import torch.nn as nn
from basicsr.archs.arch_util import to_2tuple, trunc_normal_


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, out_size):
        super(ChannelAttention, self).__init__()
        self.adp = nn.AdaptiveAvgPool1d(output_size=out_size)
        self.conv1 = nn.Conv1d(in_channels=num_feat, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.relu= nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        # y = self.attention(x)
        x = self.adp(x)
        y = self.conv1(x)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv2(y)
        # y = self.sig(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, channel, out_size):
        super(CAB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3,stride=2, padding=0)
        self.ac = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, stride=2, padding=0)
        self.ch=ChannelAttention(channel, out_size)

    def forward(self, x):
        # x = x.squeeze(0)
        x= self.conv1(x)
        x = self.ac(x)
        x = self.conv2(x)
        x = self.ch(x)
        return x#self.cab(x)

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=658, nheads=8, proj_dropout=0.2):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.nheads = nheads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(proj_dropout)
        self.scale_factor = math.sqrt(embedding_dim / nheads)

    def forward(self, x):
        q, k, v = self.qkv(x).split(self.embedding_dim, dim=-1)
        q = q.reshape(q.size(0), q.size(1) * self.nheads, -1).permute(1, 0, 2)
        k = k.reshape(k.size(0), k.size(1) * self.nheads, -1).permute(1, 2, 0)
        attn = torch.softmax((q @ k) / self.scale_factor, dim=-1)
        v = v.reshape(v.size(0), v.size(1) * self.nheads, -1).permute(1, 0, 2)
        o = attn @ v
        o = o.permute(1, 0, 2).reshape(x.shape)
        o = self.out(o)
        o = self.dropout(o)
        return o  # , attn.reshape(x.size(1), self.nheads, x.size(0), x.size(0))

class SelfAttention1D(nn.Module):
    def __init__(self, input_size=5000, nheads=4, proj_dropout=0.2):
        super(SelfAttention1D, self).__init__()
        self.input_size = input_size
        self.nheads = nheads
        self.qkv = nn.Linear(input_size, input_size * 3)
        self.out = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(proj_dropout)
        self.scale_factor = math.sqrt(input_size / nheads)

    def forward(self, x):
        q, k, v = self.qkv(x).split(self.input_size, dim=-1)
        q = q.reshape(q.size(0), q.size(1) * self.nheads, -1).permute(1, 0, 2)
        k = k.reshape(k.size(0), k.size(1) * self.nheads, -1).permute(1, 2, 0)
        attn = torch.softmax((q @ k) / self.scale_factor, dim=-1)
        v = v.reshape(v.size(0), v.size(1) * self.nheads, -1).permute(1, 0, 2)
        o = attn @ v
        o = o.permute(1, 0, 2).reshape(x.shape)
        o = self.out(o)
        o = self.dropout(o)
        return o

class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=800):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, hidden_size * 4)
        self.fc2 = Linear(hidden_size * 4, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.5)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



import pywt
class WaveletTransformer(nn.Module):
    def __init__(self, wavelet='db1', level=4):
        super(WaveletTransformer, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, inputs):
        inputs = inputs.cpu().numpy()
        coeffs = pywt.wavedec(inputs, self.wavelet, level=self.level)
        dwt = pywt.waverec(coeffs, self.wavelet)
        return dwt


class HybirdModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HybirdModel, self).__init__()
        ####Contextual Attention
        self.dwt = WaveletTransformer(wavelet='db1', level=4)
        # self.linear = nn.Conv2d()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.atten =WindowAttention1D(dim=output_size, window_size=1, num_heads=output_size)
        self.atten = SelfAttention1D(input_size=output_size)#SelfAttention(embedding_dim=output_size)
        self.ffd = Feed_Forward(input_dim=output_size, hidden_dim=output_size)
        self.mlp = Mlp(hidden_size=output_size)
        ####Channel Attention
        self.cab = CAB(channel=1,out_size=output_size)
        ###original
        self.adg = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        y = x.unsqueeze(0)
        ####Contextual Attention
        x = torch.from_numpy(self.dwt(x)).to(device)
        # x 的形状应该是 (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # # 获取最后一个时间步的输出作为特征
        # last_output = out[:, -1, :]
        x = self.fc(out).unsqueeze(0)
        x = self.atten(x)
        x = self.ffd(x)
        x = self.mlp(x)
        ###Channel Attention
        z = self.cab(y)
        ####orginal
        w = self.adg(y)
        ###final
        out = x+z+w
        return out.squeeze(0)




# model = HybirdModel(input_size=13574, hidden_size=500, num_layers=2, output_size=500).to(device)
#
#
# input_data2 = torch.randn((1, 13574)).to(device)
#
# import numpy as np
# import torch.optim as optim
# # 模拟一维数据
# data_size = 5000  # 数据点的数量
# possible_values = [0, 1, 2, -1, -2]
# # 模拟标签
# target = torch.randint(0, 5000, (1,)).to(device)
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# # 训练脉冲神经网络
# for epoch in range(100):
#     # 前向传播
#     output = model(input_data2).to(device)
#
#     # 计算损失
#     loss = criterion(output, target)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print(f'Epoch {epoch + 1}/{100}, Loss: {loss.item():.4f}')
#
# print("Training complete.")