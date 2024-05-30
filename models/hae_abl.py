import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, dim=1022):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(73408, dim)##ucec 64 * dim #luad 54272 # brca73664 #blca73408 is the computed size after convolutions and pooling
        self.fc2 = nn.Linear(dim, 500)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 73408) #luad 54272 ##ucec 64 * 1022  ##brca73664  #blca73408                     # Reshape before the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    def __init__(self, dim=566):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3,stride=2)
        self.fc1 = nn.Linear(8064, dim)  ###luad 11136 ##ucec 64 * dim # brca 10176 #blca8064 is the computed size after convolutions and pooling
        self.fc2 = nn.Linear(dim, 500)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 8064)  ##ucec 64 * 566 ##luad 11136 # brca 10176# blca8064 Reshape before the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.fc(out) #.squeeze(0)
        return x
#


import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#################################
import torch
import torch.nn as nn
import torch.optim as optim

class DenoisingAutoencoder(nn.Module):
    def __init__(self, daee = 65472, dae_l=784, encoded_space_dim=500):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(daee, 128),  ###此处需要修改
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

        # Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.linear = nn.Linear(dae_l, 500) ###此处需要修改

    def forward(self, x):
        # Encoder\
        x = x.unsqueeze(0)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)

        # Decoder
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        x = self.linear(x.view(1, -1))
        return x # 修改此处，确保输出形状为（1, 500）

# # 定义模型和损失函数
# input_dim = 9072
# # encoded_space_dim = 500
# model = DenoisingAutoencoder(36256, 784, 500)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 创建示例输入数据（batch_size=1）
# sample_input = torch.randn(1, 1, input_dim)  # 由于使用的是Conv1d，需要在输入数据上添加通道维度
# aaa = sample_input.squeeze(0)
# # 使用模型进行前向传播
# output = model(sample_input)
# print("Output Shape:", output.shape)

####################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAEEncoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.sigma = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = F.relu(self.linear(x))
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))  # Applying softplus for positive values
        return mu, sigma

class VAEDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(VAEDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_size, hidden_size, latent_size)
        self.decoder = VAEDecoder(latent_size, hidden_size, output_size)

    def forward(self, x):
        # Encode
        mu, sigma = self.encoder(x)

        # Reparameterization
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma

        # Decode
        re_x = self.decoder(z)

        return re_x#, mu, sigma

# Define input and output sizes
input_size = 16372
output_size = 500
latent_size = 50
hidden_size = 128  # You can adjust this based on your requirements

# Create VAE model
vae_model = VAE(input_size, output_size, 50, 128)

# Example input
input_data = torch.randn(1, input_size)

# # Forward pass
# output_data = vae_model(input_data)
#
# # Print shapes
# print("Output Shape:", output_data.shape)


import torch
import torch.nn as nn
from models.Hybird_Attention import SelfAttention1D, Feed_Forward, Mlp, SelfAttention
import math
import torch
class PathEmbed(nn.Module):
    def __init__(self, in_chans=658, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=1, stride=1,kernel_size=1)
        self.batch1_1 = nn.BatchNorm1d(1)
        self.relu1_1 = nn.ReLU()


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.batch1_1(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=658, nheads=658, proj_dropout=0.1):
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
        return o#, attn.reshape(x.size(1), self.nheads, x.size(0), x.size(0))


class Feed_Forward(nn.Module):
    def __init__(self,input_dim,hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,input_dim)

    def forward(self,x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output

class Mlp(nn.Module):
    def __init__(self, config=[658, 200, 100]):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config[0], config[1])#wm,786->3072
        self.fc2 = nn.Linear(config[1], config[2])#wm,3072->786
        self.act_fn = nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(0.)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x

import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim):  #inter_dims=[1000, 500, 200, 50, 5]
        super(Encoder, self).__init__()
        self.path_enmbed =PathEmbed(input_dim)
        self.attention = SelfAttention(embedding_dim=input_dim, nheads=input_dim, proj_dropout=0.1)#(d_model = 658,dim_k=10, dim_v=10, n_heads= 10)
        self.flatte = nn.Flatten()
        self.forwd = Feed_Forward(input_dim=input_dim,hidden_dim=input_dim*3)
        self.norm = Mlp([input_dim, 200, out_dim])

        self.flatte1 = nn.Flatten()
        self.batch3_1 = nn.BatchNorm1d(num_features=20)
        self.relu3_1 = nn.ReLU()
        self.drop3_1 = nn.Dropout(0.7)
        self.flatte2 = nn.Flatten()
        self.batch3_2 = nn.BatchNorm1d(num_features=20)
        self.relu3_2 = nn.ReLU()
        self.drop3_2 = nn.Dropout(0.5)
        self.lin3 = nn.Linear(20, 5)



    def forward(self, x):
        z = self.path_enmbed(x)
        z = self.attention(z)
        z = self.flatte(z)
        z = self.forwd(z)
        z = self.norm(z)
        x = self.flatte1(z)
        x = self.batch3_1(x)
        x = self.relu3_1(x)
        x = self.drop3_1(x)
        x = self.flatte2(x)
        x = self.batch3_2(x)
        x = self.relu3_2(x)
        x = self.drop3_2(x)
        x = self.lin3(x)
        z = F.softmax(x, dim=1)

        return z

# 定义示例输入和输出
input_dim = 16372
output_dim = 500
hidden_dim = 256
num_layers = 3
num_heads = 8
dropout = 0.1

input_data = torch.randn(1, input_dim)
target_data = torch.randn(1, output_dim)

# 初始化模型
# model = Encoder(input_dim, output_dim)
#
# # 计算模型输出
# output = model(input_data)
#
# print("模型输出维度：", output.shape)



###################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionTransformer1D(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=4000, nheads=10, proj_dropout=0.2):
        super(SelfAttentionTransformer1D, self).__init__()

        self.embedding_dim = embedding_dim
        self.nheads = nheads

        # Embedding layer
        self.embedding = nn.Linear(input_size, embedding_dim)

        # Self-Attention layer
        self.self_attention = SelfAttention(embedding_dim=embedding_dim, nheads=nheads, proj_dropout=proj_dropout)

        self.dropout = nn.Dropout(0.2)
        # Feed-Forward layer
        self.feed_forward = Mlp(config=[embedding_dim, embedding_dim * 3, embedding_dim])

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, output_size)

    def forward(self, x):
        # Add batch and time dimensions
        x = x.unsqueeze(0).unsqueeze(2)

        # Embedding layer
        x = self.embedding(x)

        # Self-Attention layer
        x = self.self_attention(x)

        x = self.dropout(x)
        # Flatten and apply Feed-Forward layer
        x = x.view(x.size(0), -1)
        x = self.feed_forward(x)

        # Output layer
        x = self.output_layer(x)

        # Apply softmax for classification
        x = F.softmax(x, dim=1)

        return x

# Define input and output sizes
input_size = 16372
output_size = 500

# # Create Self-Attention Transformer model
# self_attention_transformer_1d = SelfAttentionTransformer1D(input_size, output_size)
#
# # Example input
# input_data = torch.randn(1, input_size)
#
# # Forward pass
# output_data = self_attention_transformer_1d(input_data)
#
# # Print shapes
# print("Input Shape:", input_data.shape)
# print("Output Shape:", output_data.shape)


# 创建模型
# model = CNN2()
#
# # 打印模型结构
# print(model)
#
# import torch
#
# # 模型实例化
# model = LSTMModel(input_size=9072, hidden_size=1000, num_layers=2, output_size=500)
#

# model = AutoEncoder(input_dim=9072, hidden_dim=1000, output_dim=500)
#
# # # 模拟输入数据
# input_data = torch.randn(1, 1, 9072)  # 模拟一个大小为 (1, 1, 16371) 的输入张量
# # #
# # # # 前向传播
# output = model(input_data)
# # #
# # # # 打印输出的形状
# print("模型输出数据维度:", output.shape)
