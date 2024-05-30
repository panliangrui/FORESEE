from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import torch
import torch.nn as nn
import joblib
from torch_geometric.nn import SAGEConv, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.2):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Linear transformations for query, key, and value
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights =F.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and concatenate attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Linear projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        return output

class Mlp(nn.Module):
    def __init__(self, hidden_size=500, dropout_rate=0.5):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, hidden_size*4)
        self.fc2 = Linear(hidden_size*4, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)
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

class CrossScaleFusionTransformer(nn.Module):
    def __init__(self, dict, dropout_rate=0.2):
        super(CrossScaleFusionTransformer, self).__init__()
        ###############
        self.img_node = dict['img_node']
        self.img_out = dict['img_out']
        self.embed=dict['embed']

        self.img_gnn1 = SAGEConv(in_channels=self.img_node, out_channels=self.img_out)
        self.relu1 = GNN_relu_Block(self.img_out)
        self.img_gnn2 = SAGEConv(in_channels=self.img_node, out_channels=self.img_out)
        self.relu2 = GNN_relu_Block(self.img_out)
        self.img_gnn3 = SAGEConv(in_channels=self.img_node, out_channels=self.img_out)
        self.relu3 = GNN_relu_Block(self.img_out)
        # self.ps2 = nn.PixelShuffle(2)
        # self.ps4 = nn.PixelShuffle(4)
        self.dropout = nn.Dropout(0.2)
        ###LEVEL1
        self.SA_B1 = SelfAttention(embed_dim=self.embed, num_heads=4, dropout_rate=dropout_rate)
        self.gcn1 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm1 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp1 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)


        self.SA_B1_1 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn1_1 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm1_1 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp1_1 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        self.SA_B1_2 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn1_2 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm1_2 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp1_2 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        #fusion1
        self.down1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.down1_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        #level2
        self.SA_B2_1 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn2_1 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm2_1 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp2_1 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        self.SA_B2_2 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn2_2 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm2_2 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp2_2 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        self.SA_B2_3 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn2_3 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm2_3 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp2_3 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        # fusion2_1 and fusion2_2
        self.down2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.down2_1 = nn.Conv1d(in_channels=1558, out_channels=800, kernel_size=3, stride=1, padding=1)
        self.down2_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.down3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.down3_1 = nn.Conv1d(in_channels=558, out_channels=500, kernel_size=3, stride=1, padding=1)
        self.down3_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        #level3
        self.SA_B3_1 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn3_1 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm3_1 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp3_1 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        self.SA_B3_2 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn3_2 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm3_2 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp3_2 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        self.SA_B3_3 = SelfAttention(embed_dim=self.embed, num_heads=4)
        self.gcn3_3 = GCNConv(in_channels=self.embed, out_channels=self.embed)
        self.ffn_norm3_3 = LayerNorm(in_channels=self.embed, eps=1e-6)
        self.Mlp3_3 = Mlp(hidden_size=self.embed, dropout_rate=dropout_rate)

        # fusion4
        self.down4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.down4_1 = nn.Conv1d(in_channels=1414, out_channels=500, kernel_size=3, stride=1, padding=1)
        self.down4_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_img_256, x_img_512, x_img_1024, edge_index_img_256, edge_index_img_512, edge_index_img_1024):
        ##############################解析数据
        # x_img = x.x_img
        # x_edge = x.edge_index_image
        #########################################
        x1 = self.img_gnn1(x_img_256, edge_index_img_256).unsqueeze(0)
        # x1 = self.relu1(x1)
        x2 = self.img_gnn2(x_img_512, edge_index_img_512).unsqueeze(0)
        # x2 = self.relu2(x2)
        x3 = self.img_gnn3(x_img_1024, edge_index_img_1024).unsqueeze(0)
        # x3 = self.relu3(x3)
        # level 1
        atten1 = self.SA_B1(x1)
        # atten1 = self.dropout(atten1)
        normal1 = self.ffn_norm1(atten1)
        normal1 = self.gcn1(normal1, edge_index_img_256)
        mlp1 = self.Mlp1(normal1)
        mlp1 = atten1 + mlp1

        atten1_1 = self.SA_B1_1(x2)
        # atten1_1 = self.dropout(atten1_1)
        normal1_1 = self.ffn_norm1_1(atten1_1)
        normal1_1 = self.gcn1_1(normal1_1, edge_index_img_512)
        mlp1_1 = self.Mlp1_1(normal1_1)
        mlp1_1 = atten1_1 + mlp1_1

        atten1_2 = self.SA_B1_2(x3)
        # atten1_2 = self.dropout(atten1_2)
        normal1_2 = self.ffn_norm1_2(atten1_2)
        normal1_2 = self.gcn1_2(normal1_2, edge_index_img_1024)
        mlp1_2 = self.Mlp1_2(normal1_2)
        mlp1_2 = atten1_2 + mlp1_2

        ##first fusion
        # 将三个特征图进行维度变换
        # graph1 = mlp1.transpose(1, 2)  # 转换为（1, 500, 534）
        # graph1_1 = mlp1_1.transpose(1, 2)  # 转换为（1, 256, 534）
        # graph1_2 = graph_3.transpose(1, 2)  # 转换为（1, 100, 534）
        fused_graph = self.down1(torch.cat([mlp1, mlp1_1], dim=1))
        fused_graph1_1 = self.down1_1(fused_graph)
        fused_graph1 = self.dropout(fused_graph1_1)#.transpose(1, 2)

        # level 2
        atten2_1 = self.SA_B2_1(fused_graph1)
        # atten2_1 = self.dropout(atten2_1)
        normal2_1 = self.ffn_norm2_1(atten2_1)
        # normal2_1 = self.gcn1(normal2_1, x_edge)
        mlp2_1 = self.Mlp2_1(normal2_1)
        mlp2_1 = atten2_1 + mlp2_1

        atten2_2 = self.SA_B2_2(mlp1_1)
        # atten2_2 = self.dropout(atten2_1)
        normal2_2 = self.ffn_norm2_2(atten2_2)
        # normal2_2 = self.gcn2_2(normal2_2, x_edge)
        mlp2_2 = self.Mlp2_2(normal2_2)
        mlp2_2 = atten2_2 + mlp2_2

        atten2_3 = self.SA_B2_3(mlp1_2)
        # atten2_3 = self.dropout(atten2_3)
        normal2_3 = self.ffn_norm2_3(atten2_3)
        # normal2_3 = self.gcn1_2(normal2_3, x_edge)
        mlp2_3 = self.Mlp2_3(normal2_3)
        mlp2_3 = atten2_3 + mlp2_3

        # 将三个特征图进行维度变换
        # graph2 = mlp2_1.transpose(1, 2)  # 转换为（1, 500, 534）
        # graph2_1 = mlp2_2.transpose(1, 2)  # 转换为（1, 256, 534）
        # graph2_2 = mlp2_3.transpose(1, 2)  # 转换为（1, 100, 534）
        fused_graph2_1 = self.down2(torch.cat([mlp2_1, mlp2_2, mlp2_3], dim=1))
        fused_graph2_1 = self.down2_1(fused_graph2_1)
        fused_graph2 = self.dropout(fused_graph2_1)#.transpose(1, 2)

        # graph2 = mlp2_1.transpose(1, 2)  # 转换为（1, 500, 534）
        # graph2_1 = mlp2_2.transpose(1, 2)  # 转换为（1, 256, 534）
        # graph2_2 = mlp2_3.transpose(1, 2)  # 转换为（1, 100, 534）
        fused_graph2_2 = self.down3(torch.cat([mlp2_2, mlp2_3], dim=1))
        fused_graph2_2 = self.down3_1(fused_graph2_2)
        fused_graph3 = self.dropout(fused_graph2_2)#.transpose(1, 2)

        # level 3
        atten3_1 = self.SA_B3_1(fused_graph2)
        # atten3_1 = self.dropout(atten3_1)
        normal3_1 = self.ffn_norm3_1(atten3_1)
        # normal2_1 = self.gcn1(normal2_1, x_edge)
        mlp3_1 = self.Mlp3_1(normal3_1)
        mlp3_1 = atten3_1 + mlp3_1

        atten3_2 = self.SA_B3_2(fused_graph3)
        # atten3_2 = self.dropout(atten3_2)
        normal3_2 = self.ffn_norm3_2(atten3_2)
        # normal3_2 = self.gcn2_2(normal2_2, x_edge)
        mlp3_2 = self.Mlp3_2(normal3_2)
        mlp3_2 = atten3_2 + mlp3_2

        atten3_3 = self.SA_B3_3(mlp2_3)
        # atten3_3 = self.dropout(atten3_3)
        normal3_3 = self.ffn_norm3_3(atten3_3)
        # normal2_3 = self.gcn1_2(normal2_3, x_edge)
        mlp3_3 = self.Mlp3_3(normal3_3)
        mlp3_3 = atten3_3 + mlp3_3

        # 将三个特征图进行维度变换
        # graph3 = mlp3_1.transpose(1, 2)  # 转换为（1, 500, 534）
        # graph3_1 = mlp3_2.transpose(1, 2)  # 转换为（1, 256, 534）
        # graph3_2 = mlp3_3.transpose(1, 2)  # 转换为（1, 100, 534）
        fused_graph3_1 = self.down4(torch.cat([mlp3_1, mlp3_2, mlp3_3], dim=1))
        fused_graph3_1 = self.down4_1(fused_graph3_1)
        fused_graph3 = fused_graph3_1#.transpose(1, 2)
        return fused_graph3

# id = 'TCGA-44-7661'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# t_img_fea = joblib.load('./all_data.pkl')
# all_data = t_img_fea
# graph = all_data[id].to(device)
#
# # 定义模型、优化器和损失函数
# model = CrossScaleFusionTransformer(img_node=768,img_out=500).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.NLLLoss()
#
# # img = graph.x_img
# # edge = graph.edge_index_image
#
# # 模型训练
# for epoch in range(100):
#     model.train()
#     optimizer.zero_grad()
#     output = model(graph)
#     loss = criterion(output, torch.tensor([0]))  # 这里假设图分类的标签为0
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")