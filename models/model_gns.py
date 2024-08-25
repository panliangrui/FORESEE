import os
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv, LayerNorm
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
import torch.optim as optim
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode, LIAFNode
from functools import partial
from spikingjelly.clock_driven import functional
import spikingjelly.clock_driven.encoding as encoding
from models.Hybird_Attention import HybirdModel
# from models.CrossfusionTransformer import CrossScaleFusionTransformer
from models.crossVIT2_3 import CrossScaleFusionTransformer
from models.Spikformer import Spikformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.convnextv2_sparse import ConvNeXtV2
from models.vit_mae import PretrainVisionTransformer

def intensity_encode(input_data, scale_factor=1.0):
    # 强度编码，映射到脉冲发放强度
    return input_data * scale_factor

class SPS(nn.Module):
    def __init__(self, inputs=5000, embed_dims=200):
        super().__init__()
        self.proj_conv = nn.Conv1d(in_channels=inputs, out_channels=embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
        # self.flatten = nn.Flatten()

        self.rpe_conv = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.rpe_bn = nn.BatchNorm1d(embed_dims)
        self.rpe_lif = LIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        # print(x)
        x = self.maxpool(x)
        # x = self.flatten(x)

        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)

        return x

class SSA(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(tau=2.0, detach_reset=True)
        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        # x = x.flatten(2)
        # T, B, N = x.shape
        x_for_qkv = x#.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)#.reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2)#.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)#.reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2)#.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)#.reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2)#.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(1, 2)#.reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        # x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)))#.reshape(T,B,C,H,W))
        return x#, v

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNode(tau=2.0, detach_reset=True)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNode(tau=2.0, detach_reset=True)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=dim, drop=drop)

    def forward(self, x):
        x_attn = (self.attn(x))
        x = x + x_attn
        x = x + (self.mlp((x)))

        return x

# class Spikformer(nn.Module):
#     def __init__(self, inputs=128, outputs = 100, depths=1, qkv_bias=False, qk_scale=None,drop_rate=0., attn_drop_rate=0.,  mlp_ratios=4, drop_path_rate=0., norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2]
#                  ):
#         super().__init__()
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
#
#         self.patch_embed = SPS(inputs=inputs)
#
#         self.block = nn.ModuleList([Block(
#             dim=200,  mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
#             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
#             norm_layer=norm_layer, sr_ratio=sr_ratios)
#             for j in range(1)])
#
#         self.head = nn.Linear(2500, outputs) #if num_classes > 0 else nn.Identity()
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward_features(self, x):
#         x= self.patch_embed(x)
#         for blk in self.block:
#             x = blk(x)
#         return x#.flatten(3).mean(3)
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = x.flatten(0,1)
#         # x = self.head(x)
#         return x
#
# class SNNWithAttention(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(SNNWithAttention, self).__init__()
#         self.attention = SelfAttentionLayer(input_size)
#         self.snn_layer = SpikingNeuronLayer(input_size)
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         attention_output = self.attention(x)
#         spikes_0 = self.snn_layer(attention_output)
#         spikes = self.attention(x)
#         x = torch.relu(self.fc1(spikes))
#         x = self.fc2(x)
#         return x

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, stride=1, embed_dim=512):
        super().__init__()
        self.stride = stride
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(512)
        self.act = nn.GELU()
    def forward(self, x):
        a = x.unsqueeze(0)
        x = self.proj(a)#.squeeze(0)
        x = self.norm(x)
        return self.act(x)



def Mix_mlp(dim1):
    return nn.Sequential(
        nn.Linear(dim1, dim1),
        nn.GELU(),
        nn.Linear(dim1, dim1))


class MixerBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super(MixerBlock, self).__init__()

        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)

    def forward(self, x):
        x = x.transpose(0, 1)
        # z = nn.Linear(512, 3)(x)

        y = self.norm(x)
        # y = y.transpose(0,1)
        y = self.mix_mip_1(y)
        # y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        y = y.transpose(0, 1)
        z = self.mix_mip_2(y)
        z = z.transpose(0, 1)
        x = x + z
        x = x.transpose(0, 1)

        # y = self.norm(x)
        # y = y.transpose(0,1)
        # y = self.mix_mip_1(y)
        # y = y.transpose(0,1)
        # x = self.norm(y)
        return x


def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
        nn.Dropout(p=dropout))


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

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class Attention_Before(nn.Module):
    def __init__(self, config):
        super(Attention_Before, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        return query_layer, key_layer, value_layer


class Attention_After(nn.Module):
    def __init__(self, config):
        super(Attention_After, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def forward(self, query_layer, key_layer, value_layer):
        query_layer = torch.poo
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class fusion_model_mae(nn.Module):
    def __init__(self, dict, out_classes, dropout=0.2, train_type_num=3):
        super(fusion_model_mae, self).__init__()
        configs = dict
        ######################################
        ###for cross fusion
        # self.img_node = configs['img_node']#img_node = 224, img_out = 500
        self.img_out = configs['img_out']
        self.out_classes = configs['out_classes']

        ######################################
        ##for HybirdModel papr
        self.input_size = configs['input_size']
        self.hidden_size = configs['hidden_size']
        self.num_layers = configs['num_layers']
        self.output_size = configs['output_size']

        ########################################
        self.input_size_C_M = configs['input_size_C_M']
        self.hidden_size_C_M = configs['hidden_size_C_M']
        self.num_layers_C_M = configs['num_layers_C_M']
        self.output_size_C_M = configs['output_size_C_M']

        ##for Spikformer
        # self.input = configs['input']
        # self.outputs = configs['outputs']
        ######cross fusion
        self.wsi = CrossScaleFusionTransformer(dict=dict, dropout_rate=0.2)
        self.img_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_img = my_GlobalAttention(att_net_img)

        #######处理rna 降噪+transformer
        self.rna = HybirdModel(self.input_size, self.hidden_size, self.num_layers, self.output_size)
        self.rna_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_rna = my_GlobalAttention(att_net_rna)
        ###处理mut和cnv,
        self.mut_cnv_sen = HybirdModel(self.input_size_C_M, self.hidden_size_C_M, self.num_layers_C_M, self.output_size_C_M)
        self.mut_cnv_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        att_net_mut_cnv = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_mut_cnv = my_GlobalAttention(att_net_mut_cnv)
        ###脉冲注意力网络编码
        # self.mut_cnv_sen = Spikformer(inputs=self.input, outputs =self.outputs)#(input_size=10, hidden_size=128, num_classes=10)
        # self.mut_cnv_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        # att_net_mut_cnv = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        # self.mpool_mut_cnv = my_GlobalAttention(att_net_mut_cnv)
        ###特征融合
        #TransformerConv
        # self.mae = ConvNeXtV2(depths=[2, 2, 6, 6], dims=[out_classes, out_classes, out_classes, out_classes])
        self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes,
                                             decoder_embed_dim=out_classes, encoder_depth=1, decoder_depth=1,
                                             train_type_num=train_type_num)
        self.mix = MixerBlock(train_type_num, out_classes)

        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_rna = my_GlobalAttention(att_net_rna)

        att_net_mut_cnv = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_mut_cnv = my_GlobalAttention(att_net_mut_cnv)

        att_net_img_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_img_2 = my_GlobalAttention(att_net_img_2)

        att_net_rna_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_rna_2 = my_GlobalAttention(att_net_rna_2)

        att_net_mut_cnv_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_mut_cnv_2 = my_GlobalAttention(att_net_mut_cnv_2)

        self.mix = MixerBlock(train_type_num, out_classes)

        self.lin1_img = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_img = torch.nn.Linear(out_classes // 4, 1)
        self.lin1_rna = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_rna = torch.nn.Linear(out_classes // 4, 1)
        self.lin1_mut_cnv = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_mut_cnv = torch.nn.Linear(out_classes // 4, 1)

        self.norm_img = LayerNorm(out_classes // 4)
        self.norm_rna = LayerNorm(out_classes // 4)
        self.norm_mut_cnv = LayerNorm(out_classes // 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, all_thing, train_use_type=None, use_type=None, in_mask=[], in_mask1=[], in_mask2=[], mix=False):

        global mask1, mask2, x_mut_cnv, x_img
        if len(in_mask) == 0:
            mask = np.array([[[False] * len(train_use_type)]])
        else:
            mask = in_mask
            mask1 = in_mask1
            mask2 = in_mask2

        data_type = use_type
        x_img_256 = all_thing.x_img_256
        x_img_512 = all_thing.x_img_512
        x_img_1024 = all_thing.x_img_1024
        x_rna = all_thing.rna
        x_cnv = all_thing.cnv
        x_mut = all_thing.mut
        # x_cli = all_thing.x_cli

        data_id = all_thing.data_id
        edge_index_img_256 = all_thing.edge_index_image_256
        edge_index_img_512 = all_thing.edge_index_image_512
        edge_index_img_1024 = all_thing.edge_index_image_1024

        save_fea = {}
        fea_dict = {}
        num_img_256 = len(x_img_256)
        num_img_512 = len(x_img_512)
        num_img_1024 = len(x_img_1024)
        num_rna = len(x_rna)
        num_cnv = len(x_cnv)
        num_mut = len(x_mut)
        # num_cli = len(x_cli)

        att_2 = []
        pool_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            with torch.no_grad():
                # print(data_id)
                x_img = self.wsi(x_img_256, x_img_512, x_img_1024, edge_index_img_256, edge_index_img_512, edge_index_img_1024)
                x_img = self.img_relu_2(x_img.squeeze(0))
                batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
                pool_x_img, att_img_2 = self.mpool_img(x_img, batch)
                att_2.append(att_img_2)
                pool_x = torch.cat((pool_x, pool_x_img), 0)

        if 'rna' in data_type:
            # with torch.no_grad():
                x_rna = self.rna(x_rna)
                x_rna = self.rna_relu_2(x_rna)
                batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
                pool_x_rna, att_rna_2 = self.mpool_rna(x_rna, batch)
                att_2.append(att_rna_2)
                pool_x = torch.cat((pool_x, pool_x_rna), 0)

        if 'mut_cnv' in data_type:
            # with torch.no_grad():
                x_mut_cnv = torch.cat((x_mut, x_cnv), dim=1)#.unsqueeze(0)
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)#.squeeze(0)
                x_mut_cnv = self.mut_cnv_relu_2(x_mut_cnv)
                batch = torch.zeros(len(x_mut_cnv), dtype=torch.long).to(device)
                pool_x_mut_cnv, att_mut_cnv_2 = self.mpool_mut_cnv(x_mut_cnv, batch)
                att_2.append(att_mut_cnv_2)
                pool_x = torch.cat((pool_x, pool_x_mut_cnv), 0)

        fea_dict['mae_labels'] = pool_x

        if len(train_use_type) > 1:
            if use_type == train_use_type:
                mae_x = self.mae(pool_x, mask, mask1, mask2).squeeze(0)
                fea_dict['mae_out'] = mae_x
            else:
                k = 0
                tmp_x = torch.zeros((len(train_use_type), pool_x.size(1))).to(device)
                mask = np.ones(len(train_use_type), dtype=bool)
                for i, type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k += 1
                        mask[i] = False
                mask = np.expand_dims(mask, 0)
                mask = np.expand_dims(mask, 0)
                ###########################################
                k = 0
                tmp_x = torch.zeros((len(train_use_type), pool_x.size(1))).to(device)
                mask1 = np.ones(len(train_use_type), dtype=bool)
                for i, type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k += 1
                        mask1[i] = False
                mask1 = np.expand_dims(mask1, 0)
                mask1 = np.expand_dims(mask1, 0)
                #############################################
                k = 0
                tmp_x = torch.zeros((len(train_use_type), pool_x.size(1))).to(device)
                mask2 = np.ones(len(train_use_type), dtype=bool)
                for i, type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k += 1
                        mask2[i] = False
                mask2 = np.expand_dims(mask2, 0)
                mask2 = np.expand_dims(mask2, 0)


                if k == 0:
                    mask = np.array([[[False] * len(train_use_type)]])
                    mask1 = np.array([[[False] * len(train_use_type)]])
                    mask2 = np.array([[[False] * len(train_use_type)]])
                mae_x = self.mae(tmp_x, mask, mask1, mask2).squeeze(0)
                fea_dict['mae_out'] = mae_x

            save_fea['after_mae'] = mae_x.cpu().detach().numpy()
            if mix:
                mae_x = self.mix(mae_x)
                save_fea['after_mix'] = mae_x.cpu().detach().numpy()

            k = 0
            if 'img' in train_use_type and 'img' in use_type:
                x_img = x_img + mae_x[train_use_type.index('img')]
                k += 1
            if 'rna' in train_use_type and 'rna' in use_type:
                x_rna = x_rna + mae_x[train_use_type.index('rna')]
                k += 1
            if 'mut_cnv' in train_use_type and 'mut_cnv' in use_type:
                x_mut_cnv = x_mut_cnv + mae_x[train_use_type.index('mut_cnv')]
                k += 1

        att_3 = []
        pool_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
            pool_x_img, att_img_3 = self.mpool_img_2(x_img, batch)
            att_3.append(att_img_3)
            pool_x = torch.cat((pool_x, pool_x_img), 0)
        if 'rna' in data_type:
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_3 = self.mpool_rna_2(x_rna, batch)
            att_3.append(att_rna_3)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)
        if 'mut_cnv' in data_type:
            batch = torch.zeros(len(x_mut_cnv), dtype=torch.long).to(device)
            pool_x_mut_cnv, att_cli_3 = self.mpool_mut_cnv_2(x_mut_cnv, batch)
            att_3.append(att_cli_3)
            pool_x = torch.cat((pool_x, pool_x_mut_cnv), 0)

        x = pool_x

        x = F.normalize(x, dim=1)
        fea = x

        k = 0
        if 'img' in data_type:
            fea_dict['img'] = fea[k]
            k += 1
        if 'rna' in data_type:
            fea_dict['rna'] = fea[k]
            k += 1
        if 'mut_cnv' in data_type:
            fea_dict['mut_cnv'] = fea[k]
            k += 1

        k = 0
        multi_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            x_img = self.lin1_img(x[k])
            x_img = self.relu(x_img)
            x_img = self.norm_img(x_img)
            x_img = self.dropout(x_img)

            x_img = self.lin2_img(x_img).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_img), 0)
            k += 1
        if 'rna' in data_type:
            x_rna = self.lin1_rna(x[k])
            x_rna = self.relu(x_rna)
            x_rna = self.norm_rna(x_rna)
            x_rna = self.dropout(x_rna)

            x_rna = self.lin2_rna(x_rna).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_rna), 0)
            k += 1
        if 'mut_cnv' in data_type:
            x_mut_cnv = self.lin1_mut_cnv(x[k])
            x_mut_cnv = self.relu(x_mut_cnv)
            x_mut_cnv = self.norm_mut_cnv(x_mut_cnv)
            x_mut_cnv = self.dropout(x_mut_cnv)

            x_mut_cnv = self.lin2_rna(x_mut_cnv).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_mut_cnv), 0)
            k += 1
        one_x = torch.mean(multi_x, dim=0)

        return (one_x, multi_x), save_fea, (att_2, att_3), fea_dict