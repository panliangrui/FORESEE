import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode, LIAFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
from spikingjelly.clock_driven import functional
import spikingjelly.clock_driven.encoding as encoding

# 定义脉冲神经元层
# class SpikingNeuronLayer(nn.Module):
#     def __init__(self, num_inputs, threshold=1.0, decay=0.9):
#         super(SpikingNeuronLayer, self).__init__()
#         self.threshold = threshold
#         self.decay = decay
#         self.membrane_potential = torch.zeros(num_inputs, requires_grad=True)
#
#     def forward(self, x):
#         self.membrane_potential = self.membrane_potential * self.decay + x
#         spikes = torch.zeros_like(self.membrane_potential)
#         spikes[self.membrane_potential >= self.threshold] = 1.0
#         self.membrane_potential[self.membrane_potential >= self.threshold] = 0.0
#         return spikes
#
# # 定义脉冲神经网络模型
# class SNNModel(nn.Module):
#     def __init__(self):
#         super(SNNModel, self).__init__()
#         self.input_layer = SpikingNeuronLayer(num_inputs=10)
#         self.fc1 = nn.Linear(10, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.input_layer(x)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# def intensity_encode(input_data, scale_factor=1.0):
#     # 强度编码，映射到脉冲发放强度
#     return input_data * scale_factor

class SPS(nn.Module):
    def __init__(self, inputs=5000, embed_dims=1):
        super().__init__()
        self.proj_conv = nn.Conv1d(in_channels=inputs, out_channels=embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        self.proj_lif = LIFNode(tau=1.1, detach_reset=False)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
        # self.flatten = nn.Flatten()

        self.rpe_conv = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.rpe_bn = nn.BatchNorm1d(embed_dims)
        self.adp = nn.AdaptiveAvgPool1d(500)
        self.rpe_lif = LIFNode(tau=1.1, detach_reset=False)

    def forward(self, x):
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        print(x)
        x = self.maxpool(x)
        # x = self.flatten(x)

        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.adp(x)
        x = self.rpe_lif(x)
        print(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=1.1, detach_reset=True)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=1.1, detach_reset=True)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(tau=1.1, detach_reset=True)
        self.attn_lif = LIFNode(tau=1.1, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(tau=1.1, detach_reset=True)

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
        self.fc1_lif = LIFNode(tau=1.1, detach_reset=True)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNode(tau=1.1, detach_reset=True)

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

class Spikformer(nn.Module):
    def __init__(self,inputs=128, outputs = 100, depths=1, qkv_bias=False, qk_scale=None,drop_rate=0., attn_drop_rate=0.,  mlp_ratios=4, drop_path_rate=0., norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2]
                 ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.patch_embed = SPS(inputs=inputs)

        self.block = nn.ModuleList([Block(
            dim=1,  mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])
        self.head = nn.Linear(1, outputs) #if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x= self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        return x#.flatten(3).mean(3)
    def forward(self, x):
        x = self.forward_features(x)
        return x