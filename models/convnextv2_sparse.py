import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv, LayerNorm
from mae_abl import mae_utils
from mae_abl.mae_utils import get_sinusoid_encoding_table#, Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import trunc_normal_, DropPath
# from MinkowskiEngine import SparseTensor



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class MinkowskiGRN(nn.Module):
#     """ GRN layer for sparse tensors.
#     """
#
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, dim))
#
#     def forward(self, x):
#         cm = x.coordinate_manager
#         in_key = x.coordinate_map_key
#
#         Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return SparseTensor(
#             self.gamma * (x.F * Nx) + self.beta + x.F,
#             coordinate_map_key=in_key,
#             coordinate_manager=cm)
#
#
# class MinkowskiDropPath(nn.Module):
#     """ Drop Path for sparse tensors.
#     """
#
#     def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
#         super(MinkowskiDropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep
#
#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         cm = x.coordinate_manager
#         in_key = x.coordinate_map_key
#         keep_prob = 1 - self.drop_prob
#         mask = torch.cat([
#             torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
#             else torch.zeros(len(_)) for _ in x.decomposed_coordinates
#         ]).view(-1, 1).to(x.device)
#         if keep_prob > 0.0 and self.scale_by_keep:
#             mask.div_(keep_prob)
#         return SparseTensor(
#             x.F * mask,
#             coordinate_map_key=in_key,
#             coordinate_manager=cm)
#
#
# class MinkowskiLayerNorm(nn.Module):
#     """ Channel-wise layer normalization for sparse tensors.
#     """
#
#     def __init__(
#             self,
#             normalized_shape,
#             eps=1e-6,
#     ):
#         super(MinkowskiLayerNorm, self).__init__()
#         self.ln = nn.LayerNorm(normalized_shape, eps=eps)
#
#     def forward(self, input):
#         output = self.ln(input.F)
#         return SparseTensor(
#             output,
#             coordinate_map_key=input.coordinate_map_key,
#             coordinate_manager=input.coordinate_manager)

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# class LayerNorm(nn.Module):
#     """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.5):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=1, padding=0, groups=dim)  # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(dim)
        self.pwconv2 = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.squeeze(0).permute(0, 2, 1)
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths


        # self.downsample_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(3, 512, kernel_size=1, stride=1),
        #         nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)
        #     ),
        #     nn.Sequential(
        #         nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
        #         nn.Conv1d(512, 512, kernel_size=1, stride=1)
        #     ),
        #     nn.Sequential(
        #         nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
        #         nn.Conv1d(512, 512, kernel_size=1, stride=1)
        #     )
        # ])

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=1, stride=1),
            nn.LayerNorm(dims[0], eps=1e-06, elementwise_affine=True)
        )
        self.downsample_layers.append(stem)
        for i in range(2):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-06, elementwise_affine=True),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1, stride=1),
            )
            self.downsample_layers.append(downsample_layer)


        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.relax = nn.Conv1d(500,3,1,1)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], dims[-1])
        self.pos_embed = get_sinusoid_encoding_table(3, 500)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 500))

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward_features(self, x, mask):
        x_vis= x.unsqueeze(0)


        # for layer in self.downsample_layers:
        #     x_vis = layer(x_vis)
        # patch embedding
        x_vis = self.downsample_layers[0](x_vis)


        for i in range(3):
            x_vis = self.downsample_layers[i](x_vis) if i > 0 else x_vis
            x_vis = self.stages[i](x_vis)

        # x_vis = to_sparse(x)
        x_vis = self.relax(x_vis)
        return self.norm(x_vis)  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, mask):
        x_vis = self.forward_features(x, mask)
        x = self.head(x_vis)

        return x


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
        #             GATConv(in_channels=dim1,out_channels=dim2),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))


class fusion_model_convnextv2(nn.Module):
    def __init__(self, in_feats, n_hidden, out_classes, dropout=0.3, train_type_num=3):
        super(fusion_model_convnextv2, self).__init__()

        self.img_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.img_relu_2 = GNN_relu_Block(out_classes)
        self.rna_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.rna_relu_2 = GNN_relu_Block(out_classes)
        self.cli_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.cli_relu_2 = GNN_relu_Block(out_classes)
        #         TransformerConv

        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_img = my_GlobalAttention(att_net_img)

        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_rna = my_GlobalAttention(att_net_rna)

        att_net_cli = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_cli = my_GlobalAttention(att_net_cli)

        att_net_img_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_img_2 = my_GlobalAttention(att_net_img_2)

        att_net_rna_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_rna_2 = my_GlobalAttention(att_net_rna_2)

        att_net_cli_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_cli_2 = my_GlobalAttention(att_net_cli_2)

        self.mae = ConvNeXtV2(depths=[2, 2, 6, 6], dims=[512, 512, 512, 512])

        self.mix = MixerBlock(train_type_num, out_classes)

        self.lin1_img = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_img = torch.nn.Linear(out_classes // 4, 1)
        self.lin1_rna = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_rna = torch.nn.Linear(out_classes // 4, 1)
        self.lin1_cli = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_cli = torch.nn.Linear(out_classes // 4, 1)

        self.norm_img = LayerNorm(out_classes // 4)
        self.norm_rna = LayerNorm(out_classes // 4)
        self.norm_cli = LayerNorm(out_classes // 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, all_thing, train_use_type=None, use_type=None, in_mask=[], mix=False):

        if len(in_mask) == 0:
            mask = np.array([[[False] * len(train_use_type)]])
        else:
            mask = in_mask

        data_type = use_type
        x_img = all_thing.x_img
        x_rna = all_thing.x_rna
        x_cli = all_thing.x_cli

        data_id = all_thing.data_id
        edge_index_img = all_thing.edge_index_image
        edge_index_rna = all_thing.edge_index_rna
        edge_index_cli = all_thing.edge_index_cli

        save_fea = {}
        fea_dict = {}
        num_img = len(x_img)
        num_rna = len(x_rna)
        num_cli = len(x_cli)

        att_2 = []
        pool_x = torch.empty((0)).to(device)
        if 'img' in data_type:
            x_img = self.img_gnn_2(x_img, edge_index_img)
            x_img = self.img_relu_2(x_img)
            batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
            pool_x_img, att_img_2 = self.mpool_img(x_img, batch)
            att_2.append(att_img_2)
            pool_x = torch.cat((pool_x, pool_x_img), 0)
        if 'rna' in data_type:
            x_rna = self.rna_gnn_2(x_rna, edge_index_rna)
            x_rna = self.rna_relu_2(x_rna)
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_2 = self.mpool_rna(x_rna, batch)
            att_2.append(att_rna_2)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)
        if 'cli' in data_type:
            x_cli = self.cli_gnn_2(x_cli, edge_index_cli)
            x_cli = self.cli_relu_2(x_cli)
            batch = torch.zeros(len(x_cli), dtype=torch.long).to(device)
            pool_x_cli, att_cli_2 = self.mpool_cli(x_cli, batch)
            att_2.append(att_cli_2)
            pool_x = torch.cat((pool_x, pool_x_cli), 0)

        fea_dict['mae_labels'] = pool_x

        if len(train_use_type) > 1:
            if use_type == train_use_type:
                mae_x = self.mae(pool_x, mask).squeeze(0)
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
                if k == 0:
                    mask = np.array([[[False] * len(train_use_type)]])
                mae_x = self.mae(tmp_x, mask).squeeze(0)
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
            if 'cli' in train_use_type and 'cli' in use_type:
                x_cli = x_cli + mae_x[train_use_type.index('cli')]
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
        if 'cli' in data_type:
            batch = torch.zeros(len(x_cli), dtype=torch.long).to(device)
            pool_x_cli, att_cli_3 = self.mpool_cli_2(x_cli, batch)
            att_3.append(att_cli_3)
            pool_x = torch.cat((pool_x, pool_x_cli), 0)

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
        if 'cli' in data_type:
            fea_dict['cli'] = fea[k]
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
        if 'cli' in data_type:
            x_cli = self.lin1_cli(x[k])
            x_cli = self.relu(x_cli)
            x_cli = self.norm_cli(x_cli)
            x_cli = self.dropout(x_cli)

            x_cli = self.lin2_rna(x_cli).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_cli), 0)
            k += 1
        one_x = torch.mean(multi_x, dim=0)

        return (one_x, multi_x), save_fea, (att_2, att_3), fea_dict










