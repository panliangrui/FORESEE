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
from mae_abl.mae_utils import get_sinusoid_encoding_table, Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import pdb
import torch
import torch.nn as nn

# from models.vision_transformer import PatchEmbed, Block, CBlock

from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from torch_geometric.nn import GATConv
# from mae_abl.mae_models.gin import GIN
# from mae_abl.mae_models.gat import GAT
# from mae_abl.mae_models.gcn import GCN
# from mae_abl.mae_models.dot_gat import DotGAT
# from mae_abl.mae_models.loss_func import sce_loss
# from mae_abl.mae_models.utils import create_norm, drop_edge
from mae_abl.mae_utils import get_sinusoid_encoding_table
# from models.HGMAE import PreModel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from itertools import chain
from functools import partial
import torch
import torch.nn as nn
import dgl
from dgl import DropEdge
import torch.nn.functional as F

# from hgmae.models.loss_func import sce_loss
# from mae_abl.mae_models.utils import create_norm
# from models.mae_models.han import HAN
# from mae_abl.mae_models.new_han import HAN



def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss
class PreModel(nn.Module):
    def __init__(
            self, args, num_metapath: int, focused_feature_dim: int):
        super(PreModel, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        self.feat_mask_rate = args.feat_mask_rate
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn
        self.enc_dec_input_dim = self.focused_feature_dim
        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # # num head: encoder
        # if self.encoder_type in ("gat", "dotgat", "han"):
        #     enc_num_hidden = 512#self.hidden_dim // self.num_heads
        #     enc_nhead = self.num_heads
        # else:
        #     enc_num_hidden = self.hidden_dim
        #     enc_nhead = 1
        #
        # # num head: decoder
        # if self.decoder_type in ("gat", "dotgat", "han"):
        #     dec_num_hidden = self.hidden_dim // self.num_out_heads
        #     dec_nhead = self.num_out_heads
        # else:
        #     dec_num_hidden = self.hidden_dim
        #     dec_nhead = 1
        dec_in_dim = self.hidden_dim

        # encoder
        self.encoder = setup_module(m_type ='han', in_channels=1024, out_channels=1024, num_layers=1)

        # decoder
        self.decoder = setup_module(m_type ='han', in_channels=1024, out_channels=1024, num_layers=1)

        # type-specific attribute restoration
        self.alpha_l = args.alpha_l
        self.attr_restoration_loss = self.setup_loss_fn(self.loss_fn, self.alpha_l)
        self.__cache_gs = None
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))
        self.encoder_to_decoder = nn.Linear(1024, 1024, bias=False)
        self._replace_rate = args.replace_rate
        self._leave_unchanged = args.leave_unchanged
        assert self._replace_rate + self._leave_unchanged < 1, "Replace rate + leave_unchanged must be smaller than 1"

        # mp edge recon
        self.use_mp_edge_recon = args.use_mp_edge_recon
        self.mp_edge_recon_loss_weight = args.mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = args.mp_edge_mask_rate
        self.mp_edge_alpha_l = args.mp_edge_alpha_l
        self.mp_edge_recon_loss = self.setup_loss_fn(self.loss_fn, self.mp_edge_alpha_l)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # mp2vec feat pred
        self.mps_embedding_dim = args.mps_embedding_dim
        self.use_mp2vec_feat_pred = args.use_mp2vec_feat_pred
        self.mp2vec_feat_pred_loss_weight = args.mp2vec_feat_pred_loss_weight
        self.mp2vec_feat_alpha_l = args.mp2vec_feat_alpha_l
        self.mp2vec_feat_drop = args.mp2vec_feat_drop
        self.mp2vec_feat_pred_loss = self.setup_loss_fn(self.loss_fn, self.mp2vec_feat_alpha_l)
        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(
            nn.Linear(dec_in_dim, self.mps_embedding_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mps_embedding_dim, self.mps_embedding_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mps_embedding_dim, self.mps_embedding_dim)
        )

    @property
    def output_hidden_dim(self):
        return self.hidden_dim

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=0):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate:  # 0.6~0.8 Uniform sample
                mask_rate = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, feats, mps):
        # prepare for mp2vec feat pred
        if self.use_mp2vec_feat_pred:
            mp2vec_feat = feats[0][:, self.focused_feature_dim:]
            origin_feat = feats[0][:, :self.focused_feature_dim]
        else:
            origin_feat = feats#[0]

        # type-specific attribute restoration
        # gs = self.mps_to_gs(mps)
        i=0
        i=i+1
        # loss, feat_recon, enc_out, mask_nodes = self.mask_attr_restoration(origin_feat, mps, epoch=i)
        loss, feat_recon, enc_out, mask_nodes = self.mask_attr_restoration(origin_feat, mps, epoch=i)

        # mp based edge reconstruction
        if self.use_mp_edge_recon:
            edge_recon_loss = self.mask_mp_edge_reconstruction(origin_feat, mps,epoch=0)
            loss += self.mp_edge_recon_loss_weight * edge_recon_loss

        # mp2vec feat pred
        if self.use_mp2vec_feat_pred:
            # MLP decoder
            mp2vec_feat_pred = self.enc_out_to_mp2vec_feat_mapping(enc_out)

            mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss(mp2vec_feat_pred, mp2vec_feat)

            loss += self.mp2vec_feat_pred_loss_weight * mp2vec_feat_pred_loss

        return loss, feat_recon#, loss.item()

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        num_leave_nodes = int(self._leave_unchanged * num_mask_nodes)
        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_leave_nodes - num_noise_nodes
        token_nodes = mask_nodes[perm_mask[: num_real_mask_nodes]]
        noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        if num_noise_nodes > 0:
            out_x[noise_nodes] = x[noise_to_be_chosen]

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_restoration(self, feat, gs, epoch):
        cur_feat_mask_rate = self.get_mask_rate(self.feat_mask_rate, epoch=epoch)
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_feat_mask_rate)

        enc_out= self.encoder(use_x, gs)

        # ---- attribute reconstruction ----
        enc_out_mapped = self.encoder_to_decoder(enc_out)
        if self.decoder_type != "mlp":
            # re-mask
            enc_out_mapped[mask_nodes] = 0  # TODO: learnable? remove?

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(enc_out_mapped)
        else:
            feat_recon = self.decoder(enc_out_mapped, gs)

        x_init = feat[mask_nodes]
        x_rec = feat_recon[mask_nodes]
        loss = self.attr_restoration_loss(x_rec, x_init)

        return loss, feat_recon, enc_out, mask_nodes

    def mask_mp_edge_reconstruction(self, feat, mps, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_rep, _ = self.encoder(masked_gs, feat, return_hidden=False)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(rep)
        else:
            feat_recon, att_mp = self.decoder(masked_gs, rep)

        gs_recon = torch.mm(feat_recon, feat_recon.T)

        loss = None
        for i in range(len(mps)):
            if loss is None:
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
                # loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])  # loss only on masked places
            else:
                loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
                # loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon_only_masked_places_list[i], mps_only_masked_places_list[i])
        return loss

    def get_embeds(self, feats, mps, *varg):
        if self.use_mp2vec_feat_pred:
            origin_feat = feats[0][:, :self.focused_feature_dim]
        else:
            origin_feat = feats[0]
        gs = self.mps_to_gs(mps)
        rep, _ = self.encoder(gs, origin_feat)
        return rep.detach()

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            # for mp in mps:
            indices = mps#._indices()
            cur_graph = dgl.graph((indices[0], indices[1]))
            gs.append(cur_graph)
            return gs
        else:
            return self.__cache_gs


def setup_module(m_type, in_channels=1024, out_channels=512, num_layers=2) -> nn.Module:
    if m_type == "han":
        mod = HAN(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers)
    else:
        raise NotImplementedError

    return mod

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


import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GAT

class GraphMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GraphMAE, self).__init__()

        # Encoder
        self.encoder = GAT(input_dim, hidden_dim, num_layers)

        # Decoder
        self.decoder = GAT(hidden_dim, hidden_dim, num_layers)

        # Encoder to Decoder connection
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x_encoder = self.encoder(x, edge_index)
        x_decoder = self.encoder_to_decoder(x_encoder)
        x_reconstructed = self.decoder(x_decoder, edge_index)
        return x_reconstructed


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

    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths


        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(3, 512, kernel_size=1, stride=1),
                nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)
            ),
            nn.Sequential(
                nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
                nn.Conv1d(512, 512, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
                nn.Conv1d(512, 512, kernel_size=1, stride=1)
            )
        ])


        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.relax = nn.Conv1d(512,3,1,1)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], dims[-1])
        self.pos_embed = get_sinusoid_encoding_table(3, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 512))

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward_features(self, x, mask):
        x_vis= x.unsqueeze(0)


        for layer in self.downsample_layers:
            x_vis = layer(x_vis)

        for i in range(3):
            # x_vis = self.downsample_layers[i](x_vis)
            x_vis = self.stages[i](x_vis)

        # x_vis = to_sparse(x)
        x_vis = self.relax(x_vis)
        return self.norm(x_vis)  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, mask):
        x_vis = self.forward_features(x, mask)
        x = self.head(x_vis)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        # GATConv(in_channels=1024,out_channels=512),
        # nn.Linear(1024, 512),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))


def GNN_relu_Block_new(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        # GATConv(in_channels=1024,out_channels=512),
        nn.Linear(1024, 512),
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

from torch_geometric.nn import GATConv,GCNConv,GENConv, GINConv,GMMConv,GPSConv, GINEConv, GATv2Conv, APPNP, GatedGraphConv, ARMAConv

class GINNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINNet, self).__init__()

        self.gin_conv = GINConv(nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        ), eps=0.2, train_eps=True)

    def forward(self, x, edge_index):
        x = self.gin_conv(x, edge_index)
        return x

class GPSNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv=None, heads=1, dropout=0.0, attn_dropout=0.0,
                 act='relu', act_kwargs=None, norm='batch_norm', norm_kwargs=None):
        super(GPSNet, self).__init__()

        self.gps_conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True
        )

    def forward(self, x, edge_index):
        x = self.gps_conv(x, edge_index)
        return x

class GraphARMAConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stacks, num_layers):
        super(GraphARMAConv, self).__init__()

        self.armac = nn.ModuleList()
        self.armac.append(ARMAConv(input_dim, hidden_dim, num_layers))
        for _ in range(num_stacks - 1):
            self.armac.append(ARMAConv(hidden_dim, hidden_dim, num_layers))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for armac in self.armac:
            x = armac(x, edge_index)
        x = self.fc(x)
        return x
class fusion_model_our(nn.Module):
    def __init__(self, args, in_feats, n_hidden, out_classes, dropout=0.3, train_type_num=3):
        super(fusion_model_our, self).__init__()

        # self.img_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)#args, 2, 1024
        self.img_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        self.img_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.img_gnn_2 = PreModel(args, 2, 1024)
        self.img_relu_2 = GNN_relu_Block(out_classes)


        # self.rna_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.rna_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        self.rna_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.rna_gnn_2 = PreModel(args, 2, 1024)
        self.rna_relu_2 = GNN_relu_Block(out_classes)


        # self.cli_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.cli_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        self.cli_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        # self.cli_gnn_2 = PreModel(args, 2, 1024)
        self.cli_relu_2 = GNN_relu_Block(out_classes)
        #TransformerConv

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

        # self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1, decoder_depth=1, train_type_num=train_type_num)
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
            x_img = self.img_gnn_2(x_img, edge_index_img)#loss_img,
            x_img = self.img_relu_2(x_img)
            batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
            pool_x_img, att_img_2 = self.mpool_img(x_img, batch)
            att_2.append(att_img_2)
            pool_x = torch.cat((pool_x, pool_x_img), 0)
        if 'rna' in data_type:
            x_rna = self.rna_gnn_2(x_rna, edge_index_rna) #loss_rna,
            x_rna = self.rna_relu_2(x_rna)
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_2 = self.mpool_rna(x_rna, batch)
            att_2.append(att_rna_2)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)
        if 'cli' in data_type:
            x_cli = self.cli_gnn_2(x_cli, edge_index_cli) #loss_cli,
            x_cli = self.cli_relu_2(x_cli)
            batch = torch.zeros(len(x_cli), dtype=torch.long).to(device)
            pool_x_cli, att_cli_2 = self.mpool_cli(x_cli, batch)
            att_2.append(att_cli_2)
            pool_x = torch.cat((pool_x, pool_x_cli), 0)

        fea_dict['mae_labels'] = pool_x

        if len(train_use_type) > 1:
            if use_type == train_use_type:
                # edge_index_all = torch.cat((edge_index_img, edge_index_rna), 1)
                # edge_index_all = torch.cat((edge_index_all, edge_index_cli), 1)
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