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
# from models.convnextv2_sparse import ConvNeXtV2
from models.vit_mae import PretrainVisionTransformer
# from mae_abl.mae_model import PretrainVisionTransformer
# # from mae_abl.covmae_new import MaskedAutoencoderConvViT
# from mae_abl.DMAE import MaskedAutoencoderConvViT
# from mae_abl.PUT import PatchVQGAN
# from mae_abl.mage import MaskedGenerativeEncoderViT
from mae_abl.mae_utils import get_sinusoid_encoding_table,Block
from models.hae_abl import CNN1, CNN2, LSTMModel, AutoEncoder, DenoisingAutoencoder, VAE, SelfAttentionTransformer1D




class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False, train_type_num=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        #         self.patch_embed = PatchEmbed(
        #             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #         num_patches = self.patch_embed.num_patches

        self.patch_embed = nn.Linear(embed_dim, embed_dim)
        num_patches = train_type_num

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x



class AttentionModule(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query_conv = torch.nn.Conv1d(in_channels, in_channels//2, kernel_size=1)
        self.key_conv = torch.nn.Conv1d(in_channels, in_channels//2, kernel_size=1)
        self.value_conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, Q, K, V):
        proj_query = self.query_conv(Q).view(Q.size(0), -1, Q.size(2)).permute(0, 2, 1)
        proj_key = self.key_conv(K).view(K.size(0), -1, K.size(2))
        energy = torch.bmm(proj_query, proj_key)

        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(V).view(V.size(0), -1, V.size(2))
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma * out + V

        return out



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
        self.cnn_dim_r = configs['cnn_size_r']
        self.daee = configs['daee']
        self.dae_l = configs['dae_l']

        ########################################
        self.input_size_C_M = configs['input_size_C_M']
        self.hidden_size_C_M = configs['hidden_size_C_M']
        self.num_layers_C_M = configs['num_layers_C_M']
        self.output_size_C_M = configs['output_size_C_M']
        self.cnn_dim_c_m = configs['cnn_size_c_m']
        self.daee1 = configs['daee1']

        ##for Spikformer
        # self.input = configs['input']
        # self.outputs = configs['outputs']
        ######cross fusion
        self.wsi = CrossScaleFusionTransformer(dict=dict, dropout_rate=0.2)
        self.img_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_img = my_GlobalAttention(att_net_img)

        #######################################
        #######处理rna 降噪+transformer
        # self.rna = HybirdModel(self.input_size, self.hidden_size, self.num_layers, self.output_size)
        ###if hae == 'CNN':
        self.rna = CNN1(dim=self.cnn_dim_r)
        # ###elif hae == 'LSTM':
        # self.rna =LSTMModel(self.input_size-1, self.hidden_size, self.num_layers, self.output_size)#ucec, blca
        # self.rna = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.output_size)#luad, brca
        # ###elif hae == 'AutoEncoder':
        # self.rna = AutoEncoder(input_dim=self.input_size-1, hidden_dim=self.hidden_size, output_dim=self.output_size)#ucec
        # self.rna = AutoEncoder(input_dim=self.input_size, hidden_dim=self.hidden_size, output_dim=self.output_size)  # luad
        # ###elif hae == 'DAE':
        # self.rna = DenoisingAutoencoder(self.daee, self.dae_l, 500)
        # ###elif hae == 'VAE':
        # self.rna = VAE(self.input_size-1, self.output_size, 50, 128) #ucec, blca
        # self.rna = VAE(self.input_size, self.output_size, 50, 128) # luad, brca
        # ####elif hae == 'Transformer':
        # self.rna = SelfAttentionTransformer1D(self.input_size-1, self.output_size) #ucec, blca
        # self.rna = SelfAttentionTransformer1D(self.input_size, self.output_size) # luad, brca
        ####################################
        self.rna_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_rna = my_GlobalAttention(att_net_rna)
        ###处理mut和cnv,
        # self.mut_cnv_sen = HybirdModel(self.input_size_C_M, self.hidden_size_C_M, self.num_layers_C_M, self.output_size_C_M)
        ###if hae == 'CNN':
        self.mut_cnv_sen = CNN2(self.cnn_dim_c_m)
        # ###elif hae == 'LSTM':
        # self.mut_cnv_sen = LSTMModel(self.input_size_C_M, self.hidden_size_C_M, self.num_layers_C_M, self.output_size_C_M)# ucec
        # self.mut_cnv_sen = LSTMModel(self.input_size_C_M-1, self.hidden_size_C_M, self.num_layers_C_M, self.output_size_C_M) #luad, brca
        # ###elif hae == 'AutoEncoder':
        # self.mut_cnv_sen = AutoEncoder(input_dim=self.input_size_C_M, hidden_dim=self.hidden_size_C_M, output_dim=self.output_size_C_M)#ucec, blca
        # self.mut_cnv_sen = AutoEncoder(input_dim=self.input_size_C_M-1, hidden_dim=self.hidden_size_C_M, output_dim=self.output_size_C_M) #luad
        # ###elif hae == 'DAE':
        # self.mut_cnv_sen = DenoisingAutoencoder(self.daee1, self.dae_l, 500)
        # ###elif hae == 'VAE':
        # self.mut_cnv_sen = VAE(self.input_size_C_M, self.output_size, 50, 128)#ucec, blca
        # self.mut_cnv_sen = VAE(self.input_size_C_M-1, self.output_size, 50, 128) # luad, brca
        # ####elif hae == 'Transformer':
        # self.mut_cnv_sen = SelfAttentionTransformer1D(self.input_size_C_M, self.output_size) #ucec, blca
        # self.mut_cnv_sen = SelfAttentionTransformer1D(self.input_size_C_M-1, self.output_size) # luad, brca
        ####################################
        self.mut_cnv_relu_2 = GNN_relu_Block(dim2=self.out_classes)
        att_net_mut_cnv = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_mut_cnv = my_GlobalAttention(att_net_mut_cnv)

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

    def forward(self, all_thing, train_use_type=None, use_type=None, in_mask=[], in_mask1=[], in_mask2=[], mix=False, hae=None):

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
                x_img = self.wsi(x_img_256, x_img_512, x_img_1024, edge_index_img_256, edge_index_img_512, edge_index_img_1024)
                x_img = self.img_relu_2(x_img.squeeze(0))
                batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
                pool_x_img, att_img_2 = self.mpool_img(x_img, batch)
                att_2.append(att_img_2)
                pool_x = torch.cat((pool_x, pool_x_img), 0)

        if 'rna' in data_type:
            # with torch.no_grad():
            #     x_rna = self.rna(x_rna)
            ##################################################################
            ###########选择不同的CNN，LSTM，AutoEncoder，DAE，VAE，Transformer
            if hae == 'CNN':
                x_rna = self.rna(x_rna)
            elif hae == 'LSTM':
                x_rna = self.rna(x_rna)
            elif hae == 'AutoEncoder':
                x_rna = self.rna(x_rna)
            elif hae == 'DAE':
                x_rna = self.rna(x_rna)
            elif hae == 'VAE':
                x_rna = self.rna(x_rna)
            elif hae == 'Transformer':
                x_rna = self.rna(x_rna)
            else:
                x_rna = None

            x_rna = self.rna_relu_2(x_rna)
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_2 = self.mpool_rna(x_rna, batch)
            att_2.append(att_rna_2)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)

        if 'mut_cnv' in data_type:
            # with torch.no_grad():
            x_mut_cnv = torch.cat((x_mut, x_cnv), dim=1)#.unsqueeze(0)
            # x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)#.squeeze(0)
            ##################################################################
        ###########选择不同的CNN，LSTM，AutoEncoder，DAE，VAE，Transformer
            if hae == 'CNN':
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)
            elif hae == 'LSTM':
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)
            elif hae == 'AutoEncoder':
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)
            elif hae == 'DAE':
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)
            elif hae == 'VAE':
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)
            elif hae == 'Transformer':
                x_mut_cnv = self.mut_cnv_sen(x_mut_cnv)
            else:
                x_mut_cnv = None

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