import torch
import math
import random
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import drop_path
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_mask(num=3):
    mask_num = num - 1
    #     mask_num = random.randint(1,2)
    mask = np.hstack([
        np.zeros(num - mask_num, dtype=bool),
        np.ones(mask_num, dtype=bool),
    ])
    np.random.shuffle(mask)
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 0)
    return mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
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


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, train_type_num=3,
                 ):
        super().__init__()
        self.num_classes = num_classes
        #         assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #         self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]

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
class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=512,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=512,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.3,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 train_type_num=3,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            train_type_num=train_type_num)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=3,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            train_type_num=train_type_num)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #         self.mask_token = torch.zeros(1, 1, decoder_embed_dim).to(device)

        self.pos_embed = get_sinusoid_encoding_table(train_type_num, decoder_embed_dim)

        self.atten = AttentionModule(in_channels=3) ##2

        trunc_normal_(self.mask_token, std=.02)

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
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, mask1, mask2):  #, mask1, mask2
        #第一个模态掩码
        x_vis0 = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis0 = self.encoder_to_decoder(x_vis0)  # [B, N_vis, C_d]

        B, N, C = x_vis0.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed0 = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis0 = expand_pos_embed0[~mask].reshape(B, -1, C)
        pos_emd_mask0 = expand_pos_embed0[mask].reshape(B, -1, C)
        x_full0 = torch.cat([x_vis0 + pos_emd_vis0, self.mask_token + pos_emd_mask0], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x0 = self.decoder(x_full0, 0)  # [B, N_mask, 3 * 16 * 16]

        tmp_x0 = torch.zeros_like(x0).to(device)
        Mask_n0 = 0
        Truth_n0 = 0
        for i0, flag0 in enumerate(mask[0][0]):
            if flag0:
                tmp_x0[:, i0] = x0[:, pos_emd_vis0.shape[1] + Mask_n0]
                Mask_n0 += 1
            else:
                tmp_x0[:, i0] = x0[:, Truth_n0]
                Truth_n0 += 1

        ####第二个模态掩码
        x_vis1 = self.encoder(x, mask1)  # [B, N_vis, C_e]
        x_vis1 = self.encoder_to_decoder(x_vis1)  # [B, N_vis, C_d]

        B, N, C = x_vis1.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed1 = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis1 = expand_pos_embed1[~mask1].reshape(B, -1, C)
        pos_emd_mask1 = expand_pos_embed1[mask1].reshape(B, -1, C)
        x_full1 = torch.cat([x_vis1 + pos_emd_vis1, self.mask_token + pos_emd_mask1], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x1 = self.decoder(x_full1, 0)  # [B, N_mask, 3 * 16 * 16]
        tmp_x1 = torch.zeros_like(x1).to(device)
        Mask_n1 = 0
        Truth_n1 = 0
        for i1, flag1 in enumerate(mask1[0][0]):
            if flag1:
                tmp_x1[:, i1] = x1[:, pos_emd_vis1.shape[1] + Mask_n1]
                Mask_n1 += 1
            else:
                tmp_x1[:, i1] = x1[:, Truth_n1]
                Truth_n1 += 1

        #######第三个模态掩码
        x_vis2 = self.encoder(x, mask2)  # [B, N_vis, C_e]
        x_vis2 = self.encoder_to_decoder(x_vis2)  # [B, N_vis, C_d]

        B, N, C = x_vis2.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed2 = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis2 = expand_pos_embed2[~mask2].reshape(B, -1, C)
        pos_emd_mask2 = expand_pos_embed2[mask2].reshape(B, -1, C)
        x_full2 = torch.cat([x_vis2 + pos_emd_vis2, self.mask_token + pos_emd_mask2], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x2 = self.decoder(x_full2, 0)  # [B, N_mask, 3 * 16 * 16]

        tmp_x2 = torch.zeros_like(x2).to(device)
        Mask_n2 = 0
        Truth_n2 = 0
        for i2, flag2 in enumerate(mask2[0][0]):
            if flag2:
                tmp_x2[:, i2] = x2[:, pos_emd_vis2.shape[1] + Mask_n2]
                Mask_n2 += 1
            else:
                tmp_x2[:, i2] = x2[:, Truth_n2]
                Truth_n2 += 1

        ###################注意力机制提取特征
        Q=tmp_x0
        K=tmp_x1
        V=tmp_x2
        tmp_x = self.atten(Q, K, V)
        return tmp_x