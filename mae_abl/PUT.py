from audioop import bias
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# from image_synthesis.modeling.codecs.base_codec import BaseCodec
# from image_synthesis.modeling.modules.vqgan_loss.vqperceptual import VQLPIPSWithDiscriminator
from mae_abl.image_synthesis.modeling.utils.misc import distributed_sinkhorn, get_token_type
# from image_synthesis.distributed.distributed import all_reduce, get_world_size

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
from functools import partial
import pdb
import torch
import torch.nn as nn
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

def value_scheduler(init_value, dest_value, step, step_range, total_steps, scheduler_type='cosine'):
    assert scheduler_type in ['cosine', 'step'], 'scheduler {} not implemented!'.format(scheduler_type)

    step_start, step_end = tuple(step_range)
    if step_end <= 0:
        step_end = total_steps

    if step < step_start:
        return init_value
    if step > step_end:
        return dest_value

    factor = float(step - step_start) / float(max(1, step_end - step_start))
    if scheduler_type == 'cosine':
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * factor)))
    elif scheduler_type == 'step':
        factor = 1 - factor
    else:
        raise NotImplementedError('scheduler type {} not implemented!'.format(scheduler_type))
    if init_value >= dest_value:  # decrease
        value = dest_value + (init_value - dest_value) * factor
    else:  # increase
        factor = 1 - factor
        value = init_value + (dest_value - init_value) * factor
    return value


def gumbel_softmax(logits, temperature=1.0, gumbel_scale=1.0, dim=-1, hard=True):
    # gumbels = torch.distributions.gumbel.Gumbel(0,1).sample(logits.shape).to(logits)
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    # adjust the scale of gumbel noise
    gumbels = gumbels * gumbel_scale

    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


# class for quantization
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,n_e=500,e_dim = 500,beta=0.25,masked_embed_start=500,embed_init_scale=1.0,embed_ema=True,get_embed_type='retrive',distance_type='euclidean',
                 gumbel_sample=False,adjust_logits_for_gumbel='sqrt',gumbel_sample_stop_step=None,temperature_step_range=(0, 15000),temperature_scheduler_type='cosine',
                 temperature_init=1.0,temperature_dest=1 / 16.0,gumbel_scale_init=1.0, gumbel_scale_dest=1.0,gumbel_scale_step_range=(0, 1),
                 gumbel_scale_scheduler_type='cosine'):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embed_ema = embed_ema
        self.gumbel_sample = gumbel_sample
        self.adjust_logits_for_gumbel = adjust_logits_for_gumbel
        self.temperature_step_range = temperature_step_range
        self.temperature_init = temperature_init
        self.temperature_dest = temperature_dest
        self.temperature_scheduler_type = temperature_scheduler_type
        self.gumbel_scale_init = gumbel_scale_init
        self.gumbel_scale_dest = gumbel_scale_dest
        self.gumbel_scale_step_range = gumbel_scale_step_range
        self.gumbel_sample_stop_step = gumbel_sample_stop_step
        self.gumbel_scale_scheduler_type = gumbel_scale_scheduler_type
        if self.gumbel_sample_stop_step is None:
            self.gumbel_sample_stop_step = max(self.temperature_step_range[-1], self.temperature_step_range[-1])

        self.get_embed_type = get_embed_type
        self.distance_type = distance_type

        if self.embed_ema:
            self.decay = 0.99
            self.eps = 1.0e-5
            embed = torch.randn(n_e, e_dim)
            # embed = torch.zeros(n_e, e_dim)
            # embed.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)
            self.register_buffer("embedding", embed)
            self.register_buffer("cluster_size", torch.zeros(n_e))
            self.register_buffer("embedding_avg", embed.clone())
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)

        self.masked_embed_start = masked_embed_start
        if self.masked_embed_start is None:
            self.masked_embed_start = self.n_e

        if self.distance_type == 'learned':
            self.distance_fc = nn.Linear(self.e_dim, self.n_e)

    @property
    def device(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight.device
        return self.embedding.device

    @property
    def norm_feat(self):
        return self.distance_type in ['cosine', 'sinkhorn']

    @property
    def embed_weight(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight
        else:
            return self.embedding

    def get_codebook(self):
        codes = {
            'default': {
                'code': self.embedding
            }
        }

        if self.masked_embed_start < self.n_e:
            codes['unmasked'] = {'code': self.embedding[:self.masked_embed_start]}
            codes['masked'] = {'code': self.embedding[self.masked_embed_start:]}

            default_label = torch.ones((self.n_e)).to(self.device)
            default_label[self.masked_embed_start:] = 0
            codes['default']['label'] = default_label
        return codes

    def norm_embedding(self):
        if self.training:
            with torch.no_grad():
                w = self.embed_weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                if isinstance(self.embedding, nn.Embedding):
                    self.embedding.weight.copy_(w)
                else:
                    self.embedding.copy_(w)

    def get_index(self, logits, topk=1, step=None, total_steps=None):
        """
        logits: BHW x N
        topk: the topk similar codes to be sampled from

        return:
            indices: BHW
        """

        if self.gumbel_sample:
            gumbel = True
            if self.training:
                if step > self.gumbel_sample_stop_step and self.gumbel_sample_stop_step > 0:
                    gumbel = False
            else:
                gumbel = False
        else:
            gumbel = False

        if gumbel:
            temp = value_scheduler(init_value=self.temperature_init,
                                   dest_value=self.temperature_dest,
                                   step=step,
                                   step_range=self.temperature_step_range,
                                   total_steps=total_steps,
                                   scheduler_type=self.temperature_scheduler_type
                                   )
            scale = value_scheduler(init_value=self.gumbel_scale_init,
                                    dest_value=self.gumbel_scale_dest,
                                    step=step,
                                    step_range=self.gumbel_scale_step_range,
                                    total_steps=total_steps,
                                    scheduler_type=self.gumbel_scale_scheduler_type
                                    )
            if self.adjust_logits_for_gumbel == 'none':
                pass
            elif self.adjust_logits_for_gumbel == 'sqrt':
                logits = torch.sqrt(logits)
            elif self.adjust_logits_for_gumbel == 'log':
                logits = torch.log(logits)
            else:
                raise NotImplementedError

            # for logits, the larger the value is, the corresponding code shoule not be sampled, so we need to negative it
            logits = -logits
            # one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=True) # BHW x N
            logits = gumbel_softmax(logits, temperature=temp, gumbel_scale=scale, dim=1, hard=True)
        else:
            logits = -logits

        # now, the larger value should be sampled
        if topk == 1:
            indices = torch.argmax(logits, dim=1)
        else:
            assert not gumbel, 'For gumbel sample, topk may introduce some random choices of codes!'
            topk = min(logits.shape[1], topk)

            _, indices = torch.topk(logits, dim=1, k=topk)  # N x K
            chose = torch.randint(0, topk, (indices.shape[0],)).to(indices.device)  # N
            chose = torch.zeros_like(indices).scatter_(1, chose.unsqueeze(dim=1), 1.0)  # N x K
            indices = (indices * chose).sum(dim=1, keepdim=False)

            # filtered_logits = logits_top_k(logits, filter_ratio=topk, minimum=1, filter_type='count')
            # probs = F.softmax(filtered_logits * 1, dim=1)
            # indices = torch.multinomial(probs, 1).squeeze(dim=1) # BHW

        return indices

    def get_distance(self, z, code_type='all'):
        """
        z: L x D, the provided features

        return:
            d: L x N, where N is the number of tokens, the smaller distance is, the more similar it is
        """
        if self.distance_type == 'euclidean':
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embed_weight ** 2, dim=1) - 2 * \
                torch.matmul(z, self.embed_weight.t())
        elif self.distance_type == 'learned':
            d = 0 - self.distance_fc(z)  # BHW x N
        elif self.distance_type == 'sinkhorn':
            s = torch.einsum('ld,nd->ln', z, self.embed_weight)  # BHW x N
            d = 0 - distributed_sinkhorn(s.detach())
            # import pdb; pdb.set_trace()
        elif self.distance_type == 'cosine':
            d = 0 - torch.einsum('ld,nd->ln', z, self.embed_weight)  # BHW x N
        else:
            raise NotImplementedError('distance not implemented for {}'.format(self.distance_type))

        if code_type == 'masked':
            d = d[:, self.masked_embed_start:]
        elif code_type == 'unmasked':
            d = d[:, :self.masked_embed_start]

        return d
    def _quantize(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: L x D
            token_type: L, 1 denote unmasked token, other masked token
        """
        d = self.get_distance(z)

        # find closest encodings
        # import pdb; pdb.set_trace()
        if token_type is None or self.masked_embed_start == self.n_e:
            # min_encoding_indices = torch.argmin(d, dim=1) # L
            min_encoding_indices = self.get_index(d, topk=topk, step=step, total_steps=total_steps)
        else:
            min_encoding_indices = torch.zeros(z.shape[0]).long().to(z.device)
            idx = token_type == 1
            if idx.sum() > 0:
                d_ = d[idx][:, :self.masked_embed_start] # l x n
                # indices_ = torch.argmin(d_, dim=1)
                indices_ = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)
                min_encoding_indices[idx] = indices_
            idx = token_type != 1
            if idx.sum() > 0:
                d_ = d[idx][:, self.masked_embed_start:] # l x n
                # indices_ = torch.argmin(d_, dim=1)
                indices_ = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)
                indices_ += self.masked_embed_start
                min_encoding_indices[idx] = indices_

        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            # import pdb; pdb.set_trace()
            z_q = torch.matmul(min_encodings, self.embed_weight)#.view(z.shape)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(min_encoding_indices, self.embed_weight)#.view(z.shape)
        else:
            raise NotImplementedError

        return z_q, min_encoding_indices

    def forward(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: B x C x H x W
            token_type: B x 1 x H x W
        """
        if self.distance_type in ['sinkhorn', 'cosine']:
            # need to norm feat and weight embedding
            self.norm_embedding()
            z = F.normalize(z, dim=1, p=2)

        # reshape z -> (batch, height, width, channel) and flatten
        batch_size, height, width = z.shape
        # import pdb; pdb.set_trace()
        # z = z.permute(0, 2, 3, 1).contiguous()  # B x H x W x C
        z_flattened = z.view(-1, self.e_dim)  # BHW x C
        # if token_type is not None:
        #     token_type_flattened = token_type.view(-1)
        # else:
        token_type_flattened = None

        z_q, min_encoding_indices = self._quantize(z_flattened, token_type=token_type_flattened, topk=topk, step=step, total_steps=total_steps)
        # z_q = z_q.view(batch_size, height, width, -1)  # .permute(0, 2, 3, 1).contiguous()

        # if self.training and self.embed_ema:
        #     # import pdb; pdb.set_trace()
        #     assert self.distance_type in ['euclidean', 'cosine']
        #     indices_onehot = F.one_hot(min_encoding_indices, self.n_e).to(z_flattened.dtype)  # L x n_e
        #     indices_onehot_sum = indices_onehot.sum(0)  # n_e
        #     z_sum = (z_flattened.transpose(0, 1) @ indices_onehot).transpose(0, 1)  # n_e x D
        #
        #     # all_reduce(indices_onehot_sum)
        #     # all_reduce(z_sum)
        #
        #     self.cluster_size.data.mul_(self.decay).add_(indices_onehot_sum, alpha=1 - self.decay)
        #     self.embedding_avg.data.mul_(self.decay).add_(z_sum, alpha=1 - self.decay)
        #     n = self.cluster_size.sum()
        #     cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
        #     embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
        #     self.embedding.data.copy_(embed_normalized)

        if self.embed_ema:
            loss = (z_q.detach() - z).pow(2).mean()
        else:
            # compute loss for embedding
            loss = torch.mean((z_q.detach() - z).pow(2)) + self.beta * torch.mean((z_q - z.detach()).pow(2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        unique_idx = min_encoding_indices.unique()
        output = {
            'quantize': z_q,
            'used_unmasked_quantize_embed': torch.zeros_like(loss) + (unique_idx < self.masked_embed_start).sum(),
            'used_masked_quantize_embed': torch.zeros_like(loss) + (unique_idx >= self.masked_embed_start).sum(),
            'quantize_loss': loss,
            # 'index': min_encoding_indices.view(batch_size, height, width)
        }

        return output

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()

        # shape specifying (batch, height, width)
        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
            min_encodings.scatter_(1, indices[:, None], 1)
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings.float(), self.embed_weight)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(indices, self.embed_weight)
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        if shape is not None:
            z_q = z_q.view(*shape, -1)  # B x H x W x C

            if len(z_q.shape) == 4:
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none',
                 upsample_type='deconv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.upsample_type = upsample_type
        if self.upsample_type == 'deconv':
            self.deconv = nn.ConvTranspose1d(3, 3, kernel_size=1, stride=1, padding=0)
        else:
            assert self.upsample_type in ['bilinear', 'nearest'], 'upsample {} not implemented!'.format(
                self.upsample_type)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.upsample_type == 'deconv':
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_type)
            x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none', downsample_type='conv',
                 partial_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.downsample_type = downsample_type
        self.partial_conv = partial_conv
        if self.downsample_type == 'conv':
            if self.partial_conv:
                raise NotImplementedError
                self.conv = PartialConv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            assert self.downsample_type in ['bilinear', 'nearest', 'maxpool',
                                            'avgpool'], 'upsample {} not implemented!'.format(self.downsample_type)
            if self.partial_conv:
                raise NotImplementedError
                self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.downsample_type != 'conv':
            if self.downsample_type in ['nearest', 'bilinear']:
                x = F.interpolate(x, scale_factor=2.0, mode=self.downsample_type)
            elif self.downsample_type == 'maxpool':
                x = torch.max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
            elif self.downsample_type == 'avgpool':
                x = torch.avg_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
        if mask is not None:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
        if self.partial_conv:
            x = self.conv(x, mask_in=mask)
        else:
            x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x


# resblock only uses linear layer
class LinearResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(in_channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, in_channel),
        )
        self.out_channels = in_channel
        self.in_channels = in_channel

    def forward(self, x):
        out = self.layers(x)
        out = out + x

        return out


# resblock only uses conv layer
class ConvResBlock(nn.Module):
    def __init__(self, in_channel, channel, partial_conv=False):
        super().__init__()

        self.partial_conv = partial_conv
        if not partial_conv:
            self.partial_conv_args = None
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv1d(3, 3, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv1d(3, 3, 1),
            )
        else:
            raise NotImplementedError
            self.conv1 = PartialConv2d(in_channel, channel, kernel_size=3, padding=1)
            self.conv2 = PartialConv2d(channel, in_channel, kernel_size=3, padding=1)

        self.out_channels = in_channel
        self.in_channels = in_channel

    def forward(self, x, mask=None):
        if not self.partial_conv:
            out = self.conv(x)
        else:
            assert mask is not None, 'When use partial conv for inpainting, the mask should be provided!'
            mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
            out = F.relu(x)
            out = self.conv1(out, mask_in=mask)
            out = F.relu(out)
            out = self.conv2(out, mask_in=mask)
        out += x
        return out

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, stride=1, embed_dim=500):
        super().__init__()
        self.stride = stride
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(500)
        self.act = nn.GELU()
    def forward(self, x):
        a = x.unsqueeze(0)
        x = self.proj(a).squeeze(0)
        x = self.norm(x)
        return self.act(x)
class PatchEncoder2(nn.Module):
    def __init__(self, in_ch=3, res_ch=500, out_ch=500, num_res_block=2, res_block_bottleneck=2, num_post_layer=1, stride=8):
        super().__init__()
        in_dim = in_ch #* stride * stride
        self.stride = stride
        self.out_channels = out_ch
        self.patch_embed1 = PatchEmbed(in_chans=3, stride=1, embed_dim=3)
        self.pos_embed = get_sinusoid_encoding_table(3, 500)
        self.pre_layers = nn.Sequential(*[
            nn.Linear(res_ch, res_ch),
        ])

        res_layers = []
        for i in range(num_res_block):
            res_layers.append(LinearResBlock(res_ch, res_ch // res_block_bottleneck))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        if num_post_layer == 0:
            self.post_layers = nn.Identity()
        elif num_post_layer == 1:
            post_layers = [
                nn.ReLU(inplace=True),
                nn.Linear(res_ch, out_ch),
                nn.ReLU(inplace=True),
            ]
            self.post_layers = nn.Sequential(*post_layers)
        else:
            raise NotImplementedError('more post layers seems can not improve the performance!')

    def forward(self, x):
        """
        x: [B, 3, H, W]

        """
        x = self.patch_embed1(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()


        # in_size = [x.shape[-2], x.shape[-1]]
        # out_size = [s // self.stride for s in in_size]

        # x = torch.nn.functional.unfold(x, kernel_size=(self.stride, self.stride), stride=(self.stride, self.stride))  # B x 3*patch_size^2 x L
        x = torch.nn.functional.unfold(x, kernel_size=(1, 1),stride=(1, 1))#.squeeze()
        # x = x.permute(0, 2, 1).contiguous()  # B x L x 3*patch_size
        # 将展开后的张量重新调整为原始形状
        x = torch.reshape(x, (1, 3, 500))

        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = self.post_layers(x)

        # x = x.permute(0, 2, 1).contiguous()  # B x C x L
        # import pdb; pdb.set_trace()
        # x = torch.nn.functional.fold(x, output_size=512, kernel_size=(1, 1), stride=(1, 1))

        return x


class PatchConvEncoder2(nn.Module):
    def __init__(self, *,
                 in_ch=3,
                 res_ch=256,
                 out_ch,
                 num_res_block=2,
                 num_res_block_before_resolution_change=0,
                 res_block_bottleneck=2,
                 stride=8,
                 downsample_layer='downsample'):
        super().__init__()
        self.stride = stride
        self.out_channels = out_ch
        self.num_res_block_before_resolution_change = num_res_block_before_resolution_change

        # downsample with stride
        pre_layers = []
        in_ch_ = in_ch
        out_ch_ = 64
        while stride > 1:
            stride = stride // 2
            if stride == 1:
                out_ch_ = res_ch
            for i in range(self.num_res_block_before_resolution_change):
                pre_layers.append(
                    ConvResBlock(in_ch_, in_ch_ // res_block_bottleneck)
                )
            if downsample_layer == 'downsample':
                pre_layers.append(
                    DownSample(in_ch_, out_ch_, activate_before='none', activate_after='relu', downsample_type='conv'))
            elif downsample_layer == 'conv':
                pre_layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                if stride != 1:
                    pre_layers.append(nn.ReLU(inplace=True))
            else:
                raise RuntimeError('{} not impleted!'.format(downsample_layer))
            in_ch_ = out_ch_
            out_ch_ = 2 * in_ch_
        self.pre_layers = nn.Sequential(*pre_layers)

        res_layers = []
        for i in range(num_res_block):
            res_layers.append(ConvResBlock(3, 3))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        post_layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(res_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        self.post_layers = nn.Sequential(*post_layers)

    def forward(self, x):
        """
        x: [B, 3, H, W]

        """
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = self.post_layers(x)
        return x


class EncoderInPatchConvDecoder2(nn.Module):
    def __init__(self, in_ch, up_layers, with_res_block=True, res_block_bottleneck=2, downsample_layer='downsample',
                 partial_conv=False):
        super().__init__()

        out_channels = []
        for layer in up_layers:
            out_channels.append(layer.out_channels)

        layers = []
        in_ch_ = in_ch
        for l in range(len(out_channels), -1, -1):
            out_ch_ = out_channels[l - 1]
            # import pdb; pdb.set_trace()
            if l == len(out_channels):
                if partial_conv:
                    raise NotImplementedError
                    layers.append(PartialConv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
                else:
                    layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            else:
                if l == 0:
                    out_ch_ = up_layers[0].in_channels
                if isinstance(up_layers[l], UpSample):
                    if downsample_layer == 'downsample':  # recommneted
                        layers.append(DownSample(in_ch_, out_ch_, activate_before='relu', activate_after='none',
                                                 downsample_type='conv', partial_conv=partial_conv))
                    elif downsample_layer == 'conv':  # not recommented
                        if partial_conv:
                            raise NotImplementedError
                            layers.append(PartialConv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                        else:
                            layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                    else:
                        raise NotImplementedError
                elif isinstance(up_layers[l], ConvResBlock):
                    if with_res_block:
                        layers.append(ConvResBlock(in_ch_, in_ch_ // res_block_bottleneck, partial_conv=partial_conv))
                else:
                    raise NotImplementedError
            in_ch_ = out_ch_

        self.layers = nn.Sequential(*layers)
        self.downsample_layer = downsample_layer
        self.partial_conv = partial_conv

    def forward(self, x, mask=None):
        out = {}
        if self.partial_conv:
            assert mask is not None, 'When use partial conv for inpainting, the mask should be provided!'
            mask = mask.to(x)
        for l in range(len(self.layers)):  # layer in self.layers:
            layer = self.layers[l]
            if self.partial_conv:
                x = layer(x, mask)
            else:
                x = layer(x)
            if not isinstance(layer, (ConvResBlock,)):
                out[str(tuple(x.shape))] = x  # before activation, because other modules perform activativation first
            if self.downsample_layer == 'conv':
                x = F.relu(x)
        return out


class PatchConvDecoder2(nn.Module):
    def __init__(self, in_ch=500, res_ch=500,out_ch=3, num_res_block=1, res_block_bottleneck=2, num_res_block_after_resolution_change=0,stride=8, upsample_type='deconv', up_layer_with_image=True, smooth_mask_kernel_size=0,
                 # how to get the mask for merge different feature maps, only effective when up_layer_with_image is True
                 encoder_downsample_layer='conv', encoder_partial_conv=False, encoder_with_res_block=True, add_noise_to_image=False):
        super().__init__()
        self.in_channels = in_ch
        self.upsample_type = upsample_type
        self.up_layer_with_image = up_layer_with_image
        self.smooth_mask_kernel_size = smooth_mask_kernel_size
        self.requires_image = self.up_layer_with_image
        self.encoder_partial_conv = encoder_partial_conv
        self.add_noise_to_image = add_noise_to_image
        self.num_res_block_after_resolution_change = num_res_block_after_resolution_change

        if self.up_layer_with_image and self.smooth_mask_kernel_size > 1:
            self.mask_smooth_kernel = torch.ones((1, 1, self.smooth_mask_kernel_size, self.smooth_mask_kernel_size))
            self.mask_smooth_kernel = self.mask_smooth_kernel / self.mask_smooth_kernel.numel()

        self.pre_layers = nn.Sequential(*[
            torch.nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0),
        ])

        # res resblocks
        res_layers = []
        for i in range(num_res_block):
            res_layers.append(ConvResBlock(res_ch, res_ch // res_block_bottleneck))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        # upsampling in middle layers
        post_layer_in_ch = 64
        out_ch_ = post_layer_in_ch
        up_layers = []
        while stride > 1:
            stride = stride // 2
            in_ch_ = out_ch_ * 2
            if stride == 1:
                in_ch_ = res_ch
            layers_ = []
            layers_.append(UpSample(in_ch_, out_ch_, activate_before='relu', activate_after='none',
                                    upsample_type=self.upsample_type))
            for r in range(self.num_res_block_after_resolution_change):
                layers_.append(ConvResBlock(out_ch_, out_ch_ // res_block_bottleneck))
            up_layers = layers_ + up_layers
            out_ch_ *= 2
        # import pdb; pdb.set_trace()
        self.up_layers = nn.Sequential(*up_layers)

        post_layers = [
            nn.ReLU(inplace=True),
            nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0),
        ]
        self.post_layers = torch.nn.Sequential(*post_layers)

        if self.up_layer_with_image:
            self.encoder = EncoderInPatchConvDecoder2(
                in_ch=out_ch,
                up_layers=self.up_layers,
                downsample_layer=encoder_downsample_layer,
                with_res_block=encoder_with_res_block,
                partial_conv=encoder_partial_conv
            )

    def smooth_mask(self, mask, binary=True):
        """
        This function is used to expand the mask
        """
        shape = mask.shape[-2:]
        mask = F.conv2d(mask, self.mask_smooth_kernel.to(mask))
        mask = F.interpolate(mask, size=shape, mode='bilinear', align_corners=True)
        mask_ = (mask >= 0.8).to(mask)
        if binary:
            return mask_
        else:
            return mask_ * mask

    def forward(self, x, masked_image=None, mask=None):
        # pre layers
        x = self.pre_layers(x)
        x = self.res_layers(x)

        # if self.up_layer_with_image:
        #     mask = mask.to(x)
        #     if self.add_noise_to_image:
        #         masked_image = masked_image * mask + torch.randn_like(masked_image) * (1 - mask)
        #     im_x = self.encoder(masked_image, mask)
        #     for l in range(len(self.up_layers)):
        #         if isinstance(self.up_layers[l], UpSample):
        #             x_ = im_x[str(tuple(x.shape))]
        #             mask_ = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
        #             if self.smooth_mask_kernel_size > 1:
        #                 mask_ = self.smooth_mask(mask_, binary=False)
        #             x = x * (1 - mask_) + x_ * mask_
        #         x = self.up_layers[l](x)
        #     x = x * (1 - mask) + im_x[str(tuple(x.shape))] * mask
        #     x = self.post_layers(x)
        # else:
        x = self.up_layers(x)
        x = self.post_layers(x)
        return x


class FullAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self,n_embd, n_head, seq_len=None, attn_pdrop=0.1, resid_pdrop=0.1, causal=True,):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                 .view(1, 1, seq_len, seq_len))

    def forward(self, x, mask=None):
        """
        x: B x T x C
        mask: None or tensor B x T, bool type. For values with False, no attention should be attened
        """
        B, T, C = x.size()
        # import pdb; pdb.set_trace()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # print(q.shape, k.shape)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        if self.causal:
            # print(att.shape, self.mask.shape, T)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        if mask is not None:
            mask = mask.view(B, 1, 1, T)
            att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class ConvMLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop, spatial_size=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=mlp_hidden_times * n_embd, kernel_size=3, stride=1,
                               padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels=mlp_hidden_times * n_embd, out_channels=n_embd, kernel_size=3, stride=1,
                               padding=1)
        self.dropout = nn.Dropout(resid_pdrop)
        self.spatial_size = spatial_size

    def forward(self, x):
        """
        x: B x T x C
        """
        # import pdb; pdb.set_trace()
        if self.spatial_size is None:
            length = x.shape[1]
            h = int(math.sqrt(length))
            w = h
        else:
            h, w = self.spatial_size[0], self.spatial_size[1]
        x = x.view(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)  # B x C x H x W

        x = self.conv2(self.act(self.conv1(x)))
        x = x.permute(0, 2, 3, 1).view(x.shape[0], h * w, -1)  # B x L x C
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,n_embd,n_head, seq_len, attn_pdrop=0.1, resid_pdrop=0.1, causal=True, mlp_type='linear', mlp_hidden_times=4, activate='GELU'):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            causal=causal
        )
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'linear':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        elif mlp_type == 'conv':
            self.mlp = ConvMLP(
                n_embd=n_embd,
                mlp_hidden_times=mlp_hidden_times,
                act=act,
                resid_pdrop=resid_pdrop
            )

    def forward(self, x, mask=None):
        a, att = self.attn(self.ln1(x), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))

        return x#, att


class PatchVQGAN(nn.Module):
    def __init__(self,lossconfig=None,conv_before_quantize=True,ignore_keys=[], trainable=False, train_part='all', embed_dim=500, depth=2, num_heads=20,
                 ckpt_path=None, token_shape=None, resize_mask_type='pixel_shuffle', combine_rec_and_gt=False, im_process_info={'scale': 127.5, 'mean': 1.0, 'std': 1.0}):
        super().__init__()
        self.encoder = PatchEncoder2()#instantiate_from_config(encoder_config)  # Encoder(**encoder_config)
        self.decoder = PatchConvDecoder2()#instantiate_from_config(decoder_config)  # Decoder(**decoder_config)
        self.quantize = VectorQuantizer()#instantiate_from_config(quantizer_config)

        # import pdb; pdb.set_trace()
        if conv_before_quantize:
            self.quant_conv = torch.nn.Conv1d(3, 3, 1)
        else:
            assert self.encoder.out_channels == self.quantize.e_dim, "the channels for quantization shoule be the same"
            self.quant_conv = nn.Identity()
        self.post_quant_conv = torch.nn.Conv1d(3, 3, 1)
        self.im_process_info = im_process_info
        for k, v in self.im_process_info.items():
            v = torch.tensor(v).view(1, -1, 1, 1)
            if v.shape[1] != 3:
                v = v.repeat(1, 3, 1, 1)
            self.im_process_info[k] = v

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.emb_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, embed_dim, embed_dim))

        # drop for embedding
        # if embd_pdrop > 0:
        self.drop = nn.Dropout(0.5)
        # else:
        #     self.drop = None

        # transformer
        self.blocks = nn.Sequential(*[Block(
            n_embd=embed_dim,
            n_head=num_heads,
            seq_len=1200,
            attn_pdrop=0,
            resid_pdrop=0,
            causal=False,
            mlp_type='linear',
            mlp_hidden_times=4,
            activate='GELU',
        ) for n in range(self.depth)])

        # final prediction head
        # out_cls = self.content_codec.get_number_of_tokens() if num_token is None else num_token
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.to_logits = nn.Linear(embed_dim, embed_dim)



    def get_last_layer(self):
        return self.decoder.post_layers[-1].weight


    # def forward(self, batch, name='none', return_loss=True, step=0, total_steps=None, **kwargs):
    def forward(self, x, mask):

        # if name == 'generator':
        # input = self.pre_process(batch['image'])

        x = self.encoder(x)
        x = self.quant_conv(x)

        token_type_erase = torch.ones((x.shape[0], x.shape[1], x.shape[2])).long().to(device)
        quant_out = self.quantize(x, token_type=token_type_erase, step=0, total_steps=None)

        # recconstruction
        quant = self.post_quant_conv(quant_out['quantize'])

        rec = self.decoder(quant)

        x = self.emb_proj(rec)
        for block_idx in range(len(self.blocks)):
            x= self.blocks[block_idx](x)

        # 3) get logits
        x = self.layer_norm(x)
        logits = self.to_logits(x)  # B x HW x n
        return logits


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
        nn.Linear(1000, 500),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))

from mae_abl.our import PreModel
class fusion_model_PUT(nn.Module):
    def __init__(self, args, in_feats, n_hidden, out_classes, dropout=0.3, train_type_num=3):
        super(fusion_model_PUT, self).__init__()

        # self.img_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)  # args, 2, 1024
        # self.img_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        self.img_gnn_2 = PreModel(args, 2, 1024)
        self.img_relu_2 = GNN_relu_Block(out_classes)

        # self.rna_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.rna_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.rna_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        self.rna_gnn_2 = PreModel(args, 2, 1024)
        self.rna_relu_2 = GNN_relu_Block(out_classes)

        # self.cli_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.cli_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        # self.cli_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        self.cli_gnn_2 = PreModel(args, 2, 1024)
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

        self.mae = PatchVQGAN()

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
            loss_img, x_img = self.img_gnn_2(x_img, edge_index_img)
            x_img = self.img_relu_2(x_img)
            batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
            pool_x_img, att_img_2 = self.mpool_img(x_img, batch)
            att_2.append(att_img_2)
            pool_x = torch.cat((pool_x, pool_x_img), 0)
        if 'rna' in data_type:
            loss_rna, x_rna = self.rna_gnn_2(x_rna, edge_index_rna)
            x_rna = self.rna_relu_2(x_rna)
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_2 = self.mpool_rna(x_rna, batch)
            att_2.append(att_rna_2)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)
        if 'cli' in data_type:
            loss_cli, x_cli = self.cli_gnn_2(x_cli, edge_index_cli)
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


