# from image_synthesis.modeling.codecs.base_codec import BaseCodec
# from image_synthesis.modeling.modules.vqgan_loss.vqperceptual import VQLPIPSWithDiscriminator
from mae_abl.image_synthesis.modeling.utils.misc import distributed_sinkhorn, get_token_type

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv, LayerNorm
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp
from omegaconf import OmegaConf
import scipy.stats as stats
import pytorch_lightning as pl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5,
                 remap=None, unknown_index="random"):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)  # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update:
            # EMA cluster size
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            # EMA embedding average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=3, num_channels=3, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(512)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(out_channels,
                                                     out_channels,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0,
                                                     bias=False)
            else:
                self.nin_shortcut = torch.nn.Conv1d(out_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bias=False)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(h)
            else:
                x = self.nin_shortcut(h)

        return x+h


class Encoder(nn.Module):
    def __init__(self, ch=3, out_ch=512, ch_mult=(1,2,4,8), num_res_blocks=2, dropout=0.0, resamp_with_conv=False, in_channels=3, attn_resolutions=None, resolution=None, z_channels=3):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv1d(3, 3,kernel_size=1, stride=1, padding=0,bias=False)

        # curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            # block_in = ch*in_ch_mult[i_level]
            # block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=3,
                                         out_channels=3,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                # block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(512, resamp_with_conv)
                # curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=3,
                                       out_channels=3,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=3,
                                       out_channels=3,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(3)
        self.conv_out = torch.nn.Conv1d(3,
                                        3,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, maks):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        x = x.unsqueeze(0)
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            # if i_level != self.num_resolutions-1:
            #     hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, ch=3, out_ch=512, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=None, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=None, z_channels=3, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        # curr_res = resolution // 2**(self.num_resolutions-1)
        # self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                # curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQModel(pl.LightningModule):
    def __init__(self, n_embed=500,embed_dim=500,ckpt_path=None,ignore_keys=[],image_key="image",colorize_nlabels=None,
                 monitor=None,remap=None,sane_index_shape=False):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25,remap=remap, sane_index_shape=sane_index_shape)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        print("Strict load")
        self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x, mask):
        h = self.encoder(x, mask)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
        self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=500, patch_size=10, in_chans=3,
                 embed_dim=500, depth=2, num_heads=10,
                 decoder_embed_dim=500, decoder_depth=2, decoder_num_heads=10,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # VQGAN specifics
        # config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel()
        for param in self.vqgan.parameters():
            param.requires_grad = False

        # self.codebook_size = config.params.n_embed
        # vocab_size = self.codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        # self.fake_class_label = self.codebook_size + 1100 - 1024
        # self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=500,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=300+1,
                                        dropout=0.1)

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAGE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=500)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # self.initialize_weights()

    # def initialize_weights(self):
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    #
    #     decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
    #     self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    #
    #     # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #     w = self.patch_embed.proj.weight.data
    #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #
    #     # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #     torch.nn.init.normal_(self.cls_token, std=.02)
    #     torch.nn.init.normal_(self.mask_token, std=.02)
    #     torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
    #
    #     # initialize nn.Linear and nn.LayerNorm
    #     self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(x, mask)

        # _, _, token_indices = token_tuple
        # token_indices = token_indices.reshape(z_q.size(0), -1)
        # gt_indices = token_indices.clone().detach().long()

        # masking
        # bsz, seq_len = token_indices.size()
        # mask_ratio_min = self.mask_ratio_min
        # mask_rate = self.mask_ratio_generator.rvs(1)[0]
        #
        # num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        # num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        # while True:
        #     noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
        #     sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
        #     cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
        #     cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
        #     token_drop_mask = (noise <= cutoff_drop).float()
        #     token_all_mask = (noise <= cutoff_mask).float()
        #     if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
        #         break
        #     else:
        #         print("Rerandom the noise!")
        # print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        # token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        # print("Masekd num token:", torch.sum(token_indices == self.mask_token_label, dim=1))

        # concate class token
        # token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        # token_indices[:, 0] = self.fake_class_label
        # token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        # token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        # token_indices = token_indices.long()
        # # bert embedding
        # input_embeddings = self.token_emb(token_indices)
        # # print("Input embedding shape:", input_embeddings.shape)
        # bsz, seq_len, emb_dim = input_embeddings.shape
        #
        # # dropping
        # token_keep_mask = 1 - token_drop_mask
        # input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)
        #
        # # apply Transformer blocks
        # x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(z_q)
        x = self.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x#, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # if self.pad_with_cls_token:
        #     mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        # else:
        #     mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        # x_after_pad = mask_tokens.clone()
        # x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # # set undropped but masked positions with mask
        # x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        # x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        # print("Logits shape:", x.shape)

        return x

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits[:, 1:, :self.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask)
        logits = self.forward_decoder(latent)
        # loss = self.forward_loss(gt_indices, logits, token_all_mask)
        return logits









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
        nn.Linear(1024, 512),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))


from mae_abl.our import PreModel
class fusion_model_mage(nn.Module):
    def __init__(self, args,  in_feats, n_hidden, out_classes, dropout=0.3, train_type_num=3):
        super(fusion_model_mage, self).__init__()

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

        self.mae = MaskedGenerativeEncoderViT()

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


