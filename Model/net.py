import sys

sys.path.append("")
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.net_utils import DropPath, trunc_normal_
from einops import rearrange
from torch.nn import init
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1, qkv_bias=False):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.fc_k = nn.Linear(d_model, d_k, bias=qkv_bias)
        self.fc_v = nn.Linear(d_model, d_v, bias=qkv_bias)
        self.fc_o = nn.Linear(d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k // self.h).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k/h)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k // self.h).permute(0, 2, 3, 1)  # (b_s, h, d_k/h, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v // self.h).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v/h)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


# multi-head self attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


# patch embedding
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


# foldsformer encoder block
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.temporal_fc = nn.Linear(dim, dim)

        # # Fusion temproal Attention
        self.fusion_norm = norm_layer(dim)
        self.fusion_attn = ScaledDotProductAttention(dim, dim, dim, num_heads, dropout=attn_drop)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = x.size(1) // T
        H = num_spatial_tokens // W
        demoT = T - 1

        ## spatial
        xs = rearrange(x, "b (h w t) m -> (b t) (h w) m", b=B, h=H, w=W, t=T)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))
        res_spatial = rearrange(res_spatial, "(b t) (h w) m -> b (h w t) m", b=B, h=H, w=W, t=T)
        xs = x + res_spatial

        ## temporal
        xt = rearrange(xs, "b (h w t) m -> (b h w) t m", b=B, h=H, w=W, t=T)
        xstate = xt[:, :1, :]
        xfusion = xt[:, 1:, :]

        # inside fusion temproal attention
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xfusion)))
        res_temporal = rearrange(res_temporal, "(b h w) t m -> b (h w t) m", b=B, h=H, w=W, t=demoT)
        res_temporal = self.temporal_fc(res_temporal)
        xfusion = xs[:, num_spatial_tokens:, :] + res_temporal

        # state and fusion
        xfusion = rearrange(xfusion, "b (h w t) m -> (b h w) t m", b=B, h=H, w=W, t=demoT)
        res_state = self.drop_path(self.fusion_norm(self.fusion_attn(xstate, xfusion, xfusion)))
        res_state = rearrange(res_state, "(b h w) t m -> b (h w t) m", b=B, h=H, w=W, t=1)
        xstate = xs[:, :num_spatial_tokens, :] + res_state

        # cat
        xfusion = rearrange(xfusion, "(b h w) t m -> b (h w t) m", b=B, h=H, w=W, t=demoT)
        x = torch.cat((xstate, xfusion), dim=1)

        ## Mlp
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# Foldsformer Encoder
class Encoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        num_frames=8,
    ):
        super().__init__()
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        ## Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Time Embeddings
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if "Block" in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "time_embed"}

    def get_output(self):
        return self.head

    def _reshape_output(self, x, input_dim):
        x = x.view(
            x.size(0),
            int(self.img_size[0] // self.patch_size[0]),
            int(self.img_size[1] // self.patch_size[1]),
            input_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)

        ## Positional Embeddings
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            pos_embed = pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(pos_embed, size=(H, W), mode="nearest")
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode="nearest")
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        x = self.norm(x)

        # extract the first frame(current state)'s feature
        x_state = x[:, : self.num_patches, :]
        # extract the goal frame(goal state)'s feature
        x_goal = x[:, -self.num_patches :, :]
        # feedback signal
        y = torch.cat((x_state, x_goal), dim=2)

        return y

    def forward(self, x):
        x = self.forward_features(x)
        x = self._reshape_output(x, x.shape[-1])
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.decoder_net = self._init_decoder()

    def _init_decoder(self):
        intermediate_channels1 = int(self.input_dim / 2)
        intermediate_channels2 = int(self.input_dim / 4)
        in_channels = [
            self.input_dim,
            intermediate_channels1,
            intermediate_channels1,
            intermediate_channels2,
            intermediate_channels2,
        ]
        out_channels = [
            intermediate_channels1,
            intermediate_channels1,
            intermediate_channels2,
            intermediate_channels2,
            1,
        ]
        modules = []
        for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            modules.append(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=(0, 0))
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder_net(x)


class Foldsformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(
            img_size=(cfg.img_size, cfg.img_size),
            patch_size=(cfg.patch_size, cfg.patch_size),
            in_chans=1,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=nn.LayerNorm,
            num_frames=cfg.num_frames,
        )
        self.decoder_pick = Decoder(cfg.embed_dim * 2)
        self.decoder_place = Decoder(cfg.embed_dim * 2)

    def forward(self, x):
        x = self.encoder(x)
        x_pick = self.decoder_pick(x)
        x_place = self.decoder_place(x)
        return x_pick, x_place
