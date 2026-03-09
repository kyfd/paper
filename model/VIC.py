import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .gvt import pcvit_base, PosCNN
from model.points_from_den import *
from misc.layer import Gaussianlayer
from model.VGG.ResNet50_FPN import resnet50
from model.ViT.models_crossvit import Mlp, FeatureFusionModule
from model.VGG.VGG16_FPN import VGG16_FPN_Encoder
from model.ResNet.ResNet50_FPN import ResNet_50_FPN_Encoder
from model.decoder import ShareDecoder, InOutDecoder, GlobalDecoder

import cv2
import misc.transforms as own_transforms
import torchvision.transforms as standard_transforms


# 空间距离掩码函数
def get_spatial_mask(H, W, radius, device):
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([y_grid, x_grid], dim=-1).float().reshape(-1, 2).to(device)
    dist = torch.cdist(coords, coords)
    mask = torch.where(dist > radius, float('-inf'), 0.0)
    return mask.unsqueeze(0).unsqueeze(0)


# 局部约束跨帧注意力
class LocalityAwareCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., search_radius=3.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.search_radius = search_radius

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_kv):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 加入局部掩码
        spatial_mask = get_spatial_mask(H, W, self.search_radius, x.device)
        attn = attn + spatial_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 包装 Block
class LocalityBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, search_radius=3.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LocalityAwareCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, search_radius=search_radius)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_kv):
        # x 是 query (feature 1), x_kv 是 key/value (feature 2)
        x = x + self.attn(self.norm1(x), self.norm1(x_kv))
        x = x + self.mlp(self.norm2(x))
        return x


class Video_Counter(nn.Module):
    def __init__(self, cfg, cfg_data):
        super(Video_Counter, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg_data
        if cfg.encoder == 'VGG16_FPN':
            self.Extractor = VGG16_FPN_Encoder()
        elif cfg.encoder == 'PCPVT':
            self.Extractor = pcvit_base()
        elif cfg.encoder == 'ResNet_50_FPN':
            self.Extractor = ResNet_50_FPN_Encoder()
        else:
            raise Exception("The backbone is out of setting")

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.share_cross_attention = nn.ModuleList([nn.ModuleList([
            LocalityBlock(cfg.cross_attn_embed_dim, cfg.cross_attn_num_heads, cfg.mlp_ratio,
                          qkv_bias=True, qk_scale=None, norm_layer=norm_layer, search_radius=3.0)
            for _ in range(cfg.cross_attn_depth)])
            for _ in range(3)])

        self.share_cross_attention_norm = norm_layer(cfg.cross_attn_embed_dim)

        self.feature_fuse = FeatureFusionModule(self.cfg.FEATURE_DIM)
        self.global_decoder = GlobalDecoder()
        self.share_decoder = ShareDecoder()
        self.in_out_decoder = InOutDecoder()
        self.criterion = torch.nn.MSELoss()
        self.Gaussian = Gaussianlayer()

    def forward(self, img, target):
        features = self.Extractor(img)
        B, C, H, W = features[-1].shape
        self.feature_scale = H / img.shape[2]
        pre_global_den = self.global_decoder(features[-1])
        all_loss = {}

        # 初始化 GT Maps
        gt_in_out_dot_map = torch.zeros_like(pre_global_den)
        gt_share_dot_map = torch.zeros_like(pre_global_den)

        img_pair_num = img.size(0) // 2
        assert img.size(0) % 2 == 0
        share_features = None

        # Cross Attention 过程
        for l_num in range(len(self.share_cross_attention)):
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])

            for pair_idx in range(img_pair_num):
                # 准备 Query 和 Key/Value
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                kv1 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                # Forward Pass 1 (Frame t -> Frame t+1)
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv1)

                q1 = self.share_cross_attention_norm(q1)

                # 准备反向的 Query 和 Key/Value
                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                kv2 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                # Forward Pass 2 (Frame t+1 -> Frame t)
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv2)

                q2 = self.share_cross_attention_norm(q2)

                share_results.append(q1)
                share_results.append(q2)

            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        #  生成 GT 和 计算 Loss
        img_h, img_w = img.shape[2], img.shape[3]

        for pair_idx in range(img_pair_num):
            points0 = target[pair_idx * 2]['points']
            points1 = target[pair_idx * 2 + 1]['points']

            share_mask0 = target[pair_idx * 2]['share_mask0']
            outflow_mask = target[pair_idx * 2]['outflow_mask']
            share_mask1 = target[pair_idx * 2 + 1]['share_mask1']
            inflow_mask = target[pair_idx * 2 + 1]['inflow_mask']

            share_coords0 = points0[share_mask0].long()
            share_coords1 = points1[share_mask1].long()

            if len(share_coords0) > 0:
                share_coords0[:, 0] = share_coords0[:, 0].clamp(0, img_w - 1)
                share_coords0[:, 1] = share_coords0[:, 1].clamp(0, img_h - 1)
                gt_share_dot_map[pair_idx * 2, 0, share_coords0[:, 1], share_coords0[:, 0]] = 1

            if len(share_coords1) > 0:
                share_coords1[:, 0] = share_coords1[:, 0].clamp(0, img_w - 1)
                share_coords1[:, 1] = share_coords1[:, 1].clamp(0, img_h - 1)
                gt_share_dot_map[pair_idx * 2 + 1, 0, share_coords1[:, 1], share_coords1[:, 0]] = 1

            outflow_coords = points0[outflow_mask].long()
            inflow_coords = points1[inflow_mask].long()

            if len(outflow_coords) > 0:
                outflow_coords[:, 0] = outflow_coords[:, 0].clamp(0, img_w - 1)
                outflow_coords[:, 1] = outflow_coords[:, 1].clamp(0, img_h - 1)
                gt_in_out_dot_map[pair_idx * 2, 0, outflow_coords[:, 1], outflow_coords[:, 0]] = 1

            if len(inflow_coords) > 0:
                inflow_coords[:, 0] = inflow_coords[:, 0].clamp(0, img_w - 1)
                inflow_coords[:, 1] = inflow_coords[:, 1].clamp(0, img_h - 1)
                gt_in_out_dot_map[pair_idx * 2 + 1, 0, inflow_coords[:, 1], inflow_coords[:, 0]] = 1

        pre_share_den = self.share_decoder(share_features)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den)

        gt_global_dot_map = torch.zeros_like(pre_global_den)
        for i, data in enumerate(target):
            points = target[i]['points']
            p_x = points[:, 0].long().clamp(min=0, max=img_w - 1)
            p_y = points[:, 1].long().clamp(min=0, max=img_h - 1)
            gt_global_dot_map[i, 0, p_y, p_x] = 1

        gt_global_den = self.Gaussian(gt_global_dot_map)

        assert pre_global_den.size() == gt_global_den.size()
        global_mse_loss = self.criterion(pre_global_den, gt_global_den * self.cfg_data.DEN_FACTOR)
        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['global'] = global_mse_loss

        gt_share_den = self.Gaussian(gt_share_dot_map)
        assert pre_share_den.size() == gt_share_den.size()
        share_mse_loss = self.criterion(pre_share_den, gt_share_den * self.cfg_data.DEN_FACTOR)
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['share'] = share_mse_loss * 10

        gt_in_out_den = self.Gaussian(gt_in_out_dot_map)
        assert pre_in_out_den.size() == gt_in_out_den.size()
        in_out_mse_loss = self.criterion(pre_in_out_den, gt_in_out_den * self.cfg_data.DEN_FACTOR)
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['in_out'] = in_out_mse_loss

        return pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss

    def test_forward(self, img):
        features = self.Extractor(img)
        B, C, H, W = features[-1].shape
        pre_global_den = self.global_decoder(features[-1])
        img_pair_num = img.size(0) // 2
        assert img.size(0) % 2 == 0
        share_features = None
        for l_num in range(len(self.share_cross_attention)):
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])

            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv1 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv1)
                q1 = self.share_cross_attention_norm(q1)

                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv2 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv2)
                q2 = self.share_cross_attention_norm(q2)

                share_results.append(q1)
                share_results.append(q2)

            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        pre_share_den = self.share_decoder(share_features)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den)

        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR

        return pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss