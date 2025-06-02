#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import random
import time
import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from numpy import set_printoptions
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
from torch.autograd import Variable
from collections import OrderedDict 
import torch.nn.functional as F
import sys
sys.argv = ['ipykernel_launcher.py']
import argparse
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


# In[5]:


@register_model
def van_b0(num,pretrained=False,**kwargs):
    model = VAN(num_classes=num,
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b0", kwargs)
    return model


@register_model
def van_b1(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
   
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b1", kwargs)
    return model

@register_model
def van_b2(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b2", kwargs)
    return model

@register_model
def van_b3(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b3", kwargs)
    return model

@register_model
def van_b4(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
        **kwargs)
   
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b4", kwargs)
    return model


@register_model
def van_b5(pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[96, 192, 480, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 24, 3],
        **kwargs)
   
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b5", kwargs)
    return model


@register_model
def van_b6(num,pretrained=False, **kwargs):
    model = VAN(
        embed_dims=[96, 192, 384, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[6,6,90,6],
        **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "van_b6", kwargs)
    return model


# In[6]:


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=1001, patch_size=7, stride=4, in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2))
        self.norm = nn.BatchNorm1d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] ** 2 * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, HW = x.shape
        x = self.norm(x)        
        return x, HW


# In[7]:


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] ** 2 * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x        


# In[8]:


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


# In[9]:


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv1d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


# In[10]:


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
                                                           
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] ** 2 * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


# In[11]:


class VAN(nn.Module):
    def __init__(self, img_size=1001,in_chans=1, num_classes=8, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()

        
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        
        self.w1 = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )
        self.w2 = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )        
        self.rc1 = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )
        self.rc2 = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )
        self.g = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )
        self.g1 = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )        
        self.g2 = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )         
        self.phase = nn.Sequential(
            nn.Linear(embed_dims[3], 1),
         )
        self.relu=nn.ReLU()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  
        cur = 0
##  model = VAN(
 #      embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
 #      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
 #       **kwargs)
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=1001,
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] ** 2 * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x,HW = patch_embed(x)
        
            for blk in block:
                x = blk(x)
            x = x.transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, HW, -1).permute(0, 2, 1).contiguous()
        
        return x

    def forward(self, x):
        x = self.forward_features(x).permute(0, 2, 1).contiguous()
        x = self.avg_pool(x).squeeze(-1)
        #feature=x.clone()
        w1 = self.w1(x)
        w2 = self.w2(x)
        rc1 = self.rc1(x)
        rc2 = self.rc2(x)
        g = self.g(x)
        g1=self.g1(x)
        g2=self.g2(x)
        phase = self.phase(x)
        x=torch.cat((w1,w2,rc1,rc2,g,g1,g2, phase), dim=1)
        x=self.relu(x)
        return x

#



