# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init, build_norm_layer
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
import numpy as np
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# @ATTENTION.register_module()
class BEV_DEFORM_ATTN(BaseModule):
    def __init__(self,
                 q_in_embed_dims=80,
                 v_in_embed_dims=80,
                 embed_dims=80,
                 num_heads=8,
                 num_points=8,
                 num_levels=1,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 W = 128,
                 H = 128,
                 
                 ):

        super().__init__(init_cfg)
        
        ffn_cfgs=dict(
            type='FFN',
            embed_dims=embed_dims,
            feedforward_channels=embed_dims*4,
            num_fcs=2,
            ffn_drop=0.,
            act_cfg=dict(type='ReLU', inplace=True),
        )
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        dim_per_head = embed_dims // num_heads
        norm_cfg={'type': 'LN'}
        
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False

        self.im2col_step = im2col_step
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.embed_dims =embed_dims
        self.query3x3 = nn.Conv2d(q_in_embed_dims, embed_dims, kernel_size=3, stride=1, padding=1)
        self.sampling_offsets = nn.Conv2d(embed_dims, num_heads * num_points * 2, kernel_size=1, stride=1, padding=0)
        self.attention_weights = nn.Conv2d(embed_dims, num_heads * num_points, kernel_size=1, stride=1, padding=0)
        self.value_proj3x3 = nn.Conv2d(v_in_embed_dims, embed_dims, kernel_size=3, stride=1, padding=1)
        self.value_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        # self.output_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0)
        self.init_weights()
        
        self.reference_points = self.get_reference_points(int(H), int(W))
        # self.norm1 = nn.LayerNorm(self.embed_dims)
        # ffn = []
        # ffn.append(nn.Linear(embed_dims, embed_dims*4))
        # ffn.append(nn.ReLU())
        # ffn.append(nn.Linear(embed_dims*4, embed_dims))
        # self.ffn = nn.Sequential(*ffn)
        # self.norm2 = nn.LayerNorm(self.embed_dims)
        self.norm1 = build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.ffn = build_feedforward_network(ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, self.embed_dims)[1]
        
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1,
            2).repeat(1, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
        
    def get_reference_points(self, H, W,  device='cuda', dtype=torch.float):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(1, 1, 1).unsqueeze(2) # [1, 128*128 , 1, 2]
        return ref_2d

    def point_sampling(self, reference_points, pc_range,  img_metas):
        # make bev coord to polar coord
        return True
    
    
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                mask = None, 
                **kwargs):
            
        bs, embed_dims, xy_bev_h, xy_bev_w = query.shape
        bs, embed_dims, polar_bev_h, polar_bev_w = value.shape
        
        num_query = xy_bev_h * xy_bev_w
        num_value = polar_bev_h * polar_bev_w
        
        query = self.query3x3(query)
        
        if identity is None:
            identity = query.view(bs, -1, num_query).permute(0,2,1).contiguous()
        if query_pos is not None:
            query = query + query_pos
        
        
        device = query.device
        dtype = query.dtype
        
        value_proj = self.value_proj3x3(value)
        value_proj = self.value_proj(value_proj)
        value_proj = value_proj.reshape(bs, embed_dims, num_value)
        value_proj = value_proj.permute(0,2,1).reshape(bs, num_value, self.num_heads, -1).contiguous()
        
        sampling_offsets = self.sampling_offsets(query).view(bs, -1, num_query).permute(0,2,1).contiguous()
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2).contiguous()
        attention_weights = self.attention_weights(query).view(bs, -1, num_query).permute(0,2,1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        attention_weights = attention_weights.softmax(-1)
        
        # Apply the mask to attention weights
        if mask is not None:
            attention_weights = attention_weights * mask.reshape(bs, -1)[:, :, None, None, None]
            # 1, 16804, 8, 1, 8
        # if you want, you can implement multi-scale
        spatial_shapes = torch.as_tensor([[xy_bev_w, xy_bev_h]], dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        reference_points = self.reference_points.to(device).to(value_proj.dtype).repeat(bs,1,1,1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        sampling_locations -= 0.5
        # change sampling xy locations to ra locations
        # spatial_shapes = torch.as_tensor([[polar_bev_h, polar_bev_w]], dtype=torch.long, device=device)
        # xs, ys = sampling_locations[..., 0], sampling_locations[..., 1]
        # radius = torch.sqrt(ys**2 + xs**2)
        # azimuth = (torch.atan2(ys, xs)+math.pi)/(math.pi*2)
        # sampling_locations[..., 0] = radius
        # sampling_locations[..., 1] = azimuth
        if torch.cuda.is_available() and value.is_cuda:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(value_proj, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(value_proj, spatial_shapes, sampling_locations, attention_weights)\
        
        output = self.output_proj(output)        
        output = self.dropout(output) + identity
        output = self.norm1(output)
        output = self.ffn(output, output)
        output = self.norm2(output)
        output = output.permute(0, 2, 1).view(bs, self.embed_dims, xy_bev_h, xy_bev_w)
        
        return output
