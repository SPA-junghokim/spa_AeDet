# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from ops.voxel_pooling import voxel_pooling
import math
__all__ = ['LSSFPN']


def convbnsig(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels), nn.Sigmoid())

def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels))

def makePad(gks):
    pad = []
    for i in range(gks):
        for j in range(gks):
            top = i
            bottom = gks - 1 - i
            left = j
            right = gks - 1 - j
            pad.append(torch.nn.ZeroPad2d((left, right, top, bottom)))
    return pad

class DySPN(nn.Module):
    def __init__(self, in_channels, kernel_size=7, iter_times=6):
        super(DySPN, self).__init__()
        assert kernel_size == 7, 'now only support 7'
        self.kernel_size = kernel_size
        self.iter_times = iter_times
        self.affinity7 = convbn(in_channels, 7**2 - 5**2, kernel_size=3, stride=1, padding=1)
        self.affinity5 = convbn(in_channels, 5**2 - 3**2, kernel_size=3, stride=1, padding=1)
        self.affinity3 = convbn(in_channels, 3**2 - 1**2, kernel_size=3, stride=1, padding=1)

        self.attention7 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention5 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention3 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention1 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)
        self.attention0 = convbnsig(in_channels, self.iter_times, kernel_size=3, stride=1, padding=1)

    def forward(self, feature, d0, d00):
        affinity7 = self.affinity7(feature)
        affinity5 = self.affinity5(feature)
        affinity3 = self.affinity3(feature)

        attention7 = self.attention7(feature)
        attention5 = self.attention5(feature)
        attention3 = self.attention3(feature)
        attention1 = self.attention1(feature)

        zero_pad = makePad(self.kernel_size)

        weightmask_7 = [
            1, 1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1
        ]
        weightmask_5 = [
            0, 0, 0, 0, 0, 0, 0, 
            0, 1, 1, 1, 1, 1, 0, 
            0, 1, 0, 0, 0, 1, 0, 
            0, 1, 0, 0, 0, 1, 0, 
            0, 1, 0, 0, 0, 1, 0, 
            0, 1, 1, 1, 1, 1, 0, 
            0, 0, 0, 0, 0, 0, 0
        ]
        weightmask_3 = [
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 1, 1, 1, 0, 0, 
            0, 0, 1, 0, 1, 0, 0, 
            0, 0, 1, 1, 1, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0
        ]

        dt = d0
        for iters in range(self.iter_times):
            # normalization
            guide7 = attention7[:, iters:iters + 1, ...] * affinity7
            guide5 = attention5[:, iters:iters + 1, ...] * affinity5
            guide3 = attention3[:, iters:iters + 1, ...] * affinity3
            guide1 = attention1[:, iters:iters + 1, ...]
            
            guide7abs = attention7[:, iters:iters + 1, ...] * affinity7.abs()
            guide5abs = attention5[:, iters:iters + 1, ...] * affinity5.abs()
            guide3abs = attention3[:, iters:iters + 1, ...] * affinity3.abs()
            guide1abs = attention1[:, iters:iters + 1, ...]

            guide_abssum = torch.sum(guide7abs, dim=1).unsqueeze(1)
            guide_abssum += torch.sum(guide5abs, dim=1).unsqueeze(1)
            guide_abssum += torch.sum(guide3abs, dim=1).unsqueeze(1)
            guide_abssum += torch.sum(guide1abs, dim=1).unsqueeze(1)

            guide_sum = torch.sum(guide7, dim=1).unsqueeze(1)
            guide_sum += torch.sum(guide5, dim=1).unsqueeze(1)
            guide_sum += torch.sum(guide3, dim=1).unsqueeze(1)
            guide_sum += torch.sum(guide1, dim=1).unsqueeze(1)
            
            guide7 = torch.div(guide7, guide_abssum)
            guide5 = torch.div(guide5, guide_abssum)
            guide3 = torch.div(guide3, guide_abssum)
            guide1 = torch.div(guide1, guide_abssum)

            guide0 = 1 - guide_sum / guide_abssum

            # guidance
            weight_pad = []
            guide7_idx = guide5_idx = guide3_idx = 0
            for t in range(self.kernel_size * self.kernel_size):
                if weightmask_7[t]:
                    weight_pad.append(zero_pad[t](guide7[:, guide7_idx:guide7_idx + 1, :, :]))
                    guide7_idx += 1
                elif weightmask_5[t]:
                    weight_pad.append(zero_pad[t](guide5[:, guide5_idx:guide5_idx + 1, :, :]))
                    guide5_idx += 1
                elif weightmask_3[t]:
                    weight_pad.append(zero_pad[t](guide3[:, guide3_idx:guide3_idx + 1, :, :]))
                    guide3_idx += 1
                else:
                    weight_pad.append(zero_pad[t](guide1[:, 0:1, :, :]))
            weight_pad.append(zero_pad[self.kernel_size**2 // 2](guide0))

            guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size * self.kernel_size + 1)], dim=1)

            # refine
            depth_pad = []
            for t in range(self.kernel_size * self.kernel_size):
                depth_pad.append(zero_pad[t](dt))
            depth_pad.append(zero_pad[self.kernel_size**2 // 2](d00))

            depth_all = torch.cat([depth_pad[t] for t in range(self.kernel_size * self.kernel_size + 1)], dim=1)
            refined_result = torch.sum((guide_weight.mul(depth_all)), dim=1)
            refined_output = refined_result[:, (self.kernel_size - 1) // 2:-(self.kernel_size - 1) // 2,
                              (self.kernel_size - 1) // 2:-(self.kernel_size - 1) // 2].unsqueeze(dim=1)
        return refined_output
    

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, camera_channels=10):
        super(DepthNet, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.context_conv = nn.Sequential(
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128
            )),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )

        self.bn = nn.BatchNorm1d(camera_channels)
        self.context_mlp = Mlp(camera_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        ida = mats_dict['ida_mats'][:, 0:1, ...]

        mlp_input = torch.stack(
            [
                intrins[:, 0:1, ..., 0, 0],
                intrins[:, 0:1, ..., 1, 1],
                intrins[:, 0:1, ..., 0, 2],
                intrins[:, 0:1, ..., 1, 2],
                ida[:, 0:1, ..., 0, 0],
                ida[:, 0:1, ..., 0, 1],
                ida[:, 0:1, ..., 0, 3],
                ida[:, 0:1, ..., 1, 0],
                ida[:, 0:1, ..., 1, 1],
                ida[:, 0:1, ..., 1, 3]
            ],
            dim=-1,
        )

        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)

        # depth
        depth = self.depth_conv(x)

        # context
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        return torch.cat([depth, context], dim=1)


class LSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, depth_net_conf, use_cdn=False,
                 virtual_depth_bins=None, min_focal_length=None, min_ida_scale=None,
                 depth_thresh = False,
                 **kwargs):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(LSSFPN, self).__init__()

        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.depth_thresh = depth_thresh
        self.depth_thr = 1/((self.d_bound[1]-self.d_bound[0])/self.d_bound[2]) # chgd
        
        # Affinity Refinement
        # self.dyspn = DySPN(in_channels=112)
        # self.coarse_output = nn.Sequential(
        #     nn.Conv2d(in_channels=260, out_channels=260, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(260),
        #     nn.Conv2d(in_channels=260, out_channels=1, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1),
        # )
        
        # position-aware frustum
        self.to_q = nn.Linear(80, 80)
        self.to_k = nn.Linear(512, 80)
        self.to_v = nn.Linear(512, 80)
        
        self.use_cdn = use_cdn
        if min_ida_scale is None:
            self.min_focal_length = min_focal_length
        else:
            self.min_focal_length = min_focal_length * min_ida_scale

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())

        self.depth_channels = self.frustum.shape[0] if not self.use_cdn else virtual_depth_bins

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        d_coords = d_coords + self.d_bound[2] / 2.
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_image_scale(self, intrin_mat, ida_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        img_mat = ida_mat.matmul(intrin_mat)
        fx = (img_mat[:, :, 0, 0] ** 2 + img_mat[:, :, 0, 1] ** 2).sqrt()
        fy = (img_mat[:, :, 1, 0] ** 2 + img_mat[:, :, 1, 1] ** 2).sqrt()
        image_scales = ((fx ** 2 + fy ** 2) / 2.).sqrt()
        return image_scales

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def depth_sampling(self, depth_feature, indices):
        b, c, h, w = depth_feature.shape
        indices = indices[:, :, None, None].repeat(1, 1, h, w)
        indices_floor = indices.floor()
        indices_ceil = indices_floor + 1

        max_index = indices_ceil.max().long()
        if max_index >= c:
            depth_feature = torch.cat([depth_feature, depth_feature.new_zeros(b, max_index - c + 1, h, w)], 1)

        sampled_depth_feature = (indices_ceil - indices) * torch.gather(depth_feature, 1, indices_floor.long()) + \
                                (indices - indices_floor) * torch.gather(depth_feature, 1, indices_ceil.long())
        return sampled_depth_feature

    
    def prepare_pooling(self, geom_xyz, img_feat_with_depth, kept, batch_size):
        kept = kept.reshape(batch_size, -1)
        num_points = kept.shape[1]
        geom_xyz = geom_xyz.reshape(batch_size, -1, geom_xyz.shape[-1])
        img_feat_with_depth = img_feat_with_depth.reshape(batch_size, -1, img_feat_with_depth.shape[-1])
        batch_points = kept.sum(1).tolist()
        max_points = max(batch_points)
        
        geom_xyz = torch.split(geom_xyz[kept], batch_points)
        img_feat_with_depth = torch.split(img_feat_with_depth[kept], batch_points)
        
        geom_xyz_b, img_feat_with_depth_b = [], []
        for b in range(batch_size):
            res_points = max_points - batch_points[b]
            if res_points==0:
                geom_xyz_b.append(geom_xyz[b])
                img_feat_with_depth_b.append(img_feat_with_depth[b])
            else:
                geom_xyz_b.append(torch.cat([
                    geom_xyz[b], -torch.ones(res_points, geom_xyz[b].shape[-1], device=geom_xyz[b].device)
                ]))
                img_feat_with_depth_b.append(torch.cat([
                    img_feat_with_depth[b], torch.zeros(res_points, img_feat_with_depth[b].shape[-1], device=img_feat_with_depth[b].device)
                ]))
        geom_xyz_b = torch.stack(geom_xyz_b).int()
        img_feat_with_depth_b = torch.stack(img_feat_with_depth_b)

        return geom_xyz_b, img_feat_with_depth_b
    
    
    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_depth):
        return img_feat_with_depth

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...] # [B, N, 512, 16, 44]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict
        ) # [BN, 260, 16, 44]
        
        depth = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype) # [BN, 180, 16, 44]
        
        # chgd for spn
        #coarse_output = self.coarse_output(depth_feature)
        #depth = self.dyspn(depth_feature, coarse_output, depth)        
        
        if self.depth_thresh:
            kept = (depth >= self.depth_thr).view(batch_size, 
                                                num_cams, 
                                                depth.shape[1], 
                                                depth.shape[2], 
                                                depth.shape[3])

        if self.use_cdn:
            visual_depth_feature = depth_feature[:, :self.depth_channels]

            # depth mapping
            image_scales = self.get_image_scale(
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, sweep_index, ...]
            )
            offset_per_meter = self.depth_channels / self.d_bound[1] * self.min_focal_length / image_scales.view(-1, 1)
            offset = self.frustum[:, 0, 0, 2].view(1, -1) * offset_per_meter
            # print(f'offset min: {(offset[:, 1:] - offset[:, :-1]).min()}, max: {(offset[:, 1:] - offset[:, :-1]).max()}')  # todo
            real_depth_feature = self.depth_sampling(visual_depth_feature, offset)
            depth = real_depth_feature.softmax(1)
            
        img_feat_with_depth = depth.unsqueeze(
            1) * depth_feature[:, self.depth_channels:(
                self.depth_channels + self.output_channels)].unsqueeze(2) # [BN, C, D, H, W]
            
        # position-aware frustum
        query = self.to_q(img_feat_with_depth.permute(0, 3, 4, 2, 1))
        key = self.to_k(source_features.reshape(-1, source_features.shape[2],source_features.shape[3]\
                ,source_features.shape[4]).unsqueeze(2).permute(0, 3, 4, 2, 1))
        value = self.to_v(source_features.reshape(-1, source_features.shape[2],source_features.shape[3]\
                ,source_features.shape[4]).unsqueeze(2).permute(0, 3, 4, 2, 1))
        
        dot = query * key 
        att = dot.softmax(dim=-2) # along depth dim
        frustum_feature = att * value 
        img_feat_with_depth = frustum_feature.permute(0,4,3,1,2)
        
        # if self.depth_thresh:
        #     depth_thresh = 1/((self.d_bound[1]-self.d_bound[0])/self.d_bound[2])
        #     img_feat_with_depth = img_feat_with_depth.permute(0,2,3,4,1)
        #     img_feat_with_depth[depth<1/depth_thresh] = 0
        #     img_feat_with_depth = img_feat_with_depth.permute(0,4,1,2,3)
            
        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )
        
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        
        # geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).long()
        geom_xyz[geom_xyz < 0] = -1
        geom_xyz = geom_xyz.int()
        
        if self.depth_thresh:
            geom_xyz, img_feat_with_depth = self.prepare_pooling(geom_xyz, img_feat_with_depth, kept, batch_size)
        
        feature_map = voxel_pooling(geom_xyz, img_feat_with_depth.contiguous(),
                                    self.voxel_num.cuda())

        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False)
                ret_feature_list.insert(0, feature_map) # order chgd

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)
