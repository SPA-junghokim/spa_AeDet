"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/layers/heads/bev_depth_head.py`"""
import torch
import numpy as np
import math
from mmdet3d.core.utils.gaussian_polar import draw_heatmap_gaussian, gaussian_radius_polar
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import reduce_mean
from mmdet.models import build_backbone
from mmdet3d.models import build_neck
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import build_loss

__all__ = ['AeDetHead']

bev_backbone_conf = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck_conf = dict(type='SECONDFPN',
                     in_channels=[160, 320, 640],
                     upsample_strides=[2, 4, 8],
                     out_channels=[64, 64, 128])


class AeDetHead(CenterHead):
    """Head for AeDet.

    Args:
        in_channels(int): Number of channels after bev_neck.
        tasks(dict): Tasks for head.
        bbox_coder(dict): Config of bbox coder.
        common_heads(dict): Config of head for each task.
        loss_cls(dict): Config of classification loss.
        loss_bbox(dict): Config of regression loss.
        gaussian_overlap(float): Gaussian overlap used for `get_targets`.
        min_radius(int): Min radius used for `get_targets`.
        train_cfg(dict): Config used in the training process.
        test_cfg(dict): Config used in the test process.
        bev_backbone_conf(dict): Cnfig of bev_backbone.
        bev_neck_conf(dict): Cnfig of bev_neck.
    """
    def __init__(
        self,
        consistency_cfg, 
        in_channels=256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_consis=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        bev_backbone_conf=bev_backbone_conf,
        bev_neck_conf=bev_neck_conf,
        separate_head=dict(type='SeparateHead',
                           init_bias=-2.19,
                           final_kernel=3),
        azi_gaus_thr=0,
        dim_div = True,
        norm_bbox = True,
        radius_center = True,
        azimuth_center = True,
        vel_div= True
    ):
        super(AeDetHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            conv_cfg=separate_head['conv_cfg']
        )
        self.azi_gaus_thr = azi_gaus_thr
        self.dim_div = dim_div
        self.trunk = build_backbone(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = build_neck(bev_neck_conf)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_bbox = norm_bbox
        self.pc_range = self.train_cfg['point_cloud_range']
        self.voxel_size = self.train_cfg['voxel_size']
        self.out_size_factor = self.train_cfg['out_size_factor']
        self.radius_center = radius_center
        self.azimuth_center = azimuth_center
        self.vel_div = vel_div
        self.heatmap_id = 0
        self.loss_consis = build_loss(loss_consis)
        
        if consistency_cfg['bev_consistency'] or consistency_cfg['neck_consistency']:
            self.to_pers_1x1 = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
            
        if consistency_cfg['neck_consistency']:
            self.to_bev_1x1 = nn.Linear(in_channels, bev_backbone_conf['in_channels'])
        elif consistency_cfg['bev_consistency']:
            self.to_bev_1x1 = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
            
    @autocast(False)
    def forward(self, x):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        # FPN
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        ret_values = super().forward(fpn_output)
        return ret_values, fpn_output

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor'] # 64, 256

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros((len(self.class_names[idx]), feature_map_size[1],feature_map_size[0]),device='cuda')
            # 256, 64

            anno_box = gt_bboxes_3d.new_zeros((max_objs, len(self.train_cfg['code_weights'])),
                                              dtype=torch.float32,
                                              device='cuda')

            ind = gt_labels_3d.new_zeros((max_objs),
                                         dtype=torch.int64,
                                         device='cuda')
            mask = gt_bboxes_3d.new_zeros((max_objs),
                                          dtype=torch.uint8,
                                          device='cuda')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[0] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2] # 13,12, 14.05, 0.51


                    radius = torch.sqrt(x**2 + y**2) # r (19.22)
                    azimuth = torch.atan2(y, x) # a (3.77)

                    gauss_radi = gaussian_radius_polar(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'], y=y)

                    gauss_radi = max(self.train_cfg['min_radius'], int(gauss_radi))

                    #voxel_size[0] = 0.2
                    #voxel_size[1] = math.pi/(256*4)

                    coor_r = (
                        radius - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_a = (
                        azimuth - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_r, coor_a],
                                          dtype=torch.float32,
                                          device='cuda')
                    center_int = center.to(torch.int32)
                    #pc_range : [0, 0, -5, 51.2, math.pi*2, 3],

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0] # 256
                            and 0 <= center_int[1] < feature_map_size[1]): # 64
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, gauss_radi=gauss_radi,\
                        pc_range = self.pc_range, feature_map_size=feature_map_size, radius=radius,\
                        azi_gaus_thr =self.azi_gaus_thr)

                    new_idx = k
                    r, a = center_int[0], center_int[1]

                    assert a * feature_map_size[0] + r < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = a * feature_map_size[0] + r
                    mask[new_idx] = 1

                    # TODO: support other outdoor dataset
                    if len(task_boxes[idx][k]) > 7:
                        vx, vy = task_boxes[idx][k][7:] # velocity
                    rot = task_boxes[idx][k][6] # yaw angle
                    box_dim = task_boxes[idx][k][3:6] # w,l,h
                    
                    radius_on_center = radius
                    if self.radius_center == True:
                        radius_on_center = radius // (voxel_size[0]*self.train_cfg['out_size_factor'])\
                            +  voxel_size[0]*self.train_cfg['out_size_factor'] / 2 
                            
                    if self.dim_div == 'div':
                        box_dim[0] /= radius_on_center
                        box_dim[1] /= radius_on_center
                    elif self.dim_div == 'mul':
                        box_dim[0] *= radius_on_center
                        box_dim[1] *= radius_on_center
                    else:
                        pass
                        
                    if self.norm_bbox: # True
                        box_dim = box_dim.log()
                    
                    vr = torch.cos(azimuth)*vx + torch.sin(azimuth)*vy
                    if self.vel_div == "div":
                        va = (-torch.sin(azimuth)*vx + torch.cos(azimuth)* vy) / radius
                    elif self.vel_div == "mul":
                        va = (-torch.sin(azimuth)*vx + torch.cos(azimuth)* vy) * radius
                    else:
                        va = (-torch.sin(azimuth)*vx + torch.cos(azimuth)* vy)
                    
                    azimuth_on_center = azimuth
                    #breakpoint()
                    if self.azimuth_center == True:
                        if azimuth >= 0: 
                            res_azi = voxel_size[1]*self.train_cfg['out_size_factor'] / 2 
                        else:
                            res_azi = -voxel_size[1]*self.train_cfg['out_size_factor'] / 2 
                        azimuth_on_center = azimuth // (voxel_size[1]*self.train_cfg['out_size_factor'])\
                            +  res_azi
                        
                    rot = rot - azimuth_on_center
                    
                    if len(task_boxes[idx][k]) > 7:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([r+0.5, a+0.5], device='cuda'),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vr.unsqueeze(0),
                            va.unsqueeze(0),
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([r+0.5, a+0.5], device='cuda'),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])
            # if idx == 0 :
            #     import matplotlib.pyplot as plt
            #     numpy_image = heatmap.cpu().permute(1,2,0).numpy()
            #     fig = plt.figure()
            #     plt.imshow(numpy_image, cmap='viridis')  # Replace 'viridis' with your desired color map
            #     import os
            #     os.makedirs('visual_heatmap', exist_ok=True)
            #     import numpy as np
            #     fig.savefig(f'visual_heatmap/tensor_image{str(self.heatmap_id)}.png')
            #     self.heatmap_id += 1
            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        
        return heatmaps, anno_boxes, inds, masks

    def consistency_loss(self, img_feats, bev_feats, inds, task_id, anno_boxes, masks, mats_dict, backbone_conf, consistency_cfg):
        batch, bev_channel, bev_height, bev_width = bev_feats.shape
        batch, cam_N, img_channel, img_height, img_width = img_feats.shape  # B, N, C, 1, H, W
        
        inds_task = inds[task_id] % (bev_height * bev_width)
        anno_boxes_task = anno_boxes[task_id]
        masks_task = masks[task_id].bool()
        batch, max_num = inds_task.shape
        
        # make ra to xyz
        azimuth_s_idx = (inds_task.float() / torch.tensor(bev_width, dtype=torch.float)).int().float()
        radius_s_idx = (inds_task % bev_width).int().float()
        
        azimuth_s_idx = azimuth_s_idx.view(batch, max_num, 1) + anno_boxes_task[:, :, 1:2] + 0.5
        radius_s_idx = radius_s_idx.view(batch, max_num, 1) + anno_boxes_task[:, :, 0:1] + 0.5
        azimuth_s = azimuth_s_idx.view(batch, max_num,1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        radius_s = radius_s_idx.view(batch, max_num,1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]

        xs = torch.cos(azimuth_s) * radius_s
        ys = torch.sin(azimuth_s) * radius_s
        zs = anno_boxes_task[:,:,2][:,:,None]
        xyz_homo = torch.cat((xs, ys, zs, torch.ones_like(xs)),-1).unsqueeze(3)
        
        # make  xyz to uv
        sensor2ego_mats_inv = (mats_dict['sensor2ego_mats'][:, 0, ...]).inverse().unsqueeze(1)
        intrin_mats = (mats_dict['intrin_mats'][:, 0, ...]).unsqueeze(1)
        ida_mats = (mats_dict['ida_mats'][:, 0, ...]).unsqueeze(1)
        bda_mats = mats_dict.get('bda_mat', None)
        
        if bda_mats is not None:
            bda_mats_inv = bda_mats.inverse().unsqueeze(1)
            xyz_homo = bda_mats_inv@xyz_homo
        xyz_homo_n_cam = (intrin_mats@sensor2ego_mats_inv@xyz_homo.unsqueeze(2).repeat(1,1,cam_N,1,1))
        xyz_homo_n_cam[..., :2, :] = xyz_homo_n_cam[..., :2, :] / xyz_homo_n_cam[..., 2:3, :]
        xyz_homo_n_cam = ida_mats@xyz_homo_n_cam
        uv_cam_homo = xyz_homo_n_cam[..., :2].squeeze(4)
        
        # just foreground
        u_mask = (uv_cam_homo[...,0] > 0) * (uv_cam_homo[...,0] < backbone_conf['final_dim'][1]) 
        v_mask = (uv_cam_homo[...,1] > 0) * (uv_cam_homo[...,1] < backbone_conf['final_dim'][0])
        z_mask = (uv_cam_homo[...,2] > 0)
        uvz_mask = (z_mask*u_mask*v_mask)[masks_task]
        
        # bilinear
        uv_cam_homo[..., 0] = uv_cam_homo[..., 0] / backbone_conf['downsample_factor']
        uv_cam_homo[..., 1] = uv_cam_homo[..., 1] / backbone_conf['downsample_factor']
        
        # for BEV
        azimuth_s_norm = (2.0 * azimuth_s_idx.reshape(batch, -1)/ max(bev_height - 1, 1) - 1.0).unsqueeze(-1)
        radius_s_norm = (2.0 * radius_s_idx.reshape(batch, -1) / max(bev_width - 1, 1) - 1.0).unsqueeze(-1)
        
        
        if consistency_cfg['bev_consistency_type'] == "bilinear":
            img_feats_reshape = img_feats.reshape(-1, *img_feats.shape[2:])
            uv_cam = uv_cam_homo.permute(0,2,1,3).reshape(-1, max_num, 4)[..., :2].unsqueeze(2)
            uv_cam[..., 0] = (uv_cam[..., 0] / max(img_height-1,1)) * 2 - 1
            uv_cam[..., 1] = (uv_cam[..., 1] / max(img_width-1,1)) * 2 - 1
            if consistency_cfg['swap_pers_uv']:
                pass
            else:
                uv_cam[..., 0], uv_cam[..., 1] = uv_cam[..., 1], uv_cam[..., 0] 
            
            pers_gt_feats = F.grid_sample(img_feats_reshape, uv_cam, mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
            pers_gt_feats = pers_gt_feats.reshape(batch, cam_N, img_channel, max_num).permute(0, 3, 1, 2) # batch, max_num, cam_N, img_channel
            pers_gt_feats = pers_gt_feats[masks_task][uvz_mask]
            
            if consistency_cfg['swap_bev_xy']:
                grid = torch.stack((azimuth_s_norm, radius_s_norm), 3)
            else:
                grid = torch.stack((radius_s_norm, azimuth_s_norm), 3)
            bev_gt_feats = F.grid_sample(bev_feats, grid)
            bev_gt_feats = bev_gt_feats.squeeze(-1) # [B, C, 500, 1]
            bev_gt_feats = bev_gt_feats.permute(0,2,1).unsqueeze(1).repeat(1,cam_N,1,1).permute(0,2,1,3)
            bev_gt_feats = bev_gt_feats[masks_task][uvz_mask]
            
        else: # not bilinear, just corresponding index grid
            # pers
            uv_cam = uv_cam_homo[..., :2]
            u_mask = (uv_cam[...,0] < 0) + (uv_cam[...,0] >= backbone_conf['final_dim'][1]/backbone_conf['downsample_factor']) 
            v_mask = (uv_cam[...,1] < 0) + (uv_cam[...,1] >= backbone_conf['final_dim'][0]/backbone_conf['downsample_factor'])
            uv_cam[..., 0][u_mask] = 0
            uv_cam[..., 1][v_mask] = 0
            uv_cam = uv_cam.permute(0,2,1,3).reshape(-1,max_num,2)
            x = uv_cam[:,:,0].long()
            y = uv_cam[:,:,1].long()
            if consistency_cfg['swap_pers_uv']:
                x, y = y, x
            else:
                pass
            img_feats_bc = img_feats.reshape(-1, img_channel, img_height, img_width)
            pers_gt_feats = (img_feats_bc[torch.arange(batch*cam_N).unsqueeze(1), :, y, x]).reshape(batch, cam_N, max_num, img_channel).permute(0,2,1,3)
            pers_gt_feats = pers_gt_feats[masks_task][uvz_mask]
            
            # bev
            radius_s_ind = azimuth_s_idx.squeeze(2).long()
            azimuth_s_ind = radius_s_idx.squeeze(2).long()
            radius_mask = (radius_s_ind < 0) + (radius_s_ind >= bev_width)
            azimuth_mask = (azimuth_s_ind < 0) + (azimuth_s_ind >= bev_height) 
            radius_s_ind[radius_mask] = 0
            azimuth_s_ind[azimuth_mask] = 0
            
            bev_gt_feats = bev_feats.reshape(-1, bev_channel, bev_height, bev_width)
            if consistency_cfg['swap_bev_xy']:
                bev_gt_feats = bev_gt_feats[torch.arange(batch).unsqueeze(1),:, radius_s_ind, azimuth_s_ind]
            else:
                bev_gt_feats = bev_gt_feats[torch.arange(batch).unsqueeze(1),:, azimuth_s_ind, radius_s_ind]
            bev_gt_feats = bev_gt_feats.unsqueeze(2).repeat(1,1,cam_N,1)
            bev_gt_feats = bev_gt_feats[masks_task][uvz_mask]
            
        consis_loss = 0
        if consistency_cfg['consis_bidirec']:
            pers_gt_feats_detach = pers_gt_feats.detach()
            bev_gt_feats_detach = bev_gt_feats.detach()
            if consistency_cfg['neck_consistency']:
                bev_gt_feats = self.to_bev_1x1(bev_gt_feats)
            pers_gt_feats_detach = self.to_pers_1x1(pers_gt_feats_detach)
            bev_gt_feats_detach = self.to_bev_1x1(bev_gt_feats_detach)
            consis_loss += self.loss_consis(pers_gt_feats_detach, bev_gt_feats)
            consis_loss += self.loss_consis(pers_gt_feats, bev_gt_feats_detach)
        else:
            if consistency_cfg['pers_detach']:
                pers_gt_feats = pers_gt_feats.detach()
            if consistency_cfg['bev_detach']:
                bev_gt_feats = bev_gt_feats.detach()
                
            if consistency_cfg['to_pers_1x1']:
                pers_gt_feats = self.to_pers_1x1(pers_gt_feats)
            if consistency_cfg['to_bev_1x1'] or consistency_cfg['neck_consistency']:
                bev_gt_feats = self.to_bev_1x1(bev_gt_feats)
            consis_loss += self.loss_consis(pers_gt_feats, bev_gt_feats)
            
        return consis_loss


    def loss(self, targets, preds_dicts, img_feats, bev_feats, bev_neck, mats_dict, backbone_conf, consistency_cfg, **kwargs):
        """Loss function for AeDetHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = targets
        loss_heatmap_s = 0
        loss_bbox_s = 0        
        loss_consis_s = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            if consistency_cfg['bev_consistency']:
                loss_consis_s += self.consistency_loss(img_feats, bev_feats, inds, task_id, anno_boxes, masks, mats_dict, backbone_conf, consistency_cfg)

            if consistency_cfg['neck_consistency']:
                loss_consis_s += self.consistency_loss(img_feats, bev_neck[0], inds, task_id, anno_boxes, masks, mats_dict, backbone_conf, consistency_cfg)

            
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap']) # [B, N, 256, 64]
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                                         min=1).item()
            
            # if task_id == 0 :
            #     import matplotlib.pyplot as plt
            #     import os
            #     import numpy as np
            #     numpy_image = preds_dict[0]['heatmap'].cpu()[0].detach().permute(1,2,0).numpy()
            #     fig = plt.figure()
            #     rand_id = np.random.randint(1000)
            #     plt.imshow(numpy_image, cmap='viridis')  # Replace 'viridis' with your desired color map
            #     os.makedirs('visual_heatmap', exist_ok=True)
            #     fig.savefig(f'visual_heatmap/polar_pred.png')
            #     numpy_image = heatmaps[task_id].cpu()[0].permute(1,2,0).numpy()
            #     fig = plt.figure()
            #     plt.imshow(numpy_image, cmap='viridis')  # Replace 'viridis' with your desired color map
            #     fig.savefig(f'visual_heatmap/polar_tar.png')
            # 
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         avg_factor=cls_avg_factor)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (
                    preds_dict[0]['reg'],
                    preds_dict[0]['height'],
                    preds_dict[0]['dim'],
                    preds_dict[0]['rot'],
                    preds_dict[0]['vel'],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred,
                                       target_box,
                                       bbox_weights,
                                       avg_factor=num)
            loss_heatmap_s += loss_heatmap
            loss_bbox_s += loss_bbox
            
        return loss_heatmap_s, loss_bbox_s, loss_consis_s
