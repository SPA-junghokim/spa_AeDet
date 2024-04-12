"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/layers/heads/bev_depth_head.py`"""
import torch
from mmdet3d.core.utils.gaussian import draw_heatmap_gaussian, gaussian_radius
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

def transpose(x):
    return x.transpose(-2, -1)
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ transpose(positive_key)
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

class conv3x3(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, gelu = False, layer2 = False):
        super(conv3x3, self).__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        if kernel_size == 3:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False),
            )
        elif kernel_size == 1:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False),
            )
        if gelu:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False),
                nn.GELU()
            )
        if layer2:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False),
            )
            

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        return x
    

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
        self.trunk = build_backbone(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = build_neck(bev_neck_conf)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # compute the azimuth
        pc_range = self.train_cfg['point_cloud_range']
        voxel_size = self.train_cfg['voxel_size']
        out_size_factor = self.train_cfg['out_size_factor']
        cart_x = torch.arange(pc_range[0] + voxel_size[0] * out_size_factor / 2.0, -pc_range[0],
                              voxel_size[0] * out_size_factor)
        cart_y = torch.arange(-pc_range[0] - voxel_size[1] * out_size_factor / 2.0, pc_range[0],
                              -voxel_size[1] * out_size_factor)
        cart_x = cart_x.view(1, len(cart_x)).repeat(len(cart_y), 1)
        cart_y = cart_y.view(len(cart_y), 1).repeat(1, len(cart_x))
        self.azimuth = torch.atan2(cart_x, cart_y)
        
        self.pc_range = self.train_cfg['point_cloud_range']
        self.voxel_size = self.train_cfg['voxel_size']
        self.out_size_factor = self.train_cfg['out_size_factor']
        self.heatmap_id = 0
        self.loss_consis = build_loss(loss_consis)
        if consistency_cfg['pool_loss'] == 'nega_cos':
            self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
           
        if consistency_cfg['bev_consistency'] or consistency_cfg['neck_consistency']:
            self.to_pers_1x1 = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
            
        if consistency_cfg['neck_consistency'] and consistency_cfg['use_ba'] == False:
            self.to_bev_1x1 = nn.Linear(in_channels, bev_backbone_conf['in_channels'])
        elif consistency_cfg['bev_consistency']:
            self.to_bev_1x1 = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
        else:
            self.to_bev_1x1 = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
            
        if consistency_cfg['use_pa']:
            self.conv3x3_pa = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
        if consistency_cfg['use_pa_1x1']:
            self.conv3x3_pa = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], 1)
        if consistency_cfg['use_pa_gelu']:
            self.conv3x3_pa = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], gelu=True)
        if consistency_cfg['use_pa_2layer']:
            self.conv3x3_pa = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], layer2=True)
            
        if consistency_cfg['use_ba']:
            if consistency_cfg['neck_consistency']:
                self.conv3x3_ba = conv3x3(in_channels, bev_backbone_conf['in_channels'])
            if consistency_cfg['bev_consistency']:
                self.conv3x3_ba = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
                
        if consistency_cfg['use_ba_gelu']:
            if consistency_cfg['neck_consistency']:
                self.conv3x3_ba = conv3x3(in_channels, bev_backbone_conf['in_channels'], gelu=True)
            if consistency_cfg['bev_consistency']:
                self.conv3x3_ba = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], gelu=True)
                
        if consistency_cfg['use_ba_2layer']:
            if consistency_cfg['neck_consistency']:
                self.conv3x3_ba = conv3x3(in_channels, bev_backbone_conf['in_channels'], layer2=True)
            if consistency_cfg['bev_consistency']:
                self.conv3x3_ba = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], layer2=True)
                
        if consistency_cfg['use_ba_1x1']:
            if consistency_cfg['neck_consistency']:
                self.conv3x3_ba = conv3x3(in_channels, bev_backbone_conf['in_channels'], 1)
            if consistency_cfg['bev_consistency']:
                self.conv3x3_ba = conv3x3(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], 1)
            
        if consistency_cfg['pool_pa']:
            self.pool_pa = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'])
        if consistency_cfg['pool_ba']:
            self.pool_ba = nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels'], 1)
        if consistency_cfg['pool_pa_mlp']:
            self.pool_pa_mlp = nn.Sequential(
                nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels']),
                nn.ReLU(),
                nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels']),
                )
        if consistency_cfg['pool_ba_mlp']:
            self.pool_ba_mlp = nn.Sequential(
                nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels']),
                nn.ReLU(),
                nn.Linear(bev_backbone_conf['in_channels'], bev_backbone_conf['in_channels']),
                )
            
            
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

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

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
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device='cuda')

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
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
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device='cuda')
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]

                    azimuth = self.azimuth[y, x].to(gt_bboxes_3d.device)
                    azim_rot = rot - azimuth
                    reg = center - torch.tensor([x + 0.5, y + 0.5], device='cuda')
                    regx, regy = reg.split([1, 1], 0)
                    azim_regx = regx * torch.cos(azimuth) + regy * torch.sin(azimuth)
                    azim_regy = -regx * torch.sin(azimuth) + regy * torch.cos(azimuth)
                    azim_reg = torch.cat([azim_regx, azim_regy], 0)
                    azim_vx = vx * torch.cos(azimuth) + vy * torch.sin(azimuth)
                    azim_vy = -vx * torch.sin(azimuth) + vy * torch.cos(azimuth)

                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        azim_reg,
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(azim_rot).unsqueeze(0),
                        torch.cos(azim_rot).unsqueeze(0),
                        azim_vx.unsqueeze(0),
                        azim_vy.unsqueeze(0),
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def consistency_loss(self, img_feats, bev_feats, inds, task_id, anno_boxes, masks, mats_dict, backbone_conf, consistency_cfg, feature_map_pooling_consis=None):
        batch, bev_channel, bev_height, bev_width = bev_feats.shape
        batch, cam_N, img_channel, img_height, img_width = img_feats.shape  # B, N, C, 1, H, W
        
        inds_task = inds[task_id] % (bev_height * bev_width)
        anno_boxes_task = anno_boxes[task_id]
        masks_task = masks[task_id].bool()
        batch, max_num = inds_task.shape
        
        # make ra to xyz
        ys_idx = (inds_task.float() / torch.tensor(bev_width, dtype=torch.float)).int().float()
        xs_idx = (inds_task % bev_width).int().float()
        
        ys_idx = ys_idx.view(batch, max_num, 1) + anno_boxes_task[:, :, 1:2] + 0.5
        xs_idx = xs_idx.view(batch, max_num, 1) + anno_boxes_task[:, :, 0:1] + 0.5
        ys = ys_idx.view(batch, max_num,1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        xs = xs_idx.view(batch, max_num,1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]

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
        ys_norm = (2.0 * ys_idx.reshape(batch, -1) / max(bev_height - 1, 1) - 1.0).unsqueeze(-1)
        xs_norm = (2.0 * xs_idx.reshape(batch, -1) / max(bev_width - 1, 1) - 1.0).unsqueeze(-1)
    
        if consistency_cfg['bev_consistency_type'] == "bilinear":
            img_feats_reshape = img_feats.reshape(-1, *img_feats.shape[2:])
            uv_cam = uv_cam_homo.permute(0,2,1,3).reshape(-1, max_num, 4)[..., :2].unsqueeze(2)
            uv_cam[..., 0] = (uv_cam[..., 0] / max(img_width-1,1)) * 2 - 1
            uv_cam[..., 1] = (uv_cam[..., 1] / max(img_height-1,1)) * 2 - 1
            if consistency_cfg['swap_pers_uv']:
                pass
            else:
                uv_cam[..., 0], uv_cam[..., 1] = uv_cam[..., 1], uv_cam[..., 0]
            
            if consistency_cfg['use_pa']:
                img_feats_reshape = self.conv3x3_pa(img_feats_reshape.to(torch.float32))
            
            pers_gt_feats = F.grid_sample(img_feats_reshape, uv_cam, mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
            pers_gt_feats = pers_gt_feats.reshape(batch, cam_N, img_channel, max_num).permute(0, 3, 1, 2) # batch, max_num, cam_N, img_channel
            pers_gt_feats = pers_gt_feats[masks_task][uvz_mask]
            
            if consistency_cfg['swap_bev_xy']:
                grid = torch.stack((ys_norm, xs_norm), 3)
            else:
                grid = torch.stack((xs_norm, ys_norm), 3)
            
            if consistency_cfg['use_ba']:
                bev_feats = self.conv3x3_ba(bev_feats.to(torch.float32))
                
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
            xs_ind = ys_idx.squeeze(2).long() 
            ys_ind = xs_idx.squeeze(2).long()
            x_mask = (xs_ind < 0) + (xs_ind >= bev_width)
            y_mask = (ys_ind < 0) + (ys_ind >= bev_height) 
            xs_ind[x_mask] = 0
            ys_ind[y_mask] = 0
            
            bev_gt_feats = bev_feats.reshape(-1, bev_channel, bev_height, bev_width)
            if consistency_cfg['swap_bev_xy']:
                bev_gt_feats = bev_gt_feats[torch.arange(batch).unsqueeze(1),:, xs_ind, ys_ind]
            else:
                bev_gt_feats = bev_gt_feats[torch.arange(batch).unsqueeze(1),:, ys_ind, xs_ind]
            bev_gt_feats = bev_gt_feats.unsqueeze(2).repeat(1,1,cam_N,1)
            bev_gt_feats = bev_gt_feats[masks_task][uvz_mask]
            
        consis_loss = 0
        if consistency_cfg['consis_bidirec']:
            pers_gt_feats_detach = pers_gt_feats.detach()
            bev_gt_feats_detach = bev_gt_feats.detach()
            if consistency_cfg['neck_consistency'] and consistency_cfg['use_ba'] == False:
                bev_gt_feats = self.to_bev_1x1(bev_gt_feats)
            pers_gt_feats_detach = self.to_pers_1x1(pers_gt_feats_detach)
            bev_gt_feats_detach = self.to_bev_1x1(bev_gt_feats_detach)
            consis_loss += self.loss_consis(pers_gt_feats_detach, bev_gt_feats)
            consis_loss += self.loss_consis(pers_gt_feats, bev_gt_feats_detach)
            
        elif feature_map_pooling_consis is not None:
            z_bound_pooling_consis = consistency_cfg['z_bound_pooling_consis']
            x_bound = consistency_cfg['x_bound']
            x_size = (x_bound[1] - x_bound[0]) / x_bound[2]
            device = feature_map_pooling_consis.device
            
            max_z_ind = torch.tensor((z_bound_pooling_consis[1] - z_bound_pooling_consis[0]) / z_bound_pooling_consis[2])
            # z_bound_pooling_consis[0] + z_bound_pooling_consis[2] / 2.0 
            # voxel_num_pooling_consis = torch.LongTensor((z_bound_pooling_consis[1] - z_bound_pooling_consis[0]) / z_bound_pooling_consis[2])
            xs_ind = ys_idx.squeeze(2).long()
            ys_ind = xs_idx.squeeze(2).long()
            x_mask = (xs_ind < 0) + (xs_ind >= bev_width)
            y_mask = (ys_ind < 0) + (ys_ind >= bev_height) 
            xs_ind[x_mask] = 0
            ys_ind[y_mask] = 0
            
            
            bev_gt_feats = bev_feats.reshape(-1, bev_channel, bev_height, bev_width)
            if consistency_cfg['swap_bev_xy']:
                bev_gt_feats = bev_gt_feats[torch.arange(batch).unsqueeze(1),:, xs_ind, ys_ind]
            else:
                bev_gt_feats = bev_gt_feats[torch.arange(batch).unsqueeze(1),:, ys_ind, xs_ind]
            bev_gt_feats = bev_gt_feats[masks_task]
            
            pers_pooled_feat_list = []
            pers_nega_pooled_feat_list = []
            if consistency_cfg['pool_loss'] == 'nega_cos':
                cur_cos_loss = 0
                cur_cos_loss_num = 1e-7
            for batch_idx in range(feature_map_pooling_consis.shape[0]):
                mask_id = 0
                for gt_idx in range(masks_task[batch_idx].sum()):
                    while masks_task[batch_idx, mask_id] != True:
                        mask_id += 1
                    cur_x = xs_ind[batch_idx, mask_id]
                    cur_y = ys_ind[batch_idx, mask_id]
                    cur_z = ((zs[batch_idx, mask_id, 0] - z_bound_pooling_consis[0]) / z_bound_pooling_consis[2]).int()
                    if consistency_cfg['pooling_consistency']:
                        pool_z_bin_size = 0
                    if consistency_cfg['pooling_consistency_v2']:
                        pool_z_bin_size = consistency_cfg['pooling_consistency_v2_z_bin_size']
                    z_idcs = (torch.arange(pool_z_bin_size*2+1).to(device) - (pool_z_bin_size)+cur_z)
                    low_condition = z_idcs >= 0
                    up_condition = z_idcs < max_z_ind
                    condition = low_condition * up_condition
                    z_idcs = z_idcs[condition]
                    
                    negative_z_idcs = torch.arange(max_z_ind).to(device)
                    mask = ~torch.any(torch.eq(negative_z_idcs[:, None], z_idcs), dim=1)
                    negative_z_idcs = negative_z_idcs[mask]
                    
                    z_idcs = z_idcs*x_size
                    negative_z_idcs = negative_z_idcs*x_size
                
                    cur_img_feat = feature_map_pooling_consis[batch_idx, :, cur_y, cur_x + (z_idcs).long()].sum(1)
                    negative_cur_img_feat = (feature_map_pooling_consis[batch_idx, :, cur_y, cur_x + (negative_z_idcs).long()]).sum(1)
                    #negative_cur_img_feat = (feature_map_pooling_consis[batch_idx, :, cur_y, cur_x + (negative_z_idcs).long()]) # chgd
                    pers_pooled_feat_list.append(cur_img_feat)
                    pers_nega_pooled_feat_list.append(negative_cur_img_feat)
                    mask_id += 1
                    
            if len(pers_pooled_feat_list)!=0:
                pers_pooled_feats = torch.stack(pers_pooled_feat_list, 0)
                pers_nega_pooled_feats = torch.stack(pers_nega_pooled_feat_list, 0)# [num, 80]
                #pers_nega_pooled_feats = torch.cat(pers_nega_pooled_feat_list, 1).permute(1,0) # chgd [1956, 80]
                
                if consistency_cfg['pool_pers_detach']:
                    pers_pooled_feats = pers_pooled_feats.detach()
                    pers_nega_pooled_feats = pers_nega_pooled_feats.detach()
                if consistency_cfg['pool_bev_detach']:
                    bev_gt_feats = bev_gt_feats.detach()
                
                if consistency_cfg['pool_pa']:
                    pers_pooled_feats = self.pool_pa(pers_pooled_feats)
                    pers_nega_pooled_feats = self.pool_pa(pers_nega_pooled_feats)
                if consistency_cfg['pool_ba']:
                    bev_gt_feats = self.pool_ba(bev_gt_feats)
                if consistency_cfg['pool_pa_mlp']:
                    pers_pooled_feats = self.pool_pa_mlp(pers_pooled_feats)
                if consistency_cfg['pool_ba_mlp']:
                    bev_gt_feats = self.pool_ba_mlp(bev_gt_feats)
                
                if consistency_cfg['pool_loss'] == 'infonce':
                    consis_loss += info_nce(bev_gt_feats, pers_pooled_feats, pers_nega_pooled_feats) * consistency_cfg['pool_loss_coef']
                if consistency_cfg['pool_loss'] == 'nega_cos':
                    cur_cos_loss += self.cos_loss(bev_gt_feats, pers_pooled_feats).sum() * consistency_cfg['pool_loss_coef']
                    cur_cos_loss_num += 1
                else:
                    consis_loss += self.loss_consis(pers_pooled_feats, bev_gt_feats)
                    
        else:
            if consistency_cfg['pers_detach']:
                pers_gt_feats = pers_gt_feats.detach()
            if consistency_cfg['bev_detach']:
                bev_gt_feats = bev_gt_feats.detach()
                
            if consistency_cfg['to_pers_1x1']:
                pers_gt_feats = self.to_pers_1x1(pers_gt_feats)
            if consistency_cfg['to_bev_1x1'] or (consistency_cfg['neck_consistency'] \
                and consistency_cfg['use_ba'] == False):
                bev_gt_feats = self.to_bev_1x1(bev_gt_feats)
                
            consis_loss += self.loss_consis(pers_gt_feats, bev_gt_feats)
        if consistency_cfg['pool_loss'] == 'nega_cos':
            consis_loss -= cur_cos_loss / cur_cos_loss_num
        return consis_loss


    def loss(self, targets, preds_dicts, img_feats, bev_feats, bev_neck, mats_dict, backbone_conf, consistency_cfg, feature_map_pooling_consis, **kwargs):
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
                
            if consistency_cfg['pooling_consistency'] or consistency_cfg['pooling_consistency_v2']:
                loss_consis_s += self.consistency_loss(img_feats, bev_feats, inds, task_id, anno_boxes, masks, mats_dict, backbone_conf, consistency_cfg, feature_map_pooling_consis)
            
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
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
            #     fig.savefig(f'visual_heatmap/grid_pred.png')
            #     numpy_image = heatmaps[task_id].cpu()[0].permute(1,2,0).numpy()
            #     fig = plt.figure()
            #     plt.imshow(numpy_image, cmap='viridis')  # Replace 'viridis' with your desired color map
            #     fig.savefig(f'visual_heatmap/grid_tar.png')
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
