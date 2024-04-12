"""
mAP: 0.3341
mATE: 0.6744
mASE: 0.2768
mAOE: 0.5534
mAVE: 0.8970
mAAE: 0.2594
NDS: 0.4009
Eval time: 95.7s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.504	0.540	0.161	0.142	1.149	0.228
truck	0.273	0.714	0.207	0.136	0.968	0.200
bus	0.407	0.644	0.192	0.098	1.686	0.359
trailer	0.216	0.939	0.244	0.422	0.661	0.169
construction_vehicle	0.077	0.917	0.515	1.160	0.116	0.391
pedestrian	0.276	0.736	0.296	1.276	0.814	0.605
motorcycle	0.340	0.673	0.247	0.624	1.348	0.103
bicycle	0.311	0.558	0.269	0.959	0.435	0.020
traffic_cone	0.439	0.528	0.359	nan	nan	nan
barrier	0.498	0.496	0.279	0.163	nan	nan
"""
from argparse import ArgumentParser, Namespace

import mmcv
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from callbacks.ema import EMACallback
from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn
from evaluators.det_mv_evaluators import DetMVNuscEvaluator
from models.aedet_polar import AeDet
from utils.torch_dist import all_gather_object, get_rank, synchronize

import math

H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)


backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'r_bound': [0, 51.2, 0.8],
    'a_bound': [-math.pi, math.pi, 2*math.pi/256],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'use_cdn': True,
    'virtual_depth_bins': 180,
    'min_focal_length': 800,
    'min_ida_scale': 0.386 * 0.9,
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}


bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
    conv_cfg=dict(type='PolarConv')
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64],
                conv_cfg=dict(type='PolarConv', bias=False))

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='PolarBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 2*math.pi/(256*4), 8],
    pc_range=[0, -math.pi, -5, 51.2, math.pi, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[0, -math.pi, -5, 51.2, math.pi, 3],
    grid_size=[64*4, 256*4, 1],
    voxel_size=[0.2, 2*math.pi/(256*4), 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 2*math.pi/(256*4), 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'loss_consis': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
    'separate_head': dict(type='SeparateHead', init_bias=-2.19, final_kernel=3, conv_cfg=dict(type='PolarConv')),
    # 'azi_gaus_thr':0, #chgd,
}


class AeDetLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root='data/nuScenes',
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 default_root_dir='./outputs/',
                 full=False,
                 overfit=False,
                 bevdepth=False,
                 key=1,
                 dim_div=False,
                 radius_center = True,
                 azimuth_center = True,
                 vel_div= True,
                 norm_bbox=True,
                 dcnv2=False,
                 bev_consistency=False,
                 bev_consistency_type='bilinear',
                 neck_consistency=False,
                 neck_consistency_type='bilinear',
                 to_pers_1x1 = False, 
                 to_bev_1x1 = False, 
                 pers_detach = False, 
                 bev_detach = False, 
                 consis_bidirec = False,
                 swap_bev_xy = False,
                 swap_pers_uv = False,
                 consis_loss_weight=0.25,
                 polar_conv_type='normal',
                 dilate_factor=4,
                 use_da = False,
                 use_pa = False,
                 use_ba = False,
                 use_pa_1x1 = False,
                 use_ba_1x1 = False,
                 use_pa_gelu = False,
                 use_ba_gelu = False,
                 use_pa_2layer = False,
                 use_ba_2layer = False,
                 depth_gt_training=False,
                 deform_type='normal',
                 root_dir = './outputs/default',
                 xy_bev_head_with_deform_attn = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        if overfit:
            self.basic_lr_per_img = 2e-4
        else:
            self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        
        self.backbone_conf['use_da'] = use_da
        self.backbone_conf['use_pa'] = use_pa
        self.backbone_conf['xy_bev_head_with_deform_attn'] = xy_bev_head_with_deform_attn
            
        self.key_idxes = [(i+1)*-1 for i in range(key-1)]
        
        
        if xy_bev_head_with_deform_attn:
            conv_cfg=dict(type='AeConv')
            bbox_coder = dict(
                type='AeDetBBoxCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                code_size=9,
            )
            train_cfg = dict(
            point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
            )
            test_cfg = dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            )
            self.head_conf['train_cfg'] = train_cfg
            self.head_conf['test_cfg'] = test_cfg
            self.head_conf['bbox_coder'] = bbox_coder
            
            self.backbone_conf['output_channels'] = 64
            self.head_conf['bev_backbone_conf']['base_channels'] = 64*2
            
            self.head_conf['bev_backbone_conf']['in_channels'] = 64 * (
                len(self.key_idxes) + 1)
            self.head_conf['bev_neck_conf']['in_channels'] = [
                64 * (len(self.key_idxes) + 1), 64*2, 64*4, 64*8
            ]
            self.consistency_cfg=dict(
            bev_consistency = bev_consistency,
            bev_consistency_type = bev_consistency_type,
            neck_consistency = neck_consistency,
            neck_consistency_type = neck_consistency_type,
            to_pers_1x1 = to_pers_1x1, 
            to_bev_1x1 = to_bev_1x1, 
            pers_detach = pers_detach, 
            bev_detach = bev_detach, 
            consis_bidirec = consis_bidirec,
            use_pa = use_pa,
            use_ba = use_ba,
            use_pa_1x1 = use_pa_1x1,
            use_ba_1x1 = use_ba_1x1,
            use_pa_gelu = use_pa_gelu,
            use_ba_gelu = use_ba_gelu,
            use_pa_2layer = use_pa_2layer,
            use_ba_2layer = use_ba_2layer,
            )
        else:
            self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
                len(self.key_idxes) + 1)
            self.head_conf['bev_neck_conf']['in_channels'] = [
                80 * (len(self.key_idxes) + 1), 160, 320, 640
            ]
            conv_cfg=dict(type='PolarConv', 
                        polar_conv_type=polar_conv_type, 
                        dilate_factor=dilate_factor, 
                        deform_type=deform_type,
                        dcnv2=dcnv2,
                        )
            self.head_conf['dim_div']=dim_div
            self.head_conf['radius_center']=radius_center
            self.head_conf['azimuth_center']=azimuth_center
            self.head_conf['vel_div']=vel_div
            self.head_conf['norm_bbox']=norm_bbox
            
            self.head_conf['bbox_coder']['dim_div']=dim_div
            self.head_conf['bbox_coder']['radius_center']=radius_center
            self.head_conf['bbox_coder']['azimuth_center']=azimuth_center
            self.head_conf['bbox_coder']['vel_div']=vel_div
        
            self.consistency_cfg=dict(
                bev_consistency = bev_consistency,
                bev_consistency_type = bev_consistency_type,
                neck_consistency = neck_consistency,
                neck_consistency_type = neck_consistency_type,
                to_pers_1x1 = to_pers_1x1, 
                to_bev_1x1 = to_bev_1x1, 
                pers_detach = pers_detach, 
                bev_detach = bev_detach, 
                consis_bidirec = consis_bidirec,
                swap_bev_xy = swap_bev_xy,
                swap_pers_uv = swap_pers_uv,
            )
        self.head_conf['separate_head']['conv_cfg'] = conv_cfg
        self.head_conf['bev_backbone_conf']['conv_cfg'] = conv_cfg
        self.head_conf['bev_neck_conf']['conv_cfg'] = conv_cfg
        self.head_conf['loss_consis']['loss_weight'] = consis_loss_weight
        
        
        
        
        if root_dir != './outputs/default':
            default_root_dir = root_dir
            
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetMVNuscEvaluator(class_names=self.class_names,
                                            output_dir=self.default_root_dir)
        self.model = AeDet(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True,
                              consistency_cfg = self.consistency_cfg,
                              xy_bev_head_with_deform_attn = xy_bev_head_with_deform_attn,
                              )
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        

    
        self.data_return_depth = True
        self.align_camera_center = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.full = full
        self.overfit = overfit

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, depth_preds, img_feats, bev_feats, bev_neck = self(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            loss_heatmap, loss_bbox, loss_consis = self.model.module.loss(targets, preds, img_feats, bev_feats, bev_neck, mats, self.backbone_conf, self.consistency_cfg)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            loss_heatmap, loss_bbox, loss_consis = self.model.loss(targets, preds, img_feats, bev_feats, bev_neck, mats, self.backbone_conf, self.consistency_cfg)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        detection_loss = loss_heatmap + loss_bbox + loss_consis
        self.log('loss_heatmap', loss_heatmap)
        self.log('loss_bbox', loss_bbox)
        self.log('loss_consis', loss_consis)
        self.log('depth_loss', depth_loss)
        return detection_loss + depth_loss

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        inds = torch.max(depth_labels, dim=1).indices # todo

        with autocast(enabled=False):
            depth_loss = -depth_preds[fg_mask, inds[fg_mask]].log().sum() / max(1.0, fg_mask.sum())

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)

        # map the bbox from camera center to ege center
        if self.align_camera_center:
            for i in range(len(results)):
                results[i][0].tensor[:, :2] += img_metas[i]['camera_center']

        # unify the velocity prediction for the frames with different time intervals
        if img_metas[0].get('interval_scale', None) is not None:
            for i in range(len(results)):
                if img_metas[i]['interval_scale'] > 10e-8:
                    results[i][0].tensor[:, -2:] /= img_metas[i]['interval_scale']

        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
             self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-1)
        return [optimizer]

    def train_dataloader(self):
        if self.full:
            info_path = 'data/nuScenes/nuscenes_infos_train.pkl'
        elif self.overfit:
            info_path = 'data/nuScenes/nuscenes_12hz_infos_train_100data.pkl'
        else:
            info_path = 'data/nuScenes/nuscenes_infos_train_7split.pkl'
        train_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=info_path,
            is_train=True,
            use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=self.data_return_depth,
            align_camera_center=self.align_camera_center
        )
        from functools import partial

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path='data/nuScenes/nuscenes_12hz_infos_val.pkl',
            is_train=False,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
            align_camera_center=self.align_camera_center
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = AeDetLightningModel(**vars(args))
    train_dataloader = model.train_dataloader()

    if args.ckpt_path:
        ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs, ema_ckpt_path=args.ckpt_path.replace('origin', 'ema'))
    else:
        ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parent_parser.add_argument('--full', default=True)
    parent_parser.add_argument('--overfit', action='store_true')
    parent_parser.add_argument('--bevdepth', action='store_true')
    parent_parser.add_argument('--key', default = 1, type=int)
    parent_parser.add_argument('--norm_bbox', action='store_true',)
    parent_parser.add_argument('--dim_div', default = "", type=str)
    parent_parser.add_argument('--radius_center', action='store_true',)
    parent_parser.add_argument('--azimuth_center', action='store_true',)
    parent_parser.add_argument('--vel_div', default = "", type=str)
    parent_parser.add_argument('--polar_conv_type', default = "normal", type=str)
    parent_parser.add_argument('--dilate_factor', default = 4, type = float)
    parent_parser.add_argument('--deform_type', default = "normal", type=str)
    parent_parser.add_argument('--root_dir', default = './outputs/default', type=str)
    parent_parser.add_argument('--dcnv2', action='store_true',)
    parent_parser.add_argument('--bev_consistency', action='store_true',)
    parent_parser.add_argument('--bev_consistency_type', default="bilinear", type=str)
    parent_parser.add_argument('--neck_consistency', action='store_true',)
    parent_parser.add_argument('--neck_consistency_type', default="bilinear", type=str)
    parent_parser.add_argument('--to_pers_1x1', action='store_true',)
    parent_parser.add_argument('--to_bev_1x1', action='store_true',)
    parent_parser.add_argument('--pers_detach', action='store_true',)
    parent_parser.add_argument('--bev_detach', action='store_true',)
    parent_parser.add_argument('--consis_bidirec', action='store_true',)
    parent_parser.add_argument('--consis_loss_weight', default=0.25 ,type=float)
    parent_parser.add_argument('--use_da', action='store_true',)
    parent_parser.add_argument('--use_pa', action='store_true',)
    parent_parser.add_argument('--xy_bev_head_with_deform_attn', action='store_true',)
        
        
    parser = AeDetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=24,
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=1.0,
                        check_val_every_n_epoch=25,
                        enable_checkpointing=False,
                        precision=16,
                        default_root_dir='./outputs/aedet_lss_r50_256x704_128x128_24e_polar',
                        )
    
    args = parser.parse_args()
    #breakpoint()
    main(args)


if __name__ == '__main__':
    run_cli()
