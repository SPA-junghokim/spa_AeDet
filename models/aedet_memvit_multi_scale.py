"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
from torch import nn
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.aedet_head_ms_no_heatmap import AeDetHead
from .convLSTM_V2_mm import ConvLSTMV2
from mmdet.models import build_backbone
from mmcv.runner import BaseModule
from mmdet3d.models import build_neck
from layers.modules.dense_sparse_deform_attn import BEV_DEFORM_ATTN
from monai.losses.dice import DiceLoss
from layers.modules.memvit import MultiScaleBlock as MEMVIT

__all__ = ['AeDet']


class FRPN(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        mask_thre = 0.4,
    ):
        super(FRPN, self).__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=3, padding=1, stride=1),
            )
        self.upsample = nn.Upsample(scale_factor = scale_factor , mode ='bilinear',align_corners = True)
        #self.dice_loss = build_loss(dict(type='CustomDiceLoss', use_sigmoid=True, loss_weight=1.))
        self.dice_loss = DiceLoss(sigmoid=True)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))  # From lss
        self.mask_thre = mask_thre

    def forward(self, input):
        bev_mask = self.mask_net(input)            
        bev_mask = self.upsample(bev_mask)
        return bev_mask
    
    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return [mask_ce_loss, mask_dice_loss]


class AeDet(nn.Module):
    """Source code of `AeDet`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(AeDet, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = AeDetHead(**head_conf)
        self.is_train_depth = is_train_depth
        
        head_conf['bev_backbone_conf']['in_channels'] = 80
        head_conf['bev_neck_conf']['in_channels'] = [80, 160, 320, 640]
        self.trunk = build_backbone(head_conf['bev_backbone_conf'])
        self.trunk.init_weights()
        self.neck = build_neck(head_conf['bev_neck_conf'])
        self.neck.init_weights()
        del self.trunk.maxpool
        
        input_dim = backbone_conf['output_channels']
        self.channels = backbone_conf['output_channels']
        
        x_bound = backbone_conf['x_bound']
        y_bound = backbone_conf['y_bound']
        self.height = int((x_bound[1]-x_bound[0])/x_bound[2])
        self.width = int((y_bound[1]-y_bound[0])/y_bound[2])
        
        self.key_num = backbone_conf['num_keys']
        self.agg_type = backbone_conf['agg_type']
        
        
        if self.agg_type == 'cnn':
            self.agg_conv_scale1 = nn.ModuleList()
            self.agg_conv_scale2 = nn.ModuleList()
            self.agg_conv_scale3 = nn.ModuleList()
            self.agg_conv_scale4 = nn.ModuleList()
            
            self.agg_conv_scale1.append(nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, padding=1),)
            self.agg_conv_scale1.append(nn.ReLU(),)
            self.agg_conv_scale1.append(nn.Conv2d(self.channels*2, self.channels, kernel_size=3, padding=1))
            
            self.agg_conv_scale2.append(nn.Conv2d(self.channels*2 *2, self.channels*2 *2, kernel_size=3, padding=1),)
            self.agg_conv_scale2.append(nn.ReLU(),)
            self.agg_conv_scale2.append(nn.Conv2d(self.channels*2 *2, self.channels *2, kernel_size=3, padding=1))
            
            self.agg_conv_scale3.append(nn.Conv2d(self.channels*2 *4, self.channels*2 *4, kernel_size=3, padding=1),)
            self.agg_conv_scale3.append(nn.ReLU(),)
            self.agg_conv_scale3.append(nn.Conv2d(self.channels*2 *4, self.channels *4, kernel_size=3, padding=1))
            
            self.agg_conv_scale4.append(nn.Conv2d(self.channels*2 *8, self.channels*2 *8, kernel_size=3, padding=1),)
            self.agg_conv_scale4.append(nn.ReLU(),)
            self.agg_conv_scale4.append(nn.Conv2d(self.channels*2 *8, self.channels *8, kernel_size=3, padding=1))
        else:
            self.agg_conv_scale1 = nn.Conv2d(self.channels*self.key_num, self.channels, kernel_size=3, padding=1)
            self.agg_conv_scale2 = nn.Conv2d(self.channels*self.key_num *2, self.channels *2, kernel_size=3, padding=1)
            self.agg_conv_scale3 = nn.Conv2d(self.channels*self.key_num *4, self.channels *4, kernel_size=3, padding=1)
            self.agg_conv_scale4 = nn.Conv2d(self.channels*self.key_num *8, self.channels *8, kernel_size=3, padding=1)
            
        
        # self.heatmap_scale1 = FRPN(input_dim)
        # self.heatmap_scale1 = FRPN(input_dim)
        # self.heatmap_scale1 = FRPN(input_dim)
        # self.heatmap_scale1 = FRPN(input_dim)
        
        # self.deform_attn_scale1 = BEV_DEFORM_ATTN(H=self.height, W=self.width)
        # self.deform_attn_scale2 = BEV_DEFORM_ATTN(H=self.height, W=self.width)
        # self.deform_attn_scale3 = BEV_DEFORM_ATTN(H=self.height, W=self.width)
        # self.deform_attn_scale4 = BEV_DEFORM_ATTN(H=self.height, W=self.width)
        
        self.relu = nn.ReLU()
        
    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for AeDet.

        Args:
            x (Tensor): Input ferature map.
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
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,mats_dict,timestamps,is_return_depth=True)
        else:
            x, _ = self.backbone(x,mats_dict,timestamps,is_return_depth=True)
        
        bev_list_scale1 = []
        bev_list_scale2 = [] 
        bev_list_scale3 = [] 
        bev_list_scale4 = [] 
        
        cur_channels_num = 0
        for i in range(self.key_num):
            if i == 0:
                x_t = x[:,cur_channels_num:cur_channels_num+self.channels] # current t
                trunk_outs = [x_t]
                if self.trunk.deep_stem:
                    x_t = self.trunk.stem(x_t)
                else:
                    x_t = self.trunk.conv1(x_t)
                    x_t = self.trunk.norm1(x_t)
                    x_t = self.trunk.relu(x_t)
                for i, layer_name in enumerate(self.trunk.res_layers):
                    res_layer = getattr(self.trunk, layer_name)
                    x_t = res_layer(x_t)
                    if i in self.trunk.out_indices:
                        trunk_outs.append(x_t.to(torch.float32))
            
            else:
                #with torch.no_grad(): # past frames
                x_t = x[:,cur_channels_num:cur_channels_num+self.channels]
                trunk_outs = [x_t]
                if self.trunk.deep_stem:
                    x_t = self.trunk.stem(x_t)
                else:
                    x_t = self.trunk.conv1(x_t)
                    x_t = self.trunk.norm1(x_t)
                    x_t = self.trunk.relu(x_t)
                for i, layer_name in enumerate(self.trunk.res_layers):
                    res_layer = getattr(self.trunk, layer_name)
                    x_t = res_layer(x_t)
                    if i in self.trunk.out_indices:
                        trunk_outs.append(x_t.to(torch.float32))
        
            bev_list_scale1.append(trunk_outs[0])
            bev_list_scale2.append(trunk_outs[1])
            bev_list_scale3.append(trunk_outs[2])
            bev_list_scale4.append(trunk_outs[3])
            cur_channels_num += self.channels
        
        bev_features_scale1 = torch.stack(bev_list_scale1, dim=1) # [8, 2, 80, 128, 128]
        bev_features_scale2 = torch.stack(bev_list_scale2, dim=1) 
        bev_features_scale3 = torch.stack(bev_list_scale3, dim=1) 
        bev_features_scale4 = torch.stack(bev_list_scale4, dim=1)
        
        # cur_query = bev_features_scale1[:, -1]
        # cur_mask = self.heatmap(cur_query)
        # cur_mask = cur_mask > 0.4
        
        if self.agg_type == 'cnn':
            # CNN-based Temporal Aggregation
            for i in range(bev_features_scale1.shape[1]-1):
                concat_feat_scale_1 = torch.cat((bev_features_scale1[:,i], bev_features_scale1[:,i+1]),1)
                concat_feat_scale_2 = torch.cat((bev_features_scale2[:,i], bev_features_scale2[:,i+1]),1)
                concat_feat_scale_3 = torch.cat((bev_features_scale3[:,i], bev_features_scale3[:,i+1]),1)
                concat_feat_scale_4 = torch.cat((bev_features_scale4[:,i], bev_features_scale4[:,i+1]),1)

                for agg_conv in self.agg_conv_scale1:
                    concat_feat_scale_1 = agg_conv(concat_feat_scale_1)
                for agg_conv in self.agg_conv_scale2:
                    concat_feat_scale_2 = agg_conv(concat_feat_scale_2)
                for agg_conv in self.agg_conv_scale3:
                    concat_feat_scale_3 = agg_conv(concat_feat_scale_3)
                for agg_conv in self.agg_conv_scale4:
                    concat_feat_scale_4 = agg_conv(concat_feat_scale_4)
                    
                bev_features_scale1[:, i+1] = concat_feat_scale_1
                bev_features_scale2[:, i+1] = concat_feat_scale_2
                bev_features_scale3[:, i+1] = concat_feat_scale_3
                bev_features_scale4[:, i+1] = concat_feat_scale_4
                
                final_trunk = [bev_features_scale1[:,-1], bev_features_scale2[:,-1], \
                        bev_features_scale3[:,-1], bev_features_scale4[:,-1]]
                
        else:
            b, n, c, h, w = bev_features_scale1.shape
            bev_features_scale1 = self.agg_conv_scale1(bev_features_scale1.reshape(b, -1, h, w))
            bev_features_scale2 = self.agg_conv_scale2(bev_features_scale2.reshape(b, -1, h//2, w//2))
            bev_features_scale3 = self.agg_conv_scale3(bev_features_scale3.reshape(b, -1, h//4, w//4))
            bev_features_scale4 = self.agg_conv_scale4(bev_features_scale4.reshape(b, -1, h//8, w//8))

            final_trunk = [bev_features_scale1, bev_features_scale2, \
                        bev_features_scale3, bev_features_scale4]
        
        
        fpn_output = self.neck(final_trunk)
        #deform_result = self.relu(self.deform_attn_scale(cur_query, value=x, mask=cur_mask))
        #x = torch.cat((x, deform_result),1)
        fpn_output[0] = fpn_output[0].to(torch.float32)
        preds = self.head(fpn_output)
        #preds[0][0]['heatmap1'] = cur_mask
        if self.is_train_depth and self.training:
            return preds, depth_pred
        else:
            return preds
        

    def get_targets(self, gt_boxes, gt_labels):
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
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for AeDet.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
