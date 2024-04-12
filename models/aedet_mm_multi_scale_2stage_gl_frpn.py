"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
from torch import nn
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.aedet_head_multi_scale_mm_gl_frpn import AeDetHead
from .convLSTM_V2_mm import ConvLSTMV2
from mmdet.models import build_backbone
from mmcv.runner import BaseModule
from mmdet3d.models import build_neck
from layers.modules.dense_sparse_deform_attn import BEV_DEFORM_ATTN
from mmdet3d.models.builder import build_loss
from monai.losses.dice import DiceLoss

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
        """
        """
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
        
        self.trunk = build_backbone(head_conf['bev_backbone_conf'])
        self.trunk.init_weights()
        self.neck = build_neck(head_conf['bev_neck_conf'])
        self.neck.init_weights()
        del self.trunk.maxpool


        self.channels = backbone_conf['output_channels']
        num_layers = backbone_conf['convlstm_layer']
        hidden_dim = [backbone_conf['output_channels']] * num_layers
        kernel_size = (3,3) 
        self.key_num = backbone_conf['num_keys']
        self.mm_no_residual = backbone_conf['mm_no_residual']
        
        x_bound = backbone_conf['x_bound']
        y_bound = backbone_conf['y_bound']
        self.height = int((x_bound[1]-x_bound[0])/x_bound[2])
        self.width = int((y_bound[1]-y_bound[0])/y_bound[2])
        
        input_dim1 = head_conf['bev_neck_conf']['in_channels'][0]
        input_dim2 = head_conf['bev_neck_conf']['in_channels'][1]
        input_dim3 = head_conf['bev_neck_conf']['in_channels'][2]
        input_dim4 = head_conf['bev_neck_conf']['in_channels'][3]
        
        head_conf['bev_backbone_conf']['in_channels'] = 80
        head_conf['bev_neck_conf']['in_channels'] = [80, 160, 320, 640]
        
        self.rescale_conv_list = nn.ModuleList()
        scale_factor = [1,2,4,8]
        for i in range(4):
            self.rescale_conv_list.append(nn.Conv2d(in_channels=self.channels*scale_factor[i], out_channels=input_dim1, kernel_size=1))
            
        
        self.convlstmv2_scale1 = ConvLSTMV2(
                    input_dim=input_dim1,
                    hidden_dim=[input_dim1] * num_layers,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False,
                    deform_conv_lstm = backbone_conf['deform_conv_lstm'],
                    motion_gate = backbone_conf['motion_gate'],
                    bevdepth = backbone_conf['bevdepth'])
        
        self.convlstmv2_scale2 = ConvLSTMV2(
                    input_dim=input_dim2,
                    hidden_dim=[input_dim2] * num_layers,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False,
                    deform_conv_lstm = backbone_conf['deform_conv_lstm'],
                    motion_gate = backbone_conf['motion_gate'],
                    bevdepth = backbone_conf['bevdepth'])
        
        self.convlstmv2_scale3 = ConvLSTMV2(
                    input_dim=input_dim3,
                    hidden_dim=[input_dim3] * num_layers,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False,
                    deform_conv_lstm = backbone_conf['deform_conv_lstm'],
                    motion_gate = backbone_conf['motion_gate'],
                    bevdepth = backbone_conf['bevdepth'])
        
        self.convlstmv2_scale4 = ConvLSTMV2(
                    input_dim=input_dim4,
                    hidden_dim=[input_dim4] * num_layers,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False,
                    deform_conv_lstm = backbone_conf['deform_conv_lstm'],
                    motion_gate = backbone_conf['motion_gate'],
                    bevdepth = backbone_conf['bevdepth'])
        
        
        self.dropout = nn.Dropout(0.1)
        self.conv1by1_scale1 = nn.Conv2d(in_channels=input_dim1, out_channels=input_dim1*backbone_conf['num_keys'], kernel_size=1)
        self.conv1by1_scale2 = nn.Conv2d(in_channels=input_dim2, out_channels=input_dim2*backbone_conf['num_keys'], kernel_size=1)
        self.conv1by1_scale3 = nn.Conv2d(in_channels=input_dim3, out_channels=input_dim3*backbone_conf['num_keys'], kernel_size=1)
        self.conv1by1_scale4 = nn.Conv2d(in_channels=input_dim4, out_channels=input_dim4*backbone_conf['num_keys'], kernel_size=1)
        
        self.conv1by1_scale1_reduce = nn.Conv2d(in_channels=input_dim1*backbone_conf['num_keys'], out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale2_reduce = nn.Conv2d(in_channels=input_dim2*backbone_conf['num_keys'], out_channels=input_dim2, kernel_size=1)
        self.conv1by1_scale3_reduce = nn.Conv2d(in_channels=input_dim3*backbone_conf['num_keys'], out_channels=input_dim3, kernel_size=1)
        self.conv1by1_scale4_reduce = nn.Conv2d(in_channels=input_dim4*backbone_conf['num_keys'], out_channels=input_dim4, kernel_size=1)
        
        
        self.conv1by1_scale1_cat = nn.Conv2d(in_channels=input_dim1*2, out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale2_cat = nn.Conv2d(in_channels=input_dim1*2, out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale3_cat = nn.Conv2d(in_channels=input_dim1*2, out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale4_cat = nn.Conv2d(in_channels=input_dim1*2, out_channels=input_dim1, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        
        # self.deform_attn_scale1 = BEV_DEFORM_ATTN(H=self.height, W=self.width)
        # self.deform_attn_scale2 = BEV_DEFORM_ATTN(H=self.height//2, W=self.width//2)
        # self.deform_attn_scale3 = BEV_DEFORM_ATTN(H=self.height//4, W=self.width//4)
        # self.deform_attn_scale4 = BEV_DEFORM_ATTN(H=self.height//8, W=self.width//8)
        
        self.ms_heatmap_head_scale1 = FRPN(input_dim1)
        self.ms_heatmap_head_scale2 = FRPN(input_dim1)
        self.ms_heatmap_head_scale3 = FRPN(input_dim1)
        self.ms_heatmap_head_scale4 = FRPN(input_dim1)
        
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
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            
            # BEV Backbone
            bev_list = [] 
            cur_channels_num = 0
            for i in range(self.key_num):
                bev_list.append(x[:,cur_channels_num:cur_channels_num+self.channels])
                cur_channels_num += self.channels
                
            bev_list = torch.stack(bev_list, dim=1) # [8, 2, 80, 128, 128]
            batch, t, c, h, w = bev_list.shape 
            x = bev_list.reshape(-1, c, h, w) # [16, 80, 128, 128]
            
            scale_list = [self.rescale_conv_list[0](x).reshape(batch, t, c, h, w)]
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
            
            for i, layer_name in enumerate(self.trunk.res_layers):
                res_layer = getattr(self.trunk, layer_name)
                x = res_layer(x)
                pyramid_reshape = self.rescale_conv_list[i+1](x)
                if i in self.trunk.out_indices:
                    scale_list.append(pyramid_reshape.reshape(batch, t, *pyramid_reshape.shape[-3:]))
                    
            
            # Enter Multi-scale Heatmap Head
            eps = 1e-4
            f_bev_scale1_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale1(scale_list[0][:,self.key_num-1])), min=eps, max=1 - eps)
            f_bev_scale2_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale2(scale_list[1][:,self.key_num-1])), min=eps, max=1 - eps)
            f_bev_scale3_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale3(scale_list[2][:,self.key_num-1])), min=eps, max=1 - eps)
            f_bev_scale4_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale4(scale_list[3][:,self.key_num-1])), min=eps, max=1 - eps)
            
            # Generate Mask
            f_bev_scale1_mask = (f_bev_scale1_heatmap > 0.4)
            f_bev_scale2_mask = (f_bev_scale2_heatmap > 0.4)
            f_bev_scale3_mask = (f_bev_scale3_heatmap > 0.4)
            f_bev_scale4_mask = (f_bev_scale4_heatmap > 0.4)
            
            x_scale1_sparse = scale_list[0][:,self.key_num-1] * f_bev_scale1_mask
            x_scale2_sparse = scale_list[1][:,self.key_num-1] * f_bev_scale2_mask
            x_scale3_sparse = scale_list[2][:,self.key_num-1] * f_bev_scale3_mask
            x_scale4_sparse = scale_list[3][:,self.key_num-1] * f_bev_scale4_mask
            
            
            # ConvLSTM V2 Dense
            device = x_scale4_sparse.device
            res1, _ = self.convlstmv2_scale1(scale_list[0].to(device))
            res2, _ = self.convlstmv2_scale2(scale_list[1].to(device))
            res3, _ = self.convlstmv2_scale3(scale_list[2].to(device))
            res4, _ = self.convlstmv2_scale4(scale_list[3].to(device))
            
            res1 = self.dropout(res1[0]).reshape(batch, -1, h, w) # [8, 2, 80, 128, 128]
            res2 = self.dropout(res2[0]).reshape(batch, -1, h//2, w//2)
            res3 = self.dropout(res3[0]).reshape(batch, -1, h//4, w//4)
            res4 = self.dropout(res4[0]).reshape(batch, -1, h//8, w//8)
            
            bev1_residual = self.conv1by1_scale1(scale_list[0][:,-1].reshape(batch, -1, h, w)) # [B, 160, 128, 128] for t
            bev2_residual = self.conv1by1_scale2(scale_list[1][:,-1].reshape(batch, -1, h//2, w//2)) 
            bev3_residual = self.conv1by1_scale3(scale_list[2][:,-1].reshape(batch, -1, h//4, w//4)) 
            bev4_residual = self.conv1by1_scale4(scale_list[3][:,-1].reshape(batch, -1, h//8, w//8))
            
            if self.mm_no_residual:
                x_scale1 = res1
                x_scale2 = res2
                x_scale3 = res3
                x_scale4 = res4
            else:
                x_scale1 = res1 + bev1_residual
                x_scale2 = res2 + bev2_residual
                x_scale3 = res3 + bev3_residual
                x_scale4 = res4 + bev4_residual
                
            x_scale1 = self.conv1by1_scale1_reduce(x_scale1)
            x_scale2 = self.conv1by1_scale2_reduce(x_scale2)
            x_scale3 = self.conv1by1_scale3_reduce(x_scale3)
            x_scale4 = self.conv1by1_scale4_reduce(x_scale4)

            # x_scale1_deform = self.relu(self.deform_attn_scale1(x_scale1_sparse, value=x_scale1))
            # x_scale2_deform = self.relu(self.deform_attn_scale2(x_scale2_sparse, value=x_scale2))
            # x_scale3_deform = self.relu(self.deform_attn_scale3(x_scale3_sparse, value=x_scale3))
            # x_scale4_deform = self.relu(self.deform_attn_scale4(x_scale4_sparse, value=x_scale4))

            x_scale1 = self.conv1by1_scale1_cat(torch.cat((x_scale1_sparse, x_scale1), 1))
            x_scale2 = self.conv1by1_scale2_cat(torch.cat((x_scale2_sparse, x_scale2), 1))
            x_scale3 = self.conv1by1_scale3_cat(torch.cat((x_scale3_sparse, x_scale3), 1))
            x_scale4 = self.conv1by1_scale4_cat(torch.cat((x_scale4_sparse, x_scale4), 1))
            
            x = self.neck([x_scale1, x_scale2, x_scale3, x_scale4])
            x[0] = x[0].to(torch.float32)
            
            gt_heatmap_1 = torch.max(torch.cat(mats_dict['heatmap_gt_1'], dim=1), dim=1)[0] > 0
            gt_heatmap_2 = torch.max(torch.cat(mats_dict['heatmap_gt_2'], dim=1), dim=1)[0] > 0
            gt_heatmap_3 = torch.max(torch.cat(mats_dict['heatmap_gt_3'], dim=1), dim=1)[0] > 0
            gt_heatmap_4 = torch.max(torch.cat(mats_dict['heatmap_gt_4'], dim=1), dim=1)[0] > 0
            
            heatmap_loss_1 = self.ms_heatmap_head_scale1.get_bev_mask_loss(gt_heatmap_1.float(), f_bev_scale1_mask.squeeze(1).float())
            heatmap_loss_2 = self.ms_heatmap_head_scale2.get_bev_mask_loss(gt_heatmap_2.float(), f_bev_scale2_mask.squeeze(1).float())
            heatmap_loss_3 = self.ms_heatmap_head_scale3.get_bev_mask_loss(gt_heatmap_3.float(), f_bev_scale3_mask.squeeze(1).float())
            heatmap_loss_4 = self.ms_heatmap_head_scale4.get_bev_mask_loss(gt_heatmap_4.float(), f_bev_scale4_mask.squeeze(1).float())
            
            heatmap_loss_ce = heatmap_loss_1[0] + heatmap_loss_2[0] + heatmap_loss_3[0] + heatmap_loss_4[0]
            heatmap_loss_dice = heatmap_loss_1[1] + heatmap_loss_2[1] + heatmap_loss_3[1] + heatmap_loss_4[1]
            preds = self.head(x)
            
            return preds, depth_pred, [heatmap_loss_ce, heatmap_loss_dice]
        else:
            x, _ = self.backbone(x,
                            mats_dict,
                            timestamps,
                            is_return_depth=True)
            
            # BEV Backbone
            bev_list = [] 
            cur_channels_num = 0
            for i in range(self.key_num):
                bev_list.append(x[:,cur_channels_num:cur_channels_num+self.channels])
                cur_channels_num += self.channels
                
            bev_list = torch.stack(bev_list, dim=1) # [8, 2, 80, 128, 128]
            batch, t, c, h, w = bev_list.shape 
            x = bev_list.reshape(-1, c, h, w) # [16, 80, 128, 128]
            
            scale_list = [self.rescale_conv_list[0](x).reshape(batch, t, c, h, w)]
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
            
            for i, layer_name in enumerate(self.trunk.res_layers):
                res_layer = getattr(self.trunk, layer_name)
                x = res_layer(x)
                pyramid_reshape = self.rescale_conv_list[i+1](x)
                if i in self.trunk.out_indices:
                    scale_list.append(pyramid_reshape.reshape(batch, t, *pyramid_reshape.shape[-3:]))
                    
            
            # Enter Multi-scale Heatmap Head
            eps = 1e-4
            f_bev_scale1_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale1(scale_list[0][:,self.key_num-1])), min=eps, max=1 - eps)
            f_bev_scale2_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale2(scale_list[1][:,self.key_num-1])), min=eps, max=1 - eps)
            f_bev_scale3_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale3(scale_list[2][:,self.key_num-1])), min=eps, max=1 - eps)
            f_bev_scale4_heatmap = torch.clamp(self.sigmoid(self.ms_heatmap_head_scale4(scale_list[3][:,self.key_num-1])), min=eps, max=1 - eps)
            
            # Generate Mask
            f_bev_scale1_mask = (f_bev_scale1_heatmap > 0.4)
            f_bev_scale2_mask = (f_bev_scale2_heatmap > 0.4)
            f_bev_scale3_mask = (f_bev_scale3_heatmap > 0.4)
            f_bev_scale4_mask = (f_bev_scale4_heatmap > 0.4)
            
            x_scale1_sparse = scale_list[0][:,self.key_num-1] * f_bev_scale1_mask
            x_scale2_sparse = scale_list[1][:,self.key_num-1] * f_bev_scale2_mask
            x_scale3_sparse = scale_list[2][:,self.key_num-1] * f_bev_scale3_mask
            x_scale4_sparse = scale_list[3][:,self.key_num-1] * f_bev_scale4_mask
            
            
            # ConvLSTM V2 Dense
            device = x_scale4_sparse.device
            res1, _ = self.convlstmv2_scale1(scale_list[0].to(device))
            res2, _ = self.convlstmv2_scale2(scale_list[1].to(device))
            res3, _ = self.convlstmv2_scale3(scale_list[2].to(device))
            res4, _ = self.convlstmv2_scale4(scale_list[3].to(device))
            
            res1 = self.dropout(res1[0]).reshape(batch, -1, h, w) # [8, 2, 80, 128, 128]
            res2 = self.dropout(res2[0]).reshape(batch, -1, h//2, w//2)
            res3 = self.dropout(res3[0]).reshape(batch, -1, h//4, w//4)
            res4 = self.dropout(res4[0]).reshape(batch, -1, h//8, w//8)
            
            bev1_residual = self.conv1by1_scale1(scale_list[0][:,-1].reshape(batch, -1, h, w)) # [B, 160, 128, 128] for t
            bev2_residual = self.conv1by1_scale2(scale_list[1][:,-1].reshape(batch, -1, h//2, w//2)) 
            bev3_residual = self.conv1by1_scale3(scale_list[2][:,-1].reshape(batch, -1, h//4, w//4)) 
            bev4_residual = self.conv1by1_scale4(scale_list[3][:,-1].reshape(batch, -1, h//8, w//8))
            
            if self.mm_no_residual:
                x_scale1 = res1
                x_scale2 = res2
                x_scale3 = res3
                x_scale4 = res4
            else:
                x_scale1 = res1 + bev1_residual 
                x_scale2 = res2 + bev2_residual
                x_scale3 = res3 + bev3_residual
                x_scale4 = res4 + bev4_residual
                
            x_scale1 = self.conv1by1_scale1_reduce(x_scale1)
            x_scale2 = self.conv1by1_scale2_reduce(x_scale2)
            x_scale3 = self.conv1by1_scale3_reduce(x_scale3)
            x_scale4 = self.conv1by1_scale4_reduce(x_scale4)

            # BEV Neck
            # x_scale1_deform = self.relu(self.deform_attn_scale1(x_scale1_sparse, value=x_scale1))
            # x_scale2_deform = self.relu(self.deform_attn_scale2(x_scale2_sparse, value=x_scale2))
            # x_scale3_deform = self.relu(self.deform_attn_scale3(x_scale3_sparse, value=x_scale3))
            # x_scale4_deform = self.relu(self.deform_attn_scale4(x_scale4_sparse, value=x_scale4))

            x_scale1 = self.conv1by1_scale1_cat(torch.cat((x_scale1_sparse, x_scale1), 1))
            x_scale2 = self.conv1by1_scale2_cat(torch.cat((x_scale2_sparse, x_scale2), 1))
            x_scale3 = self.conv1by1_scale3_cat(torch.cat((x_scale3_sparse, x_scale3), 1))
            x_scale4 = self.conv1by1_scale4_cat(torch.cat((x_scale4_sparse, x_scale4), 1))
            
            x = self.neck([x_scale1, x_scale2, x_scale3, x_scale4])
            x[0] = x[0].to(torch.float32)
            preds = self.head(x)
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
