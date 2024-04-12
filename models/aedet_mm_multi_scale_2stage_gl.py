"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
from torch import nn
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.aedet_head_multi_scale_mm_gl import AeDetHead
from .convLSTM_V2_mm import ConvLSTMV2
from mmdet.models import build_backbone
from mmcv.runner import BaseModule
from mmdet3d.models import build_neck
from layers.modules.dense_sparse_deform_attn import BEV_DEFORM_ATTN

__all__ = ['AeDet']


class HeatmapHead(nn.Module):
    def __init__(self):
        super(HeatmapHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=80, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(x)))
        return x



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
        
        self.ms_heatmap_head_scale1 = HeatmapHead()
        self.ms_heatmap_head_scale2 = HeatmapHead()
        self.ms_heatmap_head_scale3 = HeatmapHead()
        self.ms_heatmap_head_scale4 = HeatmapHead()

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
        
        self.deform_attn_scale1 = BEV_DEFORM_ATTN(H=self.height, W=self.width)
        self.deform_attn_scale2 = BEV_DEFORM_ATTN(H=self.height//2, W=self.width//2)
        self.deform_attn_scale3 = BEV_DEFORM_ATTN(H=self.height//4, W=self.width//4)
        self.deform_attn_scale4 = BEV_DEFORM_ATTN(H=self.height//8, W=self.width//8)
        
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
            f_bev_scale1_heatmap_max = f_bev_scale1_heatmap.max(dim=1)[0].unsqueeze(1)
            f_bev_scale2_heatmap_max = f_bev_scale2_heatmap.max(dim=1)[0].unsqueeze(1)
            f_bev_scale3_heatmap_max = f_bev_scale3_heatmap.max(dim=1)[0].unsqueeze(1)
            f_bev_scale4_heatmap_max = f_bev_scale4_heatmap.max(dim=1)[0].unsqueeze(1)
            
            f_bev_scale1_mask = (f_bev_scale1_heatmap_max > 0.4)
            f_bev_scale2_mask = (f_bev_scale2_heatmap_max > 0.4)
            f_bev_scale3_mask = (f_bev_scale3_heatmap_max > 0.4)
            f_bev_scale4_mask = (f_bev_scale4_heatmap_max > 0.4)
            
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
            x_scale1_deform = self.relu(self.deform_attn_scale1(x_scale1_sparse, value=x_scale1))
            x_scale2_deform = self.relu(self.deform_attn_scale2(x_scale2_sparse, value=x_scale2))
            x_scale3_deform = self.relu(self.deform_attn_scale3(x_scale3_sparse, value=x_scale3))
            x_scale4_deform = self.relu(self.deform_attn_scale4(x_scale4_sparse, value=x_scale4))

            x_scale1 = self.conv1by1_scale1_cat(torch.cat((x_scale1_deform, x_scale1), 1))
            x_scale2 = self.conv1by1_scale2_cat(torch.cat((x_scale2_deform, x_scale2), 1))
            x_scale3 = self.conv1by1_scale3_cat(torch.cat((x_scale3_deform, x_scale3), 1))
            x_scale4 = self.conv1by1_scale4_cat(torch.cat((x_scale4_deform, x_scale4), 1))
            
            x = self.neck([x_scale1, x_scale2, x_scale3, x_scale4])
            x[0] = x[0].to(torch.float32)
            
            preds = self.head(x)
            head_num = [0, 1, 3, 5, 6, 8, 10]
            for i in range(6):
                preds[i][0]['heatmap1'] = f_bev_scale1_heatmap[:,head_num[i]:head_num[i+1]]
                preds[i][0]['heatmap2'] = f_bev_scale2_heatmap[:,head_num[i]:head_num[i+1]]
                preds[i][0]['heatmap3'] = f_bev_scale3_heatmap[:,head_num[i]:head_num[i+1]]
                preds[i][0]['heatmap4'] = f_bev_scale4_heatmap[:,head_num[i]:head_num[i+1]]
            return preds, depth_pred
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
            f_bev_scale1_heatmap_max = f_bev_scale1_heatmap.max(dim=1)[0].unsqueeze(1)
            f_bev_scale2_heatmap_max = f_bev_scale2_heatmap.max(dim=1)[0].unsqueeze(1)
            f_bev_scale3_heatmap_max = f_bev_scale3_heatmap.max(dim=1)[0].unsqueeze(1)
            f_bev_scale4_heatmap_max = f_bev_scale4_heatmap.max(dim=1)[0].unsqueeze(1)
            
            f_bev_scale1_mask = (f_bev_scale1_heatmap_max > 0.4)
            f_bev_scale2_mask = (f_bev_scale2_heatmap_max > 0.4)
            f_bev_scale3_mask = (f_bev_scale3_heatmap_max > 0.4)
            f_bev_scale4_mask = (f_bev_scale4_heatmap_max > 0.4)
            
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
            x_scale1_deform = self.relu(self.deform_attn_scale1(x_scale1_sparse, value=x_scale1))
            x_scale2_deform = self.relu(self.deform_attn_scale2(x_scale2_sparse, value=x_scale2))
            x_scale3_deform = self.relu(self.deform_attn_scale3(x_scale3_sparse, value=x_scale3))
            x_scale4_deform = self.relu(self.deform_attn_scale4(x_scale4_sparse, value=x_scale4))

            x_scale1 = self.conv1by1_scale1_cat(torch.cat((x_scale1_deform, x_scale1), 1))
            x_scale2 = self.conv1by1_scale2_cat(torch.cat((x_scale2_deform, x_scale2), 1))
            x_scale3 = self.conv1by1_scale3_cat(torch.cat((x_scale3_deform, x_scale3), 1))
            x_scale4 = self.conv1by1_scale4_cat(torch.cat((x_scale4_deform, x_scale4), 1))
            
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
