"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
from torch import nn
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.aedet_head_multi_scale_mm import AeDetHead
from .convLSTM_V2_mm import ConvLSTMV2
from mmdet.models import build_backbone
from mmdet3d.models import build_neck

__all__ = ['AeDet']


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
        self.conv1by1_scale1 = nn.Conv2d(in_channels=input_dim1*backbone_conf['num_keys'], out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale2 = nn.Conv2d(in_channels=input_dim2*backbone_conf['num_keys'], out_channels=input_dim2, kernel_size=1)
        self.conv1by1_scale3 = nn.Conv2d(in_channels=input_dim3*backbone_conf['num_keys'], out_channels=input_dim3, kernel_size=1)
        self.conv1by1_scale4 = nn.Conv2d(in_channels=input_dim4*backbone_conf['num_keys'], out_channels=input_dim4, kernel_size=1)

        
        
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
            # ConvLSTM V2
            f_bev_scale1 = scale_list[0]
            f_bev_scale2 = scale_list[1]
            f_bev_scale3 = scale_list[2]
            f_bev_scale4 = scale_list[3]

            res1, _ = self.convlstmv2_scale1(f_bev_scale1.cuda())
            res2, _ = self.convlstmv2_scale2(f_bev_scale2.cuda())
            res3, _ = self.convlstmv2_scale3(f_bev_scale3.cuda())
            res4, _ = self.convlstmv2_scale4(f_bev_scale4.cuda())
            
            res1 = self.dropout(res1[0]) # [8, 2, 80, 128, 128]
            res2 = self.dropout(res2[0]) 
            res3 = self.dropout(res3[0]) 
            res4 = self.dropout(res4[0]) 
            
            res1 = res1.reshape(res1.shape[0], -1, self.height, self.width)
            res2 = res2.reshape(res2.shape[0], -1, res2.shape[3], res2.shape[4])
            res3 = res3.reshape(res3.shape[0], -1, res3.shape[3], res3.shape[4])
            res4 = res4.reshape(res4.shape[0], -1, res4.shape[3], res4.shape[4])
                        
            bev1_residual = self.conv1by1_scale1(res1) # [B, 160, 128, 128] for t
            bev2_residual = self.conv1by1_scale2(res2) 
            bev3_residual = self.conv1by1_scale3(res3) 
            bev4_residual = self.conv1by1_scale4(res4) 
            if self.mm_no_residual:
                x_scale1 = bev1_residual
                x_scale2 = bev2_residual
                x_scale3 = bev3_residual
                x_scale4 = bev4_residual
            else:
                x_scale1 = f_bev_scale1[:, -1] + bev1_residual
                x_scale2 = f_bev_scale2[:, -1] + bev2_residual
                x_scale3 = f_bev_scale3[:, -1] + bev3_residual
                x_scale4 = f_bev_scale4[:, -1] + bev4_residual
                
                
            # BEV Neck
            x = self.neck([x_scale1, x_scale2, x_scale3, x_scale4])
            x[0] = x[0].to(torch.float32)
            
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            
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
            # ConvLSTM V2
            f_bev_scale1 = scale_list[0]
            f_bev_scale2 = scale_list[1]
            f_bev_scale3 = scale_list[2]
            f_bev_scale4 = scale_list[3]

            res1, _ = self.convlstmv2_scale1(f_bev_scale1.cuda())
            res2, _ = self.convlstmv2_scale2(f_bev_scale2.cuda())
            res3, _ = self.convlstmv2_scale3(f_bev_scale3.cuda())
            res4, _ = self.convlstmv2_scale4(f_bev_scale4.cuda())
            
            res1 = self.dropout(res1[0]) # [8, 2, 80, 128, 128]
            res2 = self.dropout(res2[0]) 
            res3 = self.dropout(res3[0]) 
            res4 = self.dropout(res4[0]) 
            
            res1 = res1.reshape(res1.shape[0], -1, self.height, self.width)
            res2 = res2.reshape(res2.shape[0], -1, res2.shape[3], res2.shape[4])
            res3 = res3.reshape(res3.shape[0], -1, res3.shape[3], res3.shape[4])
            res4 = res4.reshape(res4.shape[0], -1, res4.shape[3], res4.shape[4])
                        
            bev1_residual = self.conv1by1_scale1(res1) # [B, 160, 128, 128] for t
            bev2_residual = self.conv1by1_scale2(res2) 
            bev3_residual = self.conv1by1_scale3(res3) 
            bev4_residual = self.conv1by1_scale4(res4) 
            if self.mm_no_residual:
                x_scale1 = bev1_residual
                x_scale2 = bev2_residual
                x_scale3 = bev3_residual
                x_scale4 = bev4_residual
            else:
                x_scale1 = f_bev_scale1[:, -1] + bev1_residual
                x_scale2 = f_bev_scale2[:, -1] + bev2_residual
                x_scale3 = f_bev_scale3[:, -1] + bev3_residual
                x_scale4 = f_bev_scale4[:, -1] + bev4_residual
                
                
            # BEV Neck
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
