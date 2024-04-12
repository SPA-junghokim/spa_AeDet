"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
from torch import nn
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.aedet_head_multi_scale_mm import AeDetHead
from layers.heads.aedet_head_2stage import AeDetHead as AeDetHeadOrg
from .convLSTM_V2_mm import ConvLSTMV2
from mmdet.models import build_backbone
from mmdet3d.models import build_neck
from layers.modules.dense_sparse_deform_attn import BEV_DEFORM_ATTN

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
        
        self.trunk_sparse = build_backbone(head_conf['bev_backbone_conf'])
        self.trunk_sparse.init_weights()
        #self.neck_sparse = build_neck(head_conf['bev_neck_conf'])
        #self.neck_sparse.init_weights()
        del self.trunk_sparse.maxpool
        

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
        self.first_head = AeDetHeadOrg(**head_conf)
        
        self.rescale_conv_list = nn.ModuleList()
        scale_factor = [1,2,4,8]
        for i in range(4):
            self.rescale_conv_list.append(nn.Conv2d(in_channels=self.channels*scale_factor[i], out_channels=input_dim1, kernel_size=1))
            
        
        self.rescale_conv_list_sparse = nn.ModuleList()
        scale_factor = [1,2,4,8]
        for i in range(4):
            self.rescale_conv_list_sparse.append(nn.Conv2d(in_channels=self.channels*scale_factor[i], out_channels=input_dim1, kernel_size=1))    
        
        
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
        
        self.convlstmv2_scale1_sparse = ConvLSTMV2(
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
        
        self.convlstmv2_scale2_sparse = ConvLSTMV2(
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
        
        self.convlstmv2_scale3_sparse = ConvLSTMV2(
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
        
        self.convlstmv2_scale4_sparse = ConvLSTMV2(
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
        self.conv1by1_scale1 = nn.Conv2d(in_channels=input_dim1*num_layers, out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale2 = nn.Conv2d(in_channels=input_dim2*num_layers, out_channels=input_dim2, kernel_size=1)
        self.conv1by1_scale3 = nn.Conv2d(in_channels=input_dim3*num_layers, out_channels=input_dim3, kernel_size=1)
        self.conv1by1_scale4 = nn.Conv2d(in_channels=input_dim4*num_layers, out_channels=input_dim4, kernel_size=1)
        
        self.conv1by1_scale1_sparse = nn.Conv2d(in_channels=input_dim1*num_layers, out_channels=input_dim1, kernel_size=1)
        self.conv1by1_scale2_sparse = nn.Conv2d(in_channels=input_dim2*num_layers, out_channels=input_dim2, kernel_size=1)
        self.conv1by1_scale3_sparse = nn.Conv2d(in_channels=input_dim3*num_layers, out_channels=input_dim3, kernel_size=1)
        self.conv1by1_scale4_sparse = nn.Conv2d(in_channels=input_dim4*num_layers, out_channels=input_dim4, kernel_size=1)

        self.conv1by1_reduce = nn.Conv2d(in_channels=256, out_channels=self.channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        
        self.deform_attn_scale1 = BEV_DEFORM_ATTN(H=128, W=128)
        self.deform_attn_scale2 = BEV_DEFORM_ATTN(H=64, W=64)
        self.deform_attn_scale3 = BEV_DEFORM_ATTN(H=32, W=32)
        self.deform_attn_scale4 = BEV_DEFORM_ATTN(H=16, W=16)
        
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
            
            # Get Sparse BEV
            with torch.no_grad():
                first_preds, fpn_output_1 = self.first_head(x[:,:80]) # t-1
                
            for class_idx in range(len(first_preds)):
                if class_idx == 0:
                    heatmap = first_preds[class_idx][0]['heatmap']
                else:
                    heatmap = torch.cat((heatmap, first_preds[class_idx][0]['heatmap']), dim=1)
                                
            heatmap_max = torch.max(heatmap, dim=1)[0].unsqueeze(1)
            fpn_output_1_reduced = self.conv1by1_reduce(fpn_output_1)
            x_sparse_1 = fpn_output_1_reduced * self.sigmoid(heatmap_max)
            
            first_preds, fpn_output_2 = self.first_head(x[:,80:]) # t
            for class_idx in range(len(first_preds)):
                if class_idx == 0:
                    heatmap = first_preds[class_idx][0]['heatmap']
                else:
                    heatmap = torch.cat((heatmap, first_preds[class_idx][0]['heatmap']), dim=1)
            
            heatmap_max = torch.max(heatmap, dim=1)[0].unsqueeze(1)
            fpn_output_2_reduced = self.conv1by1_reduce(fpn_output_2)
            x_sparse_2 = fpn_output_2_reduced * self.sigmoid(heatmap_max)
            
            x_sparse_final = torch.cat((x_sparse_1, x_sparse_2), dim=1)
            
            # BEV Sparse Backbone
            bev_list = [] 
            cur_channels_num = 0
            for i in range(self.key_num):
                bev_list.append(x_sparse_final[:,cur_channels_num:cur_channels_num+self.channels])
                cur_channels_num += self.channels
                
            bev_list = torch.stack(bev_list, dim=1) # [8, 2, 80, 128, 128]
            batch, t, c, h, w = bev_list.shape 
            x_sparse = bev_list.reshape(-1, c, h, w) # [16, 80, 128, 128]
            
            scale_list_sparse = [self.rescale_conv_list_sparse[0](x_sparse).reshape(batch, t, c, h, w)]
            x_sparse = self.trunk_sparse.conv1(x_sparse)
            x_sparse = self.trunk_sparse.norm1(x_sparse)
            x_sparse = self.trunk_sparse.relu(x_sparse)
            
            for i, layer_name in enumerate(self.trunk.res_layers):
                res_layer = getattr(self.trunk_sparse, layer_name)
                x_sparse = res_layer(x_sparse)
                pyramid_reshape = self.rescale_conv_list_sparse[i+1](x_sparse)
                if i in self.trunk_sparse.out_indices:
                    scale_list_sparse.append(pyramid_reshape.reshape(batch, t, *pyramid_reshape.shape[-3:]))
            
            
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
            
            
            # ConvLSTM V2 Sparse
            f_bev_scale1_sparse = scale_list_sparse[0]
            f_bev_scale2_sparse = scale_list_sparse[1]
            f_bev_scale3_sparse = scale_list_sparse[2]
            f_bev_scale4_sparse = scale_list_sparse[3]

            res1_sparse, _ = self.convlstmv2_scale1_sparse(f_bev_scale1_sparse.cuda())
            res2_sparse, _ = self.convlstmv2_scale2_sparse(f_bev_scale2_sparse.cuda())
            res3_sparse, _ = self.convlstmv2_scale3_sparse(f_bev_scale3_sparse.cuda())
            res4_sparse, _ = self.convlstmv2_scale4_sparse(f_bev_scale4_sparse.cuda())
            
            res1_sparse = self.dropout(res1_sparse[0]) # [8, 2, 80, 128, 128]
            res2_sparse = self.dropout(res2_sparse[0]) 
            res3_sparse = self.dropout(res3_sparse[0]) 
            res4_sparse = self.dropout(res4_sparse[0]) 
            
            res1_sparse = res1_sparse.reshape(res1_sparse.shape[0], -1, self.height, self.width)
            res2_sparse = res2_sparse.reshape(res2_sparse.shape[0], -1, res2_sparse.shape[3], res2_sparse.shape[4])
            res3_sparse = res3_sparse.reshape(res3_sparse.shape[0], -1, res3_sparse.shape[3], res3_sparse.shape[4])
            res4_sparse = res4_sparse.reshape(res4_sparse.shape[0], -1, res4_sparse.shape[3], res4_sparse.shape[4])
                        
            bev1_residual_sparse = self.conv1by1_scale1_sparse(res1_sparse) # [B, 160, 128, 128] for t
            bev2_residual_sparse = self.conv1by1_scale2_sparse(res2_sparse) 
            bev3_residual_sparse = self.conv1by1_scale3_sparse(res3_sparse) 
            bev4_residual_sparse = self.conv1by1_scale4_sparse(res4_sparse)
            
            if self.mm_no_residual:
                x_scale1_sparse = bev1_residual_sparse
                x_scale2_sparse = bev2_residual_sparse
                x_scale3_sparse = bev3_residual_sparse
                x_scale4_sparse = bev4_residual_sparse
            else:
                x_scale1_sparse = f_bev_scale1_sparse[:, -1] + bev1_residual_sparse
                x_scale2_sparse = f_bev_scale2_sparse[:, -1] + bev2_residual_sparse
                x_scale3_sparse = f_bev_scale3_sparse[:, -1] + bev3_residual_sparse
                x_scale4_sparse = f_bev_scale4_sparse[:, -1] + bev4_residual_sparse
                
            
            # ConvLSTM V2 Dense
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
            
            x_scale1 = self.relu(self.deform_attn_scale1(x_scale1_sparse, value=x_scale1))
            x_scale2 = self.relu(self.deform_attn_scale2(x_scale2_sparse, value=x_scale2))
            x_scale3 = self.relu(self.deform_attn_scale3(x_scale3_sparse, value=x_scale3))
            x_scale4 = self.relu(self.deform_attn_scale4(x_scale4_sparse, value=x_scale4))
            
            # x_scale1 += x_scale1_sparse
            # x_scale2 += x_scale2_sparse
            # x_scale3 += x_scale3_sparse
            # x_scale4 += x_scale4_sparse
            
            x = self.neck([x_scale1, x_scale2, x_scale3, x_scale4])
            #x_sparse = self.neck_sparse([x_scale1_sparse, x_scale2_sparse, x_scale3_sparse, x_scale4_sparse])
            x[0] = x[0].to(torch.float32)
            #x_sparse[0] = x_sparse[0].to(torch.float32)
            
            #x += x_sparse
            
            preds = self.head(x)
            return preds, depth_pred, first_preds
        else:
            x, _ = self.backbone(x,
                            mats_dict,
                            timestamps,
                            is_return_depth=True)
            
            # Get Sparse BEV
            with torch.no_grad():
                first_preds, fpn_output_1 = self.first_head(x[:,:80]) # t-1
                
            for class_idx in range(len(first_preds)):
                if class_idx == 0:
                    heatmap = first_preds[class_idx][0]['heatmap']
                else:
                    heatmap = torch.cat((heatmap, first_preds[class_idx][0]['heatmap']), dim=1)
                                
            heatmap_max = torch.max(heatmap, dim=1)[0].unsqueeze(1)
            fpn_output_1_reduced = self.conv1by1_reduce(fpn_output_1)
            x_sparse_1 = fpn_output_1_reduced * self.sigmoid(heatmap_max)
            
            first_preds, fpn_output_2 = self.first_head(x[:,80:]) # t
            for class_idx in range(len(first_preds)):
                if class_idx == 0:
                    heatmap = first_preds[class_idx][0]['heatmap']
                else:
                    heatmap = torch.cat((heatmap, first_preds[class_idx][0]['heatmap']), dim=1)
            
            heatmap_max = torch.max(heatmap, dim=1)[0].unsqueeze(1)
            fpn_output_2_reduced = self.conv1by1_reduce(fpn_output_2)
            x_sparse_2 = fpn_output_2_reduced * self.sigmoid(heatmap_max)
            
            x_sparse_final = torch.cat((x_sparse_1, x_sparse_2), dim=1)
            
            # BEV Sparse Backbone
            bev_list = [] 
            cur_channels_num = 0
            for i in range(self.key_num):
                bev_list.append(x_sparse_final[:,cur_channels_num:cur_channels_num+self.channels])
                cur_channels_num += self.channels
                
            bev_list = torch.stack(bev_list, dim=1) # [8, 2, 80, 128, 128]
            batch, t, c, h, w = bev_list.shape 
            x_sparse = bev_list.reshape(-1, c, h, w) # [16, 80, 128, 128]
            
            scale_list_sparse = [self.rescale_conv_list_sparse[0](x_sparse).reshape(batch, t, c, h, w)]
            x_sparse = self.trunk_sparse.conv1(x_sparse)
            x_sparse = self.trunk_sparse.norm1(x_sparse)
            x_sparse = self.trunk_sparse.relu(x_sparse)
            
            for i, layer_name in enumerate(self.trunk.res_layers):
                res_layer = getattr(self.trunk_sparse, layer_name)
                x_sparse = res_layer(x_sparse)
                pyramid_reshape = self.rescale_conv_list_sparse[i+1](x_sparse)
                if i in self.trunk_sparse.out_indices:
                    scale_list_sparse.append(pyramid_reshape.reshape(batch, t, *pyramid_reshape.shape[-3:]))
            
            
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
            
            
            # ConvLSTM V2 Sparse
            f_bev_scale1_sparse = scale_list_sparse[0]
            f_bev_scale2_sparse = scale_list_sparse[1]
            f_bev_scale3_sparse = scale_list_sparse[2]
            f_bev_scale4_sparse = scale_list_sparse[3]

            res1_sparse, _ = self.convlstmv2_scale1_sparse(f_bev_scale1_sparse.cuda())
            res2_sparse, _ = self.convlstmv2_scale2_sparse(f_bev_scale2_sparse.cuda())
            res3_sparse, _ = self.convlstmv2_scale3_sparse(f_bev_scale3_sparse.cuda())
            res4_sparse, _ = self.convlstmv2_scale4_sparse(f_bev_scale4_sparse.cuda())
            
            res1_sparse = self.dropout(res1_sparse[0]) # [8, 2, 80, 128, 128]
            res2_sparse = self.dropout(res2_sparse[0]) 
            res3_sparse = self.dropout(res3_sparse[0]) 
            res4_sparse = self.dropout(res4_sparse[0]) 
            
            res1_sparse = res1_sparse.reshape(res1_sparse.shape[0], -1, self.height, self.width)
            res2_sparse = res2_sparse.reshape(res2_sparse.shape[0], -1, res2_sparse.shape[3], res2_sparse.shape[4])
            res3_sparse = res3_sparse.reshape(res3_sparse.shape[0], -1, res3_sparse.shape[3], res3_sparse.shape[4])
            res4_sparse = res4_sparse.reshape(res4_sparse.shape[0], -1, res4_sparse.shape[3], res4_sparse.shape[4])
                        
            bev1_residual_sparse = self.conv1by1_scale1_sparse(res1_sparse) # [B, 160, 128, 128] for t
            bev2_residual_sparse = self.conv1by1_scale2_sparse(res2_sparse) 
            bev3_residual_sparse = self.conv1by1_scale3_sparse(res3_sparse) 
            bev4_residual_sparse = self.conv1by1_scale4_sparse(res4_sparse)
            
            if self.mm_no_residual:
                x_scale1_sparse = bev1_residual_sparse
                x_scale2_sparse = bev2_residual_sparse
                x_scale3_sparse = bev3_residual_sparse
                x_scale4_sparse = bev4_residual_sparse
            else:
                x_scale1_sparse = f_bev_scale1_sparse[:, -1] + bev1_residual_sparse
                x_scale2_sparse = f_bev_scale2_sparse[:, -1] + bev2_residual_sparse
                x_scale3_sparse = f_bev_scale3_sparse[:, -1] + bev3_residual_sparse
                x_scale4_sparse = f_bev_scale4_sparse[:, -1] + bev4_residual_sparse
                
            
            # ConvLSTM V2 Dense
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
            
            x_scale1 = self.relu(self.deform_attn_scale1(x_scale1_sparse, value=x_scale1))
            x_scale2 = self.relu(self.deform_attn_scale2(x_scale2_sparse, value=x_scale2))
            x_scale3 = self.relu(self.deform_attn_scale3(x_scale3_sparse, value=x_scale3))
            x_scale4 = self.relu(self.deform_attn_scale4(x_scale4_sparse, value=x_scale4))
            
            # x_scale1 += x_scale1_sparse
            # x_scale2 += x_scale2_sparse
            # x_scale3 += x_scale3_sparse
            # x_scale4 += x_scale4_sparse
            
            x = self.neck([x_scale1, x_scale2, x_scale3, x_scale4])
            #x_sparse = self.neck_sparse([x_scale1_sparse, x_scale2_sparse, x_scale3_sparse, x_scale4_sparse])
            x[0] = x[0].to(torch.float32)
            #x_sparse[0] = x_sparse[0].to(torch.float32)
            
            #x += x_sparse
            
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
