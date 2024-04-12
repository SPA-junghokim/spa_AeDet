"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
from torch import nn
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.aedet_head import AeDetHead
from .convLSTM_V2_mm import ConvLSTMV2

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
        
        self.convlstmv2 = ConvLSTMV2(
                    input_dim=self.channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False,
                    deform_conv_lstm = backbone_conf['deform_conv_lstm'],
                    motion_gate = backbone_conf['motion_gate'],
                    bevdepth=backbone_conf['bevdepth'])
        
        self.dropout = nn.Dropout(0.1)
        self.conv1by1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels*backbone_conf['num_keys'], kernel_size=1)

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

            # ConvLSTM V2
            bev_list = []
            cur_channels_num = 0
            for i in range(self.key_num):
                bev_list.append(x[:,cur_channels_num:cur_channels_num+self.channels])
                cur_channels_num += self.channels
            
            f_bev = torch.stack(bev_list, dim=1) # [8, 2, 80, 128, 128]
            res, _ = self.convlstmv2(f_bev.cuda())
            res = self.dropout(res[0]) # [8, 2, 80, 128, 128]

            res = res.reshape(res.shape[0], self.channels*self.key_num, self.height, self.width)

            bev1_res = self.conv1by1(bev_list[0]) # [B, 80, 128, 128]
            
            if self.mm_no_residual:
                x = bev1_res
            else:
                x = bev1_res + res
                
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            
            # ConvLSTM V2
            bev_list = []
            cur_channels_num = 0
            for i in range(self.key_num):
                bev_list.append(x[:,cur_channels_num:cur_channels_num+self.channels])
                cur_channels_num += self.channels
            
            f_bev = torch.stack(bev_list, dim=1) # [8, 2, 80, 128, 128]

            res, _ = self.convlstmv2(f_bev.cuda())
            res = self.dropout(res[0]) # [8, 2, 80, 128, 128]

            res = res.reshape(res.shape[0], self.channels*self.key_num, self.height, self.width)

            bev1_res = self.conv1by1(bev_list[-1]) # [B, 80, 128, 128]
            
            if self.mm_no_residual:
                x = bev1_res
            else:
                x = bev1_res + res
                
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
