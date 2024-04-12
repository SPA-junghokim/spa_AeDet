"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
from torch import nn
import pdb
from layers.backbones.lss_fpn_polar import LSSFPN
from layers.modules.xy_polar_deform_attn import XY_POLAR_BEV_DEFORM_ATTN
import torch

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
    def __init__(self, backbone_conf, head_conf, is_train_depth=False, consistency_cfg=None, xy_bev_head_with_deform_attn=False):
        super(AeDet, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        if xy_bev_head_with_deform_attn:
            from layers.heads.aedet_head_consis import AeDetHead
            self.head = AeDetHead(consistency_cfg, **head_conf)
        else:
            from layers.heads.aedet_head_polar import AeDetHead
            self.head = AeDetHead(consistency_cfg, **head_conf)
        self.is_train_depth = is_train_depth
        self.xy_bev_head_with_deform_attn=xy_bev_head_with_deform_attn
        
        
        
        if xy_bev_head_with_deform_attn:
            W, H = [(row[1] - row[0]) / row[2] for row in [backbone_conf['x_bound'], backbone_conf['y_bound']]]
            self.spatial_cross_attn = XY_POLAR_BEV_DEFORM_ATTN(
                embed_dims=backbone_conf['output_channels'],
                num_heads=4,
                num_points=4,
                im2col_step=32,
                dropout=0.1,
                norm_cfg={'type': 'LN'},
                init_cfg=None, 
                W = W,
                H = H)
            
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
            polar_bev_pooled, depth_pred, img_feats, xy_bev_pooled = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            if self.xy_bev_head_with_deform_attn:
                xy_bev_pooled = self.spatial_cross_attn(xy_bev_pooled, value=polar_bev_pooled)
                preds, bev_neck = self.head(xy_bev_pooled)
                return preds, depth_pred, img_feats, xy_bev_pooled, bev_neck
            else:
                preds, bev_neck = self.head(polar_bev_pooled)
                return preds, depth_pred, img_feats, polar_bev_pooled, bev_neck
        else:
            polar_bev_pooled, xy_bev_pooled = self.backbone(x, mats_dict, timestamps)
            if self.xy_bev_head_with_deform_attn:
                xy_bev_pooled = self.spatial_cross_attn(xy_bev_pooled, value=polar_bev_pooled)
                preds, bev_neck = self.head(xy_bev_pooled)
            else:
                preds, bev_neck = self.head(polar_bev_pooled)
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

    def loss(self, targets, preds_dicts, img_feats, bev_feats, bev_neck, mats_dict, backbone_conf, bev_consistency=False):
        """Loss function for AeDet.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        return self.head.loss(targets, preds_dicts, img_feats, bev_feats, bev_neck, mats_dict, backbone_conf, bev_consistency)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
