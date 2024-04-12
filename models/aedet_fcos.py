"""Modified from `https://github.com/Megvii-BaseDetection/BEVDepth/blob/v0.0.1/models/bev_depth.py`"""
import torch
import sys
from torch import nn
from layers.backbones.lss_fpn_fcos import LSSFPN
from layers.heads.aedet_head import AeDetHead
from mmdet3d.models.dense_heads.fcos_mono3d_head import FCOSMono3DHead

__all__ = ['AeDet']


class AeDetFcos(nn.Module):
    """Source code of `AeDet`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(AeDetFcos, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = AeDetHead(**head_conf)
        self.is_train_depth = is_train_depth
        self.fcos3d_head = FCOSMono3DHead(num_classes=10, in_channels=256)
        #self.total_fcos_loss = 0

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
            pers_gt = mats_dict[1]
            mats_dict = mats_dict[0]
            total_fcos_loss=0
            
            x, depth_pred, img_feats = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            
            # Enter fcos3d 
            #cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness = self.fcos3d_head(img_feats) # cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness
            
            # post-process gt for fcos
            bboxes = []
            labels = []
            gt_bboxes_3d = []
            gt_labels_3d = []
            centers2d = []
            depths = []
            attr_labels = []
            for cam_idx in range(6):
                bboxes_tmp = []
                labels_tmp = []
                gt_bboxes_3d_tmp = []
                gt_labels_3d_tmp = []
                centers2d_tmp = []
                depths_tmp = []
                attr_labels_tmp = []
                
                for per_gt in pers_gt:
                    bboxes_tmp.append(torch.Tensor(per_gt[cam_idx]['bboxes']).cuda())
                    labels_tmp.append(torch.Tensor(per_gt[cam_idx]['labels']).cuda())
                    gt_bboxes_3d_tmp.append(per_gt[cam_idx]['gt_bboxes_3d'])
                    gt_labels_3d_tmp.append(torch.Tensor(per_gt[cam_idx]['gt_labels_3d']).cuda())
                    centers2d_tmp.append(torch.Tensor(per_gt[cam_idx]['centers2d']).cuda())
                    depths_tmp.append(torch.Tensor(per_gt[cam_idx]['depths']).cuda())
                    attr_labels_tmp.append(torch.Tensor(per_gt[cam_idx]['attr_labels']).cuda())
                bboxes.append(bboxes_tmp)
                labels.append(labels_tmp)
                gt_bboxes_3d.append(gt_bboxes_3d_tmp)
                gt_labels_3d.append(gt_labels_3d_tmp)
                centers2d.append(centers2d_tmp)
                depths.append(depths_tmp)
                attr_labels.append(attr_labels_tmp)
                
                
            # Enter fcos3d 
            for cam_idx in range(6):
                cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness = \
                    self.fcos3d_head([img_feats[0][:,cam_idx,:,:,:], img_feats[1][:,cam_idx,:,:,:],\
                        img_feats[2][:,cam_idx,:,:,:], img_feats[3][:,cam_idx,:,:,:]])
                
                
                fcos_loss = self.fcos3d_head.loss(cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness,\
                                                    bboxes[cam_idx], labels[cam_idx], gt_bboxes_3d[cam_idx],\
                                                    gt_labels_3d[cam_idx], centers2d[cam_idx], \
                                                    depths[cam_idx], attr_labels[cam_idx], None)
                
                #total_fcos_loss += fcos_loss['loss_cls'] + fcos_loss['loss_offset'] +fcos_loss['loss_depth'] \
                #    +fcos_loss['loss_size'] +fcos_loss['loss_rotsin'] +fcos_loss['loss_centerness'] +fcos_loss['loss_dir']
                total_fcos_loss += fcos_loss['loss_centerness']
            
            print(f"\r Fcos3d_loss: {total_fcos_loss}", end='')
            preds = self.head(x)
            return preds, depth_pred, total_fcos_loss
        else:
            x, img_feats = self.backbone(x, mats_dict, timestamps)
            #fcos_res = self.fcos3d_head(img_feats)
            preds = self.head(x)
            return preds, #fcos_res

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
        return self.head.get_bboxes(preds_dicts[0], img_metas, img, rescale)
