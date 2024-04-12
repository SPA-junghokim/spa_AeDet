# Copyright (c) OpenMMLab. All rights reserved.
import torch
import math
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

def adjust_within_range(radian):
    two_pi = 2 * torch.tensor(math.pi)
    radian[radian < -torch.tensor(math.pi)] = radian[radian < -torch.tensor(math.pi)] + two_pi
    radian[radian > torch.tensor(math.pi)] = radian[radian > torch.tensor(math.pi)] - two_pi
    return radian


@BBOX_CODERS.register_module()
class AeDetBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

        # compute the azimuth
        cart_x = torch.arange(self.pc_range[0] + self.out_size_factor * self.voxel_size[0] / 2.0, -self.pc_range[0],
                             self.out_size_factor * self.voxel_size[0])
        cart_y = torch.arange(-self.pc_range[0] - self.out_size_factor * self.voxel_size[0] / 2.0, self.pc_range[0],
                             -self.out_size_factor * self.voxel_size[0])
        cart_x = cart_x.view(1, len(cart_x)).repeat(len(cart_y), 1)
        cart_y = cart_y.view(len(cart_y), 1).repeat(1, len(cart_x))
        self.azimuth = torch.atan2(cart_x, cart_y)

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        # reg
        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)

        # vel
        if vel is not None:
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)

        # mapping the rotation, reg and vel from the azimuth system to Cartesian coordinates
        azimuth = self.azimuth.view(-1)[inds].view(len(inds), -1, 1).to(heat.device)
        cart_rot = rot + azimuth
        if reg is not None:
            regx, regy = reg.split([1, 1], 2)
            cart_regx = regx * torch.cos(azimuth) - regy * torch.sin(azimuth)
            cart_regy = regx * torch.sin(azimuth) + regy * torch.cos(azimuth)
            cart_reg = torch.cat([cart_regx, cart_regy], 2)
        if vel is not None:
            vx, vy = vel.split([1, 1], 2)
            cart_vx = vx * torch.cos(azimuth) - vy * torch.sin(azimuth)
            cart_vy = vx * torch.sin(azimuth) + vy * torch.cos(azimuth)
            cart_vel = torch.cat([cart_vx, cart_vy], 2)

        if reg is not None:
            xs = xs.view(batch, self.max_num, 1) + cart_reg[:, :, 0:1] + 0.5
            ys = ys.view(batch, self.max_num, 1) + cart_reg[:, :, 1:2] + 0.5
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        xs = xs.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, cart_rot], dim=2)
        else:  # exist velocity, nuscene format
            final_box_preds = torch.cat([xs, ys, hei, dim, cart_rot, cart_vel], dim=2)

        final_scores = scores
        final_preds = clses
        
        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts



@BBOX_CODERS.register_module()
class PolarBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9,
                 dim_div=None,
                 radius_center = True,
                 azimuth_center = True,
                 vel_div = True,
                 ):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size
        self.dim_div = dim_div
        self.radius_center = radius_center
        self.azimuth_center = azimuth_center
        self.vel_div = vel_div
        
    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()
        
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
        
        topk_inds = topk_inds % (height * width)
        
        topk_as = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()
        topk_rs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        topk_as = self._gather_feat(topk_as.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        topk_rs = self._gather_feat(topk_rs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_as, topk_rs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass
    
    
    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()
        
        # import matplotlib.pyplot as plt
        # numpy_image = heat[0].cpu().permute(1,2,0).numpy()
        # fig = plt.figure()
        # try:
        #     plt.imshow(numpy_image, cmap='viridis')  # Replace 'viridis' with your desired color map
        #     import os
        #     os.makedirs('visual_heatmap_pred', exist_ok=True)
        #     import numpy as np
        #     fig.savefig(f'visual_heatmap_pred/tensor_image{str(np.random.randint(1000))}.png')
        # except:
        #     pass
        scores, inds, clses, azimuth_s_idx, radius_s_idx = self._topk(heat, K=self.max_num)

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)
        
        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        # reg
        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            radius_s = radius_s_idx.view(batch, self.max_num, 1) + reg[:, :, 0:1] + 0.5
            azimuth_s = azimuth_s_idx.view(batch, self.max_num, 1) + reg[:, :, 1:2] + 0.5
        else:
            radius_s = radius_s.view(batch, self.max_num, 1) + 0.5
            azimuth_s = azimuth_s.view(batch, self.max_num, 1) + 0.5
        

        # vel
        if vel is not None:
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            vr_polar, va_polar = vel.split([1, 1], 2)

        radius_s = radius_s.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        azimuth_s = azimuth_s.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        
        
        radius_on_center = radius_s
        if self.radius_center == True:
            radius_on_center = radius_s_idx[:,:,None] * (self.voxel_size[0]*self.out_size_factor) +  self.voxel_size[0]*self.out_size_factor / 2 + self.pc_range[1]

        if self.dim_div == 'div':
            dim[:,:,0:1] *= radius_on_center # chgd
            dim[:,:,1:2] *= radius_on_center # chgd
        elif self.dim_div == 'mul':
            dim[:,:,0:1] /= radius_on_center # chgd
            dim[:,:,1:2] /= radius_on_center # chgd
        else:
            pass
        
        azimuth_on_center = azimuth_s
        if self.azimuth_center == True:
            res_azi = torch.ones_like(azimuth_s) * self.voxel_size[1]*self.out_size_factor / 2 
            res_azi[azimuth_s < 0] = -res_azi[azimuth_s < 0]
            azimuth_on_center = azimuth_s_idx[:,:,None]*(self.voxel_size[1]*self.out_size_factor) \
                + res_azi + self.pc_range[1]
                
        cart_rot = rot + azimuth_on_center 
        cart_rot = adjust_within_range(cart_rot)
        xs = torch.cos(azimuth_s) * radius_s
        ys = torch.sin(azimuth_s) * radius_s
        
        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, cart_rot], dim=2)
        else:  # exist velocity, nuscene format
            vr_xy = vr_polar
            
            if self.vel_div == "div":
                va_xy = va_polar * radius_s
            elif self.vel_div == "mul":
                va_xy = va_polar / radius_s
            else:
                va_xy = va_polar
            vx = torch.cos(azimuth_s) * vr_xy - torch.sin(azimuth_s) * va_xy
            vy = torch.sin(azimuth_s) * vr_xy + torch.cos(azimuth_s) * va_xy
            vel = torch.cat([vx, vy], 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, cart_rot, vel], dim=2)
            
        #breakpoint()

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
    
    
    
    
