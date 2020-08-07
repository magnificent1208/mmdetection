from math import ceil, log

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import normal_init
from mmcv.ops import CornerPool

import numpy as np
import cv2
import math

from mmdet.core import multi_apply, multiclass_nms, distance2bbox, force_fp32
from ..builder import build_loss, HEADS
from ..utils import gaussian_radius, gen_gaussian_target
from .corner_head import CornerHead

# from ..utils import bias_init_with_prob, Scale, ConvModule

INF = 1e8


class CascadeCornerPool(nn.Module):
    """Cascadial Corner Pooling Module (TopLeft, BottomRight, etc.)

    Args:
        in_channels (int): Input channels of module.
        out_channels (int): Output channels of module.
        feat_channels (int): Feature channels of module.
        directions (list[str]): Directions of two CornerPools.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 direction,
                 feat_channels=128,
                 out_channels=128,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(CascadeCornerPool, self).__init__()

        self.direction_dict = {
            'top': ['left', 'top'],
            'bottom': ['right', 'bottom'],
            'left': ['top', 'left'],
            'right': ['bottom', 'right']
        }
        self.tunnel1_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.tunnel2_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.aftconcat_conv = ConvModule(
            feat_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.tunnel1_pool = CornerPool(self.direction_dict[direction][0])
        self.main_pool = CornerPool(self.direction_dict[direction][1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tensor): Input feature of BiCornerPool.

        Returns:
            conv2 (tensor): Output feature of BiCornerPool.
        """
        tunnel_1 = self.tunnel1_conv(x)
        tunnel_1 = self.relu(tunnel_1)
        tunnel_1 = self.tunnel1_pool(tunnel_1)
        tunnel_2 = self.tunnel2_conv(x)

        aftconcat_conv = self.aftconcat_conv(tunnel_1 + tunnel_2)
        main = self.main_pool(afterconcat_conv)
        return main


class CenterPool(nn.Module):
    """Center Pooling Module. Pooling four times in every directions.

    Args:
        in_channels (int): Input channels of module.
        out_channels (int): Output channels of module.
        feat_channels (int): Feature channels of module.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """
    def __init__(self,
                 in_channels,
                 feat_channels=128,
                 out_channels=128,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(CenterPool, self).__init__()
        self.direction1_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.direction2_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.aftpool_conv = ConvModule(
            feat_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)
        
        self.conv1 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(
            in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.left_pool = CornerPool('left')
        self.right_pool = CornerPool('right')
        self.top_pool = CornerPool('top')
        self.bottom_pool = CornerPool('bottom')

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        direction1_conv = self.direction1_conv(x)
        direction2_conv = self.direction2_conv(x)
        direction1_feat = self.right_pool(self.left_pool(direction1_conv))
        direction2_feat = self.bottom_pool(self.top_pool(direction2_conv))
        aftpool_conv = self.aftpool_conv(direction1_feat + direction2_feat)
        conv1 = self.conv1(x)
        relu = self.relu(aftpool_conv + conv1)
        conv2 = self.conv2(relu)
        return conv2


@HEADS.register_module
class CenterHead(CornerHead):
    """Head of CenterNet: Chou maggie

    Args:
        num_classes (int): Number of detect classes. (Including background)
        in_channels (int):
        num_feat_levels (int): Number of channels from backbone.
            2 for HourglassNet-104 (Default)
            1 for HourglassNet-52
        corner_emb_channels (int): Channel of embedding vector. Defaulat: 1.

    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_feat_levels=2,
                 corner_emb_channels=1,
                 train_cfg=None,
                 test_cfg=None,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_embedding=dict(
                     type='AssociativeEmbeddingLoss',
                     pull_weight=0.25,
                     push_weight=0.25),
                 loss_offset=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1)):
        super(CenterHead, self).__init__()

        self.num_classes = num_classes
        # self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.corner_emb_channels = corner_emb_channels
        self.with_corner_emb = self.corner_emb_channels > 0
        self.corner_offset_channels = 2
        TODO: set center offset channels
        self.center_offset_channels = 2
        self.num_feat_levels = num_feat_levels
        self.loss_heatmap = build_loss(
            loss_heatmap) if loss_heatmap is not None else None
        self.loss_embedding = build_loss(
            loss_embedding) if loss_embedding is not None else None
        self.loss_offset = build_loss(
            loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()
    
    def _init_corner_kpt_layers(self):
        """Initialize corner keypoint layers.

        Including corner heatmap branch and corner offset branch. Each branch
        has two parts: prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_pool, self.br_pool = nn.ModuleList(), nn.ModuleList()
        self.tl_heat, self.br_heat = nn.ModuleList(), nn.ModuleList()
        self.tl_off, self.br_off = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.tl_pool.append(
                CascadeCornerPool(
                    self.in_channels, ['top', 'left'],
                    out_channels=self.in_channels))
            self.br_pool.append(
                CascadeCornerPool(
                    self.in_channels, ['bottom', 'right'],
                    out_channels=self.in_channels))

            self.tl_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.br_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))

            self.tl_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))
            self.br_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))
    
    def _init_center_kpt_layers(self):
        """Initialize center keypoint layers. Use CenterPool.

        Including center heatmap branch and center offset branch.
        """
        self.center_pool = nn.ModuleList()
        self.center_heat = nn.ModuleList()
        self.center_off = nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.center_pool.append(
                CenterPool(
                    self.in_channels,
                    out_channels=self.in_channels))
            self.center_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.center_off.append(
                self._make_layers(
                    out_channels=self.center_offset_channels,
                    in_channels=self.in_channels))
        
    def _init_layers(self):
        """Initialize layers for CenterHead.
        """
        self._init_corner_kpt_layers()
        if self.with_corner_emb:
            self._init_corner_emb_layers()
        self._init_center_kpt_layers()

    def init_weights(self):
        """Initialize weights of the head.
        """
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.num_feat_levels):
            self.tl_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.br_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.center_heat[i][-1].conv.bias.data.fill_(bias_init)
    
    TODO: add center information to forward output
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of corner heatmaps, offset heatmaps and
            embedding heatmaps.
                - tl_heats (list[Tensor]): Top-left corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - br_heats (list[Tensor]): Bottom-right corner heatmaps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - tl_embs (list[Tensor] | list[None]): Top-left embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - br_embs (list[Tensor] | list[None]): Bottom-right embedding
                  heatmaps for all levels, each is a 4D-tensor or None.
                  If not None, the channels number is corner_emb_channels.
                - tl_offs (list[Tensor]): Top-left offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
                - br_offs (list[Tensor]): Bottom-right offset heatmaps for all
                  levels, each is a 4D-tensor. The channels number is
                  corner_offset_channels.
        """
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind, return_pool=False):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.
            return_pool (bool): Return corner pool feature or not.

        Returns:
            tuple[Tensor]: A tuple of CornerHead's output for current feature
            level. Containing the following Tensors:

                - tl_heat (Tensor): Predicted top-left corner heatmap.
                - br_heat (Tensor): Predicted bottom-right corner heatmap.
                - tl_emb (Tensor | None): Predicted top-left embedding heatmap.
                  None for `self.with_corner_emb == False`.
                - br_emb (Tensor | None): Predicted bottom-right embedding
                  heatmap. None for `self.with_corner_emb == False`.
                - tl_off (Tensor): Predicted top-left offset heatmap.
                - br_off (Tensor): Predicted bottom-right offset heatmap.
                - tl_pool (Tensor): Top-left corner pool feature. Not must
                  have.
                - br_pool (Tensor): Bottom-right corner pool feature. Not must
                  have.
        """
        tl_pool = self.tl_pool[lvl_ind](x)
        tl_heat = self.tl_heat[lvl_ind](tl_pool)
        br_pool = self.br_pool[lvl_ind](x)
        br_heat = self.br_heat[lvl_ind](br_pool)

        tl_emb, br_emb = None, None
        if self.with_corner_emb:
            tl_emb = self.tl_emb[lvl_ind](tl_pool)
            br_emb = self.br_emb[lvl_ind](br_pool)

        tl_off = self.tl_off[lvl_ind](tl_pool)
        br_off = self.br_off[lvl_ind](br_pool)

        center_pool = self.center_pool[lvl_ind](x)
        center_heat = self.center_heat[lvl_ind](center_pool)
        center_off = self.center_off[lvl_ind](center_heat)

        result_list = [tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off]
        if return_pool:
            result_list.append(tl_pool)
            result_list.append(br_pool)
            result_list.append(center_pool)
        
        result_list.append(center_heat)
        result_list.append(center_off)

        return result_list

    def get_targets(self,
                    gt_bboxes,
                    gt_labels,
                    feat_shape,
                    img_shape,
                    with_corner_emb=False,
                    with_guiding_shift=False,
                    with_centripetal_shift=False):
        """Generate corner and center targets.

        Including corner heatmap, corner offset, center heatmap, center offset.

        Optional: corner embedding, corner guiding shift, centripetal shift.

        For CenterNet, we generate corner heatmap, corner offset, corner
        embedding and center heatmap, center offset from this function.

        For CentripetalNet, we generate corner heatmap, corner offset, guiding
        shift and centripetal shift from this function.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].
            with_corner_emb (bool): Generate corner embedding target or not.
                Default: False.
            with_guiding_shift (bool): Generate guiding shift target or not.
                Default: False.
            with_centripetal_shift (bool): Generate centripetal shift target or
                not. Default: False.

        Returns:
            dict: Ground truth of corner heatmap, corner offset, corner
            embedding, guiding shift and centripetal shift. Containing the
            following keys:

                - topleft_heatmap (Tensor): Ground truth top-left corner
                  heatmap.
                - bottomright_heatmap (Tensor): Ground truth bottom-right
                  corner heatmap.
                - topleft_offset (Tensor): Ground truth top-left corner offset.
                - bottomright_offset (Tensor): Ground truth bottom-right corner
                  offset.
                - corner_embedding (list[list[list[int]]]): Ground truth corner
                  embedding. Not must have.
                - topleft_guiding_shift (Tensor): Ground truth top-left corner
                  guiding shift. Not must have.
                - bottomright_guiding_shift (Tensor): Ground truth bottom-right
                  corner guiding shift. Not must have.
                - topleft_centripetal_shift (Tensor): Ground truth top-left
                  corner centripetal shift. Not must have.
                - bottomright_centripetal_shift (Tensor): Ground truth
                  bottom-right corner centripetal shift. Not must have.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)

        gt_tl_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_br_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_tl_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_br_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])

        gt_center_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_center_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])

        if with_corner_emb:
            match = []

        for batch_id in range(batch_size):
            # Ground truth of corner embedding per image is a list of coord set
            corner_match = []
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                # Use coords in the feature level to generate ground truth
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                # Int coords on feature map/ground truth tensor
                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))
                center_x_idx = int(scale_center_x)
                center_y_idx = int(scale_center_y)

                # Generate gaussian heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width),
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                gt_tl_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_tl_heatmap[batch_id, label], [left_idx, top_idx],
                    radius)
                gt_br_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_br_heatmap[batch_id, label], [right_idx, bottom_idx],
                    radius)
                gt_center_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_center_heatmap[batch_id, label], [center_x_idx, center_y_idx],
                    radius)

                # Generate corner offset
                left_offset = scale_left - left_idx
                top_offset = scale_top - top_idx
                right_offset = scale_right - right_idx
                bottom_offset = scale_bottom - bottom_idx
                center_x_offset = scale_center_x - center_x_idx
                center_y_offset = scale_center_y - center_y_idx

                gt_tl_offset[batch_id, 0, top_idx, left_idx] = left_offset
                gt_tl_offset[batch_id, 1, top_idx, left_idx] = top_offset
                gt_br_offset[batch_id, 0, bottom_idx, right_idx] = right_offset
                gt_br_offset[batch_id, 1, bottom_idx, right_idx] = bottom_offset
                gt_center_offset[batch_id, 0, center_x_idx, center_y_idx] = center_x_offset
                gt_center_offset[batch_id, 1, center_x_idx, center_y_idx] = center_y_offset

                # Generate corner embedding
                if with_corner_emb:
                    corner_match.append([[top_idx, left_idx],
                                         [bottom_idx, right_idx]])
                # Generate guiding shift
                if with_guiding_shift:
                    gt_tl_guiding_shift[batch_id, 0, top_idx,
                                        left_idx] = scale_center_x - left_idx
                    gt_tl_guiding_shift[batch_id, 1, top_idx,
                                        left_idx] = scale_center_y - top_idx
                    gt_br_guiding_shift[batch_id, 0, bottom_idx,
                                        right_idx] = right_idx - scale_center_x
                    gt_br_guiding_shift[
                        batch_id, 1, bottom_idx,
                        right_idx] = bottom_idx - scale_center_y
                # Generate centripetal shift
                if with_centripetal_shift:
                    gt_tl_centripetal_shift[batch_id, 0, top_idx,
                                            left_idx] = log(scale_center_x -
                                                            scale_left)
                    gt_tl_centripetal_shift[batch_id, 1, top_idx,
                                            left_idx] = log(scale_center_y -
                                                            scale_top)
                    gt_br_centripetal_shift[batch_id, 0, bottom_idx,
                                            right_idx] = log(scale_right -
                                                             scale_center_x)
                    gt_br_centripetal_shift[batch_id, 1, bottom_idx,
                                            right_idx] = log(scale_bottom -
                                                             scale_center_y)

            if with_corner_emb:
                match.append(corner_match)

        target_result = dict(
            topleft_heatmap=gt_tl_heatmap,
            topleft_offset=gt_tl_offset,
            bottomright_heatmap=gt_br_heatmap,
            bottomright_offset=gt_br_offset,
            center_heatmap=gt_center_heatmap,
            center_offset=gt_center_offset)

        if with_corner_emb:
            target_result.update(corner_embedding=match)
        if with_guiding_shift:
            target_result.update(
                topleft_guiding_shift=gt_tl_guiding_shift,
                bottomright_guiding_shift=gt_br_guiding_shift)
        if with_centripetal_shift:
            target_result.update(
                topleft_centripetal_shift=gt_tl_centripetal_shift,
                bottomright_centripetal_shift=gt_br_centripetal_shift)

        return target_result

    def loss(self,
             tl_heats,
             br_heats,
             tl_embs,
             br_embs,
             tl_offs,
             br_offs,
             gt_bboxes,
             gt_labels,
             img_metas,
             center_heats,
             center_offs,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - pull_loss (list[Tensor]): Part one of AssociativeEmbedding
                  losses of all feature levels.
                - push_loss (list[Tensor]): Part two of AssociativeEmbedding
                  losses of all feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
        """
        targets = self.get_targets(
            gt_bboxes,
            gt_labels,
            tl_heats[-1].shape,
            img_metas[0]['pad_shape'],
            with_corner_emb=self.with_corner_emb)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]

        corner_det_losses, pull_losses, push_losses, corner_off_losses, center_det_loss, center_off_loss 
            = multi_apply(self.loss_single, tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                          br_offs, center_heats, center_offs, mlvl_targets)
        
        loss_dict = dict(corner_det_loss=corner_det_losses, 
                         corner_off_loss=corner_off_losses,
                         center_det_loss=center_det_loss,
                         center_det_loss=center_det_loss)
        if self.with_corner_emb:
            loss_dict.update(pull_loss=pull_losses, push_loss=push_losses)
        return loss_dict

    def loss_single(self, tl_hmp, br_hmp, tl_emb, br_emb, tl_off, br_off,
                    center_hmp, center_off, targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            center_hmp (Tensor): Center heatmap for current level with
                shape (N, num_classes, H, W).
            center_off (Tensor): Center offset for current level with
                shape (N, num_classes, H, W).
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's differnet branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - pull_loss (Tensor): Part one of AssociativeEmbedding loss.
                - push_loss (Tensor): Part two of AssociativeEmbedding loss.
                - off_loss (Tensor): Corner offset loss.
                - center_
        """
        gt_tl_hmp = targets['topleft_heatmap']
        gt_br_hmp = targets['bottomright_heatmap']
        gt_tl_off = targets['topleft_offset']
        gt_br_off = targets['bottomright_offset']
        gt_center_hmp = targets['center_heatmap']
        gt_center_off = targets['center_offset']
        gt_embedding = targets['corner_embedding']

        # Detection loss for corner
        tl_det_loss = self.loss_heatmap(
            tl_hmp.sigmoid(),
            gt_tl_hmp,
            avg_factor=max(1, gt_tl_hmp.eq(1).sum()))
        br_det_loss = self.loss_heatmap(
            br_hmp.sigmoid(),
            gt_br_hmp,
            avg_factor=max(1, gt_br_hmp.eq(1).sum()))
        corener_det_loss = (tl_det_loss + br_det_loss) / 2.0

        # Detection loss for center
        center_det_loss = self.loss_heatmap(
            center_hmp.sigmoid(),
            gt_center_hmp,
            avg_factor=max(1, gt_center_hmp.eq(1).sum()))
        
        # AssociativeEmbedding loss
        if self.with_corner_emb and self.loss_embedding is not None:
            pull_loss, push_loss = self.loss_embedding(tl_emb, br_emb,
                                                       gt_embedding)
        else:
            pull_loss, push_loss = None, None

        # Offset loss
        # We only compute the offset loss at the real corner position.
        # The value of real corner would be 1 in heatmap ground truth.
        # The mask is computed in class agnostic mode and its shape is
        # batch * 1 * width * height.
        tl_off_mask = gt_tl_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_tl_hmp)
        br_off_mask = gt_br_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_br_hmp)
        tl_off_loss = self.loss_offset(
            tl_off,
            gt_tl_off,
            tl_off_mask,
            avg_factor=max(1, tl_off_mask.sum()))
        br_off_loss = self.loss_offset(
            br_off,
            gt_br_off,
            br_off_mask,
            avg_factor=max(1, br_off_mask.sum()))
        
        center_off_mask = gt_center_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_center_hmp)
        center_off_loss = self.loss_offset(
            center_off,
            gt_center_off,
            center_off_mask,
            avg_factor=max(1, br_off_mask.sum()))
        

        corner_off_loss = (tl_off_loss + br_off_loss) / 2.0

        return corner_det_loss, pull_loss, push_loss, corner_off_loss, 
               center_det_loss, center_off_loss

    def get_bboxes(self,
                   tl_heats,
                   br_heats,
                   tl_embs,
                   br_embs,
                   tl_offs,
                   br_offs,
                   img_metas,
                   center_heats,
                   center_offs,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == center_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    tl_heats[-1][img_id:img_id + 1, :],
                    br_heats[-1][img_id:img_id + 1, :],
                    tl_embs[-1][img_id:img_id + 1, :],
                    br_embs[-1][img_id:img_id + 1, :],
                    tl_offs[-1][img_id:img_id + 1, :],
                    br_offs[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list
