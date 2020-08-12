from math import ceil, log

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import normal_init, ConvModule, bias_init_with_prob
from mmcv.ops import CornerPool

import numpy as np
import cv2
import math

from mmdet.core import multi_apply, multiclass_nms, distance2bbox, force_fp32
from ..builder import build_loss, HEADS
from ..utils import gaussian_radius, gen_gaussian_target
from .corner_head import CornerHead


INF = 1e8


class BiCascadeCornerPool(nn.Module):
    """BiCascadial Corner Pooling Module (TopLeft, BottomRight, etc.)

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

        self.pool1 = CornerPool(direction[0])
        self.pool2 = CornerPool(direction[1])
        # pool1 conv
        self.look1_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.direct1_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.aftconcat1_conv = ConvModule(
            feat_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)            
        # pool2 conv
        self.look2_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.direct2_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.aftconcat2_conv = ConvModule(
            feat_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        # main and direct conv
        self.aftconcat_conv = ConvModule(
            feat_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        self.direct_conv = ConvModule(
            feat_channels, feat_channels, 1, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        self.out_conv = ConvModule(
            feat_channels, out_channels, 3, paddng=1, norm_cfg=norm_cfg, act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tensor): Input feature of BiCornerPool.

        Returns:
            conv2 (tensor): Output feature of BiCornerPool.
        """
        # pool 1
        look_1 = self.look1_conv(x)
        look_1 = self.pool2(look_1)
        direct_1 = self.direct1_conv(x)
        main_1 = self.aftconcat1_conv(look_1 + direct_1)
        main_1 = self.pool1(main_1)

        # pool 2
        look_2 = self.look2_conv(x)
        look_2 = self.pool1(look_2)
        direct_2 = self.direct2_conv(x)
        main_2 = self.aftconcat2_conv(look_2 + direct_2)
        main_2 = self.pool2(main_2)

        # pool1 + pool2
        main = self.aftconcat_conv(main_1 + main_2)
        direct = self.direct_conv(x)

        result = self.relu(main + direct)
        result = self.out_conv(result)
        return result


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
                    center_heats[-1][img_id:img_id + 1, :],
                    center_offs[-1][img_id:img_id + 1, :],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list

    def _get_bboxes_single(self,
                           tl_heat,
                           br_heat,
                           tl_emb,
                           br_emb,
                           tl_off,
                           br_off,
                           img_meta,
                           ct_heat,
                           ct_off,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            tl_heat=tl_heat.sigmoid(),
            br_heat=br_heat.sigmoid(),
            tl_off=tl_off,
            br_off=br_off,
            ct_heat=ct_heat,
            ct_off=ct_off,
            tl_emb=tl_emb,
            br_emb=br_emb,
            img_meta=img_meta,
            k=self.test_cfg.corner_topk,
            kernel=self.test_cfg.local_maximum_kernel,
            distance_threshold=self.test_cfg.distance_threshold)

        if rescale:
            batch_bboxes /= img_meta['scale_factor']

        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view([-1, 1])
        clses = batch_clses.view([-1, 1])

        idx = scores.argsort(dim=0, descending=True)
        bboxes = bboxes[idx].view([-1, 4])
        scores = scores[idx].view(-1)
        clses = clses[idx].view(-1)

        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = (detections[:, -1] > -0.1)
        detections = detections[keepinds]
        labels = clses[keepinds]

        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels,
                                                  self.test_cfg)

        return detections, labels

   def decode_heatmap(self,
                       tl_heat,
                       br_heat,
                       tl_off,
                       br_off,
                       ct_heat,
                       ct_off,
                       tl_emb=None,
                       br_emb=None,
                       tl_centripetal_shift=None,
                       br_centripetal_shift=None,
                       img_meta=None,
                       k=100,
                       kernel=3,
                       distance_threshold=0.5,
                       num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            tl_heat (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_heat (Tensor): Bottom-right corner heatmap for current level
                with shape (N, num_classes, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            tl_emb (Tensor | None): Top-left corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            br_emb (Tensor | None): Bottom-right corner embedding for current
                level with shape (N, corner_emb_channels, H, W).
            tl_centripetal_shift (Tensor | None): Top-left centripetal shift
                for current level with shape (N, 2, H, W).
            br_centripetal_shift (Tensor | None): Bottom-right centripetal
                shift for current level with shape (N, 2, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            k (int): Get top k corner keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            distance_threshold (float): Distance threshold. Top-left and
                bottom-right corner keypoints with feature distance less than
                the threshold will be regarded as keypoints from same object.
            num_dets (int): Num of raw boxes before doing nms.

        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = (
            tl_centripetal_shift is not None
            and br_centripetal_shift is not None)
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = tl_heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        # 1. 使用默认3x3的卷积核筛选出所有局部（在卷积核范围内）最大的点作为候选点，减少计算量
        # 2. 对heatmap进行topk排列，得到最可能为顶点的k个
        # 3. 将所有的tl和br进行全排列的配对，分别计算每个框对应的匹配分数、特征
        tl_heat = self._local_maximum(tl_heat, kernel=kernel)
        br_heat = self._local_maximum(br_heat, kernel=kernel)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(tl_heat, k=k)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(br_heat, k=k)

        tl_ys = tl_ys.view(batch, k, 1).expand(batch, k, k)
        tl_xs = tl_xs.view(batch, k, 1).expand(batch, k, k)
        br_ys = br_ys.view(batch, 1, k).expand(batch, k, k)
        br_xs = br_xs.view(batch, 1, k).expand(batch, k, k)

        tl_off = self._transpose_and_gather_feat(tl_off, tl_inds)
        tl_off = tl_off.view(batch, k, 1, 2)
        br_off = self._transpose_and_gather_feat(br_off, br_inds)
        br_off = br_off.view(batch, 1, k, 2)

        tl_xs = tl_xs + tl_off[..., 0]
        tl_ys = tl_ys + tl_off[..., 1]
        br_xs = br_xs + br_off[..., 0]
        br_ys = br_ys + br_off[..., 1]

        if with_centripetal_shift:
            tl_centripetal_shift = self._transpose_and_gather_feat(
                tl_centripetal_shift, tl_inds).view(batch, k, 1, 2).exp()
            br_centripetal_shift = self._transpose_and_gather_feat(
                br_centripetal_shift, br_inds).view(batch, 1, k, 2).exp()

            tl_ctxs = tl_xs + tl_centripetal_shift[..., 0]
            tl_ctys = tl_ys + tl_centripetal_shift[..., 1]
            br_ctxs = br_xs - br_centripetal_shift[..., 0]
            br_ctys = br_ys - br_centripetal_shift[..., 1]

        # all possible boxes based on top k corners (ignoring class)
        tl_xs *= (inp_w / width)
        tl_ys *= (inp_h / height)
        br_xs *= (inp_w / width)
        br_ys *= (inp_h / height)

        if with_centripetal_shift:
            tl_ctxs *= (inp_w / width)
            tl_ctys *= (inp_h / height)
            br_ctxs *= (inp_w / width)
            br_ctys *= (inp_h / height)

        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]

        tl_xs -= x_off
        tl_ys -= y_off
        br_xs -= x_off
        br_ys -= y_off

        # 将小于等于0的参数置零
        tl_xs *= tl_xs.gt(0.0).type_as(tl_xs)
        tl_ys *= tl_ys.gt(0.0).type_as(tl_ys)
        br_xs *= br_xs.gt(0.0).type_as(br_xs)
        br_ys *= br_ys.gt(0.0).type_as(br_ys)

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)  # shape (batch, k, k, 4)
        area_bboxes = ((br_xs - tl_xs) * (br_ys - tl_ys)).abs()

        # 生成ct点的实际位置，格式和bboxes位置对应
        TODO: 判断这里是否前后匹配，且能映射到原图对应的位置
        ct_heat = self._local_maximum(ct_heat, kernal=kernel)
        ct_scores, ct_inds, ct_class, ct_ys, ct_xs = self._topk(ct_heat, k=k)
        TODO: 判断是否需要按github上CenterNet进行修改
        # ct_ys = ct_ys.view(batch, k, 1)
        # ct_xs = ct_xs.view(batch, k, 1)
        ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
        ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)
        ct_off = self._transpose_and_gather_feat(ct_off, ct_inds)
        # ct_off = ct_off.view(batch, k, 1, 2)
        ct_off = ct_off.view(batch, 1, k, 2)
        ct_xs = ct_xs + ct_off[..., 0]
        ct_ys = ct_ys + ct_off[..., 0]

        if with_centripetal_shift:
            print('你这个写的不好使了，回来改center_triplets_head的976行')
        
        ct_xs *= (inp_w / width)
        ct_ys *= (inp_h / height)
        ct_xs -= x_off
        xt_ys -= y_off
        ct_xs *= ct_xs.gt(0.0).type_as(ct_xs) 
        ct_ys *= ct_ys.gt(0.0).type_as(ct_ys)

        if with_centripetal_shift:
            tl_ctxs -= x_off
            tl_ctys -= y_off
            br_ctxs -= x_off
            br_ctys -= y_off

            tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
            tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
            br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
            br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)

            ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys),
                                    dim=3)
            area_ct_bboxes = ((br_ctxs - tl_ctxs) * (br_ctys - tl_ctys)).abs()

            rcentral = torch.zeros_like(ct_bboxes)
            # magic nums from paper section 4.1
            mu = torch.ones_like(area_bboxes) / 2.4
            mu[area_bboxes > 3500] = 1 / 2.1  # large bbox have smaller mu

            bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
            rcentral[..., 0] = bboxes_center_x - mu * (bboxes[..., 2] -
                                                       bboxes[..., 0]) / 2
            rcentral[..., 1] = bboxes_center_y - mu * (bboxes[..., 3] -
                                                       bboxes[..., 1]) / 2
            rcentral[..., 2] = bboxes_center_x + mu * (bboxes[..., 2] -
                                                       bboxes[..., 0]) / 2
            rcentral[..., 3] = bboxes_center_y + mu * (bboxes[..., 3] -
                                                       bboxes[..., 1]) / 2
            area_rcentral = ((rcentral[..., 2] - rcentral[..., 0]) *
                             (rcentral[..., 3] - rcentral[..., 1])).abs()
            dists = area_ct_bboxes / area_rcentral

            tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (
                ct_bboxes[..., 0] >= rcentral[..., 2])
            tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (
                ct_bboxes[..., 1] >= rcentral[..., 3])
            br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (
                ct_bboxes[..., 2] >= rcentral[..., 2])
            br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (
                ct_bboxes[..., 3] >= rcentral[..., 3])

        if with_embedding:
            tl_emb = self._transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, k, 1)
            br_emb = self._transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, k)
            dists = torch.abs(tl_emb - br_emb)

        tl_scores = tl_scores.view(batch, k, 1).expand(batch, k, k)
        br_scores = br_scores.view(batch, 1, k).expand(batch, k, k)

        scores = (tl_scores + br_scores) / 2  # scores for all possible boxes

        # tl and br should have same class
        tl_clses = tl_clses.view(batch, k, 1).expand(batch, k, k)
        br_clses = br_clses.view(batch, 1, k).expand(batch, k, k)
        cls_inds = (tl_clses != br_clses)

        # reject boxes based on distances
        dist_inds = dists > distance_threshold

        # reject boxes based on widths and heights
        width_inds = (br_xs <= tl_xs)
        height_inds = (br_ys <= tl_ys)

        scores[cls_inds] = -1
        scores[width_inds] = -1
        scores[height_inds] = -1
        scores[dist_inds] = -1
        if with_centripetal_shift:
            scores[tl_ctx_inds] = -1
            scores[tl_cty_inds] = -1
            scores[br_ctx_inds] = -1
            scores[br_cty_inds] = -1

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = self._gather_feat(bboxes, inds)

        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = self._gather_feat(clses, inds).float()

        tl_scores = tl_scores.contiguous().view(batch, -1, 1)
        tl_scores = _gather_feat(tl_scores, inds).float()
        br_scores = br_scores.contiguous().view(batch, -1, 1)
        br_scores = _gather_feat(br_scores, inds).float()

        ct_xs = ct_xs[:, 0, :]
        ct_ys = ct_ys[:, 0, :]

        center = torch.cat([ct_xs.unsqueeze(2), ct_ys.unsqueeze(2), ct_clses.float().unsqueeze(2), 
                            ct_scores.unsqueeze(2)], dim=2)
        detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)

        return detections, center
        # return bboxes, scores, clses

    def _create_ct_bboxes(self, bboxes, n=3):
        """根据bbox生成中心区域框

        Args：
            bboxes (Tensor [batch, k, k, 4]): 输入的所有默认候选框各顶点坐标
            n (int): 分割的中心框占整体框的比例，default = 3
        
        Returns：
            ct_bboxes (Tensor [batch, k, k, 4]): 计算得到的中心框各顶点坐标
        """
        batch, k, k_b, c = bboxes.shape()
        assert: (k = k_b and c = 4), '输入bbox尺寸问题，检查_center_triplets_head的1077行'

        ct_bboxes = torch.zeros_like(bboxes)
        ct_bboxes[..., 0] += ((n+1)*bboxes[..., 0] + (n-1)*bboxes[..., 2]) / (2*n)
        ct_bboxes[..., 1] += ((n+1)*bboxes[..., 1] + (n-1)*bboxes[..., 3]) / (2*n)
        ct_bboxes[..., 2] += ((n-1)*bboxes[..., 0] + (n+1)*bboxes[..., 2]) / (2*n)
        ct_bboxes[..., 3] += ((n+1)*bboxes[..., 1] + (n+1)*bboxes[..., 3]) / (2*n)

        return ct_bboxes

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.
        Similar with NMS.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
     
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep
    
    def _topk(self, scores, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = (topk_inds / (height * width)).int()
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.

        Returns:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat
    
    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

