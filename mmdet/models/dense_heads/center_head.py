import math

import torch
import torch.nn as nn
import numpy as np
import cv2

from mmcv.cnn import normal_init
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmdet.core import multi_apply, multiclass_nms, distance2bbox

from ..builder import build_loss, HEADS
from ..utils import gaussian_radius, gen_gaussian_target

INF = 1e8


@HEADS.register_module()
class CenterHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=1,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_cross = False,
                 loss_hm = dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_wh = dict(
                     type="SmoothL1Loss",
                     loss_weight=0.1),
                 loss_offset=dict(
                     type='SmoothL1Loss', 
                     beta=1.0, 
                     loss_weight=1),
                 loss_rot=dict(
                     type='SmoothL1Loss',
                     loss_weight=1),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 K=100,
                 **kwargs):
        super(CenterHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.featmap_sizes = None
        self.loss_hm = build_loss(loss_hm)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_rot = build_loss(loss_rot)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.use_cross = use_cross
        self.gaussian_iou = 0.7
        self.K = K

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.wh_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
        self.rot_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.wh_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.offset_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.rot_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.center_hm = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1, bias=True)
        self.center_wh = nn.Conv2d(self.feat_channels, 2, 3, padding=1, bias=True)
        self.center_offset = nn.Conv2d(self.feat_channels, 2, 3, padding=1, bias=True)
        self.center_rot = nn.Conv2d(self.feat_channels, 1, 3, padding=1, bias=True)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):

        self.center_hm.bias.data.fill_(-2.19)
        nn.init.constant_(self.center_wh.bias, 0)
        nn.init.constant_(self.center_offset.bias, 0)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        wh_feat = x
        offset_feat = x
        rot_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.center_hm(cls_feat)

        for wh_layer in self.wh_convs:
            wh_feat = wh_layer(wh_feat)
        wh_pred = self.center_wh(wh_feat)
        
        for offset_layer in self.offset_convs:
            offset_feat = offset_layer(offset_feat)
        offset_pred = self.center_offset(offset_feat)

        for rot_layer in self.rot_convs:
            rot_feat = rot_layer(rot_feat)
        rot_pred = self.center_rot(rot_feat)
        
        return cls_score, wh_pred, offset_pred, rot_pred

    @force_fp32(apply_to=('cls_scores', 'wh_preds', 'offset_preds', 'rot_preds'))
    def loss(self,
             cls_scores,
             wh_preds,
             offset_preds,
             rot_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(wh_preds) == len(offset_preds) == len(rot_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        self.featmap_sizes = featmap_sizes
        all_level_points = self.get_points(featmap_sizes, offset_preds[0].dtype,
                                            offset_preds[0].device)

        self.tensor_dtype = offset_preds[0].dtype
        self.tensor_device = offset_preds[0].device
        heatmaps, wh_targets, offset_targets, rot_targets = self.center_target(gt_bboxes, gt_labels, img_metas, all_level_points)

        num_imgs = cls_scores[0].size(0) # batch_size

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ] # cls_scores(num_levels, batch_size, 80, h, w)  => (num_levels, batch_size * w * h, 80)
        flatten_wh_preds = [
            wh_pred.permute(0, 2, 3, 1).reshape(-1, 2) # batchsize, h, w, 2 => batchsize * h * w, 2
            for wh_pred in wh_preds
        ]
        flatten_offset_preds = [
            offset_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for offset_pred in offset_preds
        ]
        flatten_rot_preds = [
            rot_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for rot_pred in rot_preds
        ]
       
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_wh_preds = torch.cat(flatten_wh_preds)
        flatten_offset_preds = torch.cat(flatten_offset_preds)
        flatten_rot_preds = torch.cat(flatten_rot_preds)
       
        # targets
        flatten_heatmaps = torch.cat(heatmaps)
        flatten_wh_targets = torch.cat(wh_targets) # torch.Size([all_level_points, 2])
        flatten_offset_targets = torch.cat(offset_targets)
        flatten_rot_targets = torch.cat(rot_targets)
        
        # center_inds = flatten_heatmaps[..., 0].nonzero().reshape(-1) 
        center_inds = flatten_wh_targets[..., 0].nonzero().reshape(-1)
        num_center = len(center_inds)

        flatten_cls_scores = torch.clamp(flatten_cls_scores.sigmoid_(), min=1e-4, max=1-1e-4)
        loss_hm = self.loss_hm(flatten_cls_scores, flatten_heatmaps)
        
        pos_wh_targets = flatten_wh_targets[center_inds]
        pos_wh_preds = flatten_wh_preds[center_inds]
        
        pos_offset_preds = flatten_offset_preds[center_inds]
        pos_offset_targets = flatten_offset_targets[center_inds]

        pos_rot_preds = flatten_rot_preds[center_inds]
        pos_rot_targets = flatten_rot_targets[center_inds]
        
        if num_center > 0:
            # loss_wh = self.loss_wh(pos_wh_preds, pos_wh_targets, avg_factor=num_center + num_imgs)
            # loss_offset = self.loss_offset(pos_offset_preds, pos_offset_targets, avg_factor=num_center + num_imgs)
            # loss_rot = self.loss_rot(pos_rot_preds, pos_rot_targets, avg_factor=num_center + num_imgs)
            loss_wh = self.loss_wh(pos_wh_preds, pos_wh_targets)
            loss_offset = self.loss_offset(pos_offset_preds, pos_offset_targets)
            loss_rot = self.loss_rot(pos_rot_preds, pos_rot_targets)
        else:
            loss_wh = pos_wh_preds.sum()
            loss_offset = pos_offset_preds.sum()
            loss_rot = pos_rot_preds.sum()
        
        # if loss_wh > 1e5 or loss_offset > 1e5 or loss_rot >1e5:
        #     loss_wh = loss_offset = loss_rot = torch.tensor([0]).cuda()
    
        return dict(
              loss_hm = loss_hm,
              loss_wh = loss_wh,
              loss_offset = loss_offset,
              loss_rot = loss_rot)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        # out: (tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off, ct_heat, ct_off)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device) # 以一定间隔取x的值
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range) # 得到featmap的所有点
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def center_target(self, gt_bboxes_list, gt_labels_list, img_metas, all_level_points):
        # import pdb; pdb.set_trace()

        assert len(self.featmap_sizes) == len(self.regress_ranges)

        # get heatmaps and targets of each image
        # heatmaps in heatmaps_list: [num_points, 80]
        # wh_targets: [num_points, 2] => [batch_size, num_points, 2]
        heatmaps_list, wh_targets_list, offset_targets_list, rot_targets_list = multi_apply(
            self.center_target_single, gt_bboxes_list, gt_labels_list, img_metas)

        # split to per img, per level
        num_points = [center.size(0) for center in all_level_points] # 每一层多少个点 all_level_points [[12414, 2], []]
        
        heatmaps_list = [heatmaps.split(num_points, 0) for heatmaps in heatmaps_list]
        wh_targets_list = [wh_targets.split(num_points, 0) for wh_targets in wh_targets_list]
        offset_targets_list = [offset_targets.split(num_points, 0) for offset_targets in offset_targets_list]
        rot_targets_list = [rot_targets.split(num_points, 0) for rot_targets in rot_targets_list]

        # concat per level image, 同一层的concat # [(batch_size，featmap_size[1]), ...)
        concat_lvl_heatmaps = []
        concat_lvl_wh_targets = []
        concat_lvl_offset_targets = []
        concat_lvl_rot_targets = []
        num_levels = len(self.featmap_sizes)
        for i in range(num_levels):
            concat_lvl_heatmaps.append(
                torch.cat([heatmaps[i] for heatmaps in heatmaps_list])) # (num_levels, batch_size * w * h, 80)
            concat_lvl_wh_targets.append(
                torch.cat(
                    [wh_targets[i] for wh_targets in wh_targets_list]))
            concat_lvl_offset_targets.append(
                torch.cat(
                    [offset_targets[i] for offset_targets in offset_targets_list]))
            concat_lvl_rot_targets.append(
                torch.cat(
                    [rot_targets[i] for rot_targets in rot_targets_list]))        
        return concat_lvl_heatmaps, concat_lvl_wh_targets, concat_lvl_offset_targets, concat_lvl_rot_targets

    def center_target_single(self, gt_bboxes, gt_labels, img_meta):
        """
        single image
        gt_bboxes:torch.Size([6, 4])
        gt_labels:torch.Size([6]) tensor([34, 34, 34, 34, 34, 34], device='cuda:0')
        featmap_sizes:(list[tuple]): Multi-level feature map sizes.
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
        """
        # transform the gt_bboxes, gt_labels to numpy
        gt_bboxes = gt_bboxes.data.cpu().numpy()
        gt_labels = gt_labels.data.cpu().numpy()
        img_h, img_w = img_meta['img_shape'][:2]
        
        num_objs = gt_labels.shape[0]

        # heatmaps [level1, level2, level3, level4, level5]
        num_levels = len(self.featmap_sizes)

        heatmaps_targets = []
        wh_targets = []
        offset_targets = []
        rot_targets = []
        # get the target shape for each image
        for i in range(num_levels):
            h, w = self.featmap_sizes[i]
            hm = np.zeros((self.cls_out_channels, h, w), dtype=np.float32)
            heatmaps_targets.append(hm)
            wh = np.zeros((h, w, 2), dtype=np.float32)
            wh_targets.append(wh)
            offset = np.zeros((h, w, 2), dtype=np.float32)
            offset_targets.append(offset)
            rot = np.zeros((h, w, 1), dtype=np.float32)
            rot_targets.append(rot)

        for k in range(num_objs):
            bbox = gt_bboxes[k]
            cls_id = gt_labels[k]
            
            origin_h, origin_w = bbox[3], bbox[2]
            max_h_w = max(origin_h, origin_w)
            # 根据max_h_w在哪一层将output设置为当前层的
            index_levels = []
            for i in range(num_levels):
                min_regress_distance, max_regress_distance = self.regress_ranges[i]
                if not self.use_cross and (max_h_w > min_regress_distance) and (max_h_w <= max_regress_distance):
                    index_levels.append(i)
                    break
                
                if self.use_cross:
                    min_regress_distance = min_regress_distance * 0.8
                    max_regress_distance = max_regress_distance * 1.3
                    if (max_h_w > min_regress_distance) and (max_h_w <= max_regress_distance):
                        index_levels.append(i)
                    
            for index_level in index_levels:
                output_h, output_w = self.featmap_sizes[index_level]
                hm = heatmaps_targets[index_level]
                wh = wh_targets[index_level]
                offset = offset_targets[index_level]
                rot = rot_targets[index_level]

                center = np.array([img_h / 2, img_w / 2], dtype=np.float32)
                scale = max(img_h, img_w) * 1.0
              
                trans_output = get_affine_transform(center, scale, 0, [output_h, output_w])
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:4] = affine_transform(bbox[2:4], trans_output)
                h, w = bbox[3], bbox[2]
                # 转换到当层
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)
                    radius = max(0, int(radius))
                    ct = np.array([bbox[0] , bbox[1]], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm[cls_id], ct_int, radius)

                    h, w = 1. * h, 1. * w
                    offset_count = ct - ct_int # h, w

                    wh[ct_int[1], ct_int[0], 0] = w 
                    wh[ct_int[1], ct_int[0], 1] = h 
                    offset[ct_int[1], ct_int[0], 0] = offset_count[0]
                    offset[ct_int[1], ct_int[0], 1] = offset_count[1]
                    rot[ct_int[1], ct_int[0]] = bbox[4]
            
                heatmaps_targets[index_level] = hm
                wh_targets[index_level] = wh
                offset_targets[index_level] = offset
                rot_targets[index_level] = rot

        flatten_heatmaps_targets = [
            hm.transpose(1, 2, 0).reshape(-1, self.cls_out_channels)
            for hm in heatmaps_targets
        ]
        heatmaps_targets = np.concatenate(flatten_heatmaps_targets, axis=0) 
        
        flatten_wh_targets = [wh.reshape(-1, 2) for wh in wh_targets]
        wh_targets = np.concatenate(flatten_wh_targets)
        
        flatten_offset_targets = [offset.reshape(-1, 2) for offset in offset_targets]
        offset_targets = np.concatenate(flatten_offset_targets)

        flatten_rot_targets = [rot.reshape(-1, 1) for rot in rot_targets]
        rot_targets = np.concatenate(flatten_rot_targets)

        # transform the heatmaps_targets, wh_targets, offset_targets into tensor
        heatmaps_targets = torch.from_numpy(np.stack(heatmaps_targets))
        heatmaps_targets = torch.as_tensor(heatmaps_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        wh_targets = torch.from_numpy(np.stack(wh_targets))
        wh_targets = torch.as_tensor(wh_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        offset_targets = torch.from_numpy(np.stack(offset_targets))
        offset_targets = torch.as_tensor(offset_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        rot_targets = torch.from_numpy(np.stack(rot_targets))
        rot_targets = torch.as_tensor(rot_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        
        return heatmaps_targets, wh_targets, offset_targets, rot_targets

    # test use
    @force_fp32(apply_to=('cls_scores', 'wh_preds', 'offset_preds', 'rot_preds'))
    def get_bboxes(self,
                    cls_scores,
                    wh_preds,
                    offset_preds,
                    rot_preds,
                    img_metas,
                    cfg=None,
                    rescale=False,
                    with_nms=False):
        assert len(cls_scores) == len(wh_preds) == len(offset_preds) == len(rot_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        result_list = []

        for img_id in range(len(img_metas)): # 每个batch中id
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ] # =>[num_levels] => [80, h, w]
            wh_pred_list = [
                wh_preds[i][img_id].detach() for i in range(num_levels)
            ]
            offset_pred_list = [
                offset_preds[i][img_id].detach() for i in range(num_levels)
            ]
            rot_pred_list = [
                rot_preds[i][img_id].detach() for i in range(num_levels)
            ]

            scale_factor = img_metas[img_id]['scale_factor']
            img_h, img_w = img_metas[img_id]['img_shape'][:2]
            center = np.array([img_h / 2, img_w / 2], dtype=np.float32)
            scale = max(img_h, img_w) * 1.0

            det_bboxes = self.get_bboxes_single(cls_score_list,  wh_pred_list,
                                                offset_pred_list, rot_pred_list,
                                                featmap_sizes, center, scale,
                                                scale_factor, cfg) # 对每一张图像进行解调
            result_list.append(det_bboxes)
        return result_list # [batch_size]

    def get_bboxes_single(self,
                        cls_scores,
                        wh_preds,
                        offset_preds,
                        rot_preds,
                        featmap_sizes,
                        c, 
                        s,
                        scale_factor,
                        cfg):
        assert len(cls_scores) == len(wh_preds) == len(offset_preds) == len(rot_preds) == len(featmap_sizes)
        
        detections = []
        for cls_score, wh_pred, offset_pred, rot_pred, featmap_size in zip(
                cls_scores, wh_preds, offset_preds, rot_preds, featmap_sizes): # 取出每一层的点
            assert cls_score.size()[-2:] == wh_pred.size()[-2:] == offset_pred.size()[-2:] == rot_pred.size()[-2:] == featmap_size
            output_h, output_w = featmap_size
            #实际上得到了每一层的hm, wh, offset
            hm = torch.clamp(cls_score.sigmoid_(), min=1e-4, max=1-1e-4).unsqueeze(0) # 增加一个纬度
            wh = wh_pred.unsqueeze(0) # 这里需要乘以featuremap的尺度
            reg = offset_pred.unsqueeze(0)
            rot = rot_pred.unsqueeze(0)
            dets = ctdet_decode_rot(hm, reg, wh, rot, K=self.K)
            dets = post_process(dets, c, s, output_h, output_w, scale=scale_factor, num_classes=self.num_classes)
            detections.append(dets)

        results = merge_outputs(detections, self.num_classes, self.K) # 单张图的结果
        return results


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def ctdet_decode_rot(hmap, regs, w_h_, rot, K=100):
    batch, cat, height, width = hmap.shape
    # TODO: if necessary
    hmap = torch.sigmoid(hmap)
    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        rot = rot[0:1]
    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)

    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]
        
    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    rot = _tranpose_and_gather_feature(rot, inds)
    rot = rot.view(batch, K, 1)    

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    # bboxes = _get_rot_box(xs, ys, w_h_, rot)
    bboxes = torch.cat([xs, ys, w_h_, rot], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind) # 按照dim=1获取ind
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feature(feat, ind):
    # ind代表的是ground truth中设置的存在目标点的下角标
    feat = feat.permute(0, 2, 3, 1).contiguous()# from [bs c h w] to [bs, h, w, c] 
    feat = feat.view(feat.size(0), -1, feat.size(3)) # to [bs, wxh, c]
    feat = _gather_feature(feat, ind)
    return feat


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def post_process(dets, c, s, out_height, out_width, scale, num_classes):
    """将预测结果转换为CPU上的Numpy数组，并映射到原始图像上

    Args:
        dets (Tensor(GPU)): 筛选出的TopK个候选结果
        c (tuple[2]): 图片中心点坐标
        s (list[2]): 横纵维度的缩放尺寸
        out_height (int): 特征图高度
        out_width (int): 特征图宽度
        scale (): [description]
        num_classes ([type]): [description]

    Returns:
        [type]: [description]
    """
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    print(dets.shape)

    dets = ctdet_post_process(
        dets.copy(), [c], [s],
        out_height, out_width, num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j][:, 0] /= scale[0]
        dets[0][j][:, 1] /= scale[1]
        dets[0][j][:, 2] /= scale[0]
        dets[0][j][:, 3] /= scale[1]
    return dets[0]


def ctdet_post_process(dets, c, s, h, w, num_classes):
    """对结果框进行映射，并按预测类别分类
    """
    ret = []

    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (h, w))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (h, w))
        # dets[i, :, 4:6] = transform_preds(
        #     dets[i, :, 4:6], c[i], s[i], (w, h))
        # dets[i, :, 6:8] = transform_preds(
        #     dets[i, :, 6:8], c[i], s[i], (w, h))

        classes = dets[i, :, -1]

        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = dets[i, inds, :7].astype(np.float32)
        ret.append(top_preds)

    return ret


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1) 
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def merge_outputs(detections, num_classes, num_keep):

    results = {}
    max_per_image = num_keep
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        # results[j] = soft_nms(results[j], Nt=0.5, method=2, threshold=0.01)
    scores = np.hstack([results[j][:, 5] for j in range(1, num_classes + 1)])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 5] >= thresh)
            results[j] = results[j][keep_inds]

    return results2coco_boxes(results, num_classes)


def results2coco_boxes(results, num_classes): 
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """
    bboxes = [0 for i in range(num_classes)]
    for j in range(1, num_classes + 1):
        if len(results[j]) == 0:
            bboxes[j - 1] = np.zeros((0, 7), dtype=np.float32)
            continue
        bboxes[j - 1] = results[j]
    #print(bboxes) # xyxy
    return bboxes


def soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.01, method=0):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0     #cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    boxes = boxes[keep]
    return boxes
