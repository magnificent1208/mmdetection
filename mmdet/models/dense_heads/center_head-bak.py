import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Scale, normal_init

import numpy as np
import cv2
import math


from mmdet.core import multi_apply, multiclass_nms, distance2bbox, force_fp32
from ..builder import build_loss, HEADS
from .corner_head import CornerHead
from ..utils import gaussian_radius, gen_gaussian_target

# from ..utils import bias_init_with_prob, Scale, ConvModule

INF = 1e8

#x = np.empty([1,2,3,3], dtype = int) 
#print (x)

@HEADS.register_module
class CenterHead(CornerHead):

    """Head of CenterNet: Objects as Points

    Args:
        num_classes (int): Number of detect classes. (Including background)
        in_channels (int):
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. Because
            HourglassNet-104 outputs the final feature and intermediate
            supervision feature and HourglassNet-52 only outputs the final
            feature. Default: 2.
        strides (tuple): Downsample factor of each feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.        
        train_cfg (dict | None): Training config. Useless in CenterHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterHead. Default: None.
        loss_heatmap (dict | None): Config of center heatmap loss. Default:
            GaussianFocalLoss. #这里跟cornernet是完全一样的 alpha和gamma参数也是一样
        loss_offset (dict | None): Config of center offset loss. Default:
            SmoothL1Loss.

    """
    def __init__(self,
                 num_classes, # init 80
                 in_channels,
                 num_feat_levels=2,
                 feat_channels=256,                 
                 stacked_convs=1,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 train_cfg=None,
                 test_cfg=None,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_wh = dict(
                     type="SmoothL1Loss",
                     loss_weight=0.1),
                 loss_offset=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(CenterHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_feat_levels = num_feat_levels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_heatmap = build_loss(
            loss_heatmap) if loss_heatmap is not None else None
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(
            loss_offset) if loss_offset is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()
    

    def _init_layers(self):
        """Initialize layers for CenterHead.
        """
        self.cls_convs = nn.ModuleList()
        self.wh_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
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
        self.center_hm = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1, bias=True)
        self.center_wh = nn.Conv2d(self.feat_channels, 2, 3, padding=1, bias=True)
        self.center_offset = nn.Conv2d(self.feat_channels, 2, 3, padding=1, bias=True)
        #self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""

        self.center_hm.bias.data.fill_(-2.19)
        nn.init.constant_(self.center_wh.bias, 0)
        nn.init.constant_(self.center_offset.bias, 0)


    def forward(self, feats):
        """Forward features from the upstream network.
        more details on Centernet_network.png 

        Args:
            feats (list[Tensor]): Features from the upstream network, each is
            a 4D-tensor.
            hm -- n*80*128*128   Peaks as Objects Center
            wh -- n*2*128*128    Center offsets(ox,oy)
            reg -- n*2*128*128   Boxes size(w,h)

        Returns:
            tuple[Tensor]: A tuple of CenterHead's output for current feature
            level. Containing the following Tensors:
                - cls_score (Tensor): Predicted classes of input feature maps for all
                  levels, each is a 4D-tensor, the channels number is
                  num_classes.
                - wh_preds (Tensor): Predicted width&height of input feature maps.
                - offset_pred (Tensor): Predicted center offset heatmap.
                
        """
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind):
        """Forward feature of a single level.
        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.

        Returns:
            tuple[Tensor]: A tuple of CenterHead's output for current feature
            level. Containing the following Tensors:
                - cls_score (Tensor): Predicted classes of input feature maps.
                - wh_preds (Tensor): Predicted width&height of input feature maps.
                - offset_pred (Tensor): Predicted center offset heatmap.
        """
        cls_feat = x
        wh_feat = x
        offset_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer[lvl_ind](cls_feat)
        cls_score = self.center_hm[lvl_ind](cls_feat)

        for wh_layer in self.wh_convs:
            wh_feat = wh_layer[lvl_ind](wh_feat)
        wh_pred = self.center_wh[lvl_ind](wh_feat)
        
        for offset_layer in self.offset_convs:
            offset_feat = offset_layer[lvl_ind](offset_feat)
        offset_pred = self.center_offset[lvl_ind](offset_feat)

        '''
        return cls_score, wh_pred, offset_pred
        '''
        
        result_list = [cls_score, wh_pred, offset_pred]

        return result_list

#沿用cornenet的 loss &single loss
    @force_fp32(apply_to=('cls_scores', 'wh_preds', 'offset_preds'))
    def loss(self,
             cls_scores,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is num_classes.
            wh_preds (list[Tensor]): Predicted width&height of input feature maps.
            offset_preds (list[Tensor]): center offsets for each level
                with shape (N, center_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.


        Returns:
            tuple[torch.Tensor]: Losses of the head's differnet branches
            containing the following losses:
                - loss_hm (list[Tensor]): center losses of all
                  feature levels.
                - loss_wh (list[Tensor]): width&high losses of all
                  feature levels.
                - loss_offset (list[Tensor]): Center offset losses of all
                  feature levels.
        """

        targets = self.get_targets(
            gt_bboxes,
            gt_labels,
            tl_heats[-1].shape,
            img_metas[0]['pad_shape'])
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        det_losses, pull_losses, push_losses, off_losses = multi_apply(
            self.loss_single, tl_heats, br_heats, tl_embs, br_embs, tl_offs,
            br_offs, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses)

        return loss_dict


              
    def loss_single(self, cls_score, wh_pred, offset_pred, targets):
        """Compute losses for single level.

        Args:
            cls_score (Tensor):
            wh_pred (Tensor): 
            offset_pred (Tensor): 
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's differnet branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - pull_loss (Tensor): Part one of AssociativeEmbedding loss.
                - push_loss (Tensor): Part two of AssociativeEmbedding loss.
                - off_loss (Tensor): Corner offset loss.
        """
        assert len(cls_scores) == len(wh_preds) == len(offset_preds)

        gt_tl_hmp = targets['topleft_heatmap']
        gt_br_hmp = targets['bottomright_heatmap']
        gt_tl_off = targets['topleft_offset']
        gt_br_off = targets['bottomright_offset']
        gt_embedding = targets['corner_embedding']

        # Detection loss
        tl_det_loss = self.loss_heatmap(
            tl_hmp.sigmoid(),
            gt_tl_hmp,
            avg_factor=max(1,
                           gt_tl_hmp.eq(1).sum()))
        br_det_loss = self.loss_heatmap(
            br_hmp.sigmoid(),
            gt_br_hmp,
            avg_factor=max(1,
                           gt_br_hmp.eq(1).sum()))
        det_loss = (tl_det_loss + br_det_loss) / 2.0

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

        off_loss = (tl_off_loss + br_off_loss) / 2.0

        return det_loss, pull_loss, push_loss, off_loss


#论文中peak extraction 的部分
    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

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

    def _topk(self, scores, k=100):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 100. 因为centernet原文写的是100所以这里填这个数，以调用的时候实际传入的为主，可以多尝试几个数

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint. 对应值维度 n*80*K
            - topk_inds (Tensor): Indexes of each topk keypoint. 对应的索引维度是n*80*K
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.   对应的维度是n*80*K
            - topk_xs (Tensor): X-coord of each topk keypoint.   对应的维度是n*80*K
        
        **cat -- 80 num_classes
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = (topk_inds / (height * width)).int()
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
        #得到每张图片最大的Ｋ个值，有８０个类　所以８０个ｈｍｐ通道

# 我们相当于在ind中记录了目标在heatmap上的地址索引，
# 通过_tranpose_and_gather_feat以及
# def _gather_feat(feat, ind, mask=None):函数得出我们预测的宽高。

##Boxes 相关的还没改动
    def get_bboxes(self,
                   tl_heats,
                   br_heats,
                   tl_embs,
                   br_embs,
                   tl_offs,
                   br_offs,
                   img_metas,
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
        assert tl_heats[-1].shape[0] == br_heats[-1].shape[0] == len(img_metas)
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

    def get_bboxes_single(self,
                        cls_scores,
                        wh_preds,
                        offset_preds,
                        featmap_sizes,
                        c, 
                        s,
                        scale_factor,
                        cfg):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).


            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1.
        """

        assert len(cls_scores) == len(wh_preds) == len(offset_preds) == len(featmap_sizes)
        
        detections = []
        for cls_score, wh_pred, offset_pred, featmap_size in zip(
                cls_scores, wh_preds, offset_preds, featmap_sizes): # 取出每一层的点
            assert cls_score.size()[-2:] == wh_pred.size()[-2:] == offset_pred.size()[-2:] == featmap_size
            
            output_h, output_w = featmap_size
            #实际上得到了每一层的hm, wh, offset
            hm = torch.clamp(cls_score.sigmoid_(), min=1e-4, max=1-1e-4).unsqueeze(0) # 增加一个纬度
            #wh_pred[0, :, :] = wh_pred[0, :, :] * output_w
            #wh_pred[1, :, :] = wh_pred[1, :, :] * output_h # 2, output_h, output_w
            wh = wh_pred.unsqueeze(0) # 这里需要乘以featuremap的尺度
            #offset_pred[0, : ,:] =  offset_pred[0, : ,:] * output_w
            #offset_pred[1, : ,:] =  offset_pred[1, : ,:] * output_h
            reg = offset_pred.unsqueeze(0)
            
            dets = ctdet_decode(hm, wh, reg=reg, K=100)
            dets = post_process(dets, c, s, output_h, output_w, scale=scale_factor, num_classes=self.num_classes)
            detections.append(dets)
        
        results = merge_outputs(detections, self.num_classes) # 单张图的结果

        return results

    def _bboxes_nms(self, bboxes, labels, cfg):
        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.　维度：batch * (80 x 100) * 1 
            ind (Tensor): Target coord index.   维度：batch * 100
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature. 维度：batch * 100 * 1 

        *num_classes　＝　８０　
        *K = 100 *refer to method topK
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind) #Gather feature according to index
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.　　　维度：batch * C（channel） * W * H
            ind (Tensor): Target coord index.　　　　维度：batch * K

        Returns:
            feat (Tensor): Transposed and gathered feature.　　维度：feat：batch * K * C
            　　　　　　　　　含义为feat[i, j, k]为第i个batch，第k个channel的第j个最大值。
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def decode_heatmap(self,
                       heat,
                       wh,
                       reg = None,
                       #cat_spec_wh=False,  #不一定管用　from　作者
                       img_meta=None,
                       K=100,
                       kernel=3,
                       num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.

        Args:
            heat (Tensor): heatmap for current level with
                shape (N, num_classes, H, W).
            wｈ(Tensor): width and height of bbox #也可以理解为中心点的宽高
            reg (Tensor): center point offset for current level with
                shape (N, corner_offset_channels, H, W). #关键点量化误差补偿　offset
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            K (int): Get top k corner keypoints from heatmap.
            kernel (int): Max pooling kernel for extract local maximum pixels.
            num_dets (int): Num of raw boxes before doing nms.

        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """

        batch, _, height, width = heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        heat = self._local_maximum(heat, kernel=kernel)

        scores, inds, clses, ys, xs = self._topk(heat, K=K)
        # xs、ys是inds转化成在heat_map上面的行、列

        #如果用了关键点量化误差补偿，则解码并加到先前的结果上
        if regs is not None:
            regs = _tranpose_and_gather_feature(regs, inds)
            regs = regs.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
            ys = ys.view(batch, K, 1) + regs[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        wh = self._transpose_and_gather_feat(wh, inds) # inds 对应 h, w的尺度
        wh = wh.view(batch, K, 2)
      
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1) # 0, 1, 2
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)

        return bboxes, scores, clses

'''
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections    
'''

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
    
    def center_target_single(self, gt_bboxes, gt_labels, img_meta):

        '''single image
        gt_bboxes:torch.Size([6, 4])
        gt_labels:torch.Size([6]) tensor([34, 34, 34, 34, 34, 34], device='cuda:0')
        featmap_sizes:(list[tuple]): Multi-level feature map sizes.
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))

        # transform the gt_bboxes, gt_labels to numpy
        gt_bboxes = gt_bboxes.data.cpu().numpy()
        gt_labels = gt_labels.data.cpu().numpy()'''
        
        num_objs = gt_labels.shape[0]

        num_levels = len(self.featmap_sizes)

        heatmaps_targets = []
        wh_targets = []
        offset_targets = []
        # get the target shape for each image
        for i in range(num_levels):
            h, w = self.featmap_sizes[i]
            hm = np.zeros((self.cls_out_channels, h, w), dtype=np.float32)
            heatmaps_targets.append(hm)
            wh = np.zeros((h, w, 2), dtype=np.float32)
            wh_targets.append(wh)
            offset = np.zeros((h, w, 2), dtype=np.float32)
            offset_targets.append(offset)

        for k in range(num_objs):
            bbox = gt_bboxes[k]
            cls_id = gt_labels[k]
            
            if img_meta['flipped']:
                bbox[[0, 2]] = img_meta['width'] - bbox[[2, 0]] - 1
                
            # condition: in the regress_ranges
            origin_h, origin_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            #max_h_w = max(h, w) / 2
            max_h_w = max(origin_h, origin_w)
            #max_h_w = max(origin_h, origin_w) * 2 # 最长边为32在P2
            # 根据max_h_w在哪一层将output设置为当前层的
            index_levels = []
            #index_level = 0
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
                #print(output_h, output_w)
                hm = heatmaps_targets[index_level]
                wh = wh_targets[index_level]
                offset = offset_targets[index_level]
            
                # c, s is passed by meta
                trans_output = get_affine_transform(img_meta['c'], img_meta['s'], 0, [output_w, output_h])
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1) #x1, x2
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                #print(h, w)
                # 转换到当层
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    #print(ct)
                    ct_int = ct.astype(np.int32)

                    draw_umich_gaussian(hm[cls_id], ct_int, radius)

                    h, w = 1. * h, 1. * w
                    offset_count = ct - ct_int # h, w

                    wh[ct_int[1], ct_int[0], 0] = w 
                    wh[ct_int[1], ct_int[0], 1] = h 
                    offset[ct_int[1], ct_int[0], 0] = offset_count[0]
                    offset[ct_int[1], ct_int[0], 0] = offset_count[1]
            
            
                heatmaps_targets[index_level] = hm
                wh_targets[index_level] = wh
                offset_targets[index_level] = offset

        flatten_heatmaps_targets = [
            hm.transpose(1, 2, 0).reshape(-1, self.cls_out_channels)
            for hm in heatmaps_targets
        ]
            
        heatmaps_targets = np.concatenate(flatten_heatmaps_targets, axis=0) 
        
        flatten_wh_targets = [
            wh.reshape(-1, 2) for wh in wh_targets
        ]
        wh_targets = np.concatenate(flatten_wh_targets)
        
        flatten_offset_targets = [
            offset.reshape(-1, 2) for offset in offset_targets
        ]
        offset_targets = np.concatenate(flatten_offset_targets)

        # transform the heatmaps_targets, wh_targets, offset_targets into tensor
        heatmaps_targets = torch.from_numpy(np.stack(heatmaps_targets))
        heatmaps_targets = torch.tensor(heatmaps_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        wh_targets = torch.from_numpy(np.stack(wh_targets))
        wh_targets = torch.tensor(wh_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        offset_targets = torch.from_numpy(np.stack(offset_targets))
        offset_targets = torch.tensor(offset_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        
        return heatmaps_targets, wh_targets, offset_targets    

