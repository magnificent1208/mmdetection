import torch

from mmdet.core import bbox2result, bbox_mapping_back
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .two_stage import TwoStageDetector

from ..losses import CtdetLoss
from .ctdet_decetor import ctdet_decode, post_process, merge_outputs


@DETECTORS.register_module()
class CenterNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    def forward_train(self, img, img_meta, **kwargs):
        
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        return outs
    
    def forward_test(self, img, img_meta, **kwargs):

        output = self.backbone(img.type(torch.cuda.FloatTensor))[-1] # batch, c, h, m
        hm = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1-1e-4)
        wh = output['wh']
        reg = output['reg']
    #         print("hm", hm)
    #         print("wh", wh)
    #         print("reg", reg)
        dets = ctdet_decode(hm, wh, reg=reg, K=100)
    #         print("after process:\n", dets)
    #         print(img_meta)
    #         batch = kwargs
    #         print(batch)
    #        scale = img_meta[i]['scale'].detach().cpu().numpy()[0]
        dets = post_process(dets, meta = img_meta, scale=1)
    #         print("after post_process:\n", dets)
#        detections.append(dets)
        detections = [dets]
        results = merge_outputs(detections)
#         print(results)
#         det_bboxes = dets[:,:,:5].view(-1, 5)# (batch, k, 4)
#         det_labels = dets[:,:,5].view(-1) # (batch, k, 1)

#         x = self.extract_feat(img)
#         proposal_list = self.simple_test_rpn(
#             x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

# input is the output of network, return the det_bboxed, det_labels(0~L-1)
#         det_bboxes, det_labels = self.simple_test_bboxes(
#             x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

#         bbox_results = bbox2result(det_bboxes, det_labels,
#                                self.backbone.heads['hm'] + 1)

        return results
    
    def forward_dummy(self, img):
        x = self.backbone(img)
        return x

    def merge_aug_results(self, aug_results, img_metas):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes, aug_labels = [], []
        for bboxes_labels, img_info in zip(aug_results, img_metas):
            img_shape = img_info[0]['img_shape']  # using shape before padding
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes, labels = bboxes_labels
            bboxes, scores = bboxes[:, :4], bboxes[:, -1:]
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(torch.cat([bboxes, scores], dim=-1))
            aug_labels.append(labels)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        labels = torch.cat(aug_labels)

        if bboxes.shape[0] > 0:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=False):
        """Augment testing of CenterNet.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            bbox_results (tuple[np.ndarry]): Detection result of each class.
        """
        img_inds = list(range(len(imgs)))

        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, [img_metas[ind], img_metas[flip_ind]], False, False)
            aug_results.append(bbox_list[0])
            aug_results.append(bbox_list[1])

        bboxes, labels = self.merge_aug_results(aug_results, img_metas)
        bbox_results = bbox2result(bboxes, labels, self.bbox_head.num_classes)

        return bbox_results