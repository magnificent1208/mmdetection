import math

import torch
import numpy as np
import cv2

from mmdet.core import bbox2result, bbox_mapping_back
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CenterNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

        self.score_thr = test_cfg['score_thr'] if 'score_thr' in test_cfg else 0.2
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # import cv2
        # img0 = np.uint8(img[0].cpu().numpy())
        # cv2.imwrite('input.jpg', img0.transpose(1,2,0))
        # img_cv = cv2.imread('input.jpg')
        # gt_bboxes = np.array(gt_bboxes[0].cpu())
        # pos = self._get_rot_box(gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2:4], gt_bboxes[:, 4])
        # for box in pos:
        #     for k in range(4):
        #         cv2.line(img_cv, (box[2*k], box[2*k + 1]), (box[(2*k + 2) % 8], box[(2*k + 3) % 8]), (0,0,255), 2)
        # cv2.imwrite('label.jpg', img_cv)
        # import pdb; pdb.set_trace()           

        x = self.backbone(img)
        if self.neck:
            x = self.neck(x)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        """Test function for list of image without argumentation.        

        Parameters
        ----------
        img : list[torch.Tensor]
            List of imgs to test
        img_metas : list[dict]
            List of img information, with several dict.
        rescale : bool, optional
            Wheter to recale the results, by default False
        """
        # hrnet:{0.02: 16.8, 0.05: 20.0, 0.1: 19.4, 0.15:19.3, 0.2: 17.9}  
        # hourglass:{0.5: 11.1, 0.3: 20.5, 0.25: 21.1, 0.2: 19.8, 0.15:16.6, 0.1: 11.1}
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # Draw heatmap
        # import cv2 as cv
        # for i in range(16):
        #     heatmap = (outs[0][0][0][i] + 1.5)* 10
        #     heatmap = heatmap.cpu().numpy().astype(np.uint8)
        #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_HOT)
        #     # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_HOT)
        #     cv.imwrite('heatmap_{}.jpg'.format(i), heatmap)
        # import pdb; pdb.set_trace()

        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        
        for i in range(len(bbox_list)):
            for j in range(len(bbox_list[i])):
                cur_list = bbox_list[i][j]
                new_list = []

                for box in cur_list:
                    if box[5] > self.score_thr:
                        new_list.append(box)
                if new_list:
                    bbox_list[i][j] = np.array(new_list)
                else:
                    bbox_list[i][j] = np.empty([0,7])

        return bbox_list

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

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (0, 64, 64), (64, 0 , 64), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 0, 0), (0, 192, 0), (192, 0, 0), (192, 192, 0))

        if isinstance(img, str):
            img_cv = cv2.imread(img)
            img_cv = img_cv.copy()
        else:
            img_cv = img

        for i in range(self.bbox_head.num_classes):
            if len(result[i]) > 0:
                draw_boxes = self._get_rot_box(result[i][:, 0], 
                                               result[i][:, 1], 
                                               result[i][:, 2:4], 
                                               result[i][:, 4])
                for j, box in enumerate(draw_boxes):
                    if result[i][j][5] > score_thr:
                        for k in range(4):
                            cv2.line(img_cv, (box[2*k], box[2*k + 1]), (box[(2*k + 2) % 8], box[(2*k + 3) % 8]), colors[i], 2)

        cv2.imwrite(out_file, img_cv)
        return True

    def _get_rot_box(self, xs, ys, w_h_, rot):
        direction = []
        
        for angle in rot:
            cos, sin = math.cos(angle), math.sin(angle)
            direction.append([cos, sin, -sin, cos])
        # direction = torch.tensor(direction).clone().detach().cuda()
        direction = np.array(direction) 

        x0 = xs + w_h_[:,1] * direction[:, 2] / 2 + w_h_[:,0] * direction[:, 0] / 2
        y0 = ys + w_h_[:,1] * direction[:, 3] / 2 + w_h_[:,0] * direction[:, 1] / 2
        x1 = x0 - w_h_[:,0] * direction[:, 0]
        y1 = y0 - w_h_[:,0] * direction[:, 1]
        x2 = x1 - w_h_[:,1] * direction[:, 2]
        y2 = y1 - w_h_[:,1] * direction[:, 3]
        x3 = x0 - w_h_[:,1] * direction[:, 2]
        y3 = y0 - w_h_[:,1] * direction[:, 3]
        
        return np.stack([x0, y0, x1, y1, x2, y2, x3, y3], axis=1).astype(int)
            #     # Draw heatmapa
        #     # heatmap = output[0][0][i] * 10
        #     # heatmap = heatmap.cpu().numpy().astype(np.uint8)
        #     # heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_HOT)
        #     # cv.imwrite('./show_result/' + cfg.show_dir + img_id + '/' + 'heatmap_{}.jpg'.format(i), heatmap)
        
        # # os.system('mv heatmap/* show_result/' + cfg.show_dir + img_id + '/')
