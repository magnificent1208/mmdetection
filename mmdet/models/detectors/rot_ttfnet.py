import torch
import cv2
import math
import numpy as np

from mmdet.core import bbox2result
from .single_stage import SingleStageDetector
from ..builder import DETECTORS



@DETECTORS.register_module
class RTTFNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RTTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
    
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # import pdb; pdb.set_trace()
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        # if torch.onnx.is_in_onnx_export():
        import pdb; pdb.set_trace()
        return bbox_list

        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in bbox_list
        # ]
        # return bbox_results

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

        # import pdb; pdb.set_trace()
        for i in range(self.bbox_head.num_classes):
            if len(result[i]) > 0:
                draw_boxes = self._get_rot_box(result[i][:, 0], 
                                               result[i][:, 1], 
                                               result[i][:, 2:4], 
                                               result[i][:, 4])
                for j, box in enumerate(draw_boxes):
                    if result[i][j][4] > score_thr:
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