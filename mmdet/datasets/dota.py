import os
import os.path as osp

import cv2
import json
import math
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import pycocotools.coco as coco
import cv2
import numpy as np
import mmcv

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset
from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius


@DATASETS.register_module()
class DotaDataset(XMLDataset):
    
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                'small-vehicle', 'large-vehicle', 'ship', 
                'tennis-court', 'basketball-court',  
                'storage-tank', 'soccer-ball-field', 
                'roundabout', 'harbor', 
                'swimming-pool', 'helicopter','container-crane')

    def __init__(self, **kwargs):
        super(DotaDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.png'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.png'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos
    
    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('label').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('x0').text)),
                int(float(bnd_box.find('y0').text)),
                int(float(bnd_box.find('x1').text)),
                int(float(bnd_box.find('y1').text)),
                int(float(bnd_box.find('x2').text)),
                int(float(bnd_box.find('y2').text)),
                int(float(bnd_box.find('x3').text)),
                int(float(bnd_box.find('y3').text)), 
            ]
            # drop ignore and difficult
            # ignore = False
            # if self.min_size:
            #     assert not self.test_mode
            #     w = bbox[2] - bbox[0]
            #     h = bbox[3] - bbox[1]
            #     if w < self.min_size or h < self.min_size:
            #         ignore = True
            # if difficult or ignore:
            #     bboxes_ignore.append(bbox)
            #     labels_ignore.append(label)
            # else:
            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        
        n_bboxes = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :]
            cx = (bbox[0] + bbox[2] + bbox[4] + bbox[6]) / 4
            cy = (bbox[1] + bbox[3] + bbox[5] + bbox[7]) / 4
            w = math.sqrt(math.pow((bbox[0] - bbox[2]), 2) + math.pow((bbox[1] - bbox[3]), 2))
            h = math.sqrt(math.pow((bbox[2] - bbox[4]), 2) + math.pow((bbox[3] - bbox[5]), 2))

            if w < h:
                w, h = h, w
                theta = math.atan((bbox[5] - bbox[3]) / (bbox[4] - bbox[2] + 1e-3))
            else:
                theta = math.atan((bbox[3] - bbox[1]) / (bbox[2] - bbox[0] + 1e-3))
            n_bboxes.append([cx, cy, w, h, theta])

        ann = dict(
            bboxes=np.array(n_bboxes).astype(np.float32),
            labels=np.array(labels).astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, 'Annotations',
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('label').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset='dota',
                logger=logger,
                nproc=10)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            #TODO: update recall for rot-dataset
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
