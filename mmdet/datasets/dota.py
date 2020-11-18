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

import sys
from .builder import DATASETS
from .xml_style import XMLDataset

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius


# 均值和方差
DOTA_MEAN = [0.36488013, 0.36802809, 0.35458627]
DOTA_STD = [0.04308565, 0.04508483, 0.0519263 ]

# eigen: 特征
VOC_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
VOC_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                     [-0.5832747, 0.00994535, -0.81221408],
                     [-0.56089297, 0.71832671, 0.41158938]]


@DATASETS.register_module()
class DotaDataset(XMLDataset):
    
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                'small-vehicle', 'large-vehicle', 'ship', 
                'tennis-court', 'basketball-court',  
                'storage-tank', 'soccer-ball-field', 
                'roundabout', 'harbor', 
                'swimming-pool', 'helicopter','container-crane')

    # def __init__(self, **kwargs):
    #     super(DotaDataset, self).__init__()


        # self.num_classes = 16
        # self.class_names = DOTA_NAMES

        # # 处理可选择id
        # self.cat_ids = {v: i for i, v in enumerate(DOTA_NAMES)}
        # self.data_rng = np.random.RandomState(123)

        # self.mean = np.array(DOTA_MEAN, dtype=np.float32).reshape(1, 1, 3)
        # self.std = np.array(DOTA_STD, dtype=np.float32).reshape(1, 1, 3)
        
        # self.eig_val = np.array(VOC_EIGEN_VALUES, dtype=np.float32)
        # self.eig_vec = np.array(VOC_EIGEN_VECTORS, dtype=np.float32)

        # self.split = split
        # self.data_dir = os.path.join(data_dir, 'dota/train')
        # self.img_dir = os.path.join(self.data_dir, 'JPEGImages')
        # self.xml_dir = os.path.join(self.data_dir, 'Annotations')
        # self.annot_path = os.path.join(
        #     self.data_dir, 'ImageSets/Main/', '%s.txt' % split)

        # # 挑选100个
        # self.max_objs = 200
        # self.padding = 31  # 127 for hourglass
        # self.down_ratio = 4  # 降采样4倍

        # self.img_size = {'h': img_size, 'w': img_size}
        # self.fmap_size = {'h': img_size // self.down_ratio,
        #                   'w': img_size // self.down_ratio}
        # self.rand_scales = np.arange(0.6, 1.4, 0.1)
        # self.gaussian_iou = 0.7

        # print('==> initializing DOTA %s data.' % split)
        # self.images = self.get_images_name(self.annot_path)
        # self.num_samples = len(self.images)
        # print('Loaded %d %s samples' % (self.num_samples, split))

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
            name = obj.find('name').text
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
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
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
            bboxes=n_bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
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

    def __getitem__(self, index):
        import pdb; pdb.set_trace()
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name + '.png')
        ann_path = os.path.join(self.xml_dir, img_name + '.xml')
        labels, bboxes = self.read_xml(ann_path)

        labels = np.array(labels) # index: 1-15
        bboxes = np.array(bboxes) # x0y0x1y1x2y2x3y3
        bboxes = self.norm_bboxes(bboxes) # cx, cy, w, h, theta

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        # 获取中心坐标p
        center = np.array([width / 2., height / 2.],
                            dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0  # 仿射变换

        # 仿射变换
        trans_img = get_affine_transform(
            center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(
            img, trans_img, (self.img_size['w'], self.img_size['h']))

        # 归一化
        img = (img.astype(np.float32) / 255.)
        # img -= self.mean
        # img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        # 对Ground Truth heatmap进行仿射变换
        trans_fmap = get_affine_transform(
            center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']]) # 这时候已经是下采样为原来的四分之一了

        # 3个最重要的变量
        hmap = np.zeros(
            (self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        rot = np.zeros((self.max_objs, 1), dtype=np.float32)

        # indexs
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        # 具体选择哪些index
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        if len(bboxes) > self.max_objs:
            bboxes, labels = bboxes[:self.max_objs], labels[:self.max_objs]

        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            # 对检测框也进行仿射变换
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:4] = affine_transform(bbox[2:4], trans_fmap)
            w, h = bbox[2:4]

            if bbox[2] > 0 and bbox[3] > 0:
                obj_c = np.array(bbox[:2], dtype=np.float32) # 中心坐标-浮点型
                obj_c_int = obj_c.astype(np.int32) # 整型的中心坐标
                # 根据一元二次方程计算出最小的半径
                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                # 得到高斯分布
                draw_umich_gaussian(hmap[label], obj_c_int, radius)

                w_h_[k] = 1. * w, 1. * h
                
                # 记录偏移量
                regs[k] = obj_c - obj_c_int  # discretization error
                # 当前是obj序列中的第k个 = fmap_w * cy + cx = fmap中的序列数
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                # 进行mask标记
                ind_masks[k] = 1
                rot[k] = bbox[4]

        return {'image': img, 'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 
                'inds': inds, 'ind_masks': ind_masks, 'c': center, 'rot': rot,
                's': scale, 'img_id': img_name}

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
            ds_name = self.CLASSES
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
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
