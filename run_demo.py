import time
import os

import numpy as np
import torch
import cv2 as cv

from mmdet.apis import init_detector, inference_detector

config_file = './configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = './ckpts/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

if __name__ == '__main__':
    model = init_detector(config_file, checkpoint_file, device='cpu')
    model.eval()

    # imgs = ['img(2).JPG', 'img(7).JPG', 'img(12).JPG', 'img(17).JPG']
    img_dir = './data/'
    imgs = os.listdir(img_dir)

    for i, img in enumerate(imgs):
        result = inference_detector(model, img_dir + img) # result from model.simple_test
        model.show_result(img_dir + img, result, score_thr=0.7, out_file='show_dirs/result_{}.jpg'.format(i))
