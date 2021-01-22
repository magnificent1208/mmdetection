import time
import os

import numpy as np
import torch
import cv2 as cv

from mmdet.apis import init_detector, inference_detector


# imgs = ['C00510019_59.4.jpg', 'C00510020_60.4.jpg', 'C00510003_60.5.jpg', 'C00490125_66.5.jpg']
# img_dir = './'

# imgs = ['P1446_1600_1046.png', 'P1139_4800_3200.png', 'P0122_490_556.png', 'P2552_800_1947.png', 'P1308_3000_4000.png', 'P1509_2076_3200.png', 
#         'P0236_450_313.png', 'P0896_1438_800.png', 'P1413_2889_2400.png']

imgs = ['P1413_2889_2400.png']

img_dir = './data/dota/train/JPEGImages/'
work_dir_path = 'centernet_debug'
dataset = 'dota'

# imgs = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg', 'test7.jpg', 'test8.jpg']
# img_dir = './'
# work_dir_path = 'faster_rcnn_r50_210119_maggie'
# dataset = 'jsxs'

device = 'cuda'
thr = 0.5


def find_py(dir_path):
    file_names = os.listdir(dir_path)

    for name in file_names:
        if name.endswith('.py'):
            return os.path.join(dir_path, name)
    
    print('Could not find .py file')
    return None


def draw_box(imgs, wd_name, cp_name='latest.pth', device='cpu', dataset='dota', thr=0.7):

    work_dir = os.path.join('./work_dirs', dataset, wd_name)
    config_path = find_py(work_dir)
    checkpoint_path = os.path.join(work_dir, cp_name)
    out_path = os.path.join('./show_dirs', dataset, wd_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()

    for i, img in enumerate(imgs):
        result = inference_detector(model, img_dir + img) # result from model.simple_test
        
        model.show_result(img_dir + img, result, score_thr=thr, out_file=os.path.join(out_path, 'result_{}.jpg'.format(i)))


if __name__ == '__main__':
    draw_box(imgs, work_dir_path, device=device, dataset=dataset, thr=thr)
