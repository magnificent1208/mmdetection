import time

import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector

config_file = '/home/maggie/work/mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py'
# checkpoint_file = '/home/maggie/work/mmdetection/work_dirs/sim311/faster_rcnn_r50_fpn_1x/epoch_20.pth'

CLASSES = ('ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
            'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs')

# 初始化模型
# model = init_detector(config_file, checkpoint_file)
model = init_detector(config_file)
model.eval()
start_time = time.time()
input_data = np.zeros([512, 512, 3])
for _ in range(100):
    with torch.no_grad():
            result = model(input_data, return_loss=False, rescale=True, )
cal_time = time.time() - start_time
print(cal_time / 100)

# # 测试一张图片
# img = 'data/sim311/JPEGImages/DSC01062.jpg'
# result = inference_detector(model, img)
# show_result(img, result, CLASSES, out_file='test1.jpg')

# 测试一系列图片
# imgs = ['002.jpg', '003.jpg', '004.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
