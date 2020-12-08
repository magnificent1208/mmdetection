import time

import numpy as np
import torch
import cv2 as cv

from mmdet.apis import init_detector, inference_detector


config_file = '/home/maggie/work/mmdetection/configs/tunnel_obj/faster_rcnn_r50.py'
checkpoint_file = '/home/maggie/work/mmdetection/work_dirs/tunnel_obj/faster_rcnn_r50/latest.pth'

# img = '/home/maggie/work/mmdetection/data/dota/train/JPEGImages/P2739_800_1600.png'
# img = 'data/sim311/JPEGImages/DSC01062.jpg'

CLASSES = ('ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
            'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs')


def draw_rbox(img_dir, result, iou=0.1):
    img_cv = cv.imread(img_dir)
    # import pdb; pdb.set_trace()
    for i in range(16):
        for j in result[i]:
            if j[-1] > iou:
                for k in range(4):
                    cv.line(img_cv, (j[2*k], j[2*k + 1]), (j[(2*k + 2) % 8], j[(2*k + 3) % 8]), (0,0,255), 2)
                print('draw box')
    cv.imwrite('result.jpg', img_cv)


if __name__ == '__main__':
    model = init_detector(config_file, checkpoint_file)
    # model = init_detector(config_file)
    model.eval()

    # result = inference_detector(model, img)
    # draw_rbox(img, result)
    # show_result(img, result, CLASSES, out_file='test1.jpg')

    # # 计算模型时间
    # start_time = time.time()
    # input_data = np.zeros([512, 512, 3])
    # for _ in range(100):
    #     with torch.no_grad():
    #             result = model(input_data, return_loss=False, rescale=True, )
    # cal_time = time.time() - start_time
    # print(cal_time / 100)

    # 测试一系列图片
    imgs = ['002.jpg', '003.jpg', '004.jpg']
    img_dir = 'data/tunnel_obj/JPEGImages/'
    for i, img in enumerate(imgs):
        result = inference_detector(model, img_dir + img)
        model.show_result(img_dir + img, result, out_file='result_{}.jpg'.format(i))
