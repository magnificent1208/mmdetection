import time

import numpy as np
import torch
import cv2 as cv

from mmdet.apis import init_detector, inference_detector


# config_file = '/home/maggie/work/mmdetection/work_dirs/dota/center_res50_high/centernet_resnet50.py'
# checkpoint_file = '/home/maggie/work/mmdetection/work_dirs/dota/center_res50_high/latest.pth'
config_file = '/home/maggie/work/mmdetection/work_dirs/tunnel_rot/centernet_resnet50_800/centernet_resnet50.py'
checkpoint_file = '/home/maggie/work/mmdetection/work_dirs/tunnel_rot/centernet_resnet50_800/epoch_60.pth'

# img = '/home/maggie/work/mmdetection/data/dota/train/JPEGImages/P2739_800_1600.png'
# img = 'data/sim311/JPEGImages/DSC01062.jpg'

# CLASSES = ('ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
#             'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs')


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
    # imgs = ['002.jpg', '003.jpg', '004.jpg']
    # img_dir = 'data/tunnel_obj/JPEGImages/'

    # imgs = ['C00510019_59.4.jpg', 'C00510020_60.4.jpg', 'C00510003_60.5.jpg', 'C00490125_66.5.jpg']
    # img_dir = './'

    # imgs = ['P1446_1600_1046', 'P1139_4800_3200', 'P0122_490_556', 'P2552_800_1947', 'P1308_3000_4000', 'P1509_2076_3200', 
    #         'P0236_450_313', 'P0896_1438_800', 'P1361_1600_800', 'P1413_2889_2400', 'P1354_3000_1600', 'P1366_1600_0']
    # img_dir = './data/dota/train/JPEGImages/'

    imgs = ['img(2).JPG', 'img(7).JPG', 'img(12).JPG', 'img(17).JPG']
    img_dir = './data/tunnel_rot/JPEGImages/'
    for i, img in enumerate(imgs):
        result = inference_detector(model, img_dir + img) # result from model.simple_test
        model.show_result(img_dir + img, result, score_thr=0.5, out_file='show_dirs/tunnel_rot/result_{}.jpg'.format(i))
