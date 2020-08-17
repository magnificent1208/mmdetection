from mmdet.apis import init_detector, inference_detector, show_result
import time


config_file = '/home/maggie/work/mmdetection/work_dirs/sim311/faster_rcnn_r50_fpn_1x/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '/home/maggie/work/mmdetection/work_dirs/sim311/faster_rcnn_r50_fpn_1x/epoch_20.pth'

CLASSES = ('ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
            'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs')

# 初始化模型
model = init_detector(config_file, checkpoint_file)
start_time = time.time()

# 测试一张图片
img = 'data/sim311/JPEGImages/DSC01062.jpg'
result = inference_detector(model, img)
show_result(img, result, CLASSES, out_file='test1.jpg')

# 测试一系列图片
# imgs = ['002.jpg', '003.jpg', '004.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
