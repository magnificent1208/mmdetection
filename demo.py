from mmdet.apis import init_detector, inference_detector, show_result
import time

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = '/home/znjqr/mwh/mmdetection/configs/vhr/faster_rcnn_r50_fpn_1x.py'

# config_file = 'configs/rpn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/znjqr/mwh/mmdetection/work_dirs/vhr/faster_rcnn_r50_fpn_1x_vhr_2/latest.pth'

CLASSES = ('ignored regions', 'pedestrian','people', 'bicycle','car', 'van', 'truck', 
               'tricycle', 'awning-tricycle', 'bus', 'motor', 'others')

# 初始化模型
model = init_detector(config_file, checkpoint_file)
start_time = time.time()

# 测试一张图片
img = '1.jpg'
result = inference_detector(model, img)
# print(result)
show_result(img, result, CLASSES)

# 测试一系列图片
# imgs = ['002.jpg', '003.jpg', '004.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

print(time.time() - start_time)
