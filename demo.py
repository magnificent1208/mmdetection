from mmdet.apis import init_detector, inference_detector, show_result
import time

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'configs/retinanet_x101_32x4d_fpn_1x.py'
checkpoint_file = 'checkpoints/retinanet_x101_32x4d_fpn_2x_20181218-605dcd0a.pth'



# 初始化模型
model = init_detector(config_file, checkpoint_file)
start_time = time.time()

# 测试一张图片
img = '003.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)

# 测试一系列图片
# imgs = ['002.jpg', '003.jpg', '004.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

print(time.time() - start_time)
