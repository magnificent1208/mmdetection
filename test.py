import torch
import torch.nn as nn

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# from mmdet.utils import quantize_net

import pdb as ipdb


config_file = 'configs/vis/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/vis/cascade_rcnn/latest.pth'
img = '001.jpg'


model = init_detector(config_file, checkpoint_file, device='cuda:0')

for n, module in model.named_modules():
    # ipdb.set_trace()
    print(module)


quantized_model = torch.quantization.quantize_dynamic(model, {nn.Conv2d}, dtype=torch.qint8)
# print_size_of_model(model)
# print_size_of_model(quantized_model)
model = quantized_model

result = inference_detector(model, img)
# show_result_pyplot(img, result, model.CLASSES)