from mmdet.apis import init_detector, inference_detector
try:
    sys.path.remove('/home/maggie/anaconda3/envs/center/lib/python3.6/site-packages')
except:
    pass


config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')  