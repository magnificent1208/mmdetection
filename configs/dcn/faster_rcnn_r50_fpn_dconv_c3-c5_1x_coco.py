_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4)

work_dir = './work_dirs/faster_rcnn_r50_dcn'