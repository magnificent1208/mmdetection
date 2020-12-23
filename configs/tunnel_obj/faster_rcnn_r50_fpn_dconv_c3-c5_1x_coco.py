_base_ = './faster_rcnn_r50.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4)

optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/tunnel_obj/faster_rcnn_r50_dcn_stage2'
load_from = './work_dirs/tunnel_obj/faster_rcnn_r50_dcn/latest.pth'
