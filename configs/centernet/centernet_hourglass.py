model = dict(
    type='CenterNet',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=(256, 256, 384, 384, 384, 512),
        stage_blocks=(2, 2, 2, 2, 2, 4),
        feat_channel=256,
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CenterHead',
        num_classes=16,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        strides=[4, 4],
        regress_ranges=((-1, 64),(64, 1e8)),
        loss_hm=dict(type='CenterFocalLoss'),
        loss_wh=dict(type="SmoothL1Loss",loss_weight=0.5),
        loss_offset=dict(type="SmoothL1Loss",loss_weight=0.5),
        loss_rot=dict(type='SmoothL1Loss',loss_weight=2),
        K=100)
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
)
test_cfg = dict(
    a = 5
    #nms_pre=1000,
    #min_bbox_size=0,
    #score_thr=0.05,
    #nms=dict(type='nms', iou_thr=0.5),
    #max_per_img=100
)
# Dataset config
dataset_type = 'DotaDataset'
data_root = 'data/dota/'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[132.46, 137.14, 136.03], std=[76.54, 72.98, 76.86], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(768, 768), keep_ratio=True, is_rot=True),
    # dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True, is_rot=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape','img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, is_rot=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/ImageSets/Main/train.txt',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train/ImageSets/Main/val.txt',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'train/ImageSets/Main/test.txt',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='mAP', iou_thr=0.55)
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=1e-3, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 60])
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)
total_epochs = 70
# Runtime Setting
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dota/centernet_hourglass_0120'
load_from = None
resume_from = './work_dirs/dota/centernet_hourglass_0120/latest.pth'
workflow = [('train', 1)]
find_unused_parameters=True
