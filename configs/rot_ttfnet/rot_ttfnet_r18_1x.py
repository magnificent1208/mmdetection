# model settings
model = dict(
    type='TTFNet',
    pretrained='modelzoo://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='ROT_TTFHead',
        inplanes=(64, 128, 256, 512),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=1,
        # num_classes=81,
        num_classes = 2, #class+1
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.))
# cudnn_benchmark = True

# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
# train_cfg = None
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
dataset_type = 'RAIRCRAFTDataset'
data_root = 'data/r_aircraft/'
img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], 
    # std=[58.395, 57.12, 57.375], to_rgb=True)
    mean = [0.5194416012442385,0.5378052387430711,0.533462090585746],
    std = [0.3001546018824507, 0.28620901391179554, 0.3014112676161966],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False, is_rot=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='Collect', 
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape','img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, is_rot=True),
            # dict(type='Resize', keep_ratio=False),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/train.txt',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=10000, metric='mAP',) #modify

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0004)
                #  paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 5,
    step=[60, 80])
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/rot_ttfnet/test_ttfnet18_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters=True
