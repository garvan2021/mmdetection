DATASET_ROOT_PATH = '/home/clover/MightyDataset'
auto_scale_lr = dict(base_batch_size=2, enable=False)
checkpoint_config = dict(interval=1)
classes = (
    'aluminium',
    'can',
    'capacitance',
    'car_plate',
    'copper',
    'foam',
    'painted_metal',
    'pcb',
    'rubber',
    'sponge',
    'tube',
    'wire',
)
custom_hooks = [
    dict(type='NumClassCheckHook'),
]
data = dict(
    test=dict(
        ann_file=[
            '/home/clover/MightyDataset/annotations/test.json',
        ],
        classes=(
            'aluminium',
            'can',
            'capacitance',
            'car_plate',
            'copper',
            'foam',
            'painted_metal',
            'pcb',
            'rubber',
            'sponge',
            'tube',
            'wire',
        ),
        img_prefix=[
            '/home/clover/MightyDataset/images/test',
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        separate_eval=True,
        type='CocoDataset'),
    test_dataloader=dict(workers_per_gpu=2),
    train=dict(
        dataset=dict(
            ann_file=[
                '/home/clover/MightyDataset/annotations/train.json',
            ],
            classes=(
                'aluminium',
                'can',
                'capacitance',
                'car_plate',
                'copper',
                'foam',
                'painted_metal',
                'pcb',
                'rubber',
                'sponge',
                'tube',
                'wire',
            ),
            img_prefix=[
                '/home/clover/MightyDataset/images/train',
            ],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(img_scale=(
                    800,
                    1333,
                ), keep_ratio=True, type='Resize'),
                dict(level=10, prob=0.6, type='Rotate'),
                dict(
                    policies=[
                        [
                            dict(
                                level=2, prob=0.5, type='BrightnessTransform'),
                            dict(level=2, prob=0.5, type='ContrastTransform'),
                        ],
                        [
                            dict(level=2, prob=0.5, type='ColorTransform'),
                        ],
                    ],
                    type='AutoAugment'),
                dict(min_gt_bbox_wh=(
                    0.01,
                    0.01,
                ), type='FilterAnnotations'),
                dict(flip_ratio=0.5, type='RandomFlip'),
                dict(size_divisor=32, type='Pad'),
            ],
            type='CocoDataset'),
        pipeline=[
            dict(max_num_pasted=100, type='CopyPaste'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(type='DefaultFormatBundle'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_labels',
                    'gt_masks',
                ],
                type='Collect'),
        ],
        type='MultiImageMixDataset'),
    train_dataloader=dict(samples_per_gpu=4, workers_per_gpu=2),
    val=dict(
        ann_file=[
            '/home/clover/MightyDataset/annotations/val.json',
        ],
        classes=(
            'aluminium',
            'can',
            'capacitance',
            'car_plate',
            'copper',
            'foam',
            'painted_metal',
            'pcb',
            'rubber',
            'sponge',
            'tube',
            'wire',
        ),
        img_prefix=[
            '/home/clover/MightyDataset/images/val',
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='CocoDataset'),
    val_dataloader=dict(workers_per_gpu=2))
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
dist_params = dict(backend='nccl')
evaluation = dict(metric=[
    'bbox',
    'segm',
])
fp16 = dict(loss_scale=dict(init_scale=512))
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
inference = [
    'bbox',
    'segm',
]
load_from = '/home/clover/MightyDataset/checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(img_scale=(
        800,
        1333,
    ), keep_ratio=True, type='Resize'),
    dict(level=10, prob=0.6, type='Rotate'),
    dict(
        policies=[
            [
                dict(level=2, prob=0.5, type='BrightnessTransform'),
                dict(level=2, prob=0.5, type='ContrastTransform'),
            ],
            [
                dict(level=2, prob=0.5, type='ColorTransform'),
            ],
        ],
        type='AutoAugment'),
    dict(min_gt_bbox_wh=(
        0.01,
        0.01,
    ), type='FilterAnnotations'),
    dict(flip_ratio=0.5, type='RandomFlip'),
    dict(size_divisor=32, type='Pad'),
]
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        27,
        33,
    ],
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    neck=dict(
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=12,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=12,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=28, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=56,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MaskRCNN')
mp_start_method = 'fork'
opencv_num_threads = 0
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='AdamW',
    weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
resume_from = None
runner = dict(max_epochs=70, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_pipeline = [
    dict(max_num_pasted=100, type='CopyPaste'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
        'gt_masks',
    ], type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
