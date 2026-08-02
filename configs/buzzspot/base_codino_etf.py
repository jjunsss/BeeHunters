"""Shared Co-DINO/Swin-L and fixed-ETF configuration for BuzzSpot."""

_base_ = [
    '../../external/mmdetection/projects/CO-DETR/configs/codino/'
    'co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'
]

custom_imports = dict(
    imports=['codetr', 'projects.buzzspot.co_dino_fixed_etf_head'],
    allow_failed_imports=False,
)

classes = ('bee', 'bumblebee', 'hoverfly', 'moth')
metainfo = dict(
    classes=classes,
    palette=[(255, 80, 80), (80, 180, 255), (255, 200, 80), (180, 120, 255)],
)
num_classes = len(classes)
num_decoder_layers = 6
auxiliary_weight = 2.0

model = dict(
    query_head=dict(
        type='CoDINOFixedETFHead',
        num_classes=num_classes,
        etf_loss_weight=0.1,
        etf_scale=1.0,
        etf_class_weight=[1.0, 1.5, 4.0, 2.5],
    ),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_decoder_layers * auxiliary_weight,
                ),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_decoder_layers * auxiliary_weight,
                ),
            ),
        )
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_decoder_layers * auxiliary_weight,
            ),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_decoder_layers * auxiliary_weight,
            ),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_decoder_layers * auxiliary_weight,
            ),
        )
    ],
)

image_root = 'data/extracted/BuzzSet_challenge/'
annotation_root = 'data/extracted/testphase/BuzzSpot_testphase/annotations/'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root='',
        ann_file=annotation_root + 'trainval_keyframes.json',
        data_prefix=dict(img=image_root),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root='',
        ann_file=annotation_root + 'valid_keyframes.json',
        data_prefix=dict(img=image_root + 'valid/'),
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=annotation_root + 'valid_keyframes.json', classwise=True
)
test_evaluator = val_evaluator

load_from = 'checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth'
optim_wrapper = dict(optimizer=dict(lr=1e-4))
