"""Stage 2: three-epoch class-aware CropBank Mosaic fine-tuning."""

_base_ = ['stage1_trainval_12e.py']

load_from = 'outputs/train_mmdet/buzzspot_stage1_trainval_12e/epoch_12.pth'
max_epochs = 3
base_lr = 1e-5

scales = [(size, 2048) for size in range(640, 1281, 32)]
metainfo = dict(
    classes=('bee', 'bumblebee', 'hoverfly', 'moth'),
    palette=[(255, 80, 80), (80, 180, 255), (255, 200, 80), (180, 120, 255)],
)
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root='',
            ann_file=(
                'data/extracted/testphase/BuzzSpot_testphase/annotations/'
                'trainval_keyframes_mosaic_cropbank_only_c256.json'
            ),
            data_prefix=dict(img='data/extracted/BuzzSet_challenge/'),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
        ),
        pipeline=[
            dict(
                type='Mosaic',
                img_scale=(1024, 1024),
                center_ratio_range=(0.85, 1.15),
                pad_val=114.0,
                prob=0.85,
            ),
            dict(type='PhotoMetricDistortion'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomChoiceResize', scales=scales, keep_ratio=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
            dict(type='PackDetInputs'),
        ],
    ),
)

optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min_ratio=0.10,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3, save_last=True)
)
randomness = dict(seed=328038036)
work_dir = 'outputs/train_mmdet/buzzspot_stage2_cropbank_mosaic_3e'
