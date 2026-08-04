"""Stage 1: 12 epochs on all original train and validation keyframes."""

_base_ = ['base_codino_etf.py']

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min_ratio=0.05,
        convert_to_iter_based=True,
    ),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=2, save_last=True)
)
randomness = dict(seed=177070250)
work_dir = 'outputs/train_mmdet/buzzspot_stage1_trainval_12e'
