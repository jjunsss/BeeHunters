"""Stage 3: two low-LR epochs on the original train+valid keyframes."""

_base_ = ['stage1_trainval_12e.py']

load_from = 'outputs/train_mmdet/buzzspot_stage2_cropbank_mosaic_3e/epoch_3.pth'
max_epochs = 2
base_lr = 5e-6

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
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=2, save_last=True)
)
randomness = dict(seed=1384047161)
work_dir = 'outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e'
