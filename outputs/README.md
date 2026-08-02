# Output layout

The public scripts write only to these canonical paths:

```text
outputs/train_mmdet/buzzspot_stage1_trainval_12e/
outputs/train_mmdet/buzzspot_stage2_cropbank_mosaic_3e/
outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/
outputs/submissions/buzzspot_final_single_model/
```

These directories are created by a fresh reproduction. They are not model or
result downloads and remain excluded from Git.

In the author workspace, `outputs/00_final_release/` is the short,
human-facing index to the verified winning artifacts. Its entries are local
symlinks because the original July run names are referenced by report evidence
and must remain stable.

All other challenge-campaign outputs are historical research state. They are
retained for provenance, excluded from Git, and are not part of the supported
rank-1 reproduction path.
