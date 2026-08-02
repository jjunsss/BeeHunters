# Checkpoints

Download the public Objects365-to-COCO Co-DINO/Swin-L initialization with:

```bash
bash scripts/download_checkpoints.sh pretrained
```

Expected file:

```text
checkpoints/co_dino_5scale_swin_large_16e_o365tococo.pth
SHA-256 614254c94b57acbff6a4448f7aa5d6315f8483b115438796e2606ce7a62712fe
```

A reproduced three-stage run writes its final training checkpoint to:

```text
outputs/train_mmdet/buzzspot_stage3_raw_cooldown_2e/epoch_2.pth
```

The archived winning training checkpoint has SHA-256
`057722df93fa011b980154acf94fa89e670b86486e6aa2bd3ce922309169b87f`.
Its sanitized, optimizer-free inference export is 943,564,029 bytes with
SHA-256
`12705c0395abc7dd1d6937555e344e63d5f203e65b22d0b4e0d9f96590dfe31f`.
It contains only `state_dict` and the public four-class `dataset_meta`.
The export still needs its public URL. It is intended to be uploaded as a
[GitHub Release asset](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases),
not committed to Git or Git LFS. Once the release exists, pass its asset URL
through `FINAL_CHECKPOINT_URL` to `scripts/download_checkpoints.sh final`.

See [`manifest.json`](manifest.json) for the machine-readable artifact paths,
sizes, hashes, and download commands.
