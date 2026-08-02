# Third-party notices

This repository contains project-specific configuration, data preparation,
training orchestration, inference packaging, and the fixed-ETF extension. It
does not vendor the BuzzSpot dataset, upstream detector source, or model
weights.

## MMDetection and CO-DETR

The setup script clones [MMDetection](https://github.com/open-mmlab/mmdetection)
at commit
[`cfd5d3a985b0249de009b67d04f37263e11cdf3d`](https://github.com/open-mmlab/mmdetection/tree/cfd5d3a985b0249de009b67d04f37263e11cdf3d).
That checkout supplies MMDetection's CO-DETR project and Co-DINO
configuration. MMDetection is distributed under the
[Apache License 2.0](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/LICENSE).

The initialization checkpoint is downloaded from the MMDetection model zoo.
Its use remains subject to the upstream project and source-dataset terms.

## Python packages

PyTorch, torchvision, MMCV, MMEngine, MMDetection, NumPy, OpenCV, Pillow, and
pycocotools are installed as dependencies and retain their respective
licenses. They are not redistributed in this repository.

## BuzzSpot data

The two BuzzSpot archives are obtained from the challenge organizer and are
not included in Git history or model releases. Users are responsible for
complying with the organizer's dataset terms. The download script records the
organizer-provided locations solely to reproduce the challenge pipeline.

The `licenses` field in the organizer-provided train, validation, and
FinalTest COCO files declares
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This
repository does not grant additional rights to the dataset or determine the
license of derived model weights; confirm the release terms with the organizer
before publishing the final checkpoint.
