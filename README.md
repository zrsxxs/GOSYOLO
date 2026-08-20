# GOS-YOLO

Reference implementations for **GOS-YOLO: A Gradient and Orthogonal
Shape-Aware Detector for Multi-Class Road-Marking Detection**.

## Scope

This repository provides compact reference implementations of the four proposed
components and the GOS-YOLO architecture configuration. It is intended to show
the model structure and the principal tensor operations described in the paper.

The repository does not distribute the complete internal training framework,
dataset conversion utilities, trained checkpoints, experiment logs, batch
experiment scripts, or private datasets. The files are not a drop-in replacement
for an Ultralytics installation. Integration requires registering the custom
modules in the model parser and connecting the OSA-IoU component to the box loss.

## Proposed components

| Component | Position in GOS-YOLO | Reference implementation |
|---|---|---|
| SGD-Stem | Initial downsampling stage | `gos_yolo/modules/sgd_stem.py` |
| OD-SLU | Deepest backbone stage before SPPF | `gos_yolo/modules/od_slu.py` |
| GCAF | Two top-down fusion points | `gos_yolo/modules/gcaf.py` |
| OSA-IoU | Bounding-box regression component | `gos_yolo/losses/metrics.py` |

`models/GOS-YOLO.yaml` records the architecture used for the CeyMo experiments.
The class count may be overridden by an Ultralytics dataset configuration.

## Repository layout

```text
GOS-YOLO-Detector/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- models/
|   `-- GOS-YOLO.yaml
`-- gos_yolo/
    |-- __init__.py
    |-- modules/
    |   |-- __init__.py
    |   |-- sgd_stem.py
    |   |-- od_slu.py
    |   `-- gcaf.py
    `-- losses/
        |-- __init__.py
        |-- metrics.py
        `-- osa_loss.py
```

## Reference environment

The paper experiments used the following environment.

| Item | Version or configuration |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.8.0+cu129 |
| TorchVision | 0.23.0+cu129 |
| CUDA | 12.9 |
| Ultralytics | 8.3.9 for YOLO11 and GOS-YOLO |
| NumPy | 2.1.2 |
| OpenCV | 4.12.0 |
| pycocotools | 2.0.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |

## Model configuration

The public YAML uses the same locations reported in the paper.

- `SGD_Stem` replaces the layer-0 strided convolution.
- `OD_SLU` is inserted before SPPF.
- Two `GCAF` modules replace the top-down upsample-and-concatenate operations.
- The Detect head retains the three output scales of YOLO11.

The custom classes must be imported by the Ultralytics module registry before
the YAML can be parsed. GCAF receives a deep feature and a shallow feature, so
the parser must pass both input-channel counts to its constructor.

## OSA-IoU note

`metrics.py` implements the OSA-IoU score described in the paper. The calculation
uses ground-truth-dependent cross weights and treats the second box argument as
the target box.

`osa_loss.py` provides a small standalone wrapper around this score. It does not
contain the complete Ultralytics `BboxLoss` modification. In the reported
experiments, the OSA term was integrated into the YOLO11 box objective with an
OSA weight of 0.9 and a CIoU weight of 0.1. Classification and distribution focal
losses were unchanged.

## Minimal tensor check

The modules can be imported independently for tensor-shape checks.

```python
import torch

from gos_yolo.modules import GCAF, OD_SLU, SGD_Stem
from gos_yolo.losses import bbox_iou_osa

stem = SGD_Stem(3, 16).eval()
stem_output = stem(torch.randn(1, 3, 64, 64))

attention = OD_SLU(32, 32).eval()
attention_output = attention(torch.randn(1, 32, 20, 20))

fusion = GCAF(64, 32, 48).eval()
fusion_output = fusion((
    torch.randn(1, 64, 10, 10),
    torch.randn(1, 32, 20, 20),
))

boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
osa_score = bbox_iou_osa(boxes, boxes, xywh=False)
```

## Experimental setting reported in the paper

The controlled CeyMo comparison between YOLO11 and GOS-YOLO used the following
principal settings.

| Setting | Value |
|---|---|
| Input size | 640 x 640 |
| Epoch budget | 250 |
| Batch size | 32 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Momentum | 0.937 |
| Weight decay | 5e-4 |
| Early-stopping patience | 100 |
| Horizontal flip | Disabled |
| Vertical flip | Disabled |
| Main-run seed | 0 |

The test results reported for seed 0 were evaluated through a unified COCO
pipeline with a score floor of 0.001 and up to 300 detections per image.

| Model | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) | Params (M) | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11 | 95.56 | 85.45 | 88.98 | 65.16 | 2.58 | 6.30 | 227.9 |
| GOS-YOLO | 96.96 | 88.18 | 91.59 | 68.12 | 3.54 | 11.40 | 196.6 |

## Datasets

CeyMo is available from the benchmark authors at
<https://github.com/oshadajay/CeyMo>. The paper used the official 2,099-image
training portion and 788-image test split. The training portion was divided into
1,680 training images and 419 validation images.

TW-RM was obtained from its authors upon request and was used with permission.
The dataset and its annotations are not redistributed by this repository.

## Weights and experiment artifacts

Trained weights, private data, converted annotations, runtime outputs, and
internal experiment-management scripts are not distributed in this repository.

## License and attribution

This repository is distributed under the GNU Affero General Public License v3.0.
The reference implementation is designed for integration with Ultralytics, which
is also distributed under the AGPL-3.0 license. Users are responsible for meeting
the license terms of Ultralytics and other dependencies.

## Citation

```bibtex
@article{gosyolo2026,
  title   = {{GOS-YOLO}: A Gradient and Orthogonal Shape-Aware Detector for
             Multi-Class Road-Marking Detection},
  note    = {Manuscript under review},
  year    = {2026}
}
```
