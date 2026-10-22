# TinyML Framework: compression of a Faster R-CNN insulator detector for UAV inspection

An architecture-aware compression framework for PyTorch detectors and classifiers. It applies an ordered pipeline:
structured pruning (Taylor importance), recalibration of normalization statistics, knowledge distillation on
FPN features, and post-training INT8 quantization. It was built and evaluated on a Detectron2 Faster R-CNN
(ResNet-50-FPN) that detects insulators and their condition on UAV images (IDID V1.2), and it was also run on
ResNet-18 (Imagenette) to test that it is not tied to one architecture.

This repository belongs to an internship at Harmony Technology. Results are compared with a reference experimental
protocol by **retention**: the AP of the compressed model divided by the AP of the original detector.

## Main results (320 test images, one run unless stated)

| Model | Parameter reduction | AP (0 to 1) | Retention |
|---|---|---|---|
| Original detector M0 | 0 % | 0.7119 | 100 % |
| Whole backbone, critical modules protected | 41.8 % | 0.7199 | 101.1 % |
| Aggressive (mean of 3 seeds) | 57.6 % | 0.6101 | 85.7 % |
| Uniform allocation, same reduction | 57.6 % | 0.7032 | 98.8 % |
| Maximal | 61.7 % | 0.5751 | 80.8 % |
| Maximal, INT8 (stem kept in FP32), 18.2 MB | 61.7 % | 0.5712 | 80.2 % |

Retention above 100 % comes from extra training: the unpruned detector fine-tuned with the same recipe reaches 105.7 %.
At the same 57.6 % reduction, the uniform allocation beat both protected allocations in this study. ResNet-18 on
Imagenette, pruned through the framework: 95.0 % accuracy to 92.3 % after INT8, 44.8 MB to 2.19 MB.
All details, deviations from the reference protocol, and limits are in `rapport/TinyML_full_report_v12.tex` (or later).

## Repository layout

```
TinyML_Framework/
├── README.md
├── requirements.txt
├── tinyml/                     the framework
│   ├── adapters/               Detectron2, torchvision, plain PyTorch
│   ├── core/                   framework entry point, model analyzer, model registry
│   ├── dependency/             dependency graph and channel groups
│   ├── importance/             Taylor, magnitude, activation, hybrid, global ranker
│   ├── pruning/                policy, protection, structured pruner, validator
│   └── distillation/ evaluation/ quantization/ recovery/ visualization/
├── verify_m0_8000.py           gate: M0 must give AP 71.19
├── run_m1_report.py            M1 structured pruning
├── run_m2_report.py            M2 recalibration
├── run_m3_report.py            M3 distillation
├── run_m4_report.py            M4 INT8 quantization
├── evaluate_m1_report.py       evaluation of any checkpoint
├── analyze_errors.py           error analysis by class, size, and cause
├── measure_costs.py            parameters, size, GFLOPs, latency
├── run_resnet18_framework.py   ResNet-18 through the framework
├── run_resnet18_study.py       ResNet-18 with Torch-Pruning called directly (control)
├── benchmark_on_device.py      latency, memory, temperature on a target device
├── kaggle/                     cells to run the heavy steps on Kaggle GPUs
├── m1_report/                  results of the detector (logs, tables, figures; checkpoints not tracked)
├── r18_report/                 results of the ResNet-18 study
├── rapport/                    report sources and presentation text
├── docs/                       methodology notes, audit, file index, checkpoint hashes
└── tests/
```

## Installation

Python 3.11 or later and a CUDA GPU are recommended for training steps (CPU works for evaluation and quantization).

```
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

## Data and weights (not in this repository)

- **IDID V1.2 images and annotations** (`training_data.json`, `testing_data.json`): obtain them from the dataset owner
  and follow its licence. Set `TINYML_JSON_DIR` and `TINYML_IMAGES`.
- **Original detector M0** (8,000 iterations, SHA-256 `740FFEB4C02BE3658DE1E3CC486588BE16B83BC1150592DAE4D69C5984B9BB8E`):
  set `TINYML_M0` to its path. Every script checks this hash.
- Checkpoints (`*.pth`, about 70 MB each) are not tracked. Their hashes are in `docs/CHECKPOINTS_SHA256.csv`.

Environment variables used by the scripts: `TINYML_ROOT`, `TINYML_M0`, `TINYML_JSON_DIR`, `TINYML_IMAGES`,
`TINYML_FASTER`, `TINYML_DEVICE` (`cpu` to force the CPU).

## Reproducing the pipeline

```
python verify_m0_8000.py
python run_m1_report.py prune --config expose_whole_backbone --target-reduction 41.8 --clip rank
python run_m2_report.py recalibrate --m1 m1_report/checkpoints/M1R_expose_whole_backbone_t41.8_crank.pth
python run_m3_report.py distill --m2 m1_report/checkpoints/M2_expose_whole_backbone_t41.8_crank_matched.pth \
       --iters 1600 --threads 4 --min-size 640,704,768,800 --max-size 1333 --freeze-at 0 --tag v5pure
python run_m4_report.py quantize --m3 m1_report/checkpoints/M3_maximal_t61.7_crank_matched_v5pure.pth \
       --threads 4 --fp32-units stem
python evaluate_m1_report.py --checkpoint <path to a .pth>
```

The tag `v5pure` names the final recovery recipe: 1,600 iterations, training sizes 640 to 800 px, all layers
trainable, loss = detection loss + 0.5 x MSE on FPN levels P2 to P5. Pruning configurations are
`expose_residual` (18.4 %), `expose_internal` (31.5 %), `expose_whole_backbone` (41.8 %), `aggressive` (57.6 %),
`maximal` (61.7 %), `protect_head`, and `uniform_local`.

ResNet-18 study (downloads Imagenette and ImageNet weights, needs internet):

```
python run_resnet18_framework.py prepare --dataset imagenette --out r18_report --img-size 128
python run_resnet18_framework.py pipeline --engine framework --pipe-epochs 15 --dataset imagenette --out r18_report --img-size 128
```

## Using the framework on another model

```python
from tinyml.core.framework import TinyMLFramework
```

Register the model through an adapter, rank channels with an importance class and the global ranker, build a
`PruningPolicy`, and prune with `StructuredPruner`. `run_resnet18_framework.py` is a complete working example
(functions `make_framework`, `fw_register`, `prune_model_fw`, `fw_validate`).

## Running on an embedded device

The compressed model has not been measured on embedded hardware. Section "Embedded Platforms and Integration Path"
of the report describes the steps (quantize with the `qnnpack` backend, export to TorchScript or ONNX, parity check,
measurements). The script `benchmark_on_device.py` records latency, peak memory, temperature, and throttling:

```
python benchmark_on_device.py --model student_int8.ts --input 1 3 800 1200 --threads 4 --runs 30
```

## Known limits

One detector and one dataset for the detector study, 320 test images, a single seed for most configurations,
importance criteria compared on ResNet-18 only, no measurement on embedded hardware, and a stem kept in floating point
in the INT8 model (the reference quantizes it). The full list of declared deviations is in the report appendix.

## Traceability

A single M0 checkpoint is used everywhere (hash checked by every script). Every checkpoint has a JSON validation file.
No unmeasured value is replaced by an estimate.